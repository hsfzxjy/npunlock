from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .act_linear import NpuActLinear
from .model import mixer_parameters


@dataclass(slots=True)
class ActModelCache:
    convolution_states: list[np.ndarray]
    ssm_states: list[np.ndarray]

    def clone(self) -> "ActModelCache":
        return ActModelCache(
            [value.copy() for value in self.convolution_states],
            [value.copy() for value in self.ssm_states],
        )


def _empty_timings() -> dict[str, float]:
    return {
        "linear_submissions": 0.0,
        "linear_upload_ms": 0.0,
        "linear_execution_wall_ms": 0.0,
        "linear_download_ms": 0.0,
        "scan_execution_wall_ms": 0.0,
        "fused_rms_norm_calls": 0.0,
        "fused_convolution_calls": 0.0,
    }


def _add_timings(target: dict[str, float], source: dict[str, float]) -> None:
    for name, value in source.items():
        target[name] = target.get(name, 0.0) + value


def _add_linear_timings(
    target: dict[str, float], timing: dict[str, float], category: str | None = None
) -> None:
    target["linear_submissions"] += timing["submissions"]
    target["linear_upload_ms"] += timing["upload_ms"]
    target["linear_execution_wall_ms"] += timing["execution_wall_ms"]
    target["linear_download_ms"] += timing["download_ms"]
    if category is not None:
        for name in ("submissions", "upload_ms", "execution_wall_ms", "download_ms"):
            key = f"{category}_{name}"
            target[key] = target.get(key, 0.0) + timing[name]


class ActMambaRunner:
    """Batched-prefill Mamba-130m execution with neural arithmetic on one ACT graph."""

    def __init__(self, model: object, runtime: NpuActLinear):
        self.model = model
        self.runtime = runtime
        self._ssm_parameters = [
            mixer_parameters(layer.mixer) for layer in model.backbone.layers
        ]

    def step(
        self, token_id: int, cache: ActModelCache | None = None, *, compute_logits: bool = True
    ) -> tuple[np.ndarray | None, ActModelCache, dict[str, float]]:
        import torch

        with torch.no_grad():
            token = torch.tensor([[token_id]], dtype=torch.int64)
            hidden = self.model.backbone.embeddings(token).squeeze(1).numpy()
        return self._forward_hidden(hidden, cache, compute_logits=compute_logits)

    def _forward_hidden(
        self, hidden: np.ndarray, cache: ActModelCache | None, *, compute_logits: bool
    ) -> tuple[np.ndarray | None, ActModelCache, dict[str, float]]:
        totals = _empty_timings()
        next_convolution = []
        next_ssm = []
        for layer_index, layer in enumerate(self.model.backbone.layers):
            mixer = layer.mixer
            residual = hidden
            projected, timing = self.runtime.run(
                hidden,
                mixer.in_proj.weight.detach().numpy().T,
                rms_weight=layer.norm.weight.detach().numpy(),
                rms_epsilon=layer.norm.variance_epsilon,
            )
            totals["fused_rms_norm_calls"] += 1.0
            _add_linear_timings(totals, timing, "in_proj")
            x, z = np.split(projected, 2, axis=-1)
            conv_state = None if cache is None else cache.convolution_states[layer_index]
            parameters, convolution, conv_state, timing = self.runtime.run_convolution(
                x,
                mixer.conv1d.weight.detach().numpy(),
                mixer.conv1d.bias.detach().numpy(),
                mixer.x_proj.weight.detach().numpy().T,
                conv_state,
            )
            next_convolution.append(conv_state)
            totals["fused_convolution_calls"] += 1.0
            _add_linear_timings(totals, timing, "conv_x_proj")
            time_step = parameters[:, : mixer.time_step_rank]
            b_value = parameters[:, mixer.time_step_rank : mixer.time_step_rank + mixer.ssm_state_size]
            c_value = parameters[:, -mixer.ssm_state_size :]
            a, d, bias = self._ssm_parameters[layer_index]
            ssm_state = None if cache is None else cache.ssm_states[layer_index]
            scan, ssm_state, timing = self.runtime.run_scan_projected_delta(
                a,
                d,
                bias,
                mixer.dt_proj.weight.detach().numpy().T,
                time_step,
                convolution.T,
                b_value,
                c_value,
                z.T,
                ssm_state,
            )
            next_ssm.append(ssm_state)
            totals["scan_execution_wall_ms"] += timing["execution_wall_ms"]
            hidden, timing = self.runtime.run(
                scan.T, mixer.out_proj.weight.detach().numpy().T, residual=residual
            )
            _add_linear_timings(totals, timing, "out_proj")
        logits = None
        if compute_logits:
            logits, timing = self.runtime.run(
                hidden[-1:],
                self.model.lm_head.weight.detach().numpy().T,
                rms_weight=self.model.backbone.norm_f.weight.detach().numpy(),
                rms_epsilon=self.model.backbone.norm_f.variance_epsilon,
            )
            totals["fused_rms_norm_calls"] += 1.0
            _add_linear_timings(totals, timing, "lm_head")
        return logits, ActModelCache(next_convolution, next_ssm), totals

    def prefill(self, input_ids: object) -> tuple[np.ndarray, ActModelCache, dict[str, float]]:
        import torch

        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("input_ids cannot be empty")
        cache = None
        totals = _empty_timings()
        logits = None
        for begin in range(0, ids.size, 16):
            end = min(begin + 16, ids.size)
            with torch.no_grad():
                tokens = torch.from_numpy(ids[begin:end].copy()).reshape(1, -1)
                hidden = self.model.backbone.embeddings(tokens).squeeze(0).numpy()
            logits, cache, timing = self._forward_hidden(
                hidden, cache, compute_logits=end == ids.size
            )
            _add_timings(totals, timing)
        if logits is None:
            raise RuntimeError("final prefill token did not produce logits")
        return np.asarray(logits), cache, totals

    def generate_greedy(
        self, input_ids: object, new_tokens: int
    ) -> tuple[list[int], dict[str, float]]:
        if not isinstance(new_tokens, int) or new_tokens < 1:
            raise ValueError("new_tokens must be positive")
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1).tolist()
        logits, cache, totals = self.prefill(ids)
        for index in range(new_tokens):
            next_token = int(np.argmax(logits[0]))
            ids.append(next_token)
            if index + 1 < new_tokens:
                logits, cache, timing = self.step(next_token, cache)
                if logits is None:
                    raise RuntimeError("decode token did not produce logits")
                _add_timings(totals, timing)
        return ids, totals
