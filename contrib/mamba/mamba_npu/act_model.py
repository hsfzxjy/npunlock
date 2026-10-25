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
        "linear_execution_wall_ms": 0.0,
        "scan_execution_wall_ms": 0.0,
        "fused_rms_norm_calls": 0.0,
        "fused_convolution_calls": 0.0,
    }


def _add_timings(target: dict[str, float], source: dict[str, float]) -> None:
    for name, value in source.items():
        target[name] = target.get(name, 0.0) + value


class ActMambaRunner:
    """One-token Mamba-130m execution with all neural arithmetic on one ACT graph."""

    def __init__(self, model: object, runtime: NpuActLinear):
        self.model = model
        self.runtime = runtime

    def step(
        self, token_id: int, cache: ActModelCache | None = None
    ) -> tuple[np.ndarray, ActModelCache, dict[str, float]]:
        import torch

        with torch.no_grad():
            token = torch.tensor([[token_id]], dtype=torch.int64)
            hidden = self.model.backbone.embeddings(token).squeeze(1).numpy()
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
            totals["linear_submissions"] += timing["submissions"]
            totals["linear_execution_wall_ms"] += timing["execution_wall_ms"]
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
            totals["linear_submissions"] += timing["submissions"]
            totals["linear_execution_wall_ms"] += timing["execution_wall_ms"]
            time_step = parameters[:, : mixer.time_step_rank]
            b_value = parameters[:, mixer.time_step_rank : mixer.time_step_rank + mixer.ssm_state_size]
            c_value = parameters[:, -mixer.ssm_state_size :]
            delta, timing = self.runtime.run(time_step, mixer.dt_proj.weight.detach().numpy().T)
            totals["linear_submissions"] += timing["submissions"]
            totals["linear_execution_wall_ms"] += timing["execution_wall_ms"]
            a, d, bias = mixer_parameters(mixer)
            ssm_state = None if cache is None else cache.ssm_states[layer_index]
            scan, ssm_state, timing = self.runtime.run_scan(
                a, d, bias, convolution.T, delta.T, b_value, c_value, z.T, ssm_state
            )
            next_ssm.append(ssm_state)
            totals["scan_execution_wall_ms"] += timing["execution_wall_ms"]
            hidden, timing = self.runtime.run(
                scan.T, mixer.out_proj.weight.detach().numpy().T, residual=residual
            )
            totals["linear_submissions"] += timing["submissions"]
            totals["linear_execution_wall_ms"] += timing["execution_wall_ms"]
        logits, timing = self.runtime.run(
            hidden,
            self.model.lm_head.weight.detach().numpy().T,
            rms_weight=self.model.backbone.norm_f.weight.detach().numpy(),
            rms_epsilon=self.model.backbone.norm_f.variance_epsilon,
        )
        totals["fused_rms_norm_calls"] += 1.0
        totals["linear_submissions"] += timing["submissions"]
        totals["linear_execution_wall_ms"] += timing["execution_wall_ms"]
        return logits, ActModelCache(next_convolution, next_ssm), totals

    def prefill(self, input_ids: object) -> tuple[np.ndarray, ActModelCache, dict[str, float]]:
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError("input_ids cannot be empty")
        cache = None
        totals = _empty_timings()
        logits = None
        for token_id in ids:
            logits, cache, timing = self.step(int(token_id), cache)
            _add_timings(totals, timing)
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
                _add_timings(totals, timing)
        return ids, totals
