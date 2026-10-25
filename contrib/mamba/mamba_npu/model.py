from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .reference import ErrorMetrics, error_metrics
from .runtime import NpuSelectiveScan


@dataclass(frozen=True, slots=True)
class BlockResult:
    output: object
    final_state: np.ndarray
    convolution: object
    delta: np.ndarray
    b: np.ndarray
    c: np.ndarray
    scan: np.ndarray
    timings_ms: dict[str, float]


@dataclass(slots=True)
class ModelCache:
    convolution_states: list[object]
    ssm_states: list[np.ndarray]


def load_official_model(model_dir: str | Path):
    """Load the pinned Hugging Face checkpoint through its official CPU module."""

    from transformers import MambaForCausalLM

    model = MambaForCausalLM.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()
    return model


def mixer_parameters(mixer) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = -np.exp(mixer.A_log.detach().cpu().numpy().astype(np.float32))
    d = mixer.D.detach().cpu().numpy().astype(np.float32)
    delta_bias = mixer.dt_proj.bias.detach().cpu().numpy().astype(np.float32)
    return a, d, delta_bias


def prepare_scan_inputs(mixer, input_states, conv_state=None):
    """Run the official mixer up to, but not including, selective recurrence."""

    import torch
    import torch.nn.functional as functional

    with torch.no_grad():
        projected = mixer.in_proj(input_states).transpose(1, 2)
        hidden, gate = projected.chunk(2, dim=1)
        sequence_length = input_states.shape[1]
        if conv_state is None:
            convolution = mixer.act(mixer.conv1d(hidden)[..., :sequence_length])
            next_conv_state = functional.pad(hidden, (mixer.conv_kernel_size - sequence_length, 0))
        else:
            if sequence_length != 1:
                raise ValueError("an explicit convolution state supports one-token decoding only")
            next_conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
            next_conv_state[:, :, -1] = hidden[:, :, 0]
            convolution = torch.sum(next_conv_state * mixer.conv1d.weight[:, 0, :], dim=-1)
            if mixer.use_conv_bias:
                convolution = convolution + mixer.conv1d.bias
            convolution = mixer.act(convolution).unsqueeze(-1)
        parameters = mixer.x_proj(convolution.transpose(1, 2))
        time_step, b_value, c_value = torch.split(
            parameters,
            [mixer.time_step_rank, mixer.ssm_state_size, mixer.ssm_state_size],
            dim=-1,
        )
        # Keep the bias and softplus in the ACT kernel, matching the fused official path.
        raw_delta = functional.linear(time_step, mixer.dt_proj.weight, bias=None).transpose(1, 2)
    return convolution, gate, raw_delta, b_value, c_value, next_conv_state


def run_mixer_with_npu_scan(mixer, input_states, runtime: NpuSelectiveScan, *, initial_state=None, conv_state=None):
    import torch

    if input_states.shape[0] != 1:
        raise ValueError("the showcase currently supports batch size one")
    started = perf_counter()
    convolution, gate, raw_delta, b_value, c_value, next_conv_state = prepare_scan_inputs(
        mixer, input_states, conv_state
    )
    preparation_ms = (perf_counter() - started) * 1000.0
    u = convolution[0].detach().cpu().numpy().astype(np.float32)
    delta = raw_delta[0].detach().cpu().numpy().astype(np.float32)
    b_array = b_value[0].detach().cpu().numpy().astype(np.float32)
    c_array = c_value[0].detach().cpu().numpy().astype(np.float32)
    z = gate[0].detach().cpu().numpy().astype(np.float32)
    scan, final_state, scan_timings = runtime.run_sequence(u, delta, b_array, c_array, z, initial_state)
    started = perf_counter()
    scan_tensor = torch.from_numpy(scan).to(dtype=input_states.dtype).unsqueeze(0)
    output = mixer.out_proj(scan_tensor.transpose(1, 2))
    projection_ms = (perf_counter() - started) * 1000.0
    timings = {"cpu_preparation_ms": preparation_ms, "cpu_output_projection_ms": projection_ms, **scan_timings}
    return (
        BlockResult(
            output,
            final_state,
            convolution,
            delta,
            b_array,
            c_array,
            scan,
            timings,
        ),
        next_conv_state,
    )


def run_block_with_npu_scan(layer, hidden_states, runtime: NpuSelectiveScan, *, initial_state=None, conv_state=None):
    import torch

    with torch.no_grad():
        residual = hidden_states.to(torch.float32)
        normalized = layer.norm(hidden_states.to(dtype=layer.norm.weight.dtype))
        mixer_result, next_conv_state = run_mixer_with_npu_scan(
            layer.mixer,
            normalized,
            runtime,
            initial_state=initial_state,
            conv_state=conv_state,
        )
        output = residual + mixer_result.output
    return output, mixer_result, next_conv_state


def compare_tensors(actual: object, expected: object) -> ErrorMetrics:
    try:
        actual = actual.detach().cpu().numpy()
    except AttributeError:
        pass
    try:
        expected = expected.detach().cpu().numpy()
    except AttributeError:
        pass
    return error_metrics(actual, expected)


class HybridMambaRunner:
    """Official Mamba-130m weights with every selective recurrence on ACT-SHAVE.

    Embeddings, normalization, projections, depthwise convolution, and the LM
    head remain PyTorch CPU operations. The recurrent state update, D skip, and
    z gate execute in the custom NPU kernel for every layer.
    """

    def __init__(self, model, runtime: NpuSelectiveScan):
        self.model = model
        self.runtime = runtime

    def _forward(self, input_ids, cache: ModelCache | None = None):
        import torch

        with torch.no_grad():
            hidden = self.model.backbone.embeddings(input_ids)
            next_convolution = []
            next_ssm = []
            totals = {
                "model_upload_ms": 0.0,
                "cpu_preparation_ms": 0.0,
                "cpu_output_projection_ms": 0.0,
                "upload_ms": 0.0,
                "execution_wall_ms": 0.0,
                "download_ms": 0.0,
                "submissions": 0.0,
            }
            for layer_index, layer in enumerate(self.model.backbone.layers):
                a, d, bias = mixer_parameters(layer.mixer)
                totals["model_upload_ms"] += self.runtime.load_model(a, d, bias)
                initial_state = None if cache is None else cache.ssm_states[layer_index]
                conv_state = None if cache is None else cache.convolution_states[layer_index]
                hidden, details, next_conv = run_block_with_npu_scan(
                    layer,
                    hidden,
                    self.runtime,
                    initial_state=initial_state,
                    conv_state=conv_state,
                )
                next_convolution.append(next_conv.detach().clone())
                next_ssm.append(details.final_state.copy())
                for name, value in details.timings_ms.items():
                    totals[name] += value
            hidden = self.model.backbone.norm_f(hidden)
            logits = self.model.lm_head(hidden)
        return logits, ModelCache(next_convolution, next_ssm), totals

    def prefill(self, input_ids):
        return self._forward(input_ids)

    def decode(self, token_id, cache: ModelCache):
        return self._forward(token_id, cache)

    def generate_greedy(self, input_ids, new_tokens: int):
        import torch

        if not isinstance(new_tokens, int) or new_tokens < 1:
            raise ValueError("new_tokens must be positive")
        logits, cache, totals = self.prefill(input_ids)
        generated = []
        for step in range(new_tokens):
            token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(token)
            if step + 1 < new_tokens:
                logits, cache, timings = self.decode(token, cache)
                for name, value in timings.items():
                    totals[name] += value
        return torch.cat((input_ids, *generated), dim=1), totals
