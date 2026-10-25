from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ErrorMetrics:
    maximum_absolute: float
    mean_absolute: float
    maximum_relative: float


def _f32(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def softplus(value: object) -> np.ndarray:
    values = _f32(value)
    return np.maximum(values, np.float32(0.0)) + np.log1p(np.exp(-np.abs(values), dtype=np.float32))


def silu(value: object) -> np.ndarray:
    values = _f32(value)
    return values / (np.float32(1.0) + np.exp(-values, dtype=np.float32))


def selective_scan(
    u: object,
    delta: object,
    a: object,
    b: object,
    c: object,
    d: object,
    z: object,
    delta_bias: object,
    initial_state: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mamba-1 variable-B/C scan in official ``(channels, time)`` layout."""

    u_value = _f32(u)
    delta_value = _f32(delta)
    a_value = _f32(a)
    b_value = _f32(b)
    c_value = _f32(c)
    d_value = _f32(d)
    z_value = _f32(z)
    bias_value = _f32(delta_bias)
    if u_value.ndim != 2:
        raise ValueError("u must have shape (channels, sequence_length)")
    channels, sequence_length = u_value.shape
    state_size = a_value.shape[1] if a_value.ndim == 2 else 0
    expected = {
        "delta": (channels, sequence_length),
        "a": (channels, state_size),
        "b": (sequence_length, state_size),
        "c": (sequence_length, state_size),
        "d": (channels,),
        "z": (channels, sequence_length),
        "delta_bias": (channels,),
    }
    actual = {
        "delta": delta_value.shape,
        "a": a_value.shape,
        "b": b_value.shape,
        "c": c_value.shape,
        "d": d_value.shape,
        "z": z_value.shape,
        "delta_bias": bias_value.shape,
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} must have shape {shape}, got {actual[name]}")
    if initial_state is None:
        state = np.zeros((channels, state_size), dtype=np.float32)
    else:
        state = np.array(initial_state, dtype=np.float32, copy=True)
        if state.shape != (channels, state_size):
            raise ValueError(f"initial_state must have shape {(channels, state_size)}")
    transformed_delta = softplus(delta_value + bias_value[:, None])
    output = np.empty_like(u_value)
    for time_index in range(sequence_length):
        dt = transformed_delta[:, time_index, None]
        state = np.exp(dt * a_value, dtype=np.float32) * state + (
            dt * b_value[time_index][None, :] * u_value[:, time_index, None]
        )
        scan = np.sum(c_value[time_index][None, :] * state, axis=1, dtype=np.float32)
        output[:, time_index] = (scan + d_value * u_value[:, time_index]) * silu(z_value[:, time_index])
    return output, state


def error_metrics(actual: object, expected: object) -> ErrorMetrics:
    actual_value = _f32(actual)
    expected_value = _f32(expected)
    absolute = np.abs(actual_value - expected_value)
    denominator = np.maximum(np.abs(expected_value), np.float32(1.0e-6))
    return ErrorMetrics(float(absolute.max(initial=0.0)), float(absolute.mean()), float((absolute / denominator).max(initial=0.0)))
