from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HEADER_WORDS = 4
OUTPUT_WORDS_PER_SAMPLE = 3


@dataclass(frozen=True, slots=True)
class TransportSpec:
    name: str
    dtype: str
    numpy_dtype: np.dtype
    c_type: str
    chunk_elements: int
    logical_elements: int


FP16_TRANSPORT = TransportSpec("f16", "f16", np.dtype(np.float16), "__fp16", 2048, 4096)
FP32_TRANSPORT = TransportSpec("f32", "f32", np.dtype(np.float32), "float", 1024, 2048)
TRANSPORTS = {spec.name: spec for spec in (FP16_TRANSPORT, FP32_TRANSPORT)}

# Backward-compatible names for the original FP16 transport.
CHUNK_WORDS = FP16_TRANSPORT.chunk_elements
LOGICAL_WORDS = FP16_TRANSPORT.logical_elements


def transport_spec(transport: str | TransportSpec) -> TransportSpec:
    if isinstance(transport, TransportSpec):
        if TRANSPORTS.get(transport.name) != transport:
            raise ValueError("transport specification is not a supported built-in profile")
        return transport
    try:
        return TRANSPORTS[transport]
    except (KeyError, TypeError) as error:
        raise ValueError("transport must be 'f16' or 'f32'") from error


@dataclass(frozen=True, slots=True)
class Predictions:
    margins: np.ndarray
    probabilities: np.ndarray
    classes: np.ndarray


def batch_capacity(feature_count: int, *, transport: str | TransportSpec = "f16") -> int:
    if not isinstance(feature_count, int) or feature_count <= 0:
        raise ValueError("feature_count must be a positive integer")
    spec = transport_spec(transport)
    return (spec.chunk_elements - HEADER_WORDS) // feature_count


def pack_features(
    features: object,
    *,
    feature_count: int,
    transport: str | TransportSpec = "f16",
) -> np.ndarray:
    spec = transport_spec(transport)
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != feature_count:
        raise ValueError(f"features must have shape (N, {feature_count})")
    if np.isinf(values).any():
        raise ValueError("features may contain finite values or NaN, but not infinity")
    capacity = batch_capacity(feature_count, transport=spec)
    if values.shape[0] > capacity:
        raise ValueError(
            f"batch size {values.shape[0]} exceeds {spec.name} invocation capacity {capacity}"
        )
    quantized = values.astype(spec.numpy_dtype)
    if np.isinf(quantized).any():
        raise ValueError(f"a feature overflows the {spec.name.upper()} transport")
    chunk = np.zeros(spec.chunk_elements, dtype=spec.numpy_dtype)
    chunk[0] = values.shape[0]
    chunk[1] = feature_count
    chunk[HEADER_WORDS : HEADER_WORDS + quantized.size] = quantized.reshape(-1)
    return np.concatenate((chunk, chunk)).reshape(1, spec.logical_elements)


def unpack_predictions(
    output: object,
    *,
    expected_samples: int,
    feature_count: int,
    transport: str | TransportSpec = "f16",
) -> Predictions:
    spec = transport_spec(transport)
    flat = np.asarray(output, dtype=spec.numpy_dtype).reshape(-1)
    if flat.size != spec.logical_elements:
        raise ValueError(f"kernel output has {flat.size} elements, expected {spec.logical_elements}")
    first = flat[: spec.chunk_elements]
    second = flat[spec.chunk_elements :]
    byte_dtype = np.dtype(f"V{spec.numpy_dtype.itemsize}")
    if not np.array_equal(first.view(byte_dtype), second.view(byte_dtype)):
        raise ValueError("replicated ACT invocations produced different output bytes")
    header = tuple(int(float(value)) for value in first[:HEADER_WORDS])
    expected_header = (expected_samples, feature_count, OUTPUT_WORDS_PER_SAMPLE)
    if header[:3] != expected_header:
        raise ValueError(f"kernel output header {header[:3]!r} does not match {expected_header!r}")
    records = first[
        HEADER_WORDS : HEADER_WORDS + expected_samples * OUTPUT_WORDS_PER_SAMPLE
    ].reshape(expected_samples, OUTPUT_WORDS_PER_SAMPLE)
    classes = records[:, 2].astype(np.int32)
    if not np.isin(classes, (0, 1)).all():
        raise ValueError("kernel produced a non-binary class")
    return Predictions(
        margins=records[:, 0].astype(np.float32),
        probabilities=records[:, 1].astype(np.float32),
        classes=classes,
    )
