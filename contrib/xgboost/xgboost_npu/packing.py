from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHUNK_WORDS = 2048
LOGICAL_WORDS = CHUNK_WORDS * 2
HEADER_WORDS = 4
OUTPUT_WORDS_PER_SAMPLE = 3


@dataclass(frozen=True, slots=True)
class Predictions:
    margins: np.ndarray
    probabilities: np.ndarray
    classes: np.ndarray


def batch_capacity(feature_count: int) -> int:
    if not isinstance(feature_count, int) or feature_count <= 0:
        raise ValueError("feature_count must be a positive integer")
    return (CHUNK_WORDS - HEADER_WORDS) // feature_count


def pack_features(features: object, *, feature_count: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != feature_count:
        raise ValueError(f"features must have shape (N, {feature_count})")
    if np.isinf(values).any():
        raise ValueError("features may contain finite values or NaN, but not infinity")
    if values.shape[0] > batch_capacity(feature_count):
        raise ValueError(
            f"batch size {values.shape[0]} exceeds invocation capacity {batch_capacity(feature_count)}"
        )
    quantized = values.astype(np.float16)
    if np.isinf(quantized).any():
        raise ValueError("a feature overflows the FP16 transport")
    chunk = np.zeros(CHUNK_WORDS, dtype=np.float16)
    chunk[0] = np.float16(values.shape[0])
    chunk[1] = np.float16(feature_count)
    chunk[HEADER_WORDS : HEADER_WORDS + quantized.size] = quantized.reshape(-1)
    return np.concatenate((chunk, chunk)).reshape(1, LOGICAL_WORDS)


def unpack_predictions(output: object, *, expected_samples: int, feature_count: int) -> Predictions:
    flat = np.asarray(output, dtype=np.float16).reshape(-1)
    if flat.size != LOGICAL_WORDS:
        raise ValueError(f"kernel output has {flat.size} words, expected {LOGICAL_WORDS}")
    first = flat[:CHUNK_WORDS]
    second = flat[CHUNK_WORDS:]
    if not np.array_equal(first.view(np.uint16), second.view(np.uint16)):
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
