from __future__ import annotations

import numpy as np

from .reference import DetectionBatch

NMS_CHUNK_WORDS = 2048
NMS_HEADER_WORDS = 8
NMS_RECORD_WORDS = 8
NMS_CAPACITY = 255
NMS_LOGICAL_WORDS = NMS_CHUNK_WORDS * 2


def pack_nms_header(
    candidate_count: int,
    *,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> np.ndarray:
    if not isinstance(candidate_count, (int, np.integer)):
        raise TypeError("candidate_count must be an integer")
    if not 0 <= candidate_count <= NMS_CAPACITY:
        raise ValueError(f"candidate_count must be in [0, {NMS_CAPACITY}]")
    if not 0.0 <= confidence_threshold <= 1.0 or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("confidence and IoU thresholds must be in [0, 1]")
    if not isinstance(max_detections, (int, np.integer)):
        raise TypeError("max_detections must be an integer")
    if not 0 <= max_detections <= NMS_CAPACITY:
        raise ValueError(f"max_detections must be in [0, {NMS_CAPACITY}]")
    return np.asarray(
        [[candidate_count, confidence_threshold, iou_threshold, max_detections]],
        dtype=np.float16,
    )


def quantize_candidates(candidates: DetectionBatch) -> DetectionBatch:
    """Apply the exact FP16 transport quantization used by the ACT kernel."""

    return DetectionBatch(
        boxes=np.asarray(candidates.boxes, dtype=np.float16).astype(np.float32),
        scores=np.asarray(candidates.scores, dtype=np.float16).astype(np.float32),
        classes=np.asarray(candidates.classes, dtype=np.int32),
        indices=np.asarray(candidates.indices, dtype=np.int32),
    )


def pack_candidates(
    candidates: DetectionBatch,
    *,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> np.ndarray:
    count = len(candidates)
    if count > NMS_CAPACITY:
        raise ValueError(f"candidate count {count} exceeds fixed capacity {NMS_CAPACITY}")
    if candidates.boxes.shape != (count, 4):
        raise ValueError("boxes must have shape (N, 4)")
    if (
        candidates.scores.shape != (count,)
        or candidates.classes.shape != (count,)
        or candidates.indices.shape != (count,)
    ):
        raise ValueError("scores, classes, and indices must have shape (N,)")
    if not np.isfinite(candidates.boxes).all() or not np.isfinite(candidates.scores).all():
        raise ValueError("candidate boxes and scores must be finite")
    header = pack_nms_header(
        count,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
    )
    if not np.isfinite(candidates.classes).all() or not np.equal(
        candidates.classes, np.floor(candidates.classes)
    ).all():
        raise ValueError("class IDs must be finite integers")
    if not np.isfinite(candidates.indices).all() or not np.equal(
        candidates.indices, np.floor(candidates.indices)
    ).all():
        raise ValueError("original indices must be finite integers")
    if np.any(candidates.classes < 0) or np.any(candidates.classes > 255):
        raise ValueError("class IDs must be in [0, 255]")
    if np.any(candidates.indices < 0) or np.any(candidates.indices > 65535):
        raise ValueError("original indices must be in [0, 65535]")

    chunk = np.zeros(NMS_CHUNK_WORDS, dtype=np.float16)
    chunk[:4] = header[0]
    records = chunk[NMS_HEADER_WORDS:].reshape(NMS_CAPACITY, NMS_RECORD_WORDS)
    records[:count, :4] = np.asarray(candidates.boxes, dtype=np.float16)
    records[:count, 4] = np.asarray(candidates.scores, dtype=np.float16)
    records[:count, 5] = np.asarray(candidates.classes, dtype=np.float16)
    indices = np.asarray(candidates.indices, dtype=np.uint16)
    records[:count, 6] = (indices & np.uint16(0xFF)).astype(np.float16)
    records[:count, 7] = (indices >> np.uint16(8)).astype(np.float16)
    return np.concatenate((chunk, chunk)).reshape(1, NMS_LOGICAL_WORDS)


def unpack_detections(output: object) -> DetectionBatch:
    flat = np.asarray(output, dtype=np.float16).reshape(-1)
    if flat.size != NMS_LOGICAL_WORDS:
        raise ValueError(f"kernel output has {flat.size} words, expected {NMS_LOGICAL_WORDS}")
    chunk = flat[:NMS_CHUNK_WORDS]
    count = int(float(chunk[0]))
    if not 0 <= count <= NMS_CAPACITY:
        raise ValueError(f"kernel selected count {count} is outside the fixed capacity")
    records = chunk[NMS_HEADER_WORDS:].reshape(NMS_CAPACITY, NMS_RECORD_WORDS)[:count]
    low = records[:, 6].astype(np.uint16).astype(np.uint32)
    high = records[:, 7].astype(np.uint16).astype(np.uint32)
    return DetectionBatch(
        boxes=records[:, :4].astype(np.float32),
        scores=records[:, 4].astype(np.float32),
        classes=records[:, 5].astype(np.int32),
        indices=(low + high * np.uint32(256)).astype(np.int32),
    )
