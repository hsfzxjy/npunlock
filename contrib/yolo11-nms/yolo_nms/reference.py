from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    boxes: np.ndarray
    scores: np.ndarray
    classes: np.ndarray
    indices: np.ndarray

    def __len__(self) -> int:
        return int(self.scores.size)


def preselect_raw_predictions(raw_predictions: np.ndarray, topk: int) -> DetectionBatch:
    """CPU reference for class selection, xywh conversion, and deterministic TopK."""

    raw = np.asarray(raw_predictions, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[1] != 84:
        raise ValueError(f"expected YOLO11 detection output (1, 84, N), received {raw.shape!r}")
    candidate_count = raw.shape[2]
    if not 0 < topk <= candidate_count:
        raise ValueError("topk must be between one and the candidate count")
    class_scores = raw[0, 4:, :]
    classes = np.argmax(class_scores, axis=0).astype(np.int32)
    scores = class_scores[classes, np.arange(candidate_count)]
    indices = np.arange(candidate_count, dtype=np.int32)
    order = np.lexsort((indices, -scores))[:topk]
    xywh = raw[0, :4, order]
    half_size = xywh[:, 2:] * np.float32(0.5)
    boxes = np.concatenate((xywh[:, :2] - half_size, xywh[:, :2] + half_size), axis=1)
    return DetectionBatch(
        boxes=np.asarray(boxes, dtype=np.float32),
        scores=np.asarray(scores[order], dtype=np.float32),
        classes=np.asarray(classes[order], dtype=np.int32),
        indices=np.asarray(indices[order], dtype=np.int32),
    )


def _validate_candidates(candidates: DetectionBatch) -> None:
    count = len(candidates)
    if candidates.boxes.shape != (count, 4):
        raise ValueError("boxes must have shape (N, 4)")
    if candidates.classes.shape != (count,) or candidates.indices.shape != (count,):
        raise ValueError("scores, classes, and indices must have shape (N,)")
    if not np.isfinite(candidates.boxes).all() or not np.isfinite(candidates.scores).all():
        raise ValueError("candidate boxes and scores must be finite")


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection_width = min(float(left[2]), float(right[2])) - max(float(left[0]), float(right[0]))
    intersection_height = min(float(left[3]), float(right[3])) - max(float(left[1]), float(right[1]))
    left_width = float(left[2]) - float(left[0])
    left_height = float(left[3]) - float(left[1])
    right_width = float(right[2]) - float(right[0])
    right_height = float(right[3]) - float(right[1])
    if (
        intersection_width <= 0.0
        or intersection_height <= 0.0
        or left_width <= 0.0
        or left_height <= 0.0
        or right_width <= 0.0
        or right_height <= 0.0
    ):
        return 0.0
    intersection = intersection_width * intersection_height
    union = left_width * left_height + right_width * right_height - intersection
    return intersection / union if union > 0.0 else 0.0


def reference_nms(
    candidates: DetectionBatch,
    *,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> DetectionBatch:
    """Deterministic class-aware greedy NMS matching kernels/nms.c."""

    _validate_candidates(candidates)
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if max_detections < 0:
        raise ValueError("max_detections must be nonnegative")
    if max_detections == 0:
        return DetectionBatch(
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            classes=np.empty((0,), dtype=np.int32),
            indices=np.empty((0,), dtype=np.int32),
        )

    eligible = np.flatnonzero(candidates.scores > confidence_threshold)
    order = sorted(
        (int(index) for index in eligible),
        key=lambda index: (-float(candidates.scores[index]), int(candidates.indices[index]), index),
    )
    suppressed = np.zeros(len(candidates), dtype=np.bool_)
    selected: list[int] = []
    for index in order:
        if suppressed[index]:
            continue
        selected.append(index)
        if len(selected) >= max_detections:
            break
        for candidate in order:
            if candidate == index or suppressed[candidate]:
                continue
            if candidates.classes[candidate] != candidates.classes[index]:
                continue
            if _iou(candidates.boxes[index], candidates.boxes[candidate]) > iou_threshold:
                suppressed[candidate] = True

    chosen = np.asarray(selected, dtype=np.int64)
    return DetectionBatch(
        boxes=np.asarray(candidates.boxes[chosen], dtype=np.float32),
        scores=np.asarray(candidates.scores[chosen], dtype=np.float32),
        classes=np.asarray(candidates.classes[chosen], dtype=np.int32),
        indices=np.asarray(candidates.indices[chosen], dtype=np.int32),
    )
