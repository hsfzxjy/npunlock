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
    ordered = np.asarray(order, dtype=np.int64)
    boxes = np.asarray(candidates.boxes, dtype=np.float64)
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    valid_boxes = (widths > 0.0) & (heights > 0.0)
    areas = widths * heights
    suppressed = np.zeros(len(candidates), dtype=np.bool_)
    selected: list[int] = []
    for position, index in enumerate(order):
        if suppressed[index]:
            continue
        selected.append(index)
        if len(selected) >= max_detections:
            break
        remaining = ordered[position + 1 :]
        comparable = remaining[
            (~suppressed[remaining])
            & (candidates.classes[remaining] == candidates.classes[index])
            & valid_boxes[remaining]
        ]
        if not valid_boxes[index] or comparable.size == 0:
            continue
        intersection_widths = np.minimum(boxes[index, 2], boxes[comparable, 2]) - np.maximum(
            boxes[index, 0], boxes[comparable, 0]
        )
        intersection_heights = np.minimum(boxes[index, 3], boxes[comparable, 3]) - np.maximum(
            boxes[index, 1], boxes[comparable, 1]
        )
        positive = (intersection_widths > 0.0) & (intersection_heights > 0.0)
        intersections = intersection_widths[positive] * intersection_heights[positive]
        unions = areas[index] + areas[comparable[positive]] - intersections
        suppress = np.zeros(comparable.size, dtype=np.bool_)
        suppress[positive] = (unions > 0.0) & (intersections / unions > iou_threshold)
        suppressed[comparable[suppress]] = True

    chosen = np.asarray(selected, dtype=np.int64)
    return DetectionBatch(
        boxes=np.asarray(candidates.boxes[chosen], dtype=np.float32),
        scores=np.asarray(candidates.scores[chosen], dtype=np.float32),
        classes=np.asarray(candidates.classes[chosen], dtype=np.int32),
        indices=np.asarray(candidates.indices[chosen], dtype=np.int32),
    )
