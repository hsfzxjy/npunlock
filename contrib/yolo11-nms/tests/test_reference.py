from __future__ import annotations

import unittest

import numpy as np

from yolo_nms.packing import (
    pack_candidates,
    quantize_candidates,
    unpack_detections,
    validate_ranked_candidates,
)
from yolo_nms.reference import DetectionBatch, reference_nms


def candidates(
    boxes: list[list[float]],
    scores: list[float],
    classes: list[int],
    indices: list[int] | None = None,
) -> DetectionBatch:
    count = len(scores)
    return DetectionBatch(
        boxes=np.asarray(boxes, dtype=np.float32).reshape(count, 4),
        scores=np.asarray(scores, dtype=np.float32),
        classes=np.asarray(classes, dtype=np.int32),
        indices=np.arange(count, dtype=np.int32) if indices is None else np.asarray(indices, dtype=np.int32),
    )


class ReferenceNmsTests(unittest.TestCase):
    def test_class_aware_suppression(self) -> None:
        value = candidates(
            [[0, 0, 10, 10], [1, 1, 9, 9], [1, 1, 9, 9]],
            [0.9, 0.8, 0.7],
            [2, 2, 3],
        )
        result = reference_nms(
            value,
            confidence_threshold=0.25,
            iou_threshold=0.5,
            max_detections=10,
        )
        np.testing.assert_array_equal(result.indices, [0, 2])

    def test_equal_scores_use_original_index(self) -> None:
        value = candidates(
            [[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]],
            [0.5, 0.5, 0.5],
            [0, 0, 0],
            [9, 2, 5],
        )
        result = reference_nms(
            value,
            confidence_threshold=0.25,
            iou_threshold=0.5,
            max_detections=10,
        )
        np.testing.assert_array_equal(result.indices, [2, 5, 9])

    def test_transport_round_trip(self) -> None:
        value = candidates([[1, 2, 3, 4]], [0.75], [79], [8400])
        packed = pack_candidates(
            value,
            confidence_threshold=0.25,
            iou_threshold=0.7,
            max_detections=10,
        )
        synthetic_output = packed.copy()
        synthetic_output[0, 0] = np.float16(1)
        actual = unpack_detections(synthetic_output)
        expected = quantize_candidates(value)
        np.testing.assert_array_equal(actual.boxes, expected.boxes)
        np.testing.assert_array_equal(actual.scores, expected.scores)
        np.testing.assert_array_equal(actual.classes, expected.classes)
        np.testing.assert_array_equal(actual.indices, expected.indices)

    def test_ranked_kernel_input_contract(self) -> None:
        validate_ranked_candidates(candidates([], [], []))
        validate_ranked_candidates(
            candidates([[0, 0, 1, 1], [2, 2, 3, 3]], [0.9, 0.8], [0, 0])
        )
        with self.assertRaisesRegex(ValueError, "ranked"):
            validate_ranked_candidates(
                candidates([[0, 0, 1, 1], [2, 2, 3, 3]], [0.8, 0.9], [0, 0])
            )


if __name__ == "__main__":
    unittest.main()
