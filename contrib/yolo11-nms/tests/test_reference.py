from __future__ import annotations

import unittest

import numpy as np

from yolo_nms.packing import (
    NMS_CAPACITY,
    pack_candidates,
    pack_nms_header,
    quantize_candidates,
    unpack_detections,
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
    def run_reference(self, value: DetectionBatch, *, confidence: float = 0.25, iou: float = 0.5, maximum: int = 10):
        return reference_nms(
            value,
            confidence_threshold=confidence,
            iou_threshold=iou,
            max_detections=maximum,
        )

    def test_no_detections(self) -> None:
        result = self.run_reference(candidates([[0, 0, 1, 1]], [0.1], [0]))
        self.assertEqual(len(result), 0)

    def test_one_detection(self) -> None:
        result = self.run_reference(candidates([[0, 0, 1, 1]], [0.9], [3], [17]))
        np.testing.assert_array_equal(result.indices, [17])

    def test_non_overlapping_boxes_survive(self) -> None:
        result = self.run_reference(candidates([[0, 0, 2, 2], [5, 5, 7, 7]], [0.8, 0.7], [0, 0]))
        np.testing.assert_array_equal(result.indices, [0, 1])

    def test_same_class_overlap_is_suppressed(self) -> None:
        result = self.run_reference(candidates([[0, 0, 10, 10], [1, 1, 9, 9]], [0.8, 0.7], [2, 2]))
        np.testing.assert_array_equal(result.indices, [0])

    def test_different_classes_are_preserved(self) -> None:
        result = self.run_reference(candidates([[0, 0, 10, 10], [1, 1, 9, 9]], [0.8, 0.7], [2, 3]))
        np.testing.assert_array_equal(result.indices, [0, 1])

    def test_equal_scores_use_original_index(self) -> None:
        result = self.run_reference(
            candidates([[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]], [0.5, 0.5, 0.5], [0, 0, 0], [9, 2, 5])
        )
        np.testing.assert_array_equal(result.indices, [2, 5, 9])

    def test_confidence_boundary_is_strict(self) -> None:
        value = candidates([[0, 0, 1, 1], [2, 2, 3, 3]], [0.5, 0.5009765625], [0, 0])
        result = self.run_reference(value, confidence=0.5)
        np.testing.assert_array_equal(result.indices, [1])

    def test_iou_boundary_is_strict(self) -> None:
        value = candidates([[0, 0, 2, 2], [0, 0, 1, 2]], [0.9, 0.8], [0, 0])
        result = self.run_reference(value, iou=0.5)
        np.testing.assert_array_equal(result.indices, [0, 1])

    def test_maximum_output_capacity(self) -> None:
        boxes = [[float(i * 2), 0, float(i * 2 + 1), 1] for i in range(NMS_CAPACITY)]
        scores = np.linspace(1.0, 0.5, NMS_CAPACITY, dtype=np.float32).tolist()
        value = candidates(boxes, scores, [0] * NMS_CAPACITY)
        result = self.run_reference(value, maximum=37)
        self.assertEqual(len(result), 37)

    def test_pack_round_trip_layout(self) -> None:
        value = candidates([[1, 2, 3, 4]], [0.75], [79], [8400])
        packed = pack_candidates(value, confidence_threshold=0.25, iou_threshold=0.7, max_detections=10)
        synthetic_output = packed.copy()
        synthetic_output[0, 0] = np.float16(1)
        result = unpack_detections(synthetic_output)
        expected = quantize_candidates(value)
        np.testing.assert_array_equal(result.boxes, expected.boxes)
        np.testing.assert_array_equal(result.scores, expected.scores)
        np.testing.assert_array_equal(result.classes, expected.classes)
        np.testing.assert_array_equal(result.indices, expected.indices)

    def test_single_graph_header(self) -> None:
        header = pack_nms_header(
            NMS_CAPACITY,
            confidence_threshold=0.25,
            iou_threshold=0.7,
            max_detections=100,
        )
        self.assertEqual(header.shape, (1, 4))
        self.assertEqual(header.dtype, np.float16)
        np.testing.assert_array_equal(
            header,
            np.asarray([[255, 0.25, np.float16(0.7), 100]], dtype=np.float16),
        )

    def test_single_graph_header_rejects_invalid_values(self) -> None:
        for candidate_count, confidence, iou, maximum in (
            (256, 0.25, 0.7, 100),
            (255, -0.1, 0.7, 100),
            (255, 0.25, 1.1, 100),
            (255, 0.25, 0.7, 256),
        ):
            with self.subTest(values=(candidate_count, confidence, iou, maximum)), self.assertRaises(ValueError):
                pack_nms_header(
                    candidate_count,
                    confidence_threshold=confidence,
                    iou_threshold=iou,
                    max_detections=maximum,
                )


if __name__ == "__main__":
    unittest.main()
