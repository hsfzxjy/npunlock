from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np

from yolo_nms.npunlock_backend import NpuNms
from yolo_nms.packing import NMS_CAPACITY, quantize_candidates
from yolo_nms.reference import DetectionBatch, reference_nms


ROOT = Path(__file__).resolve().parents[1]


def _case(
    boxes: list[list[float]],
    scores: list[float],
    classes: list[int],
    indices: list[int] | None = None,
) -> DetectionBatch:
    count = len(scores)
    return DetectionBatch(
        np.asarray(boxes, dtype=np.float32).reshape(count, 4),
        np.asarray(scores, dtype=np.float32),
        np.asarray(classes, dtype=np.int32),
        np.arange(count, dtype=np.int32) if indices is None else np.asarray(indices, dtype=np.int32),
    )


@unittest.skipUnless(os.environ.get("NPUNLOCK_ENABLE_NPU_TESTS") == "1", "NPU test is opt-in")
class NpuOperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        movitools = os.environ.get("NPUNLOCK_MOVITOOLS_DIR")
        if not movitools:
            raise unittest.SkipTest("NPUNLOCK_MOVITOOLS_DIR is not configured")
        cls.nms = NpuNms(
            kernel_path=ROOT / "kernels" / "nms.c",
            movi_dll_dir=movitools,
            native_dir=os.environ.get("NPUNLOCK_NATIVE_DIR"),
        )

    def assert_npu_matches(
        self,
        value: DetectionBatch,
        *,
        confidence: float = 0.25,
        iou: float = 0.5,
        maximum: int = 100,
    ) -> None:
        quantized = quantize_candidates(value)
        expected = reference_nms(
            quantized,
            confidence_threshold=float(np.float16(confidence)),
            iou_threshold=float(np.float16(iou)),
            max_detections=maximum,
        )
        actual, _ = self.nms.run(
            value,
            confidence_threshold=confidence,
            iou_threshold=iou,
            max_detections=maximum,
        )
        np.testing.assert_array_equal(actual.boxes, expected.boxes)
        np.testing.assert_array_equal(actual.scores, expected.scores)
        np.testing.assert_array_equal(actual.classes, expected.classes)
        np.testing.assert_array_equal(actual.indices, expected.indices)

    def test_required_operator_cases(self) -> None:
        cases = [
            (_case([], [], []), 0.25),
            (_case([[0, 0, 1, 1]], [0.9], [0]), 0.25),
            (_case([[0, 0, 2, 2], [5, 5, 7, 7]], [0.8, 0.7], [0, 0]), 0.25),
            (_case([[0, 0, 10, 10], [1, 1, 9, 9]], [0.8, 0.7], [2, 2]), 0.25),
            (_case([[0, 0, 10, 10], [1, 1, 9, 9]], [0.8, 0.7], [2, 3]), 0.25),
            (_case([[0, 0, 1, 1], [2, 2, 3, 3]], [0.5, 0.5], [0, 0], [9, 2]), 0.25),
            (_case([[0, 0, 1, 1], [2, 2, 3, 3]], [0.5, 0.5009765625], [0, 0]), 0.5),
            (_case([[0, 0, 2, 2], [0, 0, 1, 2]], [0.9, 0.8], [0, 0]), 0.25),
        ]
        for value, confidence in cases:
            with self.subTest(count=len(value)):
                self.assert_npu_matches(value, confidence=confidence)

    def test_fixed_capacity_and_maximum(self) -> None:
        boxes = [[float(i * 2), 0, float(i * 2 + 1), 1] for i in range(NMS_CAPACITY)]
        scores = np.linspace(1.0, 0.5, NMS_CAPACITY, dtype=np.float32)
        self.assert_npu_matches(_case(boxes, scores.tolist(), [0] * NMS_CAPACITY), maximum=37)


if __name__ == "__main__":
    unittest.main()
