from __future__ import annotations

import os
from pathlib import Path
import unittest

import numpy as np
import openvino as ov

from yolo_nms.npunlock_backend import NpuNms
from yolo_nms.openvino_backend import CompiledModel, NpuYoloPreselector, letterbox, load_image
from yolo_nms.reference import preselect_raw_predictions, reference_nms
from yolo_nms.single_graph_backend import SingleGraphYoloNms


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("NPUNLOCK_ENABLE_NPU_TESTS") == "1", "NPU test is opt-in")
class YoloEndToEndTests(unittest.TestCase):
    def test_real_yolo11_predictions_match(self) -> None:
        model_dir_value = os.environ.get("YOLO11_MODEL_DIR")
        image_value = os.environ.get("YOLO11_SAMPLE_IMAGE")
        image_dir_value = os.environ.get("YOLO11_SAMPLE_DIR")
        movitools = os.environ.get("NPUNLOCK_MOVITOOLS_DIR")
        if not model_dir_value or not (image_value or image_dir_value) or not movitools:
            raise unittest.SkipTest(
                "YOLO11_MODEL_DIR, NPUNLOCK_MOVITOOLS_DIR, and YOLO11_SAMPLE_IMAGE or YOLO11_SAMPLE_DIR are required"
            )
        model_dir = Path(model_dir_value)
        image_paths = (
            sorted(Path(image_dir_value).glob("*.jpg"))
            if image_dir_value
            else [Path(image_value)]  # type: ignore[arg-type]
        )
        if not image_paths:
            raise unittest.SkipTest("the configured sample directory contains no JPEG images")
        core = ov.Core()
        raw_model = CompiledModel(core, model_dir / "yolo11n_raw.onnx", "NPU")
        self.assertEqual(raw_model.execution_devices, "NPU")
        yolo_preselector = NpuYoloPreselector(core, model_dir / "yolo11n_raw.onnx")
        nms = NpuNms(
            kernel_path=ROOT / "kernels" / "nms.c",
            movi_dll_dir=movitools,
            native_dir=os.environ.get("NPUNLOCK_NATIVE_DIR"),
        )
        cases = []
        for image_path in image_paths:
            with self.subTest(configuration="C", image=image_path.name):
                input_value, _ = letterbox(load_image(image_path))
                raw = raw_model.infer(input_value)[0][0]
                reference = reference_nms(
                    preselect_raw_predictions(raw, 255),
                    confidence_threshold=0.25,
                    iou_threshold=0.7,
                    max_detections=100,
                )
                selected, _ = yolo_preselector.run(input_value)
                actual, _ = nms.run(
                    selected,
                    confidence_threshold=0.25,
                    iou_threshold=0.7,
                    max_detections=100,
                )
                np.testing.assert_array_equal(actual.indices, reference.indices)
                np.testing.assert_array_equal(actual.classes, reference.classes)
                np.testing.assert_allclose(actual.boxes, reference.boxes, rtol=0.0, atol=0.5)
                np.testing.assert_allclose(actual.scores, reference.scores, rtol=0.0, atol=0.002)
                cases.append((image_path, input_value, reference))
        nms.close()

        single = SingleGraphYoloNms(
            raw_model=model_dir / "yolo11n_raw.onnx",
            kernel_path=ROOT / "kernels" / "nms.c",
            binding_path=ROOT / "bindings" / "npu3720-openvino-2026.3.1.json",
            artifact_dir=ROOT / "artifacts" / "test-single-graph",
            movi_dll_dir=movitools,
            native_dir=os.environ.get("NPUNLOCK_NATIVE_DIR"),
        )
        try:
            for image_path, input_value, reference in cases:
                with self.subTest(configuration="D", image=image_path.name):
                    actual, _ = single.run(
                        input_value,
                        confidence_threshold=0.25,
                        iou_threshold=0.7,
                        max_detections=100,
                    )
                    np.testing.assert_array_equal(actual.indices, reference.indices)
                    np.testing.assert_array_equal(actual.classes, reference.classes)
                    np.testing.assert_allclose(actual.boxes, reference.boxes, rtol=0.0, atol=0.5)
                    np.testing.assert_allclose(actual.scores, reference.scores, rtol=0.0, atol=0.002)
        finally:
            single.close()


if __name__ == "__main__":
    unittest.main()
