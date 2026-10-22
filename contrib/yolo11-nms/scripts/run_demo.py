from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import cv2
import numpy as np
import openvino as ov
from ultralytics import YOLO


CONTRIB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTRIB_ROOT))

from yolo_nms.npunlock_backend import NpuNms
from yolo_nms.openvino_backend import (
    CompiledModel,
    NpuPreselector,
    letterbox,
    load_image,
    scale_boxes_to_image,
)
from yolo_nms.reference import DetectionBatch, preselect_raw_predictions, reference_nms


def summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def detection_json(detections: DetectionBatch, names: dict[int, str], info: object) -> list[dict[str, object]]:
    scaled = scale_boxes_to_image(detections.boxes, info)  # type: ignore[arg-type]
    return [
        {
            "xyxy": [float(value) for value in box],
            "confidence": float(score),
            "class_id": int(class_id),
            "class_name": names[int(class_id)],
            "source_index": int(index),
        }
        for box, score, class_id, index in zip(
            scaled,
            detections.scores,
            detections.classes,
            detections.indices,
        )
    ]


def draw(image: np.ndarray, detections: list[dict[str, object]], output: Path) -> None:
    rendered = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection["xyxy"])  # type: ignore[union-attr]
        label = f"{detection['class_name']} {float(detection['confidence']):.3f}"
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (50, 220, 50), 2)
        cv2.putText(rendered, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 220, 50), 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), rendered):
        raise RuntimeError(f"failed to write visualization: {output}")


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO11n OpenVINO and npunlock NMS configurations")
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--image", type=Path, default=Path("artifacts/images/bus.jpg"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--movi-dll-dir", type=Path, required=True, help="caller-supplied MVC_DEPEND root")
    parser.add_argument("--native-dir", type=Path, help="development directory containing npunlock.dll and worker")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = options()
    model_dir = args.model_dir.resolve()
    raw_path = model_dir / "yolo11n_raw.onnx"
    embedded_path = model_dir / "yolo11n_nms.onnx"
    checkpoint = model_dir / "yolo11n.pt"
    for path in (raw_path, embedded_path, checkpoint, args.image):
        if not path.exists():
            raise SystemExit(f"required input does not exist: {path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_image(args.image)
    started = perf_counter()
    input_data, letterbox_info = letterbox(image)
    preprocessing_ms = (perf_counter() - started) * 1000.0
    names = YOLO(checkpoint).names
    core = ov.Core()

    raw_npu = CompiledModel(core, raw_path, "NPU")
    if raw_npu.execution_devices != "NPU":
        raise RuntimeError(f"raw YOLO did not execute exclusively on NPU: {raw_npu.execution_devices!r}")
    raw_cpu = CompiledModel(core, raw_path, "CPU")
    for _ in range(args.warmup):
        raw_npu.infer(input_data)
    raw_times: list[float] = []
    raw_outputs: list[np.ndarray] = []
    for _ in range(args.runs):
        outputs, elapsed = raw_npu.infer(input_data)
        raw_outputs = outputs
        raw_times.append(elapsed)
    raw_predictions = raw_outputs[0]
    cpu_raw_predictions = raw_cpu.infer(input_data)[0][0]

    cpu_preselect_times: list[float] = []
    cpu_nms_times: list[float] = []
    cpu_detections = None
    for _ in range(args.runs):
        started = perf_counter()
        cpu_candidates = preselect_raw_predictions(raw_predictions, 255)
        cpu_preselect_times.append((perf_counter() - started) * 1000.0)
        started = perf_counter()
        cpu_detections = reference_nms(
            cpu_candidates,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            max_detections=args.max_det,
        )
        cpu_nms_times.append((perf_counter() - started) * 1000.0)
    assert cpu_detections is not None

    config_a_end_to_end_times: list[float] = []
    for _ in range(args.runs):
        started = perf_counter()
        input_value, _ = letterbox(image)
        raw_value = raw_npu.infer(input_value)[0][0]
        candidates = preselect_raw_predictions(raw_value, 255)
        reference_nms(
            candidates,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            max_detections=args.max_det,
        )
        config_a_end_to_end_times.append((perf_counter() - started) * 1000.0)

    embedded_report: dict[str, object]
    embedded_npu_output: list[np.ndarray] | None = None
    try:
        embedded_npu = CompiledModel(core, embedded_path, "NPU")
    except Exception as error:
        embedded_report = {
            "status": "compile-failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    else:
        embedded_report = {
            "status": "placement-error" if embedded_npu.execution_devices != "NPU" else "compiled",
            "compile_ms": embedded_npu.compile_ms,
            "execution_devices": embedded_npu.execution_devices,
        }
        if embedded_npu.execution_devices == "NPU":
            try:
                for _ in range(args.warmup):
                    embedded_npu.infer(input_data)
                embedded_times: list[float] = []
                for _ in range(args.runs):
                    embedded_npu_output, elapsed = embedded_npu.infer(input_data)
                    embedded_times.append(elapsed)
                embedded_report.update({"status": "ok", "latency_ms": summary(embedded_times)})
            except Exception as error:
                embedded_report.update(
                    {
                        "status": "runtime-failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    embedded_cpu = CompiledModel(core, embedded_path, "CPU")
    embedded_cpu_output, embedded_cpu_ms = embedded_cpu.infer(input_data)
    embedded_report["cpu_control"] = {
        "status": "ok",
        "compile_ms": embedded_cpu.compile_ms,
        "inference_ms": embedded_cpu_ms,
        "output_shape": list(embedded_cpu_output[0].shape),
        "valid_detection_count": int(np.count_nonzero(embedded_cpu_output[0][0, :, 4] > args.confidence)),
    }
    if embedded_npu_output is not None:
        same_embedded_shape = embedded_npu_output[0].shape == embedded_cpu_output[0].shape
        embedded_delta = (
            np.abs(embedded_npu_output[0].astype(np.float32) - embedded_cpu_output[0].astype(np.float32))
            if same_embedded_shape
            else None
        )
        embedded_report["npu_vs_cpu"] = {
            "same_shape": same_embedded_shape,
            "max_abs": float(embedded_delta.max()) if embedded_delta is not None else None,
            "allclose_rtol_0_atol_0_01": bool(
                same_embedded_shape
                and np.allclose(embedded_npu_output[0], embedded_cpu_output[0], rtol=0.0, atol=0.01)
            ),
        }

    preselector = NpuPreselector(core, raw_predictions.shape[2])
    for _ in range(args.warmup):
        preselector.run(raw_predictions)
    preselector_times: list[float] = []
    npu_candidates = None
    for _ in range(args.runs):
        npu_candidates, elapsed = preselector.run(raw_predictions)
        preselector_times.append(elapsed)
    assert npu_candidates is not None

    npu_nms = NpuNms(
        kernel_path=CONTRIB_ROOT / "kernels" / "nms.c",
        movi_dll_dir=args.movi_dll_dir,
        native_dir=args.native_dir,
    )
    for _ in range(args.warmup):
        npu_nms.run(
            npu_candidates,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            max_detections=args.max_det,
        )
    component_times = {key: [] for key in ("pack_ms", "upload_ms", "execution_wall_ms", "download_ms")}
    npu_detections = None
    for _ in range(args.runs):
        npu_detections, timings = npu_nms.run(
            npu_candidates,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            max_detections=args.max_det,
        )
        for key, value in timings.items():
            component_times[key].append(value)
    assert npu_detections is not None

    end_to_end_times: list[float] = []
    for _ in range(args.runs):
        started = perf_counter()
        input_value, _ = letterbox(image)
        raw_value = raw_npu.infer(input_value)[0][0]
        selected = preselector.run(raw_value)[0]
        npu_nms.run(
            selected,
            confidence_threshold=args.confidence,
            iou_threshold=args.iou,
            max_detections=args.max_det,
        )
        end_to_end_times.append((perf_counter() - started) * 1000.0)

    same_shape = cpu_detections.boxes.shape == npu_detections.boxes.shape
    comparison = {
        "same_detection_count": same_shape,
        "same_source_indices": bool(same_shape and np.array_equal(cpu_detections.indices, npu_detections.indices)),
        "same_class_order": bool(same_shape and np.array_equal(cpu_detections.classes, npu_detections.classes)),
        "max_box_delta_network_pixels": (
            float(np.max(np.abs(cpu_detections.boxes - npu_detections.boxes))) if same_shape and len(cpu_detections) else None
        ),
        "max_confidence_delta": (
            float(np.max(np.abs(cpu_detections.scores - npu_detections.scores))) if same_shape and len(cpu_detections) else None
        ),
    }
    config_a = detection_json(cpu_detections, names, letterbox_info)
    config_c = detection_json(npu_detections, names, letterbox_info)
    draw(image, config_c, output_dir / "detections_npunlock.jpg")

    raw_delta = np.abs(cpu_raw_predictions.astype(np.float32) - raw_predictions.astype(np.float32))
    report = {
        "schema": "npunlock.yolo11.demo.v1",
        "versions": {
            "openvino": ov.__version__,
            "npu_driver": core.get_property("NPU", "NPU_DRIVER_VERSION"),
        },
        "settings": {
            "confidence": args.confidence,
            "iou": args.iou,
            "max_det": args.max_det,
            "pre_nms_topk": 255,
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "inputs": {
            "image": str(args.image.resolve()),
            "image_sha256": sha256(args.image),
            "raw_onnx_sha256": sha256(raw_path),
            "embedded_nms_onnx_sha256": sha256(embedded_path),
        },
        "configuration_a": {
            "network_device": "NPU",
            "nms_device": "CPU",
            "execution_devices": raw_npu.execution_devices,
            "network_compile_ms": raw_npu.compile_ms,
            "network_inference_ms": summary(raw_times),
            "cpu_candidate_preselection_ms": summary(cpu_preselect_times),
            "cpu_nms_ms": summary(cpu_nms_times),
            "complete_end_to_end_ms": summary(config_a_end_to_end_times),
            "detections": config_a,
        },
        "configuration_b": embedded_report,
        "configuration_c": {
            "network_device": "NPU",
            "preselector_device": "NPU",
            "nms_device": "NPU ACT-SHAVE through npunlock",
            "network_execution_devices": raw_npu.execution_devices,
            "preselector_execution_devices": preselector.model.execution_devices,
            "preselector_compile_ms": preselector.model.compile_ms,
            "preselector_ms": summary(preselector_times),
            "npunlock_build_ms": npu_nms.build_ms,
            "npunlock_components_ms": {key: summary(values) for key, values in component_times.items()},
            "kernel_only_ms": None,
            "kernel_timing_note": (
                "graphinfer exposes bounded submission/fence wall time but no device-only kernel profiling; "
                "execution_wall_ms is the closest measured value and excludes explicit shared-buffer upload/download copies"
            ),
            "complete_end_to_end_ms": summary(end_to_end_times),
            "carrier": npu_nms.carrier_explanation,
            "patch_evidence": npu_nms.patch_evidence,
            "detections": config_c,
        },
        "correctness": {
            "cpu_vs_npu_raw": {
                "max_abs": float(raw_delta.max()),
                "mean_abs": float(raw_delta.mean()),
            },
            "configuration_a_vs_c": comparison,
        },
        "preprocessing_ms_single": preprocessing_ms,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if comparison["same_source_indices"] and comparison["same_class_order"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
