from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import onnx
import openvino as ov
import torch
import ultralytics
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export official YOLO11n raw and embedded-NMS ONNX models")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    options = arguments()
    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    os.chdir(output_dir)
    try:
        model = YOLO(options.model)
        checkpoint = Path(options.model)
        if not checkpoint.exists():
            checkpoint = Path(model.ckpt_path)
        raw_export = Path(
            model.export(
                format="onnx",
                imgsz=options.imgsz,
                batch=1,
                dynamic=False,
                simplify=False,
                nms=False,
                opset=options.opset,
            )
        )
        raw_path = output_dir / "yolo11n_raw.onnx"
        raw_export.replace(raw_path)

        embedded_export = Path(
            model.export(
                format="onnx",
                imgsz=options.imgsz,
                batch=1,
                dynamic=False,
                simplify=False,
                nms=True,
                opset=options.opset,
                conf=options.confidence,
                iou=options.iou,
                max_det=options.max_det,
            )
        )
        embedded_path = output_dir / "yolo11n_nms.onnx"
        embedded_export.replace(embedded_path)
        checkpoint = checkpoint.resolve()
    finally:
        os.chdir(previous)

    raw_model = onnx.load(raw_path, load_external_data=False)
    embedded_model = onnx.load(embedded_path, load_external_data=False)
    manifest = {
        "schema": "npunlock.yolo11.export.v1",
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "openvino": ov.__version__,
        },
        "settings": {
            "model": options.model,
            "imgsz": options.imgsz,
            "opset": options.opset,
            "dynamic": False,
            "simplify": False,
            "confidence": options.confidence,
            "iou": options.iou,
            "max_det": options.max_det,
        },
        "files": {
            "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint), "size": checkpoint.stat().st_size},
            "raw_onnx": {"path": str(raw_path), "sha256": sha256(raw_path), "size": raw_path.stat().st_size},
            "embedded_nms_onnx": {
                "path": str(embedded_path),
                "sha256": sha256(embedded_path),
                "size": embedded_path.stat().st_size,
            },
        },
        "onnx": {
            "raw_opset": [item.version for item in raw_model.opset_import],
            "embedded_nms_opset": [item.version for item in embedded_model.opset_import],
            "raw_outputs": [output.name for output in raw_model.graph.output],
            "embedded_nms_outputs": [output.name for output in embedded_model.graph.output],
        },
    }
    manifest_path = output_dir / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
