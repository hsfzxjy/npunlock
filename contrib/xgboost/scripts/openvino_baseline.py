from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import openvino as ov


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attempt explicit CPU and NPU OpenVINO execution of XGBoost ONNX")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def error_record(stage: str, error: BaseException) -> dict[str, object]:
    return {"status": "failed", "stage": stage, "error_type": type(error).__name__, "error": str(error)}


def main() -> int:
    args = options()
    report: dict[str, object] = {
        "schema": "npunlock-xgboost-openvino-baseline-v1",
        "openvino_version": importlib.metadata.version("openvino"),
        "model": str(args.model.resolve()),
    }
    with np.load(args.validation, allow_pickle=False) as data:
        sample = np.asarray(data["features"][:8], dtype=np.float32)
    core = ov.Core()
    report["available_devices"] = list(core.available_devices)
    try:
        started = perf_counter()
        model = core.read_model(args.model)
        report["read_model_ms"] = (perf_counter() - started) * 1000.0
        report["operations"] = sorted({node.get_type_name() for node in model.get_ops()})
    except Exception as error:
        report["conversion"] = error_record("read_model", error)
    else:
        report["conversion"] = {"status": "ok"}
        devices: dict[str, object] = {}
        for device in ("CPU", "NPU"):
            try:
                started = perf_counter()
                compiled = core.compile_model(model, device)
                compile_ms = (perf_counter() - started) * 1000.0
                started = perf_counter()
                outputs = compiled(sample)
                infer_ms = (perf_counter() - started) * 1000.0
                devices[device] = {
                    "status": "ok",
                    "compile_ms": compile_ms,
                    "infer_ms": infer_ms,
                    "execution_devices": list(compiled.get_property("EXECUTION_DEVICES")),
                    "outputs": [list(np.asarray(value).shape) for value in outputs.values()],
                }
            except Exception as error:
                devices[device] = error_record("compile_or_infer", error)
        report["devices"] = devices
    text = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
