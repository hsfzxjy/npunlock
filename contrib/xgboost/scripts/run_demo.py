from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import numpy as np
import xgboost as xgb


CONTRIB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTRIB_ROOT))

from xgboost_npu.backend import NpuXGBoost
from xgboost_npu.model import Ensemble, load_ensemble
from xgboost_npu.packing import batch_capacity, transport_spec


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


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and benchmark the npunlock XGBoost classifier")
    parser.add_argument("--model-dir", type=Path, default=Path("contrib/xgboost/artifacts/model"))
    parser.add_argument("--output-dir", type=Path, default=Path("contrib/xgboost/outputs"))
    parser.add_argument("--movi-dll-dir", type=Path, required=True, help="caller-supplied MVC_DEPEND root")
    parser.add_argument("--native-dir", type=Path, help="directory containing npunlock.dll and workers")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--transport",
        choices=("f16", "f32", "both"),
        default="both",
        help="carrier transport profile to execute (default: both)",
    )
    return parser.parse_args()


def run_transport(
    *,
    name: str,
    ensemble: Ensemble,
    features: np.ndarray,
    labels: np.ndarray,
    booster: xgb.Booster,
    output_dir: Path,
    movi_dll_dir: Path,
    native_dir: Path | None,
    warmup: int,
    runs: int,
) -> dict[str, object]:
    spec = transport_spec(name)
    capacity = batch_capacity(ensemble.feature_count, transport=spec)
    transported = features.astype(spec.numpy_dtype).astype(np.float32)
    matrix = xgb.DMatrix(transported)
    official_margins = booster.predict(matrix, output_margin=True).astype(np.float32)
    official_probabilities = booster.predict(matrix).astype(np.float32)
    official_classes = (official_probabilities >= 0.5).astype(np.int32)
    parser_margins = ensemble.predict_margin(transported)
    kernel_path = output_dir / f"generated_xgboost_{name}.c"

    result: dict[str, object] = {
        "transport": {
            "profile": name,
            "graph_dtype": spec.dtype,
            "logical_elements": spec.logical_elements,
            "elements_per_invocation": spec.chunk_elements,
            "bytes_per_invocation": spec.chunk_elements * spec.numpy_dtype.itemsize,
            "maximum_batch_size": capacity,
            "arithmetic": "FP32 SHAVE comparisons, accumulation, and sigmoid",
        },
        "reference_parser_max_abs_margin_error": float(
            np.max(np.abs(parser_margins - official_margins))
        ),
    }
    with NpuXGBoost(
        ensemble,
        movi_dll_dir=movi_dll_dir,
        native_dir=native_dir,
        transport=spec,
    ) as backend:
        kernel_path.write_bytes(backend.source)
        npu_margins: list[np.ndarray] = []
        npu_probabilities: list[np.ndarray] = []
        npu_classes: list[np.ndarray] = []
        correctness_timings: list[dict[str, float]] = []
        for start in range(0, features.shape[0], capacity):
            predictions, timing = backend.run(features[start : start + capacity])
            npu_margins.append(predictions.margins)
            npu_probabilities.append(predictions.probabilities)
            npu_classes.append(predictions.classes)
            correctness_timings.append(timing)
        actual_margins = np.concatenate(npu_margins)
        actual_probabilities = np.concatenate(npu_probabilities)
        actual_classes = np.concatenate(npu_classes)
        margin_error = np.abs(actual_margins - official_margins)
        probability_error = np.abs(actual_probabilities - official_probabilities)
        official_class_agreement = float(np.mean(actual_classes == official_classes))
        result["correctness"] = {
            "validation_samples": int(features.shape[0]),
            "label_accuracy": float(np.mean(actual_classes == labels)),
            "official_class_agreement": official_class_agreement,
            "margin_max_abs_error": float(np.max(margin_error)),
            "margin_mean_abs_error": float(np.mean(margin_error)),
            "probability_max_abs_error": float(np.max(probability_error)),
            "probability_mean_abs_error": float(np.mean(probability_error)),
            "all_replica_outputs_identical": True,
            "chunk_timings_ms": correctness_timings,
        }
        batch_sizes = sorted({size for size in (1, 8, 32, capacity) if size <= features.shape[0]})
        benchmark: dict[str, object] = {}
        for size in batch_sizes:
            batch = features[:size]
            batch_transport = batch.astype(spec.numpy_dtype).astype(np.float32)
            batch_matrix = xgb.DMatrix(batch_transport)
            for _ in range(warmup):
                booster.predict(batch_matrix)
                backend.run(batch)
            cpu_times: list[float] = []
            end_to_end_times: list[float] = []
            pack_times: list[float] = []
            upload_times: list[float] = []
            graphinfer_times: list[float] = []
            download_times: list[float] = []
            for _ in range(runs):
                started = perf_counter()
                booster.predict(batch_matrix)
                cpu_times.append((perf_counter() - started) * 1000.0)
                started = perf_counter()
                _, timing = backend.run(batch)
                end_to_end_times.append((perf_counter() - started) * 1000.0)
                pack_times.append(timing["pack_ms"])
                upload_times.append(timing["upload_ms"])
                graphinfer_times.append(timing["graphinfer_wall_ms"])
                download_times.append(timing["download_ms"])
            benchmark[str(size)] = {
                "official_xgboost_cpu_ms": summary(cpu_times),
                "npunlock_end_to_end_ms": summary(end_to_end_times),
                "host_pack_ms": summary(pack_times),
                "host_to_shared_buffer_ms": summary(upload_times),
                "graphinfer_submission_fence_wall_ms": summary(graphinfer_times),
                "shared_buffer_to_host_ms": summary(download_times),
                "note": (
                    "Level Zero kernel timestamps are not exposed; graphinfer wall time "
                    "is not pure kernel time."
                ),
            }
        result["performance"] = benchmark
        result["build"] = {
            "kernel_path": str(kernel_path),
            "kernel_sha256": hashlib.sha256(backend.source).hexdigest(),
            "build_ms": backend.build_ms,
            "carrier": backend.carrier_explanation,
            "patch_evidence": backend.patch_evidence,
            "npunlock_manifest": backend.build_manifest,
            "ir_provenance": {
                "driver_version": backend.program.ir_provenance.driver_version,
                "vendor_id": backend.program.ir_provenance.vendor_id,
                "device_id": backend.program.ir_provenance.device_id,
                "graph_extension_version": backend.program.ir_provenance.graph_extension_version,
                "compiler_version": list(backend.program.ir_provenance.compiler_version),
            },
        }
    if official_class_agreement != 1.0:
        raise RuntimeError(f"{name} NPU classes disagree with official XGBoost")
    return result


def main() -> int:
    args = options()
    if args.warmup < 0 or args.runs <= 0:
        raise SystemExit("--warmup must be nonnegative and --runs must be positive")
    model_path = (args.model_dir / "breast_cancer_xgboost.json").resolve()
    validation_path = (args.model_dir / "validation.npz").resolve()
    for path in (model_path, validation_path):
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")
    with np.load(validation_path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int32)
    ensemble = load_ensemble(model_path)
    booster = xgb.Booster()
    booster.load_model(model_path)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    report: dict[str, object] = {
        "schema": "npunlock-xgboost-demo-v2",
        "model": {
            "path": str(model_path),
            "sha256": sha256(model_path),
            "features": ensemble.feature_count,
            "trees": len(ensemble.trees),
            "nodes": ensemble.node_count,
            "max_depth": ensemble.max_depth,
            "base_margin": ensemble.base_margin,
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "xgboost", "scikit-learn", "openvino")
        },
        "results": results,
    }
    names = ("f16", "f32") if args.transport == "both" else (args.transport,)
    for name in names:
        results[name] = run_transport(
            name=name,
            ensemble=ensemble,
            features=features,
            labels=labels,
            booster=booster,
            output_dir=output_dir,
            movi_dll_dir=args.movi_dll_dir,
            native_dir=args.native_dir,
            warmup=args.warmup,
            runs=args.runs,
        )
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
