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

from xgboost_npu.model import load_ensemble
from xgboost_npu.runtime import (
    RUNTIME_CHUNK_ELEMENTS,
    RUNTIME_KERNEL_SOURCE,
    RUNTIME_LOGICAL_ELEMENTS,
    RUNTIME_MAX_FEATURES,
    RUNTIME_MAX_NODES_PER_TREE,
    RUNTIME_MAX_TOTAL_NODES,
    RUNTIME_MODEL_HEADER_ELEMENTS,
    RUNTIME_REPLICAS,
    RuntimeNpuXGBoost,
    runtime_batch_capacity,
)


def summary(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": statistics.fmean(values),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the runtime-loaded XGBoost forest on ACT-SHAVE"
    )
    parser.add_argument("--model-dir", type=Path, default=Path("contrib/xgboost/artifacts/model"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("contrib/xgboost/outputs/runtime-report.json"),
    )
    parser.add_argument(
        "--movi-dll-dir",
        type=Path,
        required=True,
        help="caller-supplied MVC_DEPEND root",
    )
    parser.add_argument(
        "--native-dir",
        type=Path,
        help="directory containing npunlock.dll and workers",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="optional validated compiled-program cache directory",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = options()
    if args.warmup < 0 or args.runs <= 0:
        raise SystemExit("--warmup must be nonnegative and --runs must be positive")
    model_path = (args.model_dir / "breast_cancer_xgboost.json").resolve()
    validation_path = (args.model_dir / "validation.npz").resolve()
    for path in (model_path, validation_path):
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")
    ensemble = load_ensemble(model_path)
    with np.load(validation_path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.int32)
    booster = xgb.Booster()
    booster.load_model(model_path)
    matrix = xgb.DMatrix(features)
    official_margins = booster.predict(matrix, output_margin=True).astype(np.float32)
    official_probabilities = booster.predict(matrix).astype(np.float32)
    official_classes = (official_probabilities >= 0.5).astype(np.int32)
    parser_margins = ensemble.predict_margin(features)
    capacity = runtime_batch_capacity(ensemble.feature_count)

    with RuntimeNpuXGBoost(
        ensemble,
        movi_dll_dir=args.movi_dll_dir,
        native_dir=args.native_dir,
        cache_dir=args.cache_dir,
    ) as backend:
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
        class_agreement = float(np.mean(actual_classes == official_classes))
        if class_agreement != 1.0:
            raise RuntimeError("runtime-loaded NPU classes disagree with official XGBoost")
        if float(np.max(margin_error)) > 1.0e-6:
            raise RuntimeError("runtime-loaded NPU margin error exceeds FP32 tolerance")
        if float(np.max(probability_error)) > 1.0e-6:
            raise RuntimeError("runtime-loaded NPU probability error exceeds FP32 tolerance")

        benchmark: dict[str, object] = {}
        batch_sizes = sorted({size for size in (1, 8, capacity) if size <= features.shape[0]})
        for size in batch_sizes:
            batch = features[:size]
            batch_matrix = xgb.DMatrix(batch)
            for _ in range(args.warmup):
                booster.predict(batch_matrix)
                backend.run(batch)
            cpu_times: list[float] = []
            end_to_end_times: list[float] = []
            pack_times: list[float] = []
            upload_times: list[float] = []
            graphinfer_times: list[float] = []
            download_times: list[float] = []
            for _ in range(args.runs):
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
                "host_to_shared_feature_buffer_ms": summary(upload_times),
                "graphinfer_submission_fence_wall_ms": summary(graphinfer_times),
                "shared_output_to_host_ms": summary(download_times),
                "note": (
                    "The forest is copied to its persistent shared input once before timing. "
                    "Level Zero kernel timestamps are unavailable; graphinfer wall time is "
                    "not pure kernel time."
                ),
            }

        provenance = backend.runtime_provenance
        assert provenance is not None
        report: dict[str, object] = {
            "schema": "npunlock-xgboost-runtime-demo-v3",
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
                name: importlib.metadata.version(name) for name in ("numpy", "xgboost")
            },
            "runtime_representation": {
                "graph_dtype": "f32",
                "logical_elements_per_input": RUNTIME_LOGICAL_ELEMENTS,
                "elements_per_invocation_input": RUNTIME_CHUNK_ELEMENTS,
                "bytes_per_invocation_input": RUNTIME_CHUNK_ELEMENTS * 4,
                "act_invocations": RUNTIME_REPLICAS,
                "samples_per_invocation": capacity // RUNTIME_REPLICAS,
                "header_elements": RUNTIME_MODEL_HEADER_ELEMENTS,
                "elements_per_node": 2,
                "encoded_model_elements": (
                    RUNTIME_MODEL_HEADER_ELEMENTS + ensemble.node_count * 2
                ),
                "maximum_features": RUNTIME_MAX_FEATURES,
                "maximum_nodes_per_tree": RUNTIME_MAX_NODES_PER_TREE,
                "maximum_total_nodes": RUNTIME_MAX_TOTAL_NODES,
                "maximum_batch_size": capacity,
                "forest_upload": "one persistent shared-input copy before inference timing",
                "forest_pack_ms": backend.model_pack_ms,
                "forest_upload_ms": backend.model_upload_ms,
            },
            "correctness": {
                "validation_samples": int(features.shape[0]),
                "label_accuracy": float(np.mean(actual_classes == labels)),
                "official_class_agreement": class_agreement,
                "reference_parser_max_abs_margin_error": float(
                    np.max(np.abs(parser_margins - official_margins))
                ),
                "margin_max_abs_error": float(np.max(margin_error)),
                "margin_mean_abs_error": float(np.mean(margin_error)),
                "probability_max_abs_error": float(np.max(probability_error)),
                "probability_mean_abs_error": float(np.mean(probability_error)),
                "all_active_invocation_shards_validated": True,
                "inactive_invocations_match_the_first_shard": True,
                "chunk_timings_ms": correctness_timings,
            },
            "performance": benchmark,
            "build": {
                "generic_kernel_source_sha256": hashlib.sha256(
                    RUNTIME_KERNEL_SOURCE
                ).hexdigest(),
                "generic_kernel_source_bytes": len(RUNTIME_KERNEL_SOURCE),
                "packed_forest_sha256": hashlib.sha256(
                    backend.packed_model.tobytes()
                ).hexdigest(),
                "build_ms": backend.build_ms,
                "cache_status": backend.cache_status,
                "cache_load_ms": backend.cache_load_ms,
                "cache_diagnostic": backend.cache_diagnostic,
                "carrier": backend.carrier_explanation,
                "patch_evidence": backend.patch_evidence,
                "npunlock_manifest": backend.build_manifest,
                "ir_provenance": {
                    "driver_version": provenance["driver_version"],
                    "vendor_id": provenance["vendor_id"],
                    "device_id": provenance["device_id"],
                    "graph_extension_version": provenance["graph_extension_version"],
                    "compiler_version": provenance["compiler_version"],
                },
            },
        }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
