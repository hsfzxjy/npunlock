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
    RUNTIME_CHAIN_KERNEL_SOURCE,
    RUNTIME_CHUNK_ELEMENTS,
    RUNTIME_CONNECTED_FIRST_KERNEL_SOURCE,
    RUNTIME_LOGICAL_ELEMENTS,
    RUNTIME_MAX_TOTAL_NODES,
    RUNTIME_REPLICAS,
    ConnectedRuntimeNpuXGBoost,
    partitioned_runtime_batch_capacity,
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
        description="Run a connected two-part runtime-loaded XGBoost forest on ACT-SHAVE"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("contrib/xgboost/artifacts/model-64"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("contrib/xgboost/outputs/connected-runtime-report.json"),
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
    capacity = partitioned_runtime_batch_capacity(ensemble.feature_count)

    with ConnectedRuntimeNpuXGBoost(
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
            raise RuntimeError("partitioned NPU classes disagree with official XGBoost")
        if float(np.max(margin_error)) > 1.0e-6:
            raise RuntimeError("partitioned NPU margin error exceeds FP32 tolerance")
        if float(np.max(probability_error)) > 1.0e-6:
            raise RuntimeError("partitioned NPU probability error exceeds FP32 tolerance")

        benchmark: dict[str, object] = {}
        batch_sizes = sorted({size for size in (1, 8, capacity) if size <= features.shape[0]})
        for size in batch_sizes:
            batch = features[:size]
            batch_matrix = xgb.DMatrix(batch)
            for _ in range(args.warmup):
                booster.predict(batch_matrix)
                backend.run(batch)
            cpu_times: list[float] = []
            npu_times: list[float] = []
            graphinfer_times: list[float] = []
            pack_times: list[float] = []
            upload_times: list[float] = []
            download_times: list[float] = []
            for _ in range(args.runs):
                started = perf_counter()
                booster.predict(batch_matrix)
                cpu_times.append((perf_counter() - started) * 1000.0)
                started = perf_counter()
                _, timing = backend.run(batch)
                npu_times.append((perf_counter() - started) * 1000.0)
                graphinfer_times.append(timing["graphinfer_wall_ms"])
                pack_times.append(timing["pack_ms"])
                upload_times.append(timing["upload_ms"])
                download_times.append(timing["download_ms"])
            benchmark[str(size)] = {
                "official_xgboost_cpu_ms": summary(cpu_times),
                "npunlock_end_to_end_ms": summary(npu_times),
                "one_graphinfer_call_wall_ms": summary(graphinfer_times),
                "host_pack_ms": summary(pack_times),
                "host_to_shared_buffers_ms": summary(upload_times),
                "shared_outputs_to_host_ms": summary(download_times),
            }

        provenance = backend.runtime_provenance
        assert provenance is not None
        report: dict[str, object] = {
            "schema": "npunlock-xgboost-connected-runtime-demo-v5",
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
            "partitioned_runtime": {
                "contract": (
                    "exactly two connected contiguous whole-tree NPU traversal stages "
                    "in one graph execution"
                ),
                "partition_tree_counts": list(backend.partition_tree_counts),
                "partition_node_counts": list(backend.partition_node_counts),
                "single_stage_maximum_nodes": RUNTIME_MAX_TOTAL_NODES,
                "two_stage_maximum_nodes": 2 * RUNTIME_MAX_TOTAL_NODES,
                "logical_elements_per_input": RUNTIME_LOGICAL_ELEMENTS,
                "elements_per_invocation_input": RUNTIME_CHUNK_ELEMENTS,
                "act_invocations_per_stage": RUNTIME_REPLICAS,
                "samples_per_invocation": capacity // RUNTIME_REPLICAS,
                "maximum_batch_size": capacity,
                "host_role": (
                    "shards samples across four invocation-local spans, packs/copies graph "
                    "inputs, and validates every output shard; stage A forwards features and "
                    "raw margins directly to stage B inside the graph; traversal, margin "
                    "addition, sigmoid, and class selection execute on the NPU"
                ),
                "rounding_note": (
                    "the partition boundary changes FP32 addition association relative to a "
                    "single sequential accumulation"
                ),
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
                "connected_first_source_sha256": hashlib.sha256(RUNTIME_CONNECTED_FIRST_KERNEL_SOURCE).hexdigest(),
                "connected_first_source_bytes": len(RUNTIME_CONNECTED_FIRST_KERNEL_SOURCE),
                "chained_traversal_source_sha256": hashlib.sha256(RUNTIME_CHAIN_KERNEL_SOURCE).hexdigest(),
                "chained_traversal_source_bytes": len(RUNTIME_CHAIN_KERNEL_SOURCE),
                "packed_partition_sha256": [
                    hashlib.sha256(packed.tobytes()).hexdigest() for packed in backend.packed_models
                ],
                "build_ms": backend.build_ms,
                "cache_status": backend.cache_status,
                "cache_load_ms": backend.cache_load_ms,
                "cache_diagnostic": backend.cache_diagnostic,
                "carrier": backend.carrier_explanation,
                "build_plan": backend.build_explanation,
                "patch_evidence": backend.patch_evidence,
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
