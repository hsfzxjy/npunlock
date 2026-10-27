from __future__ import annotations

import argparse
from dataclasses import replace
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
from xgboost_npu.model import load_ensemble
from xgboost_npu.packing import batch_capacity, transport_spec


def parse_tree_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(sorted({int(item) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("tree counts must be comma-separated integers") from error
    if not counts or counts[0] <= 0:
        raise argparse.ArgumentTypeError("tree counts must be positive")
    return counts


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
    parser = argparse.ArgumentParser(description="Benchmark XGBoost model prefixes on CPU and ACT-SHAVE")
    parser.add_argument("--model-dir", type=Path, default=Path("contrib/xgboost/artifacts/model"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("contrib/xgboost/outputs/tree-counts.json"),
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
    parser.add_argument("--transport", choices=("f16", "f32", "both"), default="both")
    parser.add_argument("--tree-counts", type=parse_tree_counts, default=(1, 4, 8, 16, 32))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = options()
    if args.batch_size <= 0 or args.warmup < 0 or args.runs <= 0:
        raise SystemExit("--batch-size and --runs must be positive; --warmup must be nonnegative")
    model_path = (args.model_dir / "breast_cancer_xgboost.json").resolve()
    validation_path = (args.model_dir / "validation.npz").resolve()
    for path in (model_path, validation_path):
        if not path.is_file():
            raise SystemExit(f"required input does not exist: {path}")
    ensemble = load_ensemble(model_path)
    if args.tree_counts[-1] > len(ensemble.trees):
        raise SystemExit(f"requested {args.tree_counts[-1]} trees, but model contains {len(ensemble.trees)}")
    booster = xgb.Booster()
    booster.load_model(model_path)
    if booster.num_boosted_rounds() != len(ensemble.trees):
        raise RuntimeError("XGBoost boosting-round count does not match the validated tree layout")
    with np.load(validation_path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
    if args.batch_size > features.shape[0]:
        raise SystemExit(f"batch size {args.batch_size} exceeds {features.shape[0]} validation samples")
    features = features[: args.batch_size]
    names = ("f16", "f32") if args.transport == "both" else (args.transport,)

    results: dict[str, object] = {}
    report: dict[str, object] = {
        "schema": "npunlock-xgboost-tree-scaling-v1",
        "model": {
            "path": str(model_path),
            "sha256": sha256(model_path),
            "available_trees": len(ensemble.trees),
            "feature_count": ensemble.feature_count,
            "batch_size": args.batch_size,
        },
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "xgboost")},
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "results": results,
    }
    for name in names:
        spec = transport_spec(name)
        capacity = batch_capacity(ensemble.feature_count, transport=spec)
        if args.batch_size > capacity:
            raise SystemExit(f"batch size {args.batch_size} exceeds {name} profile capacity {capacity}")
        transported = features.astype(spec.numpy_dtype).astype(np.float32)
        matrix = xgb.DMatrix(transported)
        transport_results: dict[str, object] = {}
        results[name] = {
            "graph_dtype": spec.dtype,
            "batch_capacity": capacity,
            "tree_counts": transport_results,
        }
        for tree_count in args.tree_counts:
            prefix = replace(ensemble, trees=ensemble.trees[:tree_count])
            iteration_range = (0, tree_count)
            official_margins = booster.predict(
                matrix,
                output_margin=True,
                iteration_range=iteration_range,
            ).astype(np.float32)
            official_probabilities = booster.predict(
                matrix,
                iteration_range=iteration_range,
            ).astype(np.float32)
            parser_error = float(np.max(np.abs(prefix.predict_margin(transported) - official_margins)))
            with NpuXGBoost(
                prefix,
                movi_dll_dir=args.movi_dll_dir,
                native_dir=args.native_dir,
                transport=spec,
            ) as backend:
                predictions, _ = backend.run(features)
                margin_error = np.abs(predictions.margins - official_margins)
                probability_error = np.abs(predictions.probabilities - official_probabilities)
                class_agreement = float(
                    np.mean(predictions.classes == (official_probabilities >= 0.5).astype(np.int32))
                )
                for _ in range(args.warmup):
                    booster.predict(matrix, iteration_range=iteration_range)
                    backend.run(features)
                cpu_times: list[float] = []
                npu_times: list[float] = []
                graphinfer_times: list[float] = []
                for _ in range(args.runs):
                    started = perf_counter()
                    booster.predict(matrix, iteration_range=iteration_range)
                    cpu_times.append((perf_counter() - started) * 1000.0)
                    started = perf_counter()
                    _, timing = backend.run(features)
                    npu_times.append((perf_counter() - started) * 1000.0)
                    graphinfer_times.append(timing["graphinfer_wall_ms"])
                evidence = backend.patch_evidence
                provenance = backend.program.provenance
                if provenance is None:
                    raise RuntimeError("compiled XGBoost program did not retain NPU provenance")
                transport_results[str(tree_count)] = {
                    "trees": tree_count,
                    "nodes": prefix.node_count,
                    "kernel_source_bytes": len(backend.source),
                    "kernel_source_sha256": hashlib.sha256(backend.source).hexdigest(),
                    "linked_code_image_bytes": evidence["append"]["image_size"],
                    "build_ms": backend.build_ms,
                    "patched_target_count": evidence["patched_target_count"],
                    "reference_parser_max_abs_margin_error": parser_error,
                    "correctness": {
                        "official_class_agreement": class_agreement,
                        "margin_max_abs_error": float(np.max(margin_error)),
                        "margin_mean_abs_error": float(np.mean(margin_error)),
                        "probability_max_abs_error": float(np.max(probability_error)),
                        "probability_mean_abs_error": float(np.mean(probability_error)),
                    },
                    "performance_ms": {
                        "official_xgboost_cpu": summary(cpu_times),
                        "npunlock_end_to_end": summary(npu_times),
                        "graphinfer_submission_fence_wall": summary(graphinfer_times),
                    },
                    "patch_evidence": evidence,
                    "ir_provenance": provenance.to_json(),
                }
            margin_limit = 0.004 if name == "f16" else 1.0e-6
            probability_limit = 0.0005 if name == "f16" else 1.0e-6
            if class_agreement != 1.0:
                raise RuntimeError(f"{name} {tree_count}-tree NPU classes disagree with XGBoost")
            if float(np.max(margin_error)) > margin_limit:
                raise RuntimeError(f"{name} {tree_count}-tree margin error exceeds tolerance")
            if float(np.max(probability_error)) > probability_limit:
                raise RuntimeError(f"{name} {tree_count}-tree probability error exceeds tolerance")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
