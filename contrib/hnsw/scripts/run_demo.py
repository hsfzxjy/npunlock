from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import hnswlib
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


CONTRIB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTRIB_ROOT))

from hnsw_npu import MAX_BATCH_QUERIES, NpuHnsw, build_hnsw, pack_query, save_index, search_index
from hnsw_npu.backend import KERNEL_PATH
from hnsw_npu.packing import validate_capacity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run genuine bounded HNSW traversal on ACT-SHAVE and compare CPU implementations."
    )
    parser.add_argument("--movi-dll-dir", help="caller-supplied MVC_DEPEND root")
    parser.add_argument("--native-dir", help="directory containing a development npunlock.dll")
    parser.add_argument("--output-dir", default="contrib/hnsw/outputs")
    parser.add_argument("--index-size", type=int, default=48)
    parser.add_argument("--query-count", type=int, default=MAX_BATCH_QUERIES)
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--ef-construction", type=int, default=24)
    parser.add_argument("--ef-search", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=100)
    parser.add_argument("--cpu-timing-repeats", type=int, default=10_000)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=MAX_BATCH_QUERIES)
    parser.add_argument(
        "--cache-dir",
        help="persistent compiled-program cache (default: OUTPUT_DIR/program-cache)",
    )
    parser.add_argument("--cpu-only", action="store_true")
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def exact_neighbors(index_vectors: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    delta = queries[:, None, :] - index_vectors[None, :, :]
    distances = np.sum(delta * delta, axis=2, dtype=np.float32)
    node_ids = np.arange(index_vectors.shape[0])
    result = np.empty((queries.shape[0], k), dtype=np.int32)
    for row in range(queries.shape[0]):
        result[row] = np.lexsort((node_ids, distances[row]))[:k]
    return result


def recall_at_k(actual: np.ndarray, exact: np.ndarray) -> float:
    matches = sum(len(set(row).intersection(expected)) for row, expected in zip(actual, exact))
    return matches / float(exact.size)


def main() -> int:
    args = parse_args()
    if not 1 <= args.index_size <= 48:
        raise ValueError("index-size must be in [1, 48]")
    if not 1 <= args.dimensions <= 32:
        raise ValueError("dimensions must be in [1, 32]")
    if args.dimensions > args.index_size:
        raise ValueError("dimensions must not exceed index-size for the fitted PCA embedding")
    if args.index_size + args.query_count > 1797:
        raise ValueError("requested index and query count exceed the digits dataset")
    if args.query_count < 1:
        raise ValueError("query-count must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if args.timing_repeats < 1:
        raise ValueError("timing-repeats must be positive")
    if args.cpu_timing_repeats < 1:
        raise ValueError("cpu-timing-repeats must be positive")
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    if not 1 <= args.batch_size <= MAX_BATCH_QUERIES:
        raise ValueError(f"batch-size must be in [1, {MAX_BATCH_QUERIES}]")
    if not args.cpu_only and not args.movi_dll_dir:
        raise ValueError("--movi-dll-dir is required unless --cpu-only is selected")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    digits = load_digits()
    raw = digits.data.astype(np.float32)
    labels = digits.target.astype(np.int32)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(raw))[: args.index_size + args.query_count]
    selected = raw[order]
    scaler = StandardScaler()
    scaler.fit(selected[: args.index_size])
    standardized = scaler.transform(selected).astype(np.float32)
    pca = PCA(n_components=args.dimensions, svd_solver="full")
    pca.fit(standardized[: args.index_size])
    embeddings = np.ascontiguousarray(pca.transform(standardized), dtype=np.float32)
    index_vectors = embeddings[: args.index_size]
    queries = embeddings[args.index_size :]

    build_started = perf_counter()
    index = build_hnsw(
        index_vectors,
        m=args.m,
        ef_construction=args.ef_construction,
        seed=args.seed,
    )
    validate_capacity(index)
    pack_query(index, queries[0], k=args.top_k, ef_search=args.ef_search)
    host_build_ms = (perf_counter() - build_started) * 1000.0
    index_path = output_dir / "digits-pca-hnsw.npz"
    save_index(index, index_path)

    host_ids = []
    host_distances = []
    host_stats = []
    host_started = perf_counter()
    for query in queries:
        ids, distances, stats = search_index(
            index,
            query,
            k=args.top_k,
            ef_search=args.ef_search,
        )
        host_ids.append(ids)
        host_distances.append(distances)
        host_stats.append(stats)
    host_search_ms = (perf_counter() - host_started) * 1000.0
    host_ids_array = np.stack(host_ids)

    cpu_index = hnswlib.Index(space="l2", dim=args.dimensions)
    hnswlib_build_started = perf_counter()
    cpu_index.init_index(
        max_elements=args.index_size,
        ef_construction=args.ef_construction,
        M=args.m,
        random_seed=args.seed,
    )
    cpu_index.add_items(index_vectors, np.arange(args.index_size))
    hnswlib_build_ms = (perf_counter() - hnswlib_build_started) * 1000.0
    cpu_index.set_ef(args.ef_search)
    for _ in range(args.warmup):
        cpu_index.knn_query(queries[:1], k=args.top_k, num_threads=args.cpu_threads)
    hnswlib_samples_ms = []
    for _ in range(args.cpu_timing_repeats):
        hnswlib_search_started = perf_counter()
        hnswlib_ids, hnswlib_distances = cpu_index.knn_query(
            queries, k=args.top_k, num_threads=args.cpu_threads
        )
        hnswlib_samples_ms.append((perf_counter() - hnswlib_search_started) * 1000.0)
    hnswlib_search_ms = float(np.mean(hnswlib_samples_ms))
    exact_ids = exact_neighbors(index_vectors, queries, args.top_k)

    npu_ids: list[np.ndarray] = []
    npu_distances: list[np.ndarray] = []
    npu_stats = []
    patch_evidence = None
    build_manifest = None
    runtime_provenance = None
    movi_toolchain = None
    carrier_explanation = None
    npu_build_ms = None
    npu_cache_load_ms = None
    npu_cache_status = None
    npu_cache_diagnostic = None
    npu_submission_wall_ms = None
    npu_static_input_upload_ms = None
    npu_graph_samples_ms = None
    npu_end_to_end_samples_ms = None
    npu_wall_samples_ms = None
    npu_query_update_samples_ms = None
    npu_unpack_samples_ms = None
    if not args.cpu_only:
        cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "program-cache"
        batch_capacity = max(4, ((args.batch_size + 3) // 4) * 4)
        with NpuHnsw(
            index,
            movi_dll_dir=args.movi_dll_dir,
            native_dir=args.native_dir,
            cache_dir=cache_dir,
            batch_capacity=batch_capacity,
        ) as backend:
            npu_build_ms = backend.build_ms
            npu_cache_load_ms = backend.cache_load_ms
            npu_cache_status = backend.cache_status
            npu_cache_diagnostic = backend.cache_diagnostic
            npu_static_input_upload_ms = backend.static_input_upload_ms
            patch_evidence = backend.patch_evidence
            build_manifest = backend.build_manifest
            runtime_provenance = backend.runtime_provenance
            movi_toolchain = backend.movi_toolchain
            carrier_explanation = backend.carrier_explanation
            for _ in range(args.warmup):
                backend.search_many(
                    queries[: args.batch_size],
                    k=args.top_k,
                    ef_search=args.ef_search,
                    verify=False,
                )
            npu_graph_samples_ms = []
            npu_end_to_end_samples_ms = []
            npu_wall_samples_ms = []
            npu_query_update_samples_ms = []
            npu_unpack_samples_ms = []
            for repeat in range(args.timing_repeats):
                submission_started = perf_counter()
                repeat_timings = []
                for start in range(0, len(queries), args.batch_size):
                    batch = backend.search_many(
                        queries[start : start + args.batch_size],
                        k=args.top_k,
                        ef_search=args.ef_search,
                        verify=False,
                    )
                    repeat_timings.append(batch.timings)
                    if repeat == 0:
                        for result in batch.results:
                            npu_ids.append(result.ids)
                            npu_distances.append(result.distances)
                            npu_stats.append(result.stats)
                npu_wall_samples_ms.append((perf_counter() - submission_started) * 1000.0)
                npu_graph_samples_ms.append(sum(item["graphinfer_wall_ms"] for item in repeat_timings))
                npu_query_update_samples_ms.append(sum(item["upload_ms"] for item in repeat_timings))
                npu_unpack_samples_ms.append(sum(item["unpack_ms"] for item in repeat_timings))
                npu_end_to_end_samples_ms.append(
                    sum(
                        item["pack_ms"]
                        + item["upload_ms"]
                        + item["graphinfer_wall_ms"]
                        + item["download_ms"]
                        + item["unpack_ms"]
                        for item in repeat_timings
                    )
                )
            npu_submission_wall_ms = float(np.mean(npu_wall_samples_ms))
        npu_ids_array = np.stack(npu_ids)
        if not np.array_equal(npu_ids_array, host_ids_array):
            raise RuntimeError("NPU and host HNSW neighbor IDs differ")
        for npu_distance, host_distance in zip(npu_distances, host_distances):
            if not np.array_equal(npu_distance.view("V4"), host_distance.view("V4")):
                raise RuntimeError("NPU and host HNSW distances differ")
        if npu_stats != host_stats:
            raise RuntimeError("NPU and host HNSW traversal counters differ")

    index_bytes = index_path.read_bytes()
    report = {
        "schema": "npunlock.contrib.hnsw.v3",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "hnswlib": getattr(hnswlib, "__version__", "0.8.0"),
        },
        "dataset": {
            "name": "scikit-learn digits",
            "source_samples": int(len(raw)),
            "index_samples": args.index_size,
            "query_samples": args.query_count,
            "embedding": "StandardScaler followed by full-solver PCA",
            "dimensions": args.dimensions,
            "selected_source_indices": order.tolist(),
            "selected_labels": labels[order].tolist(),
            "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        },
        "index": {
            "algorithm": "HNSW with simple nearest-neighbor selection",
            "distance": "squared L2",
            "m": args.m,
            "ef_construction": args.ef_construction,
            "entry_point": index.entry_point,
            "max_level": index.max_level,
            "level_histogram": {str(level): index.levels.count(level) for level in range(index.max_level + 1)},
            "serialized_path": str(index_path),
            "serialized_sha256": sha256_bytes(index_bytes),
        },
        "search": {
            "top_k": args.top_k,
            "ef_search": args.ef_search,
            "host_hnsw_recall_at_k": recall_at_k(host_ids_array, exact_ids),
            "hnswlib_recall_at_k": recall_at_k(hnswlib_ids.astype(np.int32), exact_ids),
            "host_and_npu_exact_match": None if args.cpu_only else True,
            "mean_visited_nodes": float(np.mean([stats.visited_count for stats in host_stats])),
            "max_visited_nodes": int(max(stats.visited_count for stats in host_stats)),
            "mean_distance_evaluations": float(np.mean([stats.distance_count for stats in host_stats])),
            "total_upper_layer_hops": int(sum(stats.upper_hops for stats in host_stats)),
            "queries_terminated_by_bound": int(sum(stats.terminated_by_bound for stats in host_stats)),
            "timing_warmup_submissions": args.warmup,
            "timing_repeats": args.timing_repeats,
            "cpu_timing_repeats": args.cpu_timing_repeats,
            "hnswlib_threads": args.cpu_threads,
            "queries_per_submission": args.batch_size,
            "carrier_query_capacity": None if args.cpu_only else batch_capacity,
            "npu_submissions_per_repeat": (
                None if args.cpu_only else (args.query_count + args.batch_size - 1) // args.batch_size
            ),
        },
        "timings_ms": {
            "host_index_build": host_build_ms,
            "host_hnsw_all_queries": host_search_ms,
            "hnswlib_index_build": hnswlib_build_ms,
            "hnswlib_all_queries": hnswlib_search_ms,
            "hnswlib_all_queries_median": float(np.median(hnswlib_samples_ms)),
            "hnswlib_all_queries_p95": float(np.percentile(hnswlib_samples_ms, 95)),
            "npu_program_build": npu_build_ms,
            "npu_program_cache_load": npu_cache_load_ms,
            "npu_static_input_upload": npu_static_input_upload_ms,
            "npu_submission_loop_wall": npu_submission_wall_ms,
            "npu_submission_loop_wall_median": (None if args.cpu_only else float(np.median(npu_wall_samples_ms))),
            "npu_submission_loop_wall_p95": (None if args.cpu_only else float(np.percentile(npu_wall_samples_ms, 95))),
            "npu_graphinfer_all_queries": (None if args.cpu_only else float(np.mean(npu_graph_samples_ms))),
            "npu_query_update_all_queries": (None if args.cpu_only else float(np.mean(npu_query_update_samples_ms))),
            "npu_result_unpack_all_queries": (None if args.cpu_only else float(np.mean(npu_unpack_samples_ms))),
            "npu_end_to_end_all_queries": (None if args.cpu_only else float(np.mean(npu_end_to_end_samples_ms))),
            "npu_graphinfer_per_query_amortized": (
                None if args.cpu_only else float(np.mean(npu_graph_samples_ms)) / args.query_count
            ),
            "npu_graphinfer_all_queries_median": (None if args.cpu_only else float(np.median(npu_graph_samples_ms))),
            "npu_graphinfer_all_queries_p95": (
                None if args.cpu_only else float(np.percentile(npu_graph_samples_ms, 95))
            ),
            "npu_end_to_end_all_queries_median": (
                None if args.cpu_only else float(np.median(npu_end_to_end_samples_ms))
            ),
            "npu_end_to_end_all_queries_p95": (
                None if args.cpu_only else float(np.percentile(npu_end_to_end_samples_ms, 95))
            ),
        },
        "kernel": {
            "path": str(KERNEL_PATH),
            "source_sha256": sha256_bytes(KERNEL_PATH.read_bytes()),
            "operations_on_act_shave": [
                "squared-L2 distance evaluation",
                "upper-layer greedy descent",
                "base-layer neighbor enumeration",
                "visited-set checks and updates",
                "candidate min-heap insertion and removal",
                "bounded-result max-heap maintenance",
                "efSearch termination decision",
            ],
        },
        "npu": {
            "executed": not args.cpu_only,
            "patch_evidence": patch_evidence,
            "build_manifest": build_manifest,
            "runtime_provenance": runtime_provenance,
            "movi_toolchain": movi_toolchain,
            "carrier_explanation": carrier_explanation,
            "program_cache": {
                "status": npu_cache_status,
                "diagnostic": npu_cache_diagnostic,
            },
        },
        "results": [
            {
                "query": row,
                "host_ids": host_ids_array[row].tolist(),
                "host_distances": [float(value) for value in host_distances[row]],
                "hnswlib_ids": hnswlib_ids[row].tolist(),
                "hnswlib_distances": [float(value) for value in hnswlib_distances[row]],
                "exact_ids": exact_ids[row].tolist(),
                "stats": asdict(host_stats[row]),
            }
            for row in range(args.query_count)
        ],
    }
    report_path = output_dir / "report.json"
    temporary_path = report_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, report_path)
    print(f"wrote {report_path}")
    print(
        f"host HNSW recall@{args.top_k}={report['search']['host_hnsw_recall_at_k']:.3f}, "
        f"hnswlib={report['search']['hnswlib_recall_at_k']:.3f}"
    )
    if args.cpu_only:
        print("NPU execution skipped by --cpu-only")
    else:
        print(
            f"NPU matched all {args.query_count} host traversals; "
            f"upper-layer hops={report['search']['total_upper_layer_hops']}"
        )
        npu_graph_ms = report["timings_ms"]["npu_graphinfer_all_queries"]
        npu_api_ms = report["timings_ms"]["npu_submission_loop_wall"]
        cpu_ms = report["timings_ms"]["hnswlib_all_queries"]
        print(
            f"steady-state mean: NPU graph={npu_graph_ms:.3f} ms "
            f"({args.query_count * 1000.0 / npu_graph_ms:.0f} query/s), "
            f"NPU API={npu_api_ms:.3f} ms ({args.query_count * 1000.0 / npu_api_ms:.0f} query/s), "
            f"hnswlib={cpu_ms:.3f} ms ({args.query_count * 1000.0 / cpu_ms:.0f} query/s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
