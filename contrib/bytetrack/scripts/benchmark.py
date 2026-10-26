from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter


CONTRIB_ROOT = Path(__file__).resolve().parents[1]
if str(CONTRIB_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTRIB_ROOT))

import numpy as np

from bytetrack_npu.assignment import CpuLapjvAssignment, NpuAssignment, scipy_assignment


def timing_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(np.ceil(len(ordered) * 0.95)) - 1)]
    return {
        "median": statistics.median(values),
        "p95": p95,
        "mean": statistics.fmean(values),
    }


def measure(callable_object: object, repeats: int) -> dict[str, float]:
    values = []
    for _ in range(repeats):
        started = perf_counter()
        callable_object()
        values.append((perf_counter() - started) * 1000.0)
    return timing_summary(values)


def measure_npu(
    backend: NpuAssignment,
    costs: np.ndarray,
    threshold: float,
    repeats: int,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    total_values = []
    component_values: dict[str, list[float]] = {}
    for _ in range(repeats):
        started = perf_counter()
        result = backend.assign(costs, threshold)
        total_values.append((perf_counter() - started) * 1000.0)
        for name, value in result.timings.items():
            component_values.setdefault(name, []).append(value)
    return (
        timing_summary(total_values),
        {name: timing_summary(values) for name, values in component_values.items()},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark bounded ByteTrack assignment backends")
    parser.add_argument("--movi-dll-dir", type=Path)
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("contrib/bytetrack/outputs/benchmark.json"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--cpu-runs", type=int, default=5000)
    options = parser.parse_args()
    if options.runs <= 0 or options.cpu_runs <= 0 or options.warmup < 0:
        raise ValueError("benchmark repetition counts are invalid")
    lap = CpuLapjvAssignment()
    npu = (
        NpuAssignment(movi_dll_dir=options.movi_dll_dir, native_dir=options.native_dir)
        if options.movi_dll_dir is not None
        else None
    )
    generator = np.random.default_rng(20261026)
    cases = []
    try:
        shapes = (
            (0, 0),
            (0, 16),
            (16, 0),
            (1, 1),
            (1, 16),
            (2, 2),
            (4, 4),
            (4, 16),
            (8, 8),
            (8, 16),
            (12, 12),
            (16, 4),
            (16, 8),
            (16, 16),
        )
        for rows, columns in shapes:
            costs = generator.uniform(0.0, 1.0, size=(rows, columns)).astype(np.float32)
            threshold = 0.8
            reference = scipy_assignment(costs, threshold)
            lap_result = lap.assign(costs, threshold)
            if not np.isclose(reference.augmented_cost, lap_result.augmented_cost, atol=1e-6):
                raise RuntimeError("SciPy and LAPJV baselines disagree")
            case: dict[str, object] = {
                "shape": [rows, columns],
                "matches": len(reference.matches),
                "scipy_ms": measure(lambda: scipy_assignment(costs, threshold), options.cpu_runs),
                "lapjv_ms": measure(lambda: lap.assign(costs, threshold), options.cpu_runs),
            }
            if npu is not None:
                for _ in range(options.warmup):
                    npu.assign(costs, threshold)
                device = npu.assign(costs, threshold)
                if not np.isclose(reference.augmented_cost, device.augmented_cost, atol=1e-5):
                    raise RuntimeError("NPU assignment does not match the SciPy optimum")
                npu_call_ms, npu_components_ms = measure_npu(
                    npu,
                    costs,
                    threshold,
                    options.runs,
                )
                case["npu_call_ms"] = npu_call_ms
                case["npu_components_ms"] = npu_components_ms
            cases.append(case)
    finally:
        if npu is not None:
            npu.close()
    report = {
        "schema": "npunlock.bytetrack.assignment-benchmark.v2",
        "warmup": options.warmup,
        "npu_runs": options.runs,
        "cpu_runs": options.cpu_runs,
        "npu_build_ms": None if npu is None else npu.build_ms,
        "cases": cases,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
