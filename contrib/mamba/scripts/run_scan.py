from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from mamba_npu import NpuSelectiveScan, error_metrics, selective_scan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic Mamba-1 selective scan on ACT-SHAVE")
    parser.add_argument("--movi-dll-dir", required=True)
    parser.add_argument("--native-dir")
    parser.add_argument("--channels", type=int, default=33)
    parser.add_argument("--state-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=7)
    parser.add_argument("--output")
    args = parser.parse_args()
    rng = np.random.default_rng(20261025)
    shape = (args.channels, args.sequence_length)
    u = rng.normal(0.0, 0.4, shape).astype(np.float32)
    delta = rng.normal(-2.0, 0.6, shape).astype(np.float32)
    a = -np.exp(rng.normal(0.0, 0.4, (args.channels, args.state_size))).astype(np.float32)
    b = rng.normal(0.0, 0.3, (args.sequence_length, args.state_size)).astype(np.float32)
    c = rng.normal(0.0, 0.3, (args.sequence_length, args.state_size)).astype(np.float32)
    d = rng.normal(0.0, 0.2, args.channels).astype(np.float32)
    z = rng.normal(0.0, 0.5, shape).astype(np.float32)
    bias = rng.normal(-3.0, 0.5, args.channels).astype(np.float32)
    state = rng.normal(0.0, 0.1, (args.channels, args.state_size)).astype(np.float32)
    expected, expected_state = selective_scan(u, delta, a, b, c, d, z, bias, state)
    with NpuSelectiveScan(a, d, bias, movi_dll_dir=args.movi_dll_dir, native_dir=args.native_dir) as runtime:
        actual, actual_state, timings = runtime.run(u, delta, b, c, z, state)
        evidence = {
            "channels": args.channels,
            "state_size": args.state_size,
            "sequence_length": args.sequence_length,
            "sequence_error": asdict(error_metrics(actual, expected)),
            "state_error": asdict(error_metrics(actual_state, expected_state)),
            "timings_ms": timings,
            "build_ms": runtime.build_ms,
            "patch_report_count": len(runtime.program.patch_evidence),
            "carrier": runtime.carrier_explanation,
        }
    rendered = json.dumps(evidence, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
