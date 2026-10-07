"""Offline smoke test for an installed wheel and its bundled native runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import npunlock as npu
from npunlock.doctor import collect_diagnostics


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/npu3720/add1-shared-1x16.blob"
FIXTURE_SHA256 = "2010915f2d21e07d2afee613ffe8f38a9ce6e39523ab98e0762c1bd40e04d95f"


def main() -> None:
    native = npu.NativeLibraries()
    doctor = collect_diagnostics(
        movi_dll_dir="",
        check_driver=False,
        libraries=native,
    )
    assert doctor["native_runtime"]["status"] == "ok"  # type: ignore[index]
    assert doctor["npu_driver"]["status"] == "skipped"  # type: ignore[index]
    assert doctor["capabilities"] == {"run_saved_program": True, "compile_custom_c": False}

    graph_blob = FIXTURE.read_bytes()
    assert hashlib.sha256(graph_blob).hexdigest() == FIXTURE_SHA256
    x = npu.input("input", shape=(1, 16), dtype="f16")
    y = npu.op("Abs", x, _name="output")
    graph = npu.Graph([x], [y], name="installed_wheel_smoke")
    bundle = ROOT / "build/installed-wheel-smoke.npunlock"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with npu.load_native(graph_blob, graph=graph, libraries=native) as program:
        program.export(bundle)
    with npu.load(bundle, libraries=native) as loaded:
        assert loaded.to_bytes() == graph_blob
        assert [(tensor.name, tensor.shape, tensor.dtype) for tensor in loaded.input_contracts] == [
            ("input", (1, 16), "f16")
        ]
        assert [(tensor.name, tensor.shape, tensor.dtype) for tensor in loaded.output_contracts] == [
            ("output", (1, 16), "f16")
        ]

    print(
        json.dumps(
            {
                "doctor_schema": doctor["schema"],
                "fixture_sha256": FIXTURE_SHA256,
                "program_bundle": str(bundle),
                "status": "ok",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
