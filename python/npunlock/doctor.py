from __future__ import annotations

import os
import platform
import struct
from pathlib import Path

import numpy as np

from ._native import NativeLibraries, _native_directory
from .graph import Graph, input, op
from .ir import serialize_ir

_MOVITOOLS_FILES = (
    "bin/moviCompile64.dll",
    "bin/moviAsm64.dll",
    "bin/moviLLD64.dll",
    "lib/mlibm.a",
)

_ADD1_KERNEL = b"""
#include <npunlock/npu3720_kernel.h>

void controlled_act(unsigned layerParams) {
    act_abi_invocation invocation;
    ACT_ABI_LOAD_INVOCATION32_OR_RETURN(layerParams, invocation);
    const __fp16 *in = ACT_ABI_INPUT_PTR32(const __fp16, invocation, 0u);
    __fp16 *out = ACT_ABI_OUTPUT_PTR32(__fp16, invocation, 1u);
    for (unsigned i = 0; i < invocation.element_count; ++i) {
        out[i] = in[i] + (__fp16)1.0f;
    }
}
"""


def _file_report(path: Path) -> dict[str, object]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "size": path.stat().st_size if exists else None,
    }


def _driver_report(native: NativeLibraries, timeout_ms: int) -> dict[str, object]:
    x = input("doctor_input", shape=(1, 1), dtype="f16")
    y = op("Abs", x, _name="doctor_output")
    assert not isinstance(y, tuple)
    serialized = serialize_ir(Graph([x], [y], name="npunlock_doctor"))
    result = native.compile_ir(
        serialized.xml,
        serialized.weights,
        timeout_ms=timeout_ms,
    )
    return {
        "status": "ok",
        "driver_index": result.driver_index,
        "device_index": result.device_index,
        "driver_version": result.driver_version,
        "vendor_id": result.vendor_id,
        "device_id": result.device_id,
        "graph_extension_version": result.graph_extension_version,
        "compiler_version": list(result.compiler_version),
    }


def _full_check_report(native: NativeLibraries, movi_root: Path, timeout_ms: int) -> dict[str, object]:
    # Import lazily so the ordinary doctor path remains independent of the
    # high-level compiler orchestration and avoids a package import cycle.
    from . import compile as compile_graph
    from .graph import custom

    x = input("doctor_input", shape=(1, 32), dtype="f16")
    y = custom(x, source=_ADD1_KERNEL, carrier="Abs", _name="doctor_output")
    values = np.arange(-16, 16, dtype=np.float16).reshape(1, 32)
    expected = (values + np.float16(1)).astype(np.float16)
    with compile_graph(
        Graph([x], [y], name="npunlock_doctor_full"),
        movi_dll_dir=movi_root,
        timeout_ms=timeout_ms,
        libraries=native,
    ) as program:
        outputs = program.run({"doctor_input": values})
    actual = np.asarray(outputs["doctor_output"])
    mismatch_count = (
        int(expected.size)
        if actual.shape != expected.shape or actual.dtype != expected.dtype
        else int(np.count_nonzero(actual.view(np.uint16) != expected.view(np.uint16)))
    )
    return {
        "status": "ok" if mismatch_count == 0 else "error",
        "oracle": "fp16_add1_exact",
        "element_count": int(expected.size),
        "mismatch_count": mismatch_count,
    }


def collect_diagnostics(
    *,
    native_dir: str | os.PathLike[str] | None = None,
    movi_dll_dir: str | os.PathLike[str] | None = None,
    timeout_ms: int = 20_000,
    check_driver: bool = True,
    full_check: bool = False,
    libraries: NativeLibraries | None = None,
) -> dict[str, object]:
    """Collect installation capabilities and optionally run a custom-kernel oracle."""

    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
        raise ValueError("timeout_ms must be a positive integer")
    runtime_root = _native_directory(native_dir)
    runtime_files = tuple(_file_report(runtime_root / name) for name in ("npunlock.dll", "npunlock_worker.exe"))
    runtime: dict[str, object] = {
        "path": str(runtime_root),
        "files": list(runtime_files),
    }
    native = libraries
    if native is None:
        try:
            native = NativeLibraries(native_dir)
            runtime["status"] = "ok"
        except Exception as exc:  # Native loading raises platform-specific OSError subclasses.
            runtime["status"] = "error"
            runtime["error"] = str(exc)
    else:
        runtime["status"] = "ok"

    selected_movi = movi_dll_dir if movi_dll_dir is not None else os.environ.get("NPUNLOCK_MOVITOOLS_DIR")
    if selected_movi:
        movi_root = Path(selected_movi).expanduser().resolve()
        movi_files = tuple(_file_report(movi_root / Path(name)) for name in _MOVITOOLS_FILES)
        movitools: dict[str, object] = {
            "status": "ok" if all(bool(value["exists"]) for value in movi_files) else "incomplete",
            "path": str(movi_root),
            "files": list(movi_files),
        }
    else:
        movitools = {
            "status": "not-configured",
            "path": None,
            "files": [],
        }

    if not check_driver:
        driver: dict[str, object] = {"status": "skipped"}
    elif native is None:
        driver = {"status": "not-run", "error": "native runtime did not load"}
    else:
        try:
            driver = _driver_report(native, timeout_ms)
        except Exception as exc:
            driver = {"status": "error", "error": str(exc)}

    runtime_ok = runtime["status"] == "ok"
    driver_ok = driver["status"] == "ok"
    ready_to_run = runtime_ok and (driver_ok or not check_driver)
    if not full_check:
        custom_kernel_check: dict[str, object] = {"status": "not-requested"}
    elif not check_driver:
        custom_kernel_check = {"status": "not-run", "error": "full check requires the driver check"}
    elif native is None:
        custom_kernel_check = {"status": "not-run", "error": "native runtime did not load"}
    elif movitools["status"] != "ok":
        custom_kernel_check = {"status": "not-run", "error": "MoviTools is not fully configured"}
    elif not driver_ok:
        custom_kernel_check = {"status": "not-run", "error": "NPU driver/compiler check did not succeed"}
    else:
        try:
            assert selected_movi is not None
            custom_kernel_check = _full_check_report(native, Path(selected_movi).expanduser().resolve(), timeout_ms)
        except Exception as exc:
            custom_kernel_check = {"status": "error", "error": str(exc)}
    return {
        "schema": "npunlock.doctor.v2",
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "pointer_bits": struct.calcsize("P") * 8,
            "python": platform.python_version(),
        },
        "native_runtime": runtime,
        "movitools": movitools,
        "npu_driver": driver,
        "custom_kernel_check": custom_kernel_check,
        "capabilities": {
            "run_saved_program": ready_to_run,
            "compile_custom_c": ready_to_run and movitools["status"] == "ok",
        },
    }


def human_report(report: dict[str, object]) -> str:
    host = report["host"]
    runtime = report["native_runtime"]
    movitools = report["movitools"]
    driver = report["npu_driver"]
    custom_kernel_check = report["custom_kernel_check"]
    capabilities = report["capabilities"]
    assert isinstance(host, dict)
    assert isinstance(runtime, dict)
    assert isinstance(movitools, dict)
    assert isinstance(driver, dict)
    assert isinstance(custom_kernel_check, dict)
    assert isinstance(capabilities, dict)
    lines = [
        "npunlock doctor",
        f"host: {host['system']} {host['release']} {host['machine']} ({host['pointer_bits']}-bit Python)",
        f"native runtime: {runtime['status']} ({runtime['path']})",
        f"MoviTools: {movitools['status']}" + (f" ({movitools['path']})" if movitools["path"] else ""),
        f"NPU driver/compiler: {driver['status']}",
    ]
    if driver["status"] == "ok":
        compiler = driver["compiler_version"]
        assert isinstance(compiler, list)
        lines.append(
            f"  vendor/device: 0x{driver['vendor_id']:04x}/0x{driver['device_id']:04x}; "
            f"graph extension: 0x{driver['graph_extension_version']:x}; "
            f"compiler: {compiler[0]}.{compiler[1]}"
        )
    elif "error" in driver:
        lines.append(f"  {driver['error']}")
    if custom_kernel_check["status"] != "not-requested":
        lines.append(f"full custom-kernel oracle: {custom_kernel_check['status']}")
        if "oracle" in custom_kernel_check:
            lines.append(
                f"  {custom_kernel_check['oracle']}: "
                f"{custom_kernel_check['element_count']} elements, "
                f"{custom_kernel_check['mismatch_count']} mismatches"
            )
        elif "error" in custom_kernel_check:
            lines.append(f"  {custom_kernel_check['error']}")
    lines.extend(
        (
            f"run saved programs: {'yes' if capabilities['run_saved_program'] else 'no'}",
            f"compile custom C: {'yes' if capabilities['compile_custom_c'] else 'no'}",
        )
    )
    return "\n".join(lines)
