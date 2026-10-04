from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from .doctor import collect_diagnostics, human_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m npunlock")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="inspect the local npunlock toolchain and NPU")
    doctor.add_argument("--json", action="store_true", help="write the versioned machine-readable report")
    doctor.add_argument("--native-dir", help="development override containing npunlock.dll and its worker")
    doctor.add_argument("--movi-dll-dir", help="MVC_DEPEND root; otherwise use NPUNLOCK_MOVITOOLS_DIR")
    doctor.add_argument("--timeout-ms", type=int, default=20_000, help="finite graph-compiler timeout")
    checks = doctor.add_mutually_exclusive_group()
    checks.add_argument("--skip-driver", action="store_true", help="check package files without querying the NPU")
    checks.add_argument(
        "--full",
        action="store_true",
        help="compile and execute a custom FP16 add-one kernel against an exact host oracle",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    report = collect_diagnostics(
        native_dir=options.native_dir,
        movi_dll_dir=options.movi_dll_dir,
        timeout_ms=options.timeout_ms,
        check_driver=not options.skip_driver,
        full_check=options.full,
    )
    if options.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(human_report(report))
    capabilities = report["capabilities"]
    assert isinstance(capabilities, dict)
    custom_kernel_check = report["custom_kernel_check"]
    assert isinstance(custom_kernel_check, dict)
    if options.full:
        return 0 if custom_kernel_check["status"] == "ok" else 1
    return 0 if capabilities["run_saved_program"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
