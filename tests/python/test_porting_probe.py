from __future__ import annotations

import importlib.util
import shutil
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
MODULE_PATH = ROOT / "tools" / "porting_probe.py"
SPEC = importlib.util.spec_from_file_location("porting_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
porting_probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = porting_probe
SPEC.loader.exec_module(porting_probe)


class PortingProbeTests(unittest.TestCase):
    def test_create_and_verify_fp16_add1_bundle(self) -> None:
        BUILD.mkdir(exist_ok=True)
        graph = ROOT / "tests" / "fixtures" / "npu3720" / "add1-shared-1x16.blob"
        test_root = BUILD / "test-porting-probe"
        if test_root.exists():
            shutil.rmtree(test_root)
        try:
            bundle = test_root / "probe"
            porting_probe.create_bundle(
                graph,
                bundle,
                producer_platform="Windows x64",
                device="Meteor Lake / NPU3720",
                driver_version="test-driver",
                graph_compiler_version="8.3",
            )
            report = porting_probe.verify_bundle(bundle)
            self.assertEqual(report["schema"], porting_probe.VERIFY_SCHEMA)
            self.assertEqual(report["element_count"], 16)
            self.assertEqual(
                report["graph_sha256"],
                "2010915f2d21e07d2afee613ffe8f38a9ce6e39523ab98e0762c1bd40e04d95f",
            )

            expected = bundle / porting_probe.EXPECTED_FILE
            expected.write_bytes(expected.read_bytes()[:-1] + b"\x00")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                porting_probe.verify_bundle(bundle)
        finally:
            if test_root.exists():
                shutil.rmtree(test_root)


if __name__ == "__main__":
    unittest.main()
