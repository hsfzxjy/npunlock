from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import npunlock as npu


TUTORIAL = ROOT / "wiki" / "TUTORIAL.md"


class TutorialTests(unittest.TestCase):
    def test_python_blocks_are_syntactically_valid(self) -> None:
        text = TUTORIAL.read_text(encoding="utf-8")
        blocks = re.findall(r"```python\n(.*?)\n```", text, flags=re.DOTALL)
        self.assertGreaterEqual(len(blocks), 8)
        for index, block in enumerate(blocks):
            with self.subTest(block=index):
                ast.parse(block, filename=f"TUTORIAL.md:python-block-{index}")

    def test_relative_links_exist(self) -> None:
        text = TUTORIAL.read_text(encoding="utf-8")
        destinations = re.findall(r"\[[^]]+\]\(([^)]+)\)", text)
        self.assertGreaterEqual(len(destinations), 15)
        for destination in destinations:
            path = destination.split("#", 1)[0]
            if not path or "://" in path:
                continue
            with self.subTest(destination=destination):
                self.assertTrue((TUTORIAL.parent / path).resolve().exists())

    def test_documented_public_api_and_inference(self) -> None:
        for name in (
            "CompiledKernel",
            "Graph",
            "compile",
            "compile_kernel",
            "configure",
            "custom",
            "input",
            "load",
            "prepare",
        ):
            with self.subTest(name=name):
                self.assertTrue(hasattr(npu, name))

        f16_input = npu.input("f16_input", shape=(1, 32), dtype="f16")
        inherited = npu.custom(f16_input, source=b"kernel", carrier="Abs")
        self.assertEqual(inherited.shape, (1, 32))
        self.assertEqual(inherited.dtype, "f16")

        u8_input = npu.input("u8_input", shape=(1, 32), dtype="u8")
        converted = npu.custom(
            u8_input,
            source=b"kernel",
            carrier="Convert",
            destination_type="f32",
        )
        self.assertEqual(converted.shape, (1, 32))
        self.assertEqual(converted.dtype, "f32")


if __name__ == "__main__":
    unittest.main()
