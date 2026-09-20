from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import npunlock as npu
from npunlock import _native


class SymbolicTests(unittest.TestCase):
    def test_dynamic_unknown_operator_and_attributes(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.SomeFutureOp(
            x,
            alpha=0.5,
            mode="new",
            _shape=x.shape,
            _dtype=x.dtype,
            _name="future",
        )
        self.assertEqual(y.producer.op, "SomeFutureOp")
        self.assertEqual(y.producer.attrs, {"alpha": 0.5, "mode": "new"})
        self.assertEqual(y.shape, (1, 32))
        self.assertEqual(y.name, "future")

    def test_generic_op_handles_public_name_collision(self) -> None:
        x = npu.input("x", shape=(1,), dtype="f32")
        y = npu.op("compile", x, _shape=x.shape, _dtype=x.dtype)
        self.assertEqual(y.producer.op, "compile")

    def test_multi_output_and_sugar(self) -> None:
        x = npu.input("x", shape=(1, 4), dtype="f16")
        a, b = npu.Split(
            x,
            axis=1,
            _outputs=[npu.TensorSpec((1, 2), "f16"), npu.TensorSpec((1, 2), "f16")],
        )
        y = a + b
        graph = npu.Graph([x], [y])
        self.assertEqual([node.op for node in graph.nodes], ["Parameter", "Split", "Add"])

    def test_missing_output_contract_is_rejected(self) -> None:
        x = npu.input("x", shape=(1,), dtype="f16")
        with self.assertRaises(ValueError):
            npu.Unknown(x)

    def test_undeclared_graph_input_is_rejected(self) -> None:
        x = npu.input("x", shape=(1,), dtype="f16")
        hidden = npu.input("hidden", shape=(1,), dtype="f16")
        y = npu.Add(x, hidden, _shape=x.shape, _dtype=x.dtype)
        with self.assertRaises(ValueError):
            npu.Graph([x], [y])


class SerializationTests(unittest.TestCase):
    def test_abs_fixture_shape(self) -> None:
        x = npu.input("input", shape=(1, 32), dtype="f16")
        y = npu.Abs(x, _shape=x.shape, _dtype=x.dtype, _name="output")
        serialized = npu.serialize_ir(npu.Graph([x], [y], name="direct_ir_abs"))
        root = ET.fromstring(serialized.xml)
        layers = root.findall("./layers/layer")
        self.assertEqual([layer.attrib["type"] for layer in layers], ["Parameter", "Abs", "Result"])
        self.assertEqual(root.attrib, {"name": "direct_ir_abs", "version": "11"})
        self.assertEqual(len(root.findall("./edges/edge")), 2)
        self.assertEqual(serialized.weights, b"")

    def test_constant_bytes_and_unknown_attributes(self) -> None:
        x = npu.input("x", shape=(1, 2), dtype="f16")
        weights = np.asarray([[1.0, 2.0]], dtype=np.float16)
        w = npu.constant(weights, name="w")
        y = npu.Multiply(x, w, auto_broadcast="numpy", _shape=x.shape, _dtype=x.dtype)
        serialized = npu.serialize_ir(npu.Graph([x], [y]))
        self.assertEqual(serialized.weights, weights.tobytes())
        root = ET.fromstring(serialized.xml)
        const_data = root.find("./layers/layer[@type='Const']/data")
        self.assertIsNotNone(const_data)
        self.assertEqual(const_data.attrib["offset"], "0")
        self.assertEqual(const_data.attrib["size"], "4")
        multiply_data = root.find("./layers/layer[@type='Multiply']/data")
        self.assertEqual(multiply_data.attrib["auto_broadcast"], "numpy")

    def test_custom_lowers_only_to_carrier_ir(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        target = npu.PatchTarget(0, 0, 1, 16, 32)
        y = npu.custom(
            x,
            source=b"void controlled_act(void) {}",
            carrier="Abs",
            _shape=x.shape,
            _dtype=x.dtype,
            _patch_targets=[target],
        )
        serialized = npu.serialize_ir(npu.Graph([x], [y]))
        root = ET.fromstring(serialized.xml)
        self.assertIsNotNone(root.find("./layers/layer[@type='Abs']"))
        self.assertIsNone(root.find("./layers/layer[@type='Custom']"))
        self.assertEqual(serialized.custom_nodes, (y.producer,))


class NativeLayoutTests(unittest.TestCase):
    @unittest.skipUnless(ctypes.sizeof(ctypes.c_void_p) == 8, "MVP ABI is Windows x64")
    def test_ctypes_layouts_match_c_abi(self) -> None:
        self.assertEqual(ctypes.sizeof(_native._View), 16)
        self.assertEqual(ctypes.sizeof(_native._Buffer), 32)
        self.assertEqual(ctypes.sizeof(_native._Diagnostic), 40)
        self.assertEqual(ctypes.sizeof(_native._ShaveOptions), 112)
        self.assertEqual(ctypes.sizeof(_native._ShaveResult), 80)
        self.assertEqual(ctypes.sizeof(_native._IrOptions), 48)
        self.assertEqual(ctypes.sizeof(_native._IrResult), 136)
        self.assertEqual(ctypes.sizeof(_native._PatchOptions), 12)
        self.assertEqual(ctypes.sizeof(_native._PatchTarget), 40)
        self.assertEqual(ctypes.sizeof(_native._PatchResult), 112)

    def test_patch_target_integer_bounds(self) -> None:
        with self.assertRaises(ValueError):
            npu.PatchTarget(-1, 0, 1, 16, 32)


class FakeNative:
    def __init__(self) -> None:
        self.xml = b""
        self.weights = b""

    def compile_ir(self, xml: bytes, weights: bytes, **kwargs: object) -> npu.IrCompileResult:
        self.xml = xml
        self.weights = weights
        return npu.IrCompileResult(b"native", 0, 0, 1, 0x8086, 0x7D1D, 0x10012, (8, 3))

    def compile_shave(self, source: bytes, **kwargs: object) -> bytes:
        self.assertions = (source, kwargs)
        return b"elf"

    def patch_graph(self, graph_blob: bytes, shave_elf: bytes, targets: object) -> npu.PatchResult:
        self.patch_args = (graph_blob, shave_elf, tuple(targets))
        return npu.PatchResult(b"patched", b"{}")


class CompilationFlowTests(unittest.TestCase):
    def test_thin_compile_flow(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        target = npu.PatchTarget(0, 0, 1, 16, 32)
        y = npu.custom(
            x,
            source=b"kernel",
            carrier="Abs",
            _shape=x.shape,
            _dtype=x.dtype,
            _patch_targets=[target],
        )
        fake = FakeNative()
        program = npu.compile(
            npu.Graph([x], [y]),
            native_dir="unused",
            movi_dll_dir="movi",
            linker_script=b"script",
            libraries=fake,  # type: ignore[arg-type]
        )
        self.assertEqual(program.graph_blob, b"patched")
        self.assertIn(b'type="Abs"', fake.xml)
        self.assertEqual(fake.patch_args, (b"native", b"elf", (target,)))
        with self.assertRaises(NotImplementedError):
            program.run({"x": np.zeros((1, 32), dtype=np.float16)})


if __name__ == "__main__":
    unittest.main()
