from __future__ import annotations

import ctypes
import io
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
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

    def test_f32_custom_marks_carrier_precision_sensitive(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f32")
        y = npu.custom(
            x,
            source=b"void controlled_act(void) {}",
            carrier="Abs",
            _shape=x.shape,
            _dtype=x.dtype,
            _name="custom_f32",
        )
        serialized = npu.serialize_ir(npu.Graph([x], [y]))
        root = ET.fromstring(serialized.xml)
        attribute = root.find(
            "./layers/layer[@name='custom_f32']/rt_info/attribute"
        )
        self.assertIsNotNone(attribute)
        self.assertEqual(
            attribute.attrib,
            {
                "name": "DisablePrecisionConversion",
                "version": "0",
                "value": "dynamic:f16",
            },
        )


class NativeLayoutTests(unittest.TestCase):
    def test_native_directory_precedence(self) -> None:
        bundled = Path(_native.__file__).resolve().parent / "_bin"
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(_native._native_directory(None), bundled)
        with patch.dict("os.environ", {"NPUNLOCK_NATIVE_DIR": "environment-native"}, clear=True):
            self.assertEqual(
                _native._native_directory(None), Path("environment-native").resolve()
            )
            self.assertEqual(
                _native._native_directory("explicit-native"), Path("explicit-native").resolve()
            )

    @unittest.skipUnless(ctypes.sizeof(ctypes.c_void_p) == 8, "MVP ABI is Windows x64")
    def test_ctypes_layouts_match_c_abi(self) -> None:
        self.assertEqual(ctypes.sizeof(_native._View), 16)
        self.assertEqual(ctypes.sizeof(_native._Buffer), 32)
        self.assertEqual(ctypes.sizeof(_native._Diagnostic), 40)
        self.assertEqual(ctypes.sizeof(_native._ShaveOptions), 112)
        self.assertEqual(ctypes.sizeof(_native._ShaveResult), 144)
        self.assertEqual(ctypes.sizeof(_native._IrOptions), 48)
        self.assertEqual(ctypes.sizeof(_native._IrResult), 200)
        self.assertEqual(ctypes.sizeof(_native._PatchOptions), 12)
        self.assertEqual(ctypes.sizeof(_native._PatchTarget), 40)
        self.assertEqual(ctypes.sizeof(_native._PatchDiscoveredTarget), 48)
        self.assertEqual(ctypes.sizeof(_native._PatchDiscoveryResult), 72)
        self.assertEqual(ctypes.sizeof(_native._PatchResult), 112)
        self.assertEqual(ctypes.sizeof(_native._InferOptions), 32)
        self.assertEqual(ctypes.sizeof(_native._InferInput), 40)
        self.assertEqual(ctypes.sizeof(_native._InferOutput), 104)
        self.assertEqual(ctypes.sizeof(_native._InferResult), 144)

    def test_failed_worker_streams_are_written_verbatim(self) -> None:
        class BinaryStream:
            def __init__(self) -> None:
                self.buffer = io.BytesIO()

        stdout = BinaryStream()
        stderr = BinaryStream()
        with patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", stderr):
            _native._report_worker_streams(b"plain stdout\n", b"plain stderr\n", failed=True)
        self.assertEqual(stdout.buffer.getvalue(), b"plain stdout\n")
        self.assertEqual(stderr.buffer.getvalue(), b"plain stderr\n")

    def test_successful_worker_stderr_raises_warning(self) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            _native._report_worker_streams(b"ignored stdout\n", b"worker warning\n", failed=False)
        self.assertEqual(len(captured), 1)
        self.assertEqual(str(captured[0].message), "worker warning\n")
        self.assertIs(captured[0].category, RuntimeWarning)

    def test_patch_target_integer_bounds(self) -> None:
        with self.assertRaises(ValueError):
            npu.PatchTarget(-1, 0, 1, 16, 32)


class FakeNative:
    def __init__(self) -> None:
        self.xml = b""
        self.weights = b""
        self.patch_calls: list[tuple[bytes, bytes, tuple[npu.PatchTarget, ...]]] = []
        self.discovery_groups = ((npu.PatchTarget(0, 0, 1, 8, 16),),)
        self.infer_dtype = "f16"

    def compile_ir(self, xml: bytes, weights: bytes, **kwargs: object) -> npu.IrCompileResult:
        self.xml = xml
        self.weights = weights
        self.compile_ir_kwargs = kwargs
        return npu.IrCompileResult(b"native", 0, 0, 1, 0x8086, 0x7D1D, 0x10012, (8, 3))

    def compile_shave(self, source: bytes, **kwargs: object) -> bytes:
        self.assertions = (source, kwargs)
        return b"elf"

    def patch_graph(self, graph_blob: bytes, shave_elf: bytes, targets: object) -> npu.PatchResult:
        self.patch_args = (graph_blob, shave_elf, tuple(targets))
        self.patch_calls.append(self.patch_args)
        graph = b"patched" if len(self.patch_calls) == 1 else b"patched" + str(len(self.patch_calls)).encode()
        return npu.PatchResult(graph, b"{}")

    def discover_patch_targets(self, graph_blob: bytes) -> tuple[tuple[npu.PatchTarget, ...], ...]:
        self.discovery_blob = graph_blob
        return self.discovery_groups

    def infer_graph(self, graph_blob: bytes, inputs: object, **kwargs: object) -> npu.InferenceResult:
        values = tuple(inputs)  # type: ignore[arg-type]
        self.infer_args = (graph_blob, values, kwargs)
        dtype = np.dtype({"f16": "float16", "f32": "float32"}[self.infer_dtype])
        source = np.frombuffer(values[0].data, dtype=dtype)
        output = (source + dtype.type(1)).astype(dtype).tobytes()
        return npu.InferenceResult(
            (npu.InferenceOutput(1, "Result_0", (1, 32), self.infer_dtype, output),),
            0,
            0,
            1,
            0x8086,
            0x7D1D,
        )


class CompilationFlowTests(unittest.TestCase):
    def tearDown(self) -> None:
        npu.configure(movi_dll_dir=None)

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
            libraries=fake,  # type: ignore[arg-type]
        )
        self.assertIsNone(fake.assertions[1]["linker_script"])
        self.assertEqual(program.graph_blob, b"patched")
        self.assertIn(b'type="Abs"', fake.xml)
        self.assertEqual(fake.patch_args, (b"native", b"elf", (target,)))
        result = program.run({"x": np.zeros((1, 32), dtype=np.float16)})
        np.testing.assert_array_equal(result["Result_0"], np.ones((1, 32), dtype=np.float16))
        self.assertEqual(fake.infer_args[1][0].selector, "x")

    def test_custom_only_graph_infers_targets_by_position(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(
            x,
            source=b"kernel",
            carrier="Abs",
            _shape=x.shape,
            _dtype=x.dtype,
        )
        fake = FakeNative()
        program = npu.compile(
            npu.Graph([x], [y]),
            native_dir="unused",
            movi_dll_dir="movi",
            linker_script=b"script",
            libraries=fake,  # type: ignore[arg-type]
        )
        self.assertEqual(fake.discovery_blob, b"native")
        self.assertEqual(fake.patch_args[2], (npu.PatchTarget(0, 0, 1, 8, 16),))
        self.assertEqual(program.graph_blob, b"patched")

    def test_f32_custom_enables_accuracy_mode_and_runs_f32(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f32")
        y = npu.custom(
            x,
            source=b"kernel",
            carrier="Abs",
            _shape=x.shape,
            _dtype=x.dtype,
        )
        fake = FakeNative()
        fake.discovery_groups = ((npu.PatchTarget(0, 0, 1, 32, 128, 0x3B),),)
        fake.infer_dtype = "f32"
        program = npu.compile(
            npu.Graph([x], [y]),
            native_dir="unused",
            movi_dll_dir="movi",
            libraries=fake,  # type: ignore[arg-type]
        )
        self.assertEqual(
            fake.compile_ir_kwargs["build_flags"],
            '--config EXECUTION_MODE_HINT="ACCURACY"',
        )
        value = np.linspace(-1, 1, 32, dtype=np.float32).reshape(1, 32)
        result = program.run({"x": value})
        np.testing.assert_array_equal(result["Result_0"], value + np.float32(1))
        self.assertEqual(fake.patch_args[2], fake.discovery_groups[0])
        self.assertEqual(program.graph_blob, b"patched")

    def test_mixed_act_graph_maps_custom_node_position(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        ordinary = npu.Exp(x, _shape=x.shape, _dtype=x.dtype)
        y = npu.custom(
            ordinary,
            source=b"kernel",
            carrier="Abs",
            _shape=x.shape,
            _dtype=x.dtype,
        )
        ordinary_group = (npu.PatchTarget(0, 0, 1, 16, 32),)
        custom_group = (npu.PatchTarget(1, 1, 1, 16, 32),)
        fake = FakeNative()
        fake.discovery_groups = (ordinary_group, custom_group)
        npu.compile(
            npu.Graph([x], [y]),
            native_dir="unused",
            movi_dll_dir="movi",
            linker_script=b"script",
            libraries=fake,  # type: ignore[arg-type]
        )
        self.assertEqual(fake.patch_args[2], custom_group)

    def test_multilayer_binary_custom_maps_two_input_group(self) -> None:
        shape = (1, 32)
        x = npu.input("x", shape=shape, dtype="f16")
        y = npu.input("y", shape=shape, dtype="f16")
        abs_x = npu.Abs(x, _shape=shape, _dtype="f16")
        abs_y = npu.Abs(y, _shape=shape, _dtype="f16")
        mixed = npu.custom(
            abs_x,
            abs_y,
            source=b"kernel",
            carrier="Maximum",
            _shape=shape,
            _dtype="f16",
        )
        output = npu.Sqrt(mixed, _shape=shape, _dtype="f16")
        unary_zero = (npu.PatchTarget(0, 0, 1, 16, 32),)
        unary_one = (npu.PatchTarget(1, 1, 1, 16, 32),)
        binary = (npu.PatchTarget(2, 2, 2, 8, 16),)
        unary_three = (npu.PatchTarget(3, 3, 1, 8, 16),)
        fake = FakeNative()
        fake.discovery_groups = (unary_zero, unary_one, binary, unary_three)
        npu.compile(
            npu.Graph([x, y], [output]),
            native_dir="unused",
            movi_dll_dir="movi",
            libraries=fake,  # type: ignore[arg-type]
        )
        self.assertEqual(fake.patch_args[2], binary)

    def test_ambiguous_positional_group_count_is_rejected(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        ordinary = npu.Exp(x, _shape=x.shape, _dtype=x.dtype)
        y = npu.custom(
            ordinary,
            source=b"kernel",
            carrier="Abs",
            _shape=x.shape,
            _dtype=x.dtype,
        )
        with self.assertRaisesRegex(ValueError, "one-to-one positional mapping"):
            npu.compile(
                npu.Graph([x], [y]),
                native_dir="unused",
                movi_dll_dir="movi",
                linker_script=b"script",
                libraries=FakeNative(),  # type: ignore[arg-type]
            )

    def test_custom_chain_maps_topological_positions_to_act_groups(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        first = npu.custom(
            x,
            source=b"first",
            carrier="Abs",
            _shape=x.shape,
            _dtype=x.dtype,
            _name="first",
        )
        second = npu.custom(
            first,
            source=b"second",
            carrier="Abs",
            _shape=x.shape,
            _dtype=x.dtype,
            _name="second",
        )
        group_zero = (npu.PatchTarget(0, 0, 1, 16, 32),)
        group_one = (npu.PatchTarget(1, 1, 1, 16, 32),)
        fake = FakeNative()
        fake.discovery_groups = (group_zero, group_one)
        program = npu.compile(
            npu.Graph([x], [second]),
            native_dir="unused",
            movi_dll_dir="movi",
            linker_script=b"script",
            libraries=fake,  # type: ignore[arg-type]
        )
        self.assertEqual(fake.patch_calls[0], (b"native", b"elf", group_zero))
        self.assertEqual(fake.patch_calls[1], (b"patched", b"elf", group_one))
        self.assertEqual(program.graph_blob, b"patched2")

    def test_movitools_directory_from_environment(self) -> None:
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
        with patch.dict("os.environ", {"NPUNLOCK_MOVITOOLS_DIR": "environment-movi"}):
            npu.compile(
                npu.Graph([x], [y]),
                native_dir="unused",
                linker_script=b"script",
                libraries=fake,  # type: ignore[arg-type]
            )
        self.assertEqual(fake.assertions[1]["movi_dll_dir"], "environment-movi")

    def test_configured_movitools_directory_overrides_environment(self) -> None:
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
        npu.configure(movi_dll_dir="configured-movi")
        with patch.dict("os.environ", {"NPUNLOCK_MOVITOOLS_DIR": "environment-movi"}):
            npu.compile(
                npu.Graph([x], [y]),
                native_dir="unused",
                linker_script=b"script",
                libraries=fake,  # type: ignore[arg-type]
            )
        self.assertEqual(fake.assertions[1]["movi_dll_dir"], "configured-movi")

    def test_program_run_rejects_wrong_tensor_contract(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.Abs(x, _shape=x.shape, _dtype=x.dtype)
        fake = FakeNative()
        program = npu.compile(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            program.run({"x": np.zeros((32,), dtype=np.float16)})


if __name__ == "__main__":
    unittest.main()
