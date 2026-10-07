from __future__ import annotations

import ctypes
import hashlib
import io
import json
import sys
import unittest
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import npunlock as npu
from npunlock import _native
from npunlock.doctor import collect_diagnostics, human_report


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
        with self.assertRaisesRegex(ValueError, "limited to documented same-spec operators"):
            npu.Unknown(x)

    def test_documented_same_spec_operators_infer_output_contracts(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.input("y", shape=(1, 32), dtype="f16")

        unary = npu.Sqrt(npu.Abs(x))
        binary = npu.Maximum(x, y)

        self.assertEqual(unary.spec, x.spec)
        self.assertEqual(binary.spec, x.spec)

    def test_same_spec_inference_rejects_mismatch_and_partial_override(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        different_shape = npu.input("different", shape=(1, 1), dtype="f16")
        with self.assertRaisesRegex(ValueError, "requires _shape and _dtype"):
            npu.Add(x, different_shape)
        with self.assertRaisesRegex(ValueError, "require both _shape and _dtype"):
            npu.Abs(x, _shape=x.shape)

    def test_custom_infers_single_output_contract_from_first_input(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        inherited = npu.custom(x, source=b"kernel", carrier="Abs")
        shape_override = npu.custom(
            x,
            source=b"kernel",
            carrier="Abs",
            _shape=(2, 16),
        )
        dtype_override = npu.custom(
            x,
            source=b"kernel",
            carrier="Abs",
            _dtype="f32",
        )

        self.assertEqual(inherited.spec, x.spec)
        self.assertEqual(shape_override.shape, (2, 16))
        self.assertEqual(shape_override.dtype, "f16")
        self.assertEqual(dtype_override.shape, (1, 32))
        self.assertEqual(dtype_override.dtype, "f32")

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
            _name="custom_f32",
        )
        serialized = npu.serialize_ir(npu.Graph([x], [y]))
        root = ET.fromstring(serialized.xml)
        attribute = root.find("./layers/layer[@name='custom_f32']/rt_info/attribute")
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
            self.assertEqual(_native._native_directory(None), Path("environment-native").resolve())
            self.assertEqual(
                _native._native_directory("explicit-native"),
                Path("explicit-native").resolve(),
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
        self.assertEqual(ctypes.sizeof(_native._InferOptions), 16)
        self.assertEqual(ctypes.sizeof(_native._InferInput), 40)
        self.assertEqual(ctypes.sizeof(_native._InferOutput), 104)
        self.assertEqual(ctypes.sizeof(_native._InferResult), 80)
        self.assertEqual(ctypes.sizeof(_native._InferSessionResult), 72)
        self.assertEqual(ctypes.sizeof(_native._InferSharedBuffer), 72)
        self.assertEqual(ctypes.sizeof(_native._InferSharedTensor), 32)
        self.assertEqual(ctypes.sizeof(_native._InferSessionRunResult), 48)

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


class DoctorTests(unittest.TestCase):
    def test_doctor_distinguishes_run_and_custom_compile_capabilities(self) -> None:
        fake = FakeNative()
        root = ROOT / "build" / "doctor-test-movitools"
        files = tuple(
            root / name
            for name in (
                "bin/moviCompile64.dll",
                "bin/moviAsm64.dll",
                "bin/moviLLD64.dll",
                "lib/mlibm.a",
            )
        )
        try:
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            report = collect_diagnostics(
                native_dir="unused",
                movi_dll_dir=root,
                libraries=fake,  # type: ignore[arg-type]
            )
            full_report = collect_diagnostics(
                native_dir="unused",
                movi_dll_dir=root,
                full_check=True,
                libraries=fake,  # type: ignore[arg-type]
            )
        finally:
            for path in files:
                path.unlink(missing_ok=True)
            for path in (root / "bin", root / "lib", root):
                if path.exists():
                    path.rmdir()

        self.assertEqual(report["schema"], "npunlock.doctor.v2")
        self.assertEqual(report["npu_driver"]["compiler_version"], [8, 3])  # type: ignore[index]
        self.assertEqual(
            report["capabilities"],
            {"run_saved_program": True, "compile_custom_c": True},
        )
        self.assertIn("compiler: 8.3", human_report(report))
        self.assertEqual(
            full_report["custom_kernel_check"],
            {
                "status": "ok",
                "oracle": "fp16_add1_exact",
                "element_count": 32,
                "mismatch_count": 0,
            },
        )
        self.assertIn("full custom-kernel oracle: ok", human_report(full_report))

        without_movi = collect_diagnostics(
            native_dir="unused",
            movi_dll_dir="",
            check_driver=False,
            libraries=fake,  # type: ignore[arg-type]
        )
        self.assertEqual(without_movi["movitools"]["status"], "not-configured")  # type: ignore[index]
        self.assertEqual(
            without_movi["capabilities"],
            {"run_saved_program": True, "compile_custom_c": False},
        )
        full_without_movi = collect_diagnostics(
            native_dir="unused",
            movi_dll_dir="",
            full_check=True,
            libraries=fake,  # type: ignore[arg-type]
        )
        self.assertEqual(full_without_movi["custom_kernel_check"]["status"], "not-run")  # type: ignore[index]


class FakeNative:
    def __init__(self) -> None:
        self.xml = b""
        self.weights = b""
        self.compile_ir_count = 0
        self.shave_calls: list[tuple[bytes, dict[str, object]]] = []
        self.patch_calls: list[tuple[bytes, bytes, tuple[npu.PatchTarget, ...]]] = []
        self.discovery_groups = ((npu.PatchTarget(0, 0, 1, 8, 16),),)
        self.infer_dtype = "f16"
        self.sessions: list[FakeInferenceSession] = []

    def compile_ir(self, xml: bytes, weights: bytes, **kwargs: object) -> npu.IrCompileResult:
        self.compile_ir_count += 1
        self.xml = xml
        self.weights = weights
        self.compile_ir_kwargs = kwargs
        return npu.IrCompileResult(b"native", 0, 0, 1, 0x8086, 0x7D1D, 0x10012, (8, 3))

    def compile_shave(self, source: bytes, **kwargs: object) -> bytes:
        self.shave_calls.append((source, dict(kwargs)))
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

    def discover_patch_targets_v2(self, graph_blob: bytes) -> tuple[tuple[npu.PatchTargetV2, ...], ...]:
        self.discovery_v2_blob = graph_blob
        return self.discovery_groups_v2

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

    def create_inference_session(self, graph_blob: bytes, **kwargs: object) -> object:
        self.shared_session_args = (graph_blob, kwargs)
        self.shared_session = FakeInferenceSession(self, graph_blob)
        self.sessions.append(self.shared_session)
        return self.shared_session


class FakeSharedBuffer:
    def __init__(self, session: object, size: int):
        self._session = session
        self._storage = ctypes.create_string_buffer(size)
        self.address = ctypes.addressof(self._storage)
        self.size = size


class FakeInferenceSession:
    def __init__(self, libraries: FakeNative, graph_blob: bytes):
        self._libraries = libraries
        self.graph_blob = graph_blob
        self.calls: list[tuple[object, object]] = []
        self.copied_calls: list[tuple[npu.InferenceInput, ...]] = []
        self.closed = False

    def create_buffer(self, size: int) -> FakeSharedBuffer:
        return FakeSharedBuffer(self, size)

    def infer(self, inputs: object, outputs: object) -> None:
        if self.closed:
            raise RuntimeError("inference session is closed")
        input_values = tuple(inputs)  # type: ignore[arg-type]
        output_values = tuple(outputs)  # type: ignore[arg-type]
        self.calls.append((input_values, output_values))
        source_buffer = input_values[0][1]
        output_buffer = output_values[0][1]
        dtype = np.dtype({"f16": "float16", "f32": "float32"}[self._libraries.infer_dtype])
        source_bytes = (ctypes.c_uint8 * source_buffer.size).from_address(source_buffer.address)
        output_bytes = (ctypes.c_uint8 * output_buffer.size).from_address(output_buffer.address)
        source = np.frombuffer(source_bytes, dtype=dtype)
        destination = np.frombuffer(output_bytes, dtype=dtype)
        destination[:] = source + dtype.type(1)

    def infer_copied(self, inputs: object) -> npu.InferenceResult:
        if self.closed:
            raise RuntimeError("inference session is closed")
        values = tuple(inputs)  # type: ignore[arg-type]
        self.copied_calls.append(values)
        return self._libraries.infer_graph(self.graph_blob, values)

    def close(self) -> None:
        self.closed = True


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

    def test_large_custom_op_accepts_exact_cover_partition_groups(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f32")
        y = npu.custom(x, source=b"kernel", carrier="Abs")
        targets = tuple(npu.PatchTarget(index, index, 1, 8, 32, 0x3B) for index in range(4))
        fake = FakeNative()
        fake.discovery_groups = ((targets[0], targets[1]), (targets[2], targets[3]))

        npu.compile(
            npu.Graph([x], [y]),
            native_dir="unused",
            movi_dll_dir="movi",
            libraries=fake,  # type: ignore[arg-type]
        )

        self.assertEqual(fake.patch_args[2], targets)

    def test_prepared_graph_explains_and_builds_ambiguous_branches_once(self) -> None:
        shape = (1, 32)
        x = npu.input("x", shape=shape, dtype="f32")
        a = npu.input("a", shape=shape, dtype="f16")
        b = npu.input("b", shape=shape, dtype="f16")
        scaled = npu.custom(x, source=b"scale", carrier="Abs", _name="scaled")
        mixed = npu.custom(a, b, source=b"mix", carrier="Maximum", _name="mixed")
        graph = npu.Graph([x, a, b], [mixed, scaled], name="ambiguous_branches")

        unary = (npu.PatchTarget(0, 0, 1, 32, 128, 0x3B),)
        binary = (npu.PatchTarget(1, 1, 2, 32, 64, 0x1F),)
        fake = FakeNative()
        fake.discovery_groups = (unary, binary)

        prepared = npu.prepare(graph, native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        self.assertEqual(fake.compile_ir_count, 1)
        self.assertEqual(tuple(mapping.status for mapping in prepared.mappings), ("explicit-required",) * 2)
        self.assertIn("does not match its input arity", prepared.explain())
        (unary_group,) = prepared.find_groups(input_count=1, dtype="f32")
        (binary_group,) = prepared.find_groups(input_count=2, dtype="f16")

        plan = prepared.plan(
            bindings={scaled: unary_group, mixed: binary_group},
            kernels={
                scaled: npu.KernelSpec(b"scale override", definitions=("GAIN=2",)),
                mixed: npu.KernelSpec(b"mix override", linker_script=b"script"),
            },
        )
        explanation = plan.explain()

        self.assertEqual(fake.shave_calls, [])
        self.assertIn("build plan for graph 'ambiguous_branches'", explanation)
        self.assertIn("scaled: groups=0", explanation)
        self.assertIn("mixed: groups=1", explanation)
        self.assertIn("GAIN=2", explanation)
        self.assertIn(hashlib.sha256(b"mix override").hexdigest(), explanation)

        program = plan.build(movi_dll_dir="movi")

        self.assertEqual(fake.compile_ir_count, 1)
        self.assertEqual(
            fake.shave_calls,
            [
                (
                    b"mix override",
                    {
                        "movi_dll_dir": "movi",
                        "linker_script": b"script",
                        "definitions": (),
                        "timeout_ms": 20_000,
                        "worker": None,
                    },
                ),
                (
                    b"scale override",
                    {
                        "movi_dll_dir": "movi",
                        "linker_script": None,
                        "definitions": ("GAIN=2",),
                        "timeout_ms": 20_000,
                        "worker": None,
                    },
                ),
            ],
        )
        self.assertEqual(fake.patch_calls[0][2], binary)
        self.assertEqual(fake.patch_calls[1][2], unary)
        self.assertEqual(program.graph_blob, b"patched2")

    def test_build_plan_reuses_identical_kernel_compilation(self) -> None:
        shape = (1, 32)
        x = npu.input("x", shape=shape, dtype="f16")
        first = npu.custom(x, source=b"same", carrier="Abs", _name="first")
        second = npu.custom(first, source=b"same", carrier="Sqrt", _name="second")
        fake = FakeNative()
        fake.discovery_groups = (
            (npu.PatchTarget(0, 0, 1, 32, 64, 0x1F),),
            (npu.PatchTarget(1, 1, 1, 32, 64, 0x1F),),
        )

        prepared = npu.prepare(
            npu.Graph([x], [second], name="shared_kernel_plan"),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )
        plan = prepared.plan(definitions=("MODE=1",))
        program = plan.build(movi_dll_dir="movi")

        self.assertEqual(plan.group_indices, ((0,), (1,)))
        self.assertEqual(len(fake.shave_calls), 1)
        self.assertEqual(fake.shave_calls[0][1]["definitions"], ("MODE=1",))
        self.assertEqual(len(fake.patch_calls), 2)
        self.assertEqual(program.graph_blob, b"patched2")

    def test_kernel_spec_snapshots_files_and_rejects_invalid_definitions(self) -> None:
        source_path = ROOT / "build" / "kernel-spec-test.c"
        linker_path = ROOT / "build" / "kernel-spec-test.ld"
        source_path.write_bytes(b"source v1")
        linker_path.write_bytes(b"linker v1")
        try:
            kernel = npu.KernelSpec(
                source_path,
                definitions=("COUNT=32",),
                linker_script=linker_path,
            )
            source_path.write_bytes(b"source v2")
            linker_path.write_bytes(b"linker v2")
        finally:
            source_path.unlink(missing_ok=True)
            linker_path.unlink(missing_ok=True)

        self.assertEqual(kernel.source, b"source v1")
        self.assertEqual(kernel.linker_script, b"linker v1")
        self.assertEqual(kernel.source_sha256, hashlib.sha256(b"source v1").hexdigest())
        with self.assertRaisesRegex(ValueError, "NAME=DECIMAL"):
            npu.KernelSpec(b"source", definitions=("lower=1",))

    def test_prepared_graph_rejects_incomplete_duplicate_and_incompatible_bindings(self) -> None:
        shape = (1, 32)
        x = npu.input("x", shape=shape, dtype="f32")
        a = npu.input("a", shape=shape, dtype="f16")
        b = npu.input("b", shape=shape, dtype="f16")
        scaled = npu.custom(x, source=b"scale", carrier="Abs", _name="scaled")
        mixed = npu.custom(a, b, source=b"mix", carrier="Maximum", _name="mixed")
        fake = FakeNative()
        fake.discovery_groups = (
            (npu.PatchTarget(0, 0, 1, 32, 128, 0x3B),),
            (npu.PatchTarget(1, 1, 2, 32, 64, 0x1F),),
        )
        prepared = npu.prepare(
            npu.Graph([x, a, b], [mixed, scaled]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        with self.assertRaisesRegex(ValueError, "needs explicit bindings"):
            prepared.build(bindings={scaled: prepared.groups[0]}, movi_dll_dir="movi")
        with self.assertRaisesRegex(ValueError, "input arity"):
            prepared.build(
                bindings={scaled: prepared.groups[1], mixed: prepared.groups[0]},
                movi_dll_dir="movi",
            )
        with self.assertRaisesRegex(ValueError, "assigned to both"):
            prepared.build(
                bindings={scaled: prepared.groups[1], mixed: prepared.groups[1]},
                movi_dll_dir="movi",
            )

    def test_prepared_graph_accepts_explicit_partition_group_sequence(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f32")
        y = npu.custom(x, source=b"kernel", carrier="Abs", _name="partitioned")
        targets = tuple(npu.PatchTarget(index, index, 1, 8, 32, 0x3B) for index in range(4))
        fake = FakeNative()
        fake.discovery_groups = tuple((target,) for target in targets)

        prepared = npu.prepare(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        program = prepared.build(bindings={y: prepared.groups}, movi_dll_dir="movi")

        self.assertEqual(fake.patch_args[2], targets)
        self.assertEqual(program.graph_blob, b"patched")

    def test_prepared_graph_exposes_mixed_precision_tensor_contracts(self) -> None:
        class MixedNative(FakeNative):
            def discover_patch_targets(self, graph_blob: bytes) -> tuple[tuple[npu.PatchTarget, ...], ...]:
                raise npu.NativeError("patchblob discovery", 5, "unsupported", b"mixed precision")

        tensors = (
            npu.PatchTensorContract("input", 0, "f32", 16, 64, 0x07),
            npu.PatchTensorContract("output", 0, "f16", 16, 32, 0x0F),
        )
        targets = tuple(npu.PatchTargetV2(index, index, tensors, 1) for index in range(2))
        reverse_tensors = (
            npu.PatchTensorContract("input", 0, "f16", 16, 32, 0x07),
            npu.PatchTensorContract("output", 0, "f32", 16, 64, 0x0F),
        )
        reverse_targets = tuple(npu.PatchTargetV2(index + 2, index + 2, reverse_tensors, 1) for index in range(2))
        fake = MixedNative()
        fake.discovery_groups_v2 = (targets, reverse_targets)
        x = npu.input("x", shape=(1, 16), dtype="f32")
        y = npu.custom(x, source=b"convert", carrier="Abs", _dtype="f16", _name="converted")

        prepared = npu.prepare(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]

        self.assertEqual(fake.discovery_v2_blob, b"native")
        self.assertEqual(prepared.mappings[0].status, "explicit-required")
        self.assertEqual(prepared.mappings[0].group_indices, ())
        (group,) = prepared.find_groups(input_dtypes=("f32",), output_dtype="f16")
        self.assertTrue(group.unary_conversion)
        self.assertIn("f32->f16", prepared.explain())
        with self.assertRaisesRegex(ValueError, "tensor precisions"):
            prepared.build(bindings={y: prepared.groups[1]}, movi_dll_dir="movi")
        program = prepared.build(bindings={y: group}, movi_dll_dir="movi")
        self.assertEqual(fake.patch_args[2], targets)
        self.assertEqual(program.graph_blob, b"patched")

    def test_partition_groups_must_exactly_cover_custom_output(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f32")
        y = npu.custom(x, source=b"kernel", carrier="Abs")
        fake = FakeNative()
        fake.discovery_groups = (
            (npu.PatchTarget(0, 0, 1, 8, 32, 0x3B),),
            (npu.PatchTarget(1, 1, 1, 8, 32, 0x3B),),
        )

        with self.assertRaisesRegex(ValueError, "unique exact-cover partition mapping"):
            npu.compile(
                npu.Graph([x], [y]),
                native_dir="unused",
                movi_dll_dir="movi",
                libraries=fake,  # type: ignore[arg-type]
            )

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

    def test_shared_arrays_support_numpy_and_shared_inference(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.Abs(x, _shape=x.shape, _dtype=x.dtype)
        fake = FakeNative()
        program = npu.compile(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        shared_input = program.shared_array(x.shape, x.dtype)
        shared_output = program.shared_array(y.shape, y.dtype)
        values = np.arange(32, dtype=np.float16).reshape(1, 32)

        np.add(values, np.float16(2), out=shared_input)
        squared = np.square(shared_input)
        result = program.run({"x": shared_input}, outputs={"Result_0": shared_output})

        self.assertIs(result["Result_0"], shared_output)
        np.testing.assert_array_equal(shared_input, values + np.float16(2))
        np.testing.assert_array_equal(squared, np.square(values + np.float16(2)))
        np.testing.assert_array_equal(shared_output, shared_input + np.float16(1))
        self.assertEqual(len(fake.shared_session.calls), 1)

    def test_program_reuses_session_for_ordinary_inference(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.Abs(x)
        fake = FakeNative()
        program = npu.compile(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        value = np.zeros((1, 32), dtype=np.float16)

        first = program.run({"x": value})
        second = program.run({"x": value + np.float16(2)})

        self.assertEqual(len(fake.sessions), 1)
        self.assertEqual(len(fake.sessions[0].copied_calls), 2)
        np.testing.assert_array_equal(first["Result_0"], value + np.float16(1))
        np.testing.assert_array_equal(second["Result_0"], value + np.float16(3))

    def test_program_shared_tensor_dictionaries_and_close_lifecycle(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.Abs(x)
        fake = FakeNative()
        program = npu.compile(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        inputs = program.shared_inputs()
        outputs = program.shared_outputs()
        inputs["x"][:] = np.arange(32, dtype=np.float16).reshape(1, 32)

        result = program.run(inputs, outputs=outputs)
        program.close()
        program.close()

        self.assertTrue(program.closed)
        self.assertTrue(fake.sessions[0].closed)
        self.assertIs(result["Result_0"], outputs["Result_0"])
        np.testing.assert_array_equal(outputs["Result_0"], inputs["x"] + np.float16(1))
        with self.assertRaisesRegex(RuntimeError, "program is closed"):
            program.run({"x": np.zeros((1, 32), dtype=np.float16)})
        with self.assertRaisesRegex(RuntimeError, "program is closed"):
            program.shared_inputs()

    def test_program_context_manager_closes_session(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.Abs(x)
        fake = FakeNative()
        program = npu.compile(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]

        with program as active:
            active.run({"x": np.zeros((1, 32), dtype=np.float16)})

        self.assertTrue(program.closed)
        with self.assertRaisesRegex(RuntimeError, "program is closed"):
            with program:
                pass

    def test_concurrent_program_runs_share_one_session(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.Abs(x)
        fake = FakeNative()
        program = npu.compile(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        values = tuple(np.full((1, 32), index, dtype=np.float16) for index in range(4))

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = tuple(executor.map(lambda value: program.run({"x": value}), values))

        self.assertEqual(len(fake.sessions), 1)
        self.assertEqual(len(fake.sessions[0].copied_calls), 4)
        for index, result in enumerate(results):
            np.testing.assert_array_equal(result["Result_0"], values[index] + np.float16(1))

    def test_native_blob_bytes_file_and_reload(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.Abs(x, _shape=x.shape, _dtype=x.dtype)
        graph = npu.Graph([x], [y])
        fake = FakeNative()
        program = npu.compile(graph, native_dir="unused", libraries=fake)  # type: ignore[arg-type]

        self.assertEqual(program.to_bytes(), b"native")
        path = ROOT / "build" / "python-api-native-blob-test.bin"
        try:
            program.save(path)
            self.assertEqual(path.read_bytes(), b"native")

            from_bytes = npu.load_native(
                bytearray(program.to_bytes()), graph=graph, libraries=fake  # type: ignore[arg-type]
            )
            from_file = npu.load_native_file(path, graph=graph, libraries=fake)  # type: ignore[arg-type]
        finally:
            path.unlink(missing_ok=True)

        self.assertIsNone(from_bytes.serialized_ir)
        self.assertIsNone(from_bytes.ir_provenance)
        self.assertEqual(from_bytes.patch_reports, ())
        self.assertEqual(from_file.to_bytes(), b"native")
        value = np.zeros((1, 32), dtype=np.float16)
        np.testing.assert_array_equal(from_bytes.run({"x": value})["Result_0"], value + 1)
        np.testing.assert_array_equal(from_file.run({"x": value})["Result_0"], value + 1)

    def test_native_blob_load_validation(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.Abs(x, _shape=x.shape, _dtype=x.dtype)
        graph = npu.Graph([x], [y])
        fake = FakeNative()
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            npu.load_native(b"", graph=graph, libraries=fake)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "bytes-like"):
            npu.load_native("graph.blob", graph=graph, libraries=fake)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "requires a Graph"):
            npu.load_native(b"native", graph=object(), libraries=fake)  # type: ignore[arg-type]

    def test_program_bundle_round_trip_needs_no_symbolic_graph(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs", _name="y")
        fake = FakeNative()
        program = npu.compile(
            npu.Graph([x], [y], name="bundle_test"),
            native_dir="unused",
            movi_dll_dir="movi",
            libraries=fake,  # type: ignore[arg-type]
        )
        first_path = ROOT / "build" / "python-api-program-test.npunlock"
        second_path = ROOT / "build" / "python-api-program-test-2.npunlock"
        try:
            program.export(first_path)
            program.export(second_path)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            with zipfile.ZipFile(first_path, "r") as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["schema"], "npunlock.program.v1")
                self.assertEqual(manifest["inputs"], [{"dtype": "f16", "name": "x", "shape": [1, 32]}])
                self.assertEqual(manifest["outputs"], [{"dtype": "f16", "name": "y", "shape": [1, 32]}])
                self.assertEqual(archive.read("patch-reports/000.json"), b"{}")

            loaded = npu.load(first_path, libraries=fake)  # type: ignore[arg-type]
        finally:
            first_path.unlink(missing_ok=True)
            second_path.unlink(missing_ok=True)

        self.assertIsNone(loaded.graph)
        self.assertIsNone(loaded.serialized_ir)
        self.assertIsNone(loaded.ir_provenance)
        self.assertEqual(loaded.patch_reports, (b"{}",))
        self.assertIsNotNone(loaded.artifact_manifest)
        self.assertEqual(loaded.input_contracts, (npu.TensorContract("x", (1, 32), "f16"),))
        self.assertEqual(loaded.output_contracts, (npu.TensorContract("y", (1, 32), "f16"),))
        value = np.zeros((1, 32), dtype=np.float16)
        np.testing.assert_array_equal(loaded.run({"x": value})["y"], value + np.float16(1))

    def test_program_bundle_rejects_graph_hash_mismatch(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.Abs(x)
        fake = FakeNative()
        program = npu.compile(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        valid_path = ROOT / "build" / "python-api-valid-program.npunlock"
        corrupt_path = ROOT / "build" / "python-api-corrupt-program.npunlock"
        try:
            program.export(valid_path)
            with zipfile.ZipFile(valid_path, "r") as source, zipfile.ZipFile(corrupt_path, "w") as destination:
                for name in source.namelist():
                    data = b"corrupt" if name == "graph.blob" else source.read(name)
                    destination.writestr(name, data)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                npu.load(corrupt_path, libraries=fake)  # type: ignore[arg-type]
        finally:
            valid_path.unlink(missing_ok=True)
            corrupt_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
