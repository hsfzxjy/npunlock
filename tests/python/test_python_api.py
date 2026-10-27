from __future__ import annotations

import ast
import ctypes
import hashlib
import io
import json
import sys
import unittest
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
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
    def test_public_contribs_stay_above_the_npunlock_native_boundary(self) -> None:
        forbidden_attributes = {
            "NativeLibraries",
            "artifact_manifest",
            "compile_ir",
            "compile_shave",
            "configure",
            "ir_provenance",
            "patch_graph",
            "patch_reports",
            "report_json",
        }
        violations: list[str] = []
        for path in sorted((ROOT / "contrib").glob("**/*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            aliases: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "npunlock":
                            aliases.add(alias.asname or alias.name)
                        elif alias.name.startswith("npunlock._"):
                            violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: imports {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    if node.module.startswith("npunlock._"):
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: imports {node.module}")
                    elif node.module == "npunlock":
                        for alias in node.names:
                            if alias.name.startswith("_") or alias.name in forbidden_attributes:
                                violations.append(
                                    f"{path.relative_to(ROOT)}:{node.lineno}: imports npunlock.{alias.name}"
                                )
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and (
                    node.attr in forbidden_attributes
                    or (isinstance(node.value, ast.Name) and node.value.id in aliases and node.attr.startswith("_"))
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: accesses {ast.unparse(node)}")
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in aliases
                    and node.func.attr == "custom"
                    and any(keyword.arg == "_name" for keyword in node.keywords)
                ):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: uses legacy npu.custom(_name=...)")

        self.assertEqual(violations, [])

    def test_every_explicit_public_export_has_a_stub_declaration(self) -> None:
        tree = ast.parse((ROOT / "python/npunlock/__init__.pyi").read_text(encoding="utf-8"))
        declared: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                declared.add(node.name)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                declared.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                declared.update(alias.asname or alias.name for alias in node.names)
        self.assertEqual(sorted(set(npu.__all__) - declared), [])

    def test_act_group_reports_per_invocation_tensor_counts(self) -> None:
        dense = npu.ActGroup(
            0,
            (
                npu.PatchTarget(0, 0, 2, 8, 16, 0x1F),
                npu.PatchTarget(1, 1, 2, 8, 16, 0x1F),
            ),
        )
        scalar = npu.ActGroup(1, (npu.PatchTarget(2, 2, 2, 8, 16, 0x5F),))
        five_input = npu.ActGroup(2, (npu.PatchTarget(3, 3, 5, 8, 16, 0x1F),))
        varying = npu.ActGroup(
            3,
            (
                npu.PatchTarget(4, 4, 1, 8, 16, 0x1F),
                npu.PatchTarget(5, 5, 1, 16, 32, 0x1F),
            ),
        )
        conversion_tensors = (
            npu.PatchTensorContract("input", 0, "f32", 16, 64, 0x07),
            npu.PatchTensorContract("output", 0, "f16", 16, 32, 0x0F),
        )
        conversion = npu.ActGroup(
            4,
            (
                npu.PatchTargetV2(6, 6, conversion_tensors, 1),
                npu.PatchTargetV2(7, 7, conversion_tensors, 1),
            ),
        )
        scalar_v2_tensors = (
            npu.PatchTensorContract("input", 0, "u32", 8, 32, 0x07),
            npu.PatchTensorContract("input", 1, "u32", 1, 4, 0x07),
            npu.PatchTensorContract("output", 0, "u32", 8, 32, 0x0F),
        )
        scalar_v2 = npu.ActGroup(
            5,
            (
                npu.PatchTargetV2(8, 8, scalar_v2_tensors),
                npu.PatchTargetV2(9, 9, scalar_v2_tensors),
            ),
        )
        scalar_first_v2_tensors = (
            npu.PatchTensorContract("input", 0, "f16", 1, 2, 0x07),
            npu.PatchTensorContract("input", 1, "f16", 8, 16, 0x07),
            npu.PatchTensorContract("output", 0, "f16", 8, 16, 0x0F),
        )
        scalar_first_v2 = npu.ActGroup(
            6,
            (
                npu.PatchTargetV2(10, 10, scalar_first_v2_tensors),
                npu.PatchTargetV2(11, 11, scalar_first_v2_tensors),
            ),
        )

        self.assertEqual(dense.per_invocation_input_element_counts, (8, 8))
        self.assertEqual(dense.per_invocation_output_element_count, 8)
        self.assertEqual(scalar.per_invocation_input_element_counts, (8, 1))
        self.assertEqual(scalar.per_invocation_output_element_count, 8)
        self.assertEqual(five_input.per_invocation_input_element_counts, (8, 8, 8, 8, 8))
        self.assertIsNone(varying.per_invocation_input_element_counts)
        self.assertIsNone(varying.per_invocation_output_element_count)
        self.assertEqual(conversion.element_count, 32)
        self.assertEqual(conversion.span_bytes, 64)
        self.assertEqual(conversion.per_invocation_output_element_count, 16)
        self.assertTrue(scalar_v2.input_1_scalar)
        self.assertEqual(scalar_v2.scalar_input_indices, (1,))
        self.assertFalse(scalar_first_v2.input_1_scalar)
        self.assertEqual(scalar_first_v2.scalar_input_indices, (0,))

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

    def test_rank2_matmul_inference_transpose_and_rejections(self) -> None:
        left = npu.input("left", shape=(2, 3), dtype="f16")
        right = npu.input("right", shape=(3, 4), dtype="f16")
        product = left @ right
        self.assertEqual(product.shape, (2, 4))
        self.assertEqual(product.dtype, "f16")
        self.assertEqual(product.producer.attrs, {"transpose_a": False, "transpose_b": False})

        transposed_left = npu.input("transposed_left", shape=(3, 2), dtype="f16")
        transposed_right = npu.input("transposed_right", shape=(4, 3), dtype="f16")
        transposed = npu.MatMul(transposed_left, transposed_right, transpose_a=True, transpose_b=True)
        self.assertEqual(transposed.shape, (2, 4))

        with self.assertRaisesRegex(ValueError, "inner dimensions"):
            npu.MatMul(left, npu.input("bad_inner", shape=(2, 4), dtype="f16"))
        with self.assertRaisesRegex(ValueError, "matching input dtypes"):
            npu.MatMul(left, npu.input("bad_dtype", shape=(3, 4), dtype="f32"))
        with self.assertRaisesRegex(ValueError, "two rank-2 tensors"):
            npu.MatMul(npu.input("batched", shape=(1, 2, 3), dtype="f16"), right)
        with self.assertRaisesRegex(TypeError, "transpose_a must be bool"):
            npu.MatMul(left, right, transpose_a=1)

        explicitly_shaped = npu.MatMul(
            npu.input("batched_left", shape=(1, 2, 3), dtype="f16"),
            npu.input("batched_right", shape=(1, 3, 4), dtype="f16"),
            transpose_a=False,
            transpose_b=False,
            _shape=(1, 2, 4),
            _dtype="f16",
        )
        self.assertEqual(explicitly_shaped.shape, (1, 2, 4))

    def test_structural_operator_inference(self) -> None:
        left = npu.input("left", shape=(2, 3), dtype="i32")
        right = npu.input("right", shape=(1, 3), dtype="i32")
        joined = npu.Concat(left, right, axis=0)
        transposed = npu.Transpose(joined, npu.constant(np.asarray([1, 0], dtype=np.int64)))
        converted = npu.Convert(transposed, destination_type="f32")

        self.assertEqual(joined.spec, npu.TensorSpec((3, 3), "i32"))
        self.assertEqual(transposed.spec, npu.TensorSpec((3, 3), "i32"))
        self.assertEqual(converted.spec, npu.TensorSpec((3, 3), "f32"))

        with self.assertRaisesRegex(TypeError, "integer axis"):
            npu.Concat(left, right, axis=True)
        with self.assertRaisesRegex(ValueError, "out of range"):
            npu.Concat(left, right, axis=2)
        with self.assertRaisesRegex(ValueError, "equal non-axis dimensions"):
            npu.Concat(left, npu.input("bad", shape=(1, 4), dtype="i32"), axis=0)
        with self.assertRaisesRegex(ValueError, "constant order"):
            npu.Transpose(left, npu.input("order", shape=(2,), dtype="i64"))
        with self.assertRaisesRegex(ValueError, "permutation"):
            npu.Transpose(left, npu.constant(np.asarray([0, 0], dtype=np.int64)))

    def test_static_reshape_inference_and_rejections(self) -> None:
        data = npu.input("data", shape=(2, 3, 4), dtype="u8")
        reshaped = npu.Reshape(data, npu.constant(np.asarray([3, 8], dtype=np.int64)))
        inferred = npu.Reshape(data, npu.constant(np.asarray([6, -1], dtype=np.int64)))
        copied = npu.Reshape(
            data,
            npu.constant(np.asarray([0, 12], dtype=np.int64)),
            special_zero=True,
        )

        self.assertEqual(reshaped.spec, npu.TensorSpec((3, 8), "u8"))
        self.assertEqual(inferred.spec, npu.TensorSpec((6, 4), "u8"))
        self.assertEqual(copied.spec, npu.TensorSpec((2, 12), "u8"))
        self.assertEqual(reshaped.producer.attrs, {"special_zero": False})

        with self.assertRaisesRegex(ValueError, "constant pattern"):
            npu.Reshape(data, npu.input("pattern", shape=(2,), dtype="i64"))
        with self.assertRaisesRegex(ValueError, "one-dimensional integer"):
            npu.Reshape(data, npu.constant(np.asarray([[3, 8]], dtype=np.int64)))
        with self.assertRaisesRegex(ValueError, "at most one inferred"):
            npu.Reshape(data, npu.constant(np.asarray([-1, -1], dtype=np.int64)))
        with self.assertRaisesRegex(ValueError, "element counts do not match"):
            npu.Reshape(data, npu.constant(np.asarray([5, 5], dtype=np.int64)))
        with self.assertRaisesRegex(ValueError, "zero dimensions require"):
            npu.Reshape(data, npu.constant(np.asarray([0, 12], dtype=np.int64)))
        with self.assertRaisesRegex(TypeError, "special_zero must be bool"):
            npu.Reshape(data, npu.constant(np.asarray([3, 8], dtype=np.int64)), special_zero=1)

    def test_static_squeeze_unsqueeze_inference_and_rejections(self) -> None:
        data = npu.input("data", shape=(1, 2, 1, 3), dtype="i8")
        axes = npu.constant(np.asarray([0, -2], dtype=np.int64))
        squeezed = npu.Squeeze(data, axes)
        restored = npu.Unsqueeze(squeezed, axes)

        self.assertEqual(squeezed.spec, npu.TensorSpec((2, 3), "i8"))
        self.assertEqual(restored.spec, data.spec)

        with self.assertRaisesRegex(ValueError, "constant axes"):
            npu.Squeeze(data, npu.input("axes", shape=(2,), dtype="i64"))
        with self.assertRaisesRegex(ValueError, "dimensions of size one"):
            npu.Squeeze(data, npu.constant(np.asarray([1], dtype=np.int64)))
        with self.assertRaisesRegex(ValueError, "axes must be unique"):
            npu.Unsqueeze(squeezed, npu.constant(np.asarray([0, 0], dtype=np.int64)))
        with self.assertRaisesRegex(ValueError, "axis is out of range"):
            npu.Unsqueeze(squeezed, npu.constant(np.asarray([4], dtype=np.int64)))

    def test_static_split_inference_and_rejections(self) -> None:
        data = npu.input("data", shape=(2, 6), dtype="i16")
        outputs = npu.Split(data, npu.constant(np.asarray(-1, dtype=np.int64)), num_splits=3)

        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(output.spec == npu.TensorSpec((2, 2), "i16") for output in outputs))

        with self.assertRaisesRegex(ValueError, "constant axis"):
            npu.Split(data, npu.input("axis", shape=(1,), dtype="i64"), num_splits=3)
        with self.assertRaisesRegex(ValueError, "scalar integer"):
            npu.Split(data, npu.constant(np.asarray([1], dtype=np.int64)), num_splits=3)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            npu.Split(data, npu.constant(np.asarray(1, dtype=np.int64)), num_splits=0)
        with self.assertRaisesRegex(ValueError, "divisible"):
            npu.Split(data, npu.constant(np.asarray(1, dtype=np.int64)), num_splits=4)

    def test_static_gather_inference_and_rejections(self) -> None:
        data = npu.input("data", shape=(3, 4), dtype="u16")
        indices = npu.input("indices", shape=(2,), dtype="i32")
        gathered = npu.Gather(data, indices, npu.constant(np.asarray(0, dtype=np.int64)))

        self.assertEqual(gathered.spec, npu.TensorSpec((2, 4), "u16"))

        matrix_indices = npu.input("matrix_indices", shape=(2, 2), dtype="i32")
        trailing = npu.Gather(data, matrix_indices, npu.constant(np.asarray(-1, dtype=np.int64)))
        self.assertEqual(trailing.spec, npu.TensorSpec((3, 2, 2), "u16"))

        with self.assertRaisesRegex(ValueError, "I32 or I64 indices"):
            npu.Gather(data, npu.input("bad_indices", shape=(2,), dtype="u16"), npu.constant(np.asarray(0)))
        with self.assertRaisesRegex(ValueError, "constant axis"):
            npu.Gather(data, indices, npu.input("axis", shape=(), dtype="i64"))
        with self.assertRaisesRegex(ValueError, "scalar integer"):
            npu.Gather(data, indices, npu.constant(np.asarray([0], dtype=np.int64)))
        with self.assertRaisesRegex(ValueError, "batch_dims=0"):
            npu.Gather(data, indices, npu.constant(np.asarray(0)), batch_dims=1)
        with self.assertRaisesRegex(ValueError, "axis is out of range"):
            npu.Gather(data, indices, npu.constant(np.asarray(2)))
        with self.assertRaisesRegex(TypeError, "batch_dims must be an integer"):
            npu.Gather(data, indices, npu.constant(np.asarray(0)), batch_dims=False)

    def test_static_variadic_split_inference_and_rejections(self) -> None:
        data = npu.input("data", shape=(2, 7), dtype="u8")
        axis = npu.constant(np.asarray(-1, dtype=np.int64))
        outputs = npu.VariadicSplit(data, axis, npu.constant(np.asarray([2, -1, 1], dtype=np.int64)))

        self.assertEqual(
            tuple(output.spec for output in outputs),
            (
                npu.TensorSpec((2, 2), "u8"),
                npu.TensorSpec((2, 4), "u8"),
                npu.TensorSpec((2, 1), "u8"),
            ),
        )

        with self.assertRaisesRegex(ValueError, "constant axis"):
            npu.VariadicSplit(data, npu.input("axis", shape=(), dtype="i64"), npu.constant(np.asarray([3, 4])))
        with self.assertRaisesRegex(ValueError, "constant split lengths"):
            npu.VariadicSplit(data, axis, npu.input("lengths", shape=(2,), dtype="i64"))
        with self.assertRaisesRegex(ValueError, "at most one inferred"):
            npu.VariadicSplit(data, axis, npu.constant(np.asarray([-1, -1, 1])))
        with self.assertRaisesRegex(ValueError, "positive or one -1"):
            npu.VariadicSplit(data, axis, npu.constant(np.asarray([2, 0, 5])))
        with self.assertRaisesRegex(ValueError, "must sum"):
            npu.VariadicSplit(data, axis, npu.constant(np.asarray([2, 2, 2])))
        with self.assertRaisesRegex(ValueError, "inferred length must be positive"):
            npu.VariadicSplit(data, axis, npu.constant(np.asarray([7, -1])))

    def test_static_slice_inference_and_rejections(self) -> None:
        data = npu.input("data", shape=(3, 8), dtype="u8")
        sliced = npu.Slice(
            data,
            npu.constant(np.asarray([1, 1], dtype=np.int64)),
            npu.constant(np.asarray([3, 8], dtype=np.int64)),
            npu.constant(np.asarray([1, 2], dtype=np.int64)),
            npu.constant(np.asarray([0, -1], dtype=np.int64)),
        )
        clipped = npu.Slice(
            data,
            npu.constant(np.asarray([-3], dtype=np.int64)),
            npu.constant(np.asarray([-1], dtype=np.int64)),
            npu.constant(np.asarray([1], dtype=np.int64)),
            npu.constant(np.asarray([1], dtype=np.int64)),
        )

        self.assertEqual(sliced.spec, npu.TensorSpec((2, 4), "u8"))
        self.assertEqual(clipped.spec, npu.TensorSpec((3, 2), "u8"))

        vector = npu.constant(np.asarray([0], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "constant starts"):
            npu.Slice(data, npu.input("starts", shape=(1,), dtype="i64"), vector, vector, vector)
        with self.assertRaisesRegex(ValueError, "same nonzero length"):
            npu.Slice(
                data,
                npu.constant(np.asarray([0, 0], dtype=np.int64)),
                vector,
                vector,
                vector,
            )
        with self.assertRaisesRegex(ValueError, "only positive steps"):
            npu.Slice(data, vector, vector, npu.constant(np.asarray([0], dtype=np.int64)), vector)
        with self.assertRaisesRegex(ValueError, "axes must be unique"):
            npu.Slice(
                data,
                npu.constant(np.asarray([0, 0], dtype=np.int64)),
                npu.constant(np.asarray([1, 1], dtype=np.int64)),
                npu.constant(np.asarray([1, 1], dtype=np.int64)),
                npu.constant(np.asarray([0, 0], dtype=np.int64)),
            )
        with self.assertRaisesRegex(ValueError, "axis is out of range"):
            npu.Slice(
                data,
                vector,
                npu.constant(np.asarray([1], dtype=np.int64)),
                npu.constant(np.asarray([1], dtype=np.int64)),
                npu.constant(np.asarray([2])),
            )
        with self.assertRaisesRegex(ValueError, "zero-sized dimensions"):
            npu.Slice(data, vector, vector, npu.constant(np.asarray([1], dtype=np.int64)), vector)

    def test_static_pad_inference_and_rejections(self) -> None:
        data = npu.input("data", shape=(2, 3), dtype="u8")
        pads_begin = npu.constant(np.asarray([1, 2], dtype=np.int64))
        pads_end = npu.constant(np.asarray([1, 2], dtype=np.int64))
        pad_value = npu.constant(np.asarray(255, dtype=np.uint8))
        padded = npu.Pad(data, pads_begin, pads_end, pad_value)

        self.assertEqual(padded.spec, npu.TensorSpec((4, 7), "u8"))
        self.assertEqual(padded.producer.attrs, {"mode": "constant"})

        with self.assertRaisesRegex(ValueError, "only constant mode"):
            npu.Pad(data, pads_begin, pads_end, pad_value, mode="reflect")
        with self.assertRaisesRegex(ValueError, "one value per data axis"):
            npu.Pad(data, npu.constant(np.asarray([1])), pads_end, pad_value)
        with self.assertRaisesRegex(ValueError, "only nonnegative pads"):
            npu.Pad(data, npu.constant(np.asarray([-1, 0])), pads_end, pad_value)
        with self.assertRaisesRegex(ValueError, "scalar constant pad_value"):
            npu.Pad(data, pads_begin, pads_end, npu.input("fill", shape=(), dtype="u8"))
        with self.assertRaisesRegex(ValueError, "dtypes must match"):
            npu.Pad(data, pads_begin, pads_end, npu.constant(np.asarray(1, dtype=np.int32)))
        with self.assertRaisesRegex(TypeError, "mode must be a string"):
            npu.Pad(data, pads_begin, pads_end, pad_value, mode=1)

    def test_structural_operator_attributes_are_serialized(self) -> None:
        left = npu.input("left", shape=(1, 2), dtype="i32")
        right = npu.input("right", shape=(1, 2), dtype="i32")
        joined = npu.Concat(left, right, axis=0)
        converted = npu.Convert(joined, destination_type="f32")
        indices = npu.input("indices", shape=(1,), dtype="i32")
        axis = npu.constant(np.asarray(0, dtype=np.int64))
        gathered = npu.Gather(left, indices, axis)
        root = ET.fromstring(npu.serialize_ir(npu.Graph([left, right, indices], [converted, gathered])).xml)

        concat_data = root.find("./layers/layer[@type='Concat']/data")
        convert_data = root.find("./layers/layer[@type='Convert']/data")
        gather_data = root.find("./layers/layer[@type='Gather']/data")
        self.assertEqual(concat_data.attrib, {"axis": "0"})
        self.assertEqual(convert_data.attrib, {"destination_type": "f32"})
        self.assertEqual(gather_data.attrib, {"batch_dims": "0"})

    def test_slice_uses_its_required_ir_opset(self) -> None:
        data = npu.input("data", shape=(3, 8), dtype="u8")
        sliced = npu.Slice(
            data,
            npu.constant(np.asarray([1], dtype=np.int64)),
            npu.constant(np.asarray([3], dtype=np.int64)),
            npu.constant(np.asarray([1], dtype=np.int64)),
            npu.constant(np.asarray([0], dtype=np.int64)),
        )
        root = ET.fromstring(npu.serialize_ir(npu.Graph([data], [sliced])).xml)

        self.assertEqual(root.find("./layers/layer[@type='Slice']").attrib["version"], "opset8")
        self.assertEqual(root.find("./layers/layer[@type='Parameter']").attrib["version"], "opset1")

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

    def test_custom_infers_single_output_contract_and_binary_broadcast_shape(self) -> None:
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

        row_scalar = npu.input("row_scalar", shape=(4, 1), dtype="f16")
        matrix = npu.input("matrix", shape=(4, 8), dtype="f16")
        scalar_first = npu.custom(row_scalar, matrix, source=b"kernel", carrier="Subtract")
        scalar_second = npu.custom(matrix, row_scalar, source=b"kernel", carrier="Subtract")
        self.assertEqual(scalar_first.shape, (4, 8))
        self.assertEqual(scalar_second.shape, (4, 8))
        incompatible = npu.input("incompatible", shape=(3, 2), dtype="f16")
        with self.assertRaisesRegex(ValueError, "cannot infer a broadcast output shape"):
            npu.custom(x, incompatible, source=b"kernel", carrier="Subtract")

        source_free = npu.custom(x, carrier="Abs", _name="compiled_later")
        self.assertIsNone(source_free.producer.metadata["_kernel"])
        self.assertEqual(source_free.spec, x.spec)

        u8 = npu.input("u8", shape=(1, 32), dtype="u8")
        converted = npu.custom(u8, source=b"kernel", carrier="Convert", destination_type="f32")
        self.assertEqual(converted.shape, u8.shape)
        self.assertEqual(converted.dtype, "f32")

        compiled = npu.CompiledKernel(b"linked elf")
        attached = npu.custom(x, kernel=compiled, carrier="Abs")
        self.assertIs(attached.producer.metadata["_kernel"], compiled)
        with self.assertRaisesRegex(ValueError, "source or kernel"):
            npu.custom(x, source=b"source", kernel=compiled, carrier="Abs")
        with self.assertRaisesRegex(TypeError, "KernelSpec or CompiledKernel"):
            npu.custom(x, kernel=object(), carrier="Abs")  # type: ignore[arg-type]

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

    def test_custom_carrier_override_is_non_mutating(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs", _name="custom")
        graph = npu.Graph([x], [y])

        baseline = npu.serialize_ir(graph)
        candidate = npu.serialize_ir(graph, _carrier_overrides={y.producer: "Sqrt"})

        self.assertIn(b'type="Abs"', baseline.xml)
        self.assertNotIn(b'type="Sqrt"', baseline.xml)
        self.assertIn(b'type="Sqrt"', candidate.xml)
        self.assertNotIn(b'type="Abs"', candidate.xml)
        self.assertEqual(candidate.weights, baseline.weights)
        self.assertEqual(y.producer.metadata["_carrier"], "Abs")

    def test_custom_accepts_a_public_name(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs", name="named_custom")

        self.assertEqual(y.name, "named_custom")
        self.assertEqual(y.producer.name, "named_custom")
        self.assertIn(b'name="named_custom"', npu.serialize_ir(npu.Graph([x], [y])).xml)
        with self.assertRaisesRegex(ValueError, "name or _name"):
            npu.custom(x, source=b"kernel", carrier="Abs", name="public", _name="legacy")
        with self.assertRaisesRegex(TypeError, "name must be a string"):
            npu.custom(x, source=b"kernel", carrier="Abs", name=1)  # type: ignore[arg-type]

    def test_custom_carrier_override_rejects_unrelated_nodes(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.Abs(x)
        graph = npu.Graph([x], [y])

        with self.assertRaisesRegex(ValueError, "custom nodes in this graph"):
            npu.serialize_ir(graph, _carrier_overrides={y.producer: "Sqrt"})

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

    def test_default_native_runtime_is_shared_per_resolved_directory(self) -> None:
        first_path = ROOT / "build" / "runtime-a"
        second_path = ROOT / "build" / "runtime-b"
        npu._native_library_cache.clear()
        with (
            patch.object(npu, "_native_directory", side_effect=(first_path, first_path, second_path)),
            patch.object(npu, "NativeLibraries", side_effect=(object(), object())) as constructor,
        ):
            first = npu._default_native_libraries(None)
            self.assertIs(npu._default_native_libraries(None), first)
            self.assertIsNot(npu._default_native_libraries(None), first)
        self.assertEqual(constructor.call_count, 2)
        npu._native_library_cache.clear()

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
        self.assertEqual(ctypes.sizeof(_native._PatchComparisonResult), 72)
        self.assertEqual(ctypes.sizeof(_native._PatchGroupCompleteness), 36)
        self.assertEqual(ctypes.sizeof(_native._PatchGroupCompletenessResult), 64)
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
        self.ir_calls: list[tuple[bytes, bytes, dict[str, object]]] = []
        self.compile_ir_blobs: list[bytes] = []
        self.shave_calls: list[tuple[bytes, dict[str, object]]] = []
        self.patch_calls: list[tuple[bytes, bytes, tuple[npu.PatchTarget, ...]]] = []
        self.discovery_groups = ((npu.PatchTarget(0, 0, 1, 32, 64),),)
        self.infer_dtype = "f16"
        self.sessions: list[FakeInferenceSession] = []
        self.comparison = npu.GraphComparison(0, (1,), True, True)
        self.comparisons: list[npu.GraphComparison | Exception] = []
        self.completeness = (npu.ActGroupCompleteness(0, "complete", 1, 2, 2, 0, 0, 0),)
        self.coverage = (
            npu.ActGroupCoverage(
                0,
                1,
                "not_observed",
                "complete",
                "not_observed",
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
            ),
        )

    def compile_ir(self, xml: bytes, weights: bytes, **kwargs: object) -> npu.IrCompileResult:
        self.compile_ir_count += 1
        self.xml = xml
        self.weights = weights
        self.compile_ir_kwargs = kwargs
        self.ir_calls.append((xml, weights, dict(kwargs)))
        graph_blob = self.compile_ir_blobs.pop(0) if self.compile_ir_blobs else b"native"
        return npu.IrCompileResult(graph_blob, 0, 0, 1, 0x8086, 0x7D1D, 0x10012, (8, 3))

    def query_ir_provenance(self, **kwargs: object) -> npu.IrCompileResult:
        self.query_ir_kwargs = kwargs
        return npu.IrCompileResult(b"", 0, 0, 1, 0x8086, 0x7D1D, 0x10012, (8, 3))

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

    def compare_graphs(self, baseline_graph_blob: bytes, candidate_graph_blob: bytes) -> npu.GraphComparison:
        self.compare_args = (baseline_graph_blob, candidate_graph_blob)
        result = self.comparisons.pop(0) if self.comparisons else self.comparison
        if isinstance(result, Exception):
            raise result
        return result

    def inspect_group_completeness(self, graph_blob: bytes) -> tuple[npu.ActGroupCompleteness, ...]:
        self.completeness_blob = graph_blob
        return self.completeness

    def inspect_group_coverage(self, graph_blob: bytes) -> tuple[npu.ActGroupCoverage, ...]:
        self.coverage_blob = graph_blob
        return self.coverage

    def infer_graph(self, graph_blob: bytes, inputs: object, **kwargs: object) -> npu.InferenceResult:
        values = tuple(inputs)  # type: ignore[arg-type]
        self.infer_args = (graph_blob, values, kwargs)
        dtype = np.dtype(
            {
                "f16": "float16",
                "f32": "float32",
                "u8": "uint8",
                "i8": "int8",
                "u16": "uint16",
                "i16": "int16",
                "u32": "uint32",
                "i32": "int32",
                "i64": "int64",
            }[self.infer_dtype]
        )
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
        dtype = np.dtype(
            {
                "f16": "float16",
                "f32": "float32",
                "u8": "uint8",
                "i8": "int8",
                "u16": "uint16",
                "i16": "int16",
                "u32": "uint32",
                "i32": "int32",
                "i64": "int64",
            }[self._libraries.infer_dtype]
        )
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
        self.assertIsInstance(fake.infer_args[1][0].data, memoryview)
        self.assertFalse(fake.infer_args[1][0].data.readonly)

    def test_compile_accepts_source_free_compiled_kernel(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, carrier="Abs", _name="compiled")
        fake = FakeNative()

        with patch.dict("os.environ", {}, clear=True):
            program = npu.compile(
                npu.Graph([x], [y], name="one_call_compiled"),
                native_dir="unused",
                kernels={y: npu.CompiledKernel(b"linked elf")},
                libraries=fake,  # type: ignore[arg-type]
            )

        self.assertEqual(fake.compile_ir_count, 1)
        self.assertEqual(fake.shave_calls, [])
        self.assertEqual(fake.patch_calls[0][1], b"linked elf")
        self.assertEqual(program.graph_blob, b"patched")

    def test_compile_accepts_compiled_kernel_attached_to_custom_node(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        compiled = npu.CompiledKernel(b"attached linked elf")
        y = npu.custom(x, kernel=compiled, carrier="Abs", _name="attached")
        fake = FakeNative()

        with patch.dict("os.environ", {}, clear=True):
            program = npu.compile(
                npu.Graph([x], [y], name="attached_compiled"),
                native_dir="unused",
                libraries=fake,  # type: ignore[arg-type]
            )

        self.assertEqual(fake.compile_ir_count, 1)
        self.assertEqual(fake.shave_calls, [])
        self.assertEqual(fake.patch_calls[0][1], b"attached linked elf")
        self.assertEqual(program.graph_blob, b"patched")

    def test_kernel_spec_attached_to_custom_node_keeps_its_settings(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        spec = npu.KernelSpec(
            b"attached source",
            definitions=("MODE=7",),
            linker_script=b"attached linker",
        )
        y = npu.custom(x, kernel=spec, carrier="Abs", _name="attached_spec")
        fake = FakeNative()
        prepared = npu.prepare(
            npu.Graph([x], [y], name="attached_spec_graph"),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        plan = prepared.plan(definitions=("IGNORED=1",), linker_script=b"ignored")
        program = plan.build(movi_dll_dir="movi")

        self.assertIn("MODE=7", plan.explain())
        self.assertEqual(
            fake.shave_calls[0][1],
            {
                "movi_dll_dir": "movi",
                "linker_script": b"attached linker",
                "definitions": ("MODE=7",),
                "timeout_ms": 20_000,
                "worker": None,
            },
        )
        self.assertEqual(program.graph_blob, b"patched")

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
        fake.discovery_groups = ((npu.PatchTarget(0, 0, 1, 32, 64),),)
        program = npu.compile(
            npu.Graph([x], [y]),
            native_dir="unused",
            movi_dll_dir="movi",
            linker_script=b"script",
            libraries=fake,  # type: ignore[arg-type]
        )
        self.assertEqual(fake.discovery_blob, b"native")
        self.assertEqual(fake.patch_args[2], (npu.PatchTarget(0, 0, 1, 32, 64),))
        self.assertEqual(program.graph_blob, b"patched")

    def test_prepared_carrier_probe_returns_evidence_without_mutating_graph(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs", _name="custom")
        fake = FakeNative()
        fake.compile_ir_blobs = [b"baseline", b"candidate"]
        fake.comparison = npu.GraphComparison(0, (0,), True, True)
        prepared = npu.prepare(
            npu.Graph([x], [y]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        probe = prepared.probe_carrier(y, "Sqrt", ir_worker="worker.exe")

        self.assertTrue(probe.accepted)
        self.assertEqual(probe.group_index, 0)
        self.assertEqual((probe.baseline_carrier, probe.candidate_carrier), ("Abs", "Sqrt"))
        self.assertEqual(fake.compare_args, (b"baseline", b"candidate"))
        self.assertEqual(fake.compile_ir_count, 2)
        self.assertEqual(fake.ir_calls[1][2]["worker"], "worker.exe")
        self.assertIn(b'type="Abs"', prepared.serialized_ir.xml)
        self.assertIn(b'type="Sqrt"', probe.candidate_ir.xml)
        self.assertEqual(y.producer.metadata["_carrier"], "Abs")
        self.assertEqual(probe.baseline_ir_sha256, hashlib.sha256(prepared.serialized_ir.xml).hexdigest())
        self.assertEqual(probe.candidate_ir_sha256, hashlib.sha256(probe.candidate_ir.xml).hexdigest())
        self.assertEqual(probe.baseline_graph_blob_sha256, hashlib.sha256(b"baseline").hexdigest())
        self.assertEqual(probe.candidate_graph_blob_sha256, hashlib.sha256(b"candidate").hexdigest())

    def test_prepared_carrier_probe_rejects_non_custom_output(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs")
        ordinary = npu.Abs(y)
        prepared = npu.prepare(
            npu.Graph([x], [ordinary]),
            native_dir="unused",
            libraries=FakeNative(),  # type: ignore[arg-type]
        )

        with self.assertRaisesRegex(ValueError, "custom node"):
            prepared.probe_carrier(ordinary, "Sqrt")

    def test_prepared_mapping_probe_requires_two_consistent_acceptances(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs")
        fake = FakeNative()
        fake.compile_ir_blobs = [b"baseline", b"sqrt", b"tanh", b"relu"]
        fake.comparisons = [
            npu.GraphComparison(0, (0,), True, True),
            npu.GraphComparison(1, (), False, False),
            npu.GraphComparison(0, (0,), True, True),
        ]
        prepared = npu.prepare(
            npu.Graph([x], [y]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        evidence = prepared.probe_mapping(y, ("Sqrt", "Tanh", "Relu"))

        self.assertEqual(evidence.status, "consistent")
        self.assertEqual(evidence.group_index, 0)
        self.assertEqual(tuple(probe.candidate_carrier for probe in evidence.accepted_probes), ("Sqrt", "Relu"))
        self.assertIn("accepted=2/3", evidence.explain())
        self.assertIn("Abs->Tanh: rejected", evidence.explain())

    def test_prepared_mapping_probe_reports_conflict(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs")
        fake = FakeNative()
        fake.compile_ir_blobs = [b"baseline", b"sqrt", b"tanh"]
        fake.comparisons = [
            npu.GraphComparison(0, (0,), True, True),
            npu.GraphComparison(0, (1,), True, True),
        ]
        prepared = npu.prepare(
            npu.Graph([x], [y]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        evidence = prepared.probe_mapping(y, ("Sqrt", "Tanh"))

        self.assertEqual(evidence.status, "conflicting")
        self.assertIsNone(evidence.group_index)

    def test_prepared_group_completeness_is_read_only(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs")
        fake = FakeNative()
        prepared = npu.prepare(
            npu.Graph([x], [y]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        completeness = prepared.inspect_group_completeness()

        self.assertEqual(completeness, fake.completeness)
        self.assertEqual(fake.completeness_blob, b"native")
        self.assertTrue(completeness[0].observed)
        self.assertTrue(completeness[0].complete)
        self.assertEqual(completeness[0].unresolved_names, ())

    def test_prepared_group_coverage_keeps_route_kinds_independent(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs")
        fake = FakeNative()
        prepared = npu.prepare(
            npu.Graph([x], [y]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        coverage = prepared.inspect_group_coverage()

        self.assertEqual(coverage, fake.coverage)
        self.assertEqual(fake.coverage_blob, b"native")
        self.assertEqual(coverage[0].observed_kinds, ("act_consumer",))
        self.assertEqual(coverage[0].complete_kinds, ("act_consumer",))
        self.assertEqual(coverage[0].direct_output_unresolved_names, ())
        self.assertEqual(coverage[0].act_consumer_unresolved_names, ())
        self.assertEqual(coverage[0].dpu_consumer_unresolved_names, ())
        self.assertEqual(coverage[0].dma_dpu_consumer_unresolved_names, ())
        self.assertEqual(coverage[0].dma_act_consumer_unresolved_names, ())

    def test_reviewed_binding_accepts_complete_dma_dpu_coverage(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs")
        fake = FakeNative()
        fake.compile_ir_blobs = [b"baseline", b"sqrt", b"tanh"]
        fake.comparisons = [
            npu.GraphComparison(0, (0,), True, True),
            npu.GraphComparison(0, (0,), True, True),
        ]
        fake.coverage = (
            npu.ActGroupCoverage(
                0,
                1,
                "not_observed",
                "not_observed",
                "not_observed",
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                dma_dpu_consumer_state="complete",
                dma_dpu_tile_count_expected=2,
                dma_dpu_translated_tile_count=2,
                dma_dpu_matched_input_count=2,
            ),
        )
        prepared = npu.prepare(
            npu.Graph([x], [y]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        evidence = prepared.probe_mapping(y, ("Sqrt", "Tanh"))
        reviewed = prepared.review_binding(y, evidence, coverage_kind="dma_dpu_consumer")

        self.assertEqual(reviewed.group_index, 0)
        self.assertEqual(reviewed.coverage.complete_kinds, ("dma_dpu_consumer",))

    def test_reviewed_binding_accepts_complete_dma_act_coverage(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs")
        fake = FakeNative()
        fake.compile_ir_blobs = [b"baseline", b"sqrt", b"tanh"]
        fake.comparisons = [
            npu.GraphComparison(0, (0,), True, True),
            npu.GraphComparison(0, (0,), True, True),
        ]
        fake.coverage = (
            npu.ActGroupCoverage(
                0,
                1,
                "not_observed",
                "not_observed",
                "not_observed",
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                dma_act_consumer_state="complete",
                dma_act_routed_invocation_count=1,
                dma_act_replica_invocation_count=0,
                dma_act_consumer_group_count=1,
                dma_act_consumer_input_count=2,
                dma_act_ddr_span_bytes=64,
            ),
        )
        prepared = npu.prepare(
            npu.Graph([x], [y]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        evidence = prepared.probe_mapping(y, ("Sqrt", "Tanh"))
        reviewed = prepared.review_binding(y, evidence, coverage_kind="dma_act_consumer")

        self.assertEqual(reviewed.group_index, 0)
        self.assertEqual(reviewed.coverage.complete_kinds, ("dma_act_consumer",))

    def test_reviewed_binding_combines_consistent_mapping_and_complete_coverage(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs")
        fake = FakeNative()
        fake.compile_ir_blobs = [b"baseline", b"sqrt", b"tanh"]
        fake.comparisons = [
            npu.GraphComparison(0, (0,), True, True),
            npu.GraphComparison(0, (0,), True, True),
        ]
        prepared = npu.prepare(
            npu.Graph([x], [y]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        evidence = prepared.probe_mapping(y, ("Sqrt", "Tanh"))
        reviewed = prepared.review_binding(y, evidence, coverage_kind="act_consumer")
        plan = prepared.plan(bindings={y: reviewed})

        self.assertEqual(reviewed.group_index, 0)
        self.assertEqual(reviewed.coverage_kind, "act_consumer")
        self.assertEqual(reviewed.baseline_graph_blob_sha256, hashlib.sha256(b"baseline").hexdigest())
        self.assertIn("accepted_carriers=Sqrt,Tanh", reviewed.explain())
        self.assertEqual(plan.group_indices, ((0,),))

    def test_probe_binding_selects_the_only_complete_coverage_kind(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs")
        fake = FakeNative()
        fake.comparison = npu.GraphComparison(0, (0,), True, True)
        prepared = npu.prepare(
            npu.Graph([x], [y]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        reviewed = prepared.probe_binding(y, ("Sqrt", "Tanh"))

        self.assertEqual(reviewed.group_index, 0)
        self.assertEqual(reviewed.coverage_kind, "act_consumer")
        self.assertEqual([probe.candidate_carrier for probe in reviewed.evidence.probes], ["Sqrt", "Tanh"])

    def test_probe_binding_requires_explicit_kind_for_ambiguous_coverage(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs")
        fake = FakeNative()
        fake.comparison = npu.GraphComparison(0, (0,), True, True)
        fake.coverage = (replace(fake.coverage[0], direct_output_state="complete"),)
        prepared = npu.prepare(
            npu.Graph([x], [y]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        with self.assertRaisesRegex(ValueError, "exactly one complete kind"):
            prepared.probe_binding(y, ("Sqrt", "Tanh"))

        reviewed = prepared.probe_binding(y, ("Sqrt", "Tanh"), coverage_kind="act_consumer")
        self.assertEqual(reviewed.coverage_kind, "act_consumer")

    def test_probe_candidates_retains_failures_without_repeating_successes(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs")
        fake = FakeNative()
        fake.comparisons = [
            npu.NativeError("patchblob comparison", 5, "unsupported", b'{"message":"bad carrier"}', b"out", b"err"),
            npu.GraphComparison(0, (0,), True, True),
            npu.GraphComparison(0, (0,), True, True),
        ]
        prepared = npu.prepare(
            npu.Graph([x], [y]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        report = prepared.probe_candidates(y, ("Relu", "Sqrt", "Tanh"))
        reviewed = prepared.probe_binding(y, report)

        self.assertEqual(report.status, "consistent")
        self.assertEqual([probe.candidate_carrier for probe in report.probes], ["Sqrt", "Tanh"])
        self.assertEqual(len(report.failures), 1)
        self.assertEqual(report.failures[0].candidate_carrier, "Relu")
        self.assertEqual(report.failures[0].stage, "patchblob comparison")
        self.assertEqual(report.failures[0].diagnostic, b'{"message":"bad carrier"}')
        self.assertEqual(report.failures[0].stdout_log, b"out")
        self.assertEqual(report.failures[0].stderr_log, b"err")
        self.assertIn("Relu: error", report.explain())
        self.assertEqual(reviewed.coverage_kind, "act_consumer")
        self.assertEqual(fake.compile_ir_count, 4)

    def test_reviewed_binding_requires_complete_selected_coverage(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs")
        fake = FakeNative()
        fake.compile_ir_blobs = [b"baseline", b"sqrt", b"tanh"]
        fake.comparisons = [
            npu.GraphComparison(0, (0,), True, True),
            npu.GraphComparison(0, (0,), True, True),
        ]
        prepared = npu.prepare(
            npu.Graph([x], [y]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )
        evidence = prepared.probe_mapping(y, ("Sqrt", "Tanh"))

        with self.assertRaisesRegex(ValueError, "not_observed, not complete"):
            prepared.review_binding(y, evidence, coverage_kind="direct_output")

    def test_plan_revalidates_reviewed_binding(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs")
        fake = FakeNative()
        fake.compile_ir_blobs = [b"baseline", b"sqrt", b"tanh"]
        fake.comparisons = [
            npu.GraphComparison(0, (0,), True, True),
            npu.GraphComparison(0, (0,), True, True),
        ]
        prepared = npu.prepare(
            npu.Graph([x], [y]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )
        evidence = prepared.probe_mapping(y, ("Sqrt", "Tanh"))
        reviewed = prepared.review_binding(y, evidence, coverage_kind="act_consumer")
        altered = npu.ReviewedBinding(
            reviewed.output,
            reviewed.group_index,
            reviewed.coverage_kind,
            "0" * 64,
            reviewed.evidence,
            reviewed.coverage,
        )

        with self.assertRaisesRegex(ValueError, "no longer matches"):
            prepared.plan(bindings={y: altered})

    def test_positional_mapping_rejects_dtype_mismatch(self) -> None:
        f32 = npu.input("f32", shape=(1, 32), dtype="f32")
        converted = npu.custom(f32, source=b"kernel", carrier="Convert", _dtype="f16", _name="converted")
        dtype_fake = FakeNative()
        dtype_fake.discovery_groups = ((npu.PatchTarget(0, 0, 1, 32, 64, 0x1F),),)
        dtype_prepared = npu.prepare(
            npu.Graph([f32], [converted]),
            native_dir="unused",
            libraries=dtype_fake,  # type: ignore[arg-type]
        )

        self.assertEqual(dtype_prepared.mappings[0].status, "explicit-required")
        self.assertIn("tensor precisions", dtype_prepared.mappings[0].reason)
        with self.assertRaisesRegex(ValueError, "tensor precisions"):
            dtype_prepared.plan(bindings={converted: dtype_prepared.groups[0]})

    def test_one_group_may_duplicate_partition_coverage_across_tiles(self) -> None:
        f16 = npu.input("f16", shape=(1, 32), dtype="f16")
        output = npu.custom(f16, source=b"kernel", carrier="Abs", _name="output")
        targets = tuple(npu.PatchTarget(index, index, 1, 16, 32, 0x1F) for index in range(4))
        fake = FakeNative()
        fake.discovery_groups = (targets,)
        prepared = npu.prepare(
            npu.Graph([f16], [output]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        self.assertEqual(prepared.mappings[0].status, "automatic")
        self.assertEqual(prepared.plan()._target_groups, (targets,))

    def test_positional_discovery_requires_validated_compiler_version(self) -> None:
        class UnvalidatedCompilerNative(FakeNative):
            def compile_ir(self, xml: bytes, weights: bytes, **kwargs: object) -> npu.IrCompileResult:
                result = super().compile_ir(xml, weights, **kwargs)
                return npu.IrCompileResult(
                    result.graph_blob,
                    result.driver_index,
                    result.device_index,
                    result.driver_version,
                    result.vendor_id,
                    result.device_id,
                    result.graph_extension_version,
                    (8, 4),
                )

        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs", _name="y")
        with self.assertRaisesRegex(ValueError, "validated only for graph compiler 8.3"):
            npu.prepare(
                npu.Graph([x], [y]),
                native_dir="unused",
                libraries=UnvalidatedCompilerNative(),  # type: ignore[arg-type]
            )

        target = npu.PatchTarget(0, 0, 1, 32, 64)
        explicit = npu.custom(x, source=b"kernel", carrier="Abs", _name="explicit", _patch_targets=[target])
        prepared = npu.prepare(
            npu.Graph([x], [explicit]),
            native_dir="unused",
            libraries=UnvalidatedCompilerNative(),  # type: ignore[arg-type]
        )
        self.assertEqual(prepared.mappings[0].status, "raw-explicit")

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

    def test_equal_span_binary_f32_custom_group_maps_automatically(self) -> None:
        targets = tuple(npu.PatchTarget(index, index, 2, 8, 32, 0x3B) for index in range(4))
        fake = FakeNative()
        fake.discovery_groups = (targets,)
        a = npu.input("a", shape=(1, 32), dtype="f32")
        b = npu.input("b", shape=(1, 32), dtype="f32")
        y = npu.custom(a, b, source=b"binary f32 kernel", carrier="Subtract", _name="y")

        prepared = npu.prepare(
            npu.Graph([a, b], [y], name="binary_f32_custom"),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        self.assertEqual(prepared.mappings[0].status, "automatic")
        self.assertEqual(prepared.find_group(input_count=2, dtype="f32").targets, targets)
        prepared.build(movi_dll_dir="movi")
        self.assertEqual(fake.patch_args[2], targets)
        self.assertIn('EXECUTION_MODE_HINT="ACCURACY"', fake.compile_ir_kwargs["build_flags"])

    def test_f32_row_broadcast_group_maps_by_dense_descriptor_contract(self) -> None:
        targets = tuple(npu.PatchTarget(index, index, 2, 8, 32, 0x3B) for index in range(4))
        fake = FakeNative()
        fake.discovery_groups = (targets,)
        data = npu.input("data", shape=(4, 8), dtype="f32")
        bias = npu.input("bias", shape=(1, 8), dtype="f32")
        output = npu.custom(data, bias, source=b"row kernel", carrier="Subtract", _name="output")

        prepared = npu.prepare(
            npu.Graph([data, bias], [output], name="f32_row_broadcast"),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        group = prepared.find_group(input_count=2, dtype="f32")
        self.assertEqual(group.per_invocation_input_element_counts, (8, 8))
        self.assertEqual(group.per_invocation_output_element_count, 8)
        self.assertEqual(prepared.mappings[0].status, "automatic")
        prepared.build(movi_dll_dir="movi")
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
        self.assertIn("per-invocation=(32,32)->32", prepared.explain())
        unary_group = prepared.find_group(input_count=1, dtype="f32")
        binary_group = prepared.find_group(input_count=2, dtype="f16")
        with self.assertRaisesRegex(ValueError, "is ambiguous") as ambiguous:
            prepared.find_group()
        self.assertIn("group[0]", str(ambiguous.exception))
        self.assertIn("group[1]", str(ambiguous.exception))
        with self.assertRaisesRegex(ValueError, "no ACT group matches") as missing:
            prepared.find_group(element_count=7)
        self.assertIn("available groups: group[0]", str(missing.exception))
        self.assertIn("group[1]", str(missing.exception))

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

    def test_build_plan_failure_identifies_symbolic_kernel_and_preserves_streams(self) -> None:
        class FailingNative(FakeNative):
            def compile_shave(self, source: bytes, **kwargs: object) -> bytes:
                raise npu.NativeError(
                    "shavecc",
                    5,
                    "unsupported",
                    b'{"message":"bad source"}',
                    b"compiler stdout\n",
                    b"compiler stderr\n",
                )

        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"broken", carrier="Abs", _name="broken_kernel")
        fake = FailingNative()
        prepared = npu.prepare(
            npu.Graph([x], [y], name="failing_plan"),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        with self.assertRaises(npu.NativeError) as raised:
            prepared.plan().build(movi_dll_dir="movi")

        error = raised.exception
        self.assertEqual(error.stage, "kernel 'broken_kernel' groups=0: shavecc")
        self.assertIn("kernel 'broken_kernel' groups=0: shavecc failed", str(error))
        self.assertEqual(error.diagnostic, b'{"message":"bad source"}')
        self.assertEqual(error.stdout_log, b"compiler stdout\n")
        self.assertEqual(error.stderr_log, b"compiler stderr\n")

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

    def test_compiled_kernel_snapshots_elf_path(self) -> None:
        elf_path = ROOT / "build" / "compiled-kernel-test.elf"
        saved_path = ROOT / "build" / "compiled-kernel-saved.elf"
        elf_path.write_bytes(b"elf v1")
        try:
            kernel = npu.CompiledKernel(elf_path)
            elf_path.write_bytes(b"elf v2")
            kernel.save(saved_path)
            saved_bytes = saved_path.read_bytes()
        finally:
            elf_path.unlink(missing_ok=True)
            saved_path.unlink(missing_ok=True)

        self.assertEqual(kernel.elf, b"elf v1")
        self.assertEqual(saved_bytes, b"elf v1")
        self.assertEqual(kernel.elf_name, str(elf_path))
        self.assertEqual(kernel.elf_sha256, hashlib.sha256(b"elf v1").hexdigest())
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            npu.CompiledKernel(b"")
        with self.assertRaisesRegex(TypeError, "ELF bytes or a filesystem path"):
            npu.CompiledKernel(bytearray(b"elf"))  # type: ignore[arg-type]

    def test_compile_kernel_forwards_snapshotted_options(self) -> None:
        fake = FakeNative()
        spec = npu.KernelSpec(
            b"kernel source",
            definitions=("MODE=3",),
            linker_script=b"linker script",
        )

        compiled = npu.compile_kernel(
            spec,
            native_dir="unused",
            movi_dll_dir="movi",
            timeout_ms=1234,
            movi_worker="worker.exe",
            libraries=fake,  # type: ignore[arg-type]
        )

        self.assertEqual(compiled.elf, b"elf")
        self.assertEqual(compiled.elf_name, "<compiled from <bytes>>")
        self.assertEqual(
            fake.shave_calls,
            [
                (
                    b"kernel source",
                    {
                        "movi_dll_dir": "movi",
                        "linker_script": b"linker script",
                        "definitions": ("MODE=3",),
                        "timeout_ms": 1234,
                        "worker": "worker.exe",
                    },
                )
            ],
        )

        direct = npu.compile_kernel(
            b"direct source",
            definitions=("MODE=4",),
            linker_script=b"direct linker",
            movi_dll_dir="movi",
            libraries=fake,  # type: ignore[arg-type]
        )
        self.assertEqual(direct.elf_name, "<compiled from <bytes>>")
        self.assertEqual(fake.shave_calls[-1][0], b"direct source")
        self.assertEqual(fake.shave_calls[-1][1]["definitions"], ("MODE=4",))
        self.assertEqual(fake.shave_calls[-1][1]["linker_script"], b"direct linker")

        source_path = ROOT / "build" / "compile-kernel-direct.c"
        source_path.write_bytes(b"path source")
        try:
            from_path = npu.compile_kernel(
                source_path,
                movi_dll_dir="movi",
                libraries=fake,  # type: ignore[arg-type]
            )
        finally:
            source_path.unlink(missing_ok=True)
        self.assertEqual(from_path.elf_name, f"<compiled from {source_path}>")
        self.assertEqual(fake.shave_calls[-1][0], b"path source")

        call_count = len(fake.shave_calls)
        with self.assertRaisesRegex(ValueError, "cannot override a KernelSpec"):
            npu.compile_kernel(
                spec,
                definitions=("MODE=9",),
                movi_dll_dir="movi",
                libraries=fake,  # type: ignore[arg-type]
            )
        self.assertEqual(len(fake.shave_calls), call_count)
        with self.assertRaisesRegex(TypeError, "KernelSpec, source bytes, or a source path"):
            npu.compile_kernel(object(), movi_dll_dir="movi", libraries=fake)  # type: ignore[arg-type]
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "requires an MVC_DEPEND root"):
                npu.compile_kernel(spec, libraries=fake)  # type: ignore[arg-type]

    def test_compiled_kernel_build_does_not_require_movitools(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, carrier="Abs", _name="precompiled")
        fake = FakeNative()
        prepared = npu.prepare(
            npu.Graph([x], [y], name="precompiled_plan"),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )
        kernel = npu.CompiledKernel(b"linked shave elf")
        plan = prepared.plan(kernels={y: kernel})

        with patch.dict("os.environ", {}, clear=True):
            program = prepared.build(kernels={y: kernel})

        self.assertEqual(fake.shave_calls, [])
        self.assertEqual(fake.patch_calls[0][1], b"linked shave elf")
        self.assertEqual(program.graph_blob, b"patched")
        self.assertIn("compiled_elf='<bytes>'", plan.explain())
        self.assertIn(kernel.elf_sha256, plan.explain())
        with self.assertRaisesRegex(TypeError, "KernelSpec or CompiledKernel"):
            prepared.plan(kernels={y: object()})  # type: ignore[dict-item]

    def test_source_free_custom_node_requires_kernel_override(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, carrier="Abs", _name="compiled_later")
        fake = FakeNative()
        prepared = npu.prepare(
            npu.Graph([x], [y], name="missing_kernel_override"),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        with self.assertRaisesRegex(
            ValueError,
            "custom node 'compiled_later' has no C source; provide a KernelSpec or CompiledKernel override",
        ):
            prepared.plan()
        self.assertEqual(fake.shave_calls, [])

    def test_mixed_kernel_plan_compiles_only_source_entries(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        first = npu.custom(x, carrier="Abs", _name="first")
        second = npu.custom(first, source=b"source to compile", carrier="Sqrt", _name="second")
        fake = FakeNative()
        fake.discovery_groups = (
            (npu.PatchTarget(0, 0, 1, 32, 64, 0x1F),),
            (npu.PatchTarget(1, 1, 1, 32, 64, 0x1F),),
        )
        prepared = npu.prepare(
            npu.Graph([x], [second], name="mixed_kernel_inputs"),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )
        program = prepared.plan(
            kernels={
                first: npu.CompiledKernel(b"precompiled elf"),
                second: npu.KernelSpec(b"source to compile", definitions=("MODE=2",)),
            }
        ).build(movi_dll_dir="movi")

        self.assertEqual(len(fake.shave_calls), 1)
        self.assertEqual(fake.shave_calls[0][0], b"source to compile")
        self.assertEqual(fake.shave_calls[0][1]["definitions"], ("MODE=2",))
        self.assertEqual(tuple(call[1] for call in fake.patch_calls), (b"precompiled elf", b"elf"))
        self.assertEqual(program.graph_blob, b"patched2")

    def test_prepared_bundle_restores_reviewed_bindings_without_graph_compile(self) -> None:
        def make_graph(source: bytes) -> tuple[npu.Graph, npu.Tensor, npu.Tensor]:
            x = npu.input("x", shape=(1, 32), dtype="f32")
            a = npu.input("a", shape=(1, 32), dtype="f16")
            b = npu.input("b", shape=(1, 32), dtype="f16")
            scaled = npu.custom(x, source=source, carrier="Abs", _name="scaled")
            mixed = npu.custom(a, b, source=b"mix", carrier="Maximum", _name="mixed")
            return npu.Graph([x, a, b], [mixed, scaled], name="prepared_bundle"), scaled, mixed

        fake = FakeNative()
        fake.discovery_groups = (
            (npu.PatchTarget(0, 0, 1, 32, 128, 0x3B),),
            (npu.PatchTarget(1, 1, 2, 32, 64, 0x1F),),
        )
        graph, scaled, mixed = make_graph(b"old source")
        prepared = npu.prepare(graph, native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        plan = prepared.plan(bindings={scaled: prepared.groups[0], mixed: prepared.groups[1]})
        path = ROOT / "build" / "python-api-prepared-test.npunlock"
        legacy_path = ROOT / "build" / "python-api-prepared-v1-test.npunlock"
        try:
            prepared.export(path, plan=plan)
            with zipfile.ZipFile(path, "r") as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["schema"], "npunlock.prepared.v2")
                self.assertNotIn(b"old source", archive.read("manifest.json"))
                self.assertEqual(manifest["bindings"][0]["group_indices"], [1])
                self.assertEqual(manifest["bindings"][1]["group_indices"], [0])
                self.assertIsNone(manifest["bindings"][0]["review"])
                self.assertIsNone(manifest["bindings"][1]["review"])

            with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(legacy_path, "w") as destination:
                manifest = json.loads(source.read("manifest.json"))
                manifest["schema"] = "npunlock.prepared.v1"
                for binding in manifest["bindings"]:
                    del binding["review"]
                destination.writestr(
                    "manifest.json",
                    (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
                )
                destination.writestr("carrier.blob", source.read("carrier.blob"))

            reloaded_graph, _, _ = make_graph(b"new source")
            loaded = npu.load_prepared(path, graph=reloaded_graph, libraries=fake)  # type: ignore[arg-type]
            legacy_loaded = npu.load_prepared(
                legacy_path,
                graph=reloaded_graph,
                libraries=fake,  # type: ignore[arg-type]
            )
        finally:
            path.unlink(missing_ok=True)
            legacy_path.unlink(missing_ok=True)

        self.assertEqual(fake.compile_ir_count, 1)
        self.assertEqual(tuple(mapping.status for mapping in loaded.mappings), ("saved", "saved"))
        self.assertEqual(tuple(mapping.status for mapping in legacy_loaded.mappings), ("saved", "saved"))
        loaded_plan = loaded.plan()
        self.assertEqual(loaded_plan.group_indices, ((1,), (0,)))
        self.assertEqual(tuple(kernel.source for kernel in loaded_plan.kernels), (b"mix", b"new source"))
        loaded_plan.build(movi_dll_dir="movi")
        self.assertEqual(fake.compile_ir_count, 1)

    def test_prepared_bundle_revalidates_evidence_backed_binding(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs", _name="reviewed")
        graph = npu.Graph([x], [y], name="reviewed_bundle")
        fake = FakeNative()
        fake.compile_ir_blobs = [b"baseline", b"sqrt", b"tanh"]
        fake.comparisons = [
            npu.GraphComparison(0, (0,), True, True),
            npu.GraphComparison(0, (0,), True, True),
        ]
        fake.coverage = (
            replace(
                fake.coverage[0],
                act_consumer_state="not_observed",
                dma_dpu_consumer_state="complete",
                dma_dpu_tile_count_expected=2,
                dma_dpu_translated_tile_count=2,
                dma_dpu_matched_input_count=2,
            ),
        )
        prepared = npu.prepare(graph, native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        evidence = prepared.probe_mapping(y, ("Sqrt", "Tanh"))
        reviewed = prepared.review_binding(y, evidence, coverage_kind="dma_dpu_consumer")
        plan = prepared.plan(bindings={y: reviewed})
        path = ROOT / "build" / "python-api-reviewed-prepared.npunlock"
        corrupt_path = ROOT / "build" / "python-api-reviewed-prepared-corrupt.npunlock"
        try:
            prepared.export(path, plan=plan)
            with zipfile.ZipFile(path, "r") as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["schema"], "npunlock.prepared.v2")
                self.assertEqual(manifest["bindings"][0]["review"]["coverage_kind"], "dma_dpu_consumer")
                self.assertEqual(
                    [probe["carrier"] for probe in manifest["bindings"][0]["review"]["probes"]],
                    ["Sqrt", "Tanh"],
                )
                self.assertIn("reviews/0/0.blob", archive.namelist())
                self.assertIn("reviews/0/1.blob", archive.namelist())

            fake.comparisons = [
                npu.GraphComparison(0, (0,), True, True),
                npu.GraphComparison(0, (0,), True, True),
            ]
            loaded = npu.load_prepared(path, graph=graph, libraries=fake)  # type: ignore[arg-type]
            loaded_plan = loaded.plan()

            self.assertEqual(fake.compile_ir_count, 3)
            self.assertEqual(loaded.mappings[0].status, "saved-reviewed")
            self.assertIsNotNone(loaded_plan.reviewed_bindings[0])
            self.assertIn("reviewed binding: group=0", loaded_plan.explain())

            with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(corrupt_path, "w") as destination:
                for name in source.namelist():
                    data = source.read(name)
                    if name == "reviews/0/0.blob":
                        data += b"corrupt"
                    destination.writestr(name, data)
            with self.assertRaisesRegex(ValueError, "reviewed candidate 0:0 does not match"):
                npu.load_prepared(corrupt_path, graph=graph, libraries=fake)  # type: ignore[arg-type]
        finally:
            path.unlink(missing_ok=True)
            corrupt_path.unlink(missing_ok=True)

    def test_prepared_bundle_rejects_graph_and_act_contract_mismatch(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs", _name="y")
        graph = npu.Graph([x], [y], name="prepared_validation")
        fake = FakeNative()
        prepared = npu.prepare(graph, native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        valid_path = ROOT / "build" / "python-api-prepared-valid.npunlock"
        corrupt_path = ROOT / "build" / "python-api-prepared-corrupt.npunlock"
        try:
            prepared.export(valid_path)
            other_x = npu.input("x", shape=(1, 16), dtype="f16")
            other_y = npu.custom(other_x, source=b"kernel", carrier="Abs", _name="y")
            with self.assertRaisesRegex(ValueError, "IR XML"):
                npu.load_prepared(
                    valid_path,
                    graph=npu.Graph([other_x], [other_y], name="prepared_validation"),
                    libraries=fake,  # type: ignore[arg-type]
                )

            class StaleNative(FakeNative):
                def query_ir_provenance(self, **kwargs: object) -> npu.IrCompileResult:
                    return npu.IrCompileResult(b"", 0, 0, 2, 0x8086, 0x7D1D, 0x10012, (8, 4))

            with self.assertRaisesRegex(ValueError, "provenance is stale"):
                npu.load_prepared(valid_path, graph=graph, libraries=StaleNative())  # type: ignore[arg-type]

            with zipfile.ZipFile(valid_path, "r") as source, zipfile.ZipFile(corrupt_path, "w") as destination:
                for name in source.namelist():
                    data = source.read(name)
                    if name == "manifest.json":
                        manifest = json.loads(data)
                        manifest["act_groups"][0][0]["span_bytes"] = 999
                        data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
                    destination.writestr(name, data)
            with self.assertRaisesRegex(ValueError, "ACT contract"):
                npu.load_prepared(corrupt_path, graph=graph, libraries=fake)  # type: ignore[arg-type]
        finally:
            valid_path.unlink(missing_ok=True)
            corrupt_path.unlink(missing_ok=True)

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
        self.assertEqual(group.per_invocation_input_element_counts, (16,))
        self.assertEqual(group.per_invocation_output_element_count, 16)
        self.assertIn("f32->f16", prepared.explain())
        with self.assertRaisesRegex(ValueError, "tensor precisions"):
            prepared.build(bindings={y: prepared.groups[1]}, movi_dll_dir="movi")
        program = prepared.build(bindings={y: group}, movi_dll_dir="movi")
        self.assertEqual(fake.patch_args[2], targets)
        self.assertEqual(program.graph_blob, b"patched")

    def test_u8_to_f32_custom_group_requires_explicit_binding(self) -> None:
        class MixedNative(FakeNative):
            def discover_patch_targets(self, graph_blob: bytes) -> tuple[tuple[npu.PatchTarget, ...], ...]:
                raise npu.NativeError("patchblob discovery", 5, "unsupported", b"mixed precision")

        tensors = (
            npu.PatchTensorContract("input", 0, "u8", 32, 32, 0x07),
            npu.PatchTensorContract("output", 0, "f32", 32, 128, 0x0F),
        )
        targets = tuple(npu.PatchTargetV2(index, index, tensors, 1) for index in range(2))
        fake = MixedNative()
        fake.discovery_groups_v2 = (targets,)
        x = npu.input("x", shape=(1, 32), dtype="u8")
        y = npu.custom(
            x,
            source=b"convert",
            carrier="Convert",
            destination_type="f32",
            _name="converted",
        )

        prepared = npu.prepare(
            npu.Graph([x], [y]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        self.assertEqual(y.dtype, "f32")
        self.assertEqual(prepared.mappings[0].status, "explicit-required")
        group = prepared.find_group(input_dtypes=("u8",), output_dtype="f32")
        self.assertTrue(group.unary_conversion)
        self.assertIn("u8->f32", prepared.explain())
        program = prepared.build(bindings={y: group}, movi_dll_dir="movi")
        self.assertEqual(fake.patch_args[2], targets)
        self.assertEqual(program.graph_blob, b"patched")

    def test_integer_custom_groups_use_version_two_contract_automatically(self) -> None:
        class IntegerNative(FakeNative):
            def discover_patch_targets(self, graph_blob: bytes) -> tuple[tuple[npu.PatchTarget, ...], ...]:
                raise npu.NativeError("patchblob discovery", 5, "unsupported", b"integer requires version 2")

        for dtype, span in (("i32", 64), ("i16", 32), ("i8", 16)):
            with self.subTest(dtype=dtype):
                tensors = (
                    npu.PatchTensorContract("input", 0, dtype, 16, span, 0x07),
                    npu.PatchTensorContract("output", 0, dtype, 16, span, 0x0F),
                )
                targets = tuple(npu.PatchTargetV2(index, index, tensors) for index in range(4))
                fake = IntegerNative()
                fake.discovery_groups_v2 = (targets,)
                x = npu.input("x", shape=(1, 32), dtype=dtype)
                y = npu.custom(x, source=b"integer kernel", carrier="Abs", _name="scaled")

                prepared = npu.prepare(
                    npu.Graph([x], [y], name=f"{dtype}_custom"),
                    native_dir="unused",
                    libraries=fake,  # type: ignore[arg-type]
                )

                self.assertEqual(prepared.mappings[0].status, "automatic")
                self.assertEqual(prepared.mappings[0].group_indices, (0,))
                self.assertEqual(prepared.find_group(dtype=dtype).targets, targets)
                self.assertEqual(prepared.find_group(input_dtypes=(dtype,), output_dtype=dtype).targets, targets)
                program = prepared.build(movi_dll_dir="movi")
                self.assertEqual(fake.patch_args[2], targets)
                self.assertEqual(program.graph_blob, b"patched")

    def test_binary_integer_custom_groups_use_version_two_contract_automatically(self) -> None:
        class IntegerNative(FakeNative):
            def discover_patch_targets(self, graph_blob: bytes) -> tuple[tuple[npu.PatchTarget, ...], ...]:
                raise npu.NativeError("patchblob discovery", 5, "unsupported", b"integer requires version 2")

        for dtype, span, carrier in (
            ("i32", 32, "Minimum"),
            ("i16", 16, "Maximum"),
            ("i8", 8, "Subtract"),
            ("u16", 16, "Subtract"),
            ("u32", 32, "Subtract"),
        ):
            with self.subTest(dtype=dtype):
                tensors = (
                    npu.PatchTensorContract("input", 0, dtype, 8, span, 0x07),
                    npu.PatchTensorContract("input", 1, dtype, 8, span, 0x07),
                    npu.PatchTensorContract("output", 0, dtype, 8, span, 0x0F),
                )
                targets = tuple(npu.PatchTargetV2(index, index, tensors) for index in range(4))
                fake = IntegerNative()
                fake.discovery_groups_v2 = (targets,)
                a = npu.input("a", shape=(1, 32), dtype=dtype)
                b = npu.input("b", shape=(1, 32), dtype=dtype)
                y = npu.custom(a, b, source=b"integer kernel", carrier=carrier, _name="weighted")

                prepared = npu.prepare(
                    npu.Graph([a, b], [y], name=f"binary_{dtype}_custom"),
                    native_dir="unused",
                    libraries=fake,  # type: ignore[arg-type]
                )

                self.assertEqual(prepared.mappings[0].status, "automatic")
                self.assertEqual(prepared.mappings[0].group_indices, (0,))
                self.assertEqual(prepared.find_group(input_count=2, dtype=dtype).targets, targets)
                self.assertEqual(
                    prepared.find_group(input_dtypes=(dtype, dtype), output_dtype=dtype).targets,
                    targets,
                )
                program = prepared.build(movi_dll_dir="movi")
                self.assertEqual(fake.patch_args[2], targets)
                self.assertEqual(program.graph_blob, b"patched")

    def test_unsigned_scalar_custom_groups_map_automatically(self) -> None:
        class IntegerNative(FakeNative):
            def discover_patch_targets(self, graph_blob: bytes) -> tuple[tuple[npu.PatchTarget, ...], ...]:
                raise npu.NativeError("patchblob discovery", 5, "unsupported", b"integer requires version 2")

        for dtype, span in (("u16", 2), ("u32", 4)):
            with self.subTest(dtype=dtype):
                tensors = (
                    npu.PatchTensorContract("input", 0, dtype, 8, 8 * span, 0x07),
                    npu.PatchTensorContract("input", 1, dtype, 1, span, 0x07),
                    npu.PatchTensorContract("output", 0, dtype, 8, 8 * span, 0x0F),
                )
                targets = tuple(npu.PatchTargetV2(index, index, tensors) for index in range(4))
                fake = IntegerNative()
                fake.discovery_groups_v2 = (targets,)
                data = npu.input("data", shape=(1, 32), dtype=dtype)
                scalar = npu.input("scalar", shape=(1, 1), dtype=dtype)
                output = npu.custom(data, scalar, source=b"unsigned scalar kernel", carrier="Subtract")

                prepared = npu.prepare(
                    npu.Graph([data, scalar], [output], name=f"scalar_{dtype}_custom"),
                    native_dir="unused",
                    libraries=fake,  # type: ignore[arg-type]
                )

                self.assertEqual(prepared.mappings[0].status, "automatic")
                self.assertTrue(prepared.groups[0].input_1_scalar)
                self.assertEqual(prepared.find_group(dtype=dtype, input_1_scalar=True).targets, targets)
                program = prepared.build(movi_dll_dir="movi")
                self.assertEqual(fake.patch_args[2], targets)
                self.assertEqual(program.graph_blob, b"patched")

    def test_scalar_first_custom_groups_map_automatically(self) -> None:
        class ScalarFirstNative(FakeNative):
            def discover_patch_targets(self, graph_blob: bytes) -> tuple[tuple[npu.PatchTarget, ...], ...]:
                raise npu.NativeError("patchblob discovery", 5, "unsupported", b"scalar-first requires version 2")

        for dtype, span in (("f16", 2), ("f32", 4), ("u16", 2), ("u32", 4)):
            with self.subTest(dtype=dtype):
                tensors = (
                    npu.PatchTensorContract("input", 0, dtype, 1, span, 0x07),
                    npu.PatchTensorContract("input", 1, dtype, 8, 8 * span, 0x07),
                    npu.PatchTensorContract("output", 0, dtype, 8, 8 * span, 0x0F),
                )
                targets = tuple(npu.PatchTargetV2(index, index, tensors) for index in range(4))
                fake = ScalarFirstNative()
                fake.discovery_groups_v2 = (targets,)
                scalar = npu.input("scalar", shape=(1, 1), dtype=dtype)
                data = npu.input("data", shape=(1, 32), dtype=dtype)
                output = npu.custom(
                    scalar,
                    data,
                    source=b"scalar-first kernel",
                    carrier="Subtract",
                )

                prepared = npu.prepare(
                    npu.Graph([scalar, data], [output], name=f"scalar_first_{dtype}_custom"),
                    native_dir="unused",
                    libraries=fake,  # type: ignore[arg-type]
                )

                self.assertEqual(prepared.mappings[0].status, "automatic")
                self.assertEqual(prepared.groups[0].scalar_input_indices, (0,))
                self.assertEqual(prepared.find_group(scalar_input_indices=(0,)).targets, targets)
                program = prepared.build(movi_dll_dir="movi")
                self.assertEqual(fake.patch_args[2], targets)
                self.assertEqual(program.graph_blob, b"patched")

        with self.assertRaisesRegex(ValueError, "scalar_input_indices"):
            prepared.find_groups(scalar_input_indices=(1, 0))

        wrong_tensors = (
            npu.PatchTensorContract("input", 0, "f16", 8, 16, 0x07),
            npu.PatchTensorContract("input", 1, "f16", 1, 2, 0x07),
            npu.PatchTensorContract("output", 0, "f16", 8, 16, 0x0F),
        )
        wrong_native = ScalarFirstNative()
        wrong_native.discovery_groups_v2 = (
            tuple(npu.PatchTargetV2(index, index, wrong_tensors) for index in range(4)),
        )
        scalar = npu.input("scalar", shape=(1, 1), dtype="f16")
        data = npu.input("data", shape=(1, 32), dtype="f16")
        output = npu.custom(
            scalar,
            data,
            source=b"scalar-first kernel",
            carrier="Subtract",
        )
        wrong_prepared = npu.prepare(
            npu.Graph([scalar, data], [output]),
            native_dir="unused",
            libraries=wrong_native,  # type: ignore[arg-type]
        )
        self.assertEqual(wrong_prepared.mappings[0].status, "explicit-required")
        self.assertIn("scalar input positions", wrong_prepared.mappings[0].reason)

    def test_exact_per_row_scalar_groups_map_in_either_input_position(self) -> None:
        class PerRowNative(FakeNative):
            def discover_patch_targets(self, graph_blob: bytes) -> tuple[tuple[npu.PatchTarget, ...], ...]:
                raise npu.NativeError("patchblob discovery", 5, "unsupported", b"requires version 2")

        for dtype, span in (("f16", 2), ("f32", 4), ("u16", 2), ("u32", 4)):
            for scalar_index in (0, 1):
                with self.subTest(dtype=dtype, scalar_index=scalar_index):
                    input_counts = (1, 8) if scalar_index == 0 else (8, 1)
                    tensors = (
                        npu.PatchTensorContract("input", 0, dtype, input_counts[0], input_counts[0] * span, 0x07),
                        npu.PatchTensorContract("input", 1, dtype, input_counts[1], input_counts[1] * span, 0x07),
                        npu.PatchTensorContract("output", 0, dtype, 8, 8 * span, 0x0F),
                    )
                    targets = tuple(npu.PatchTargetV2(index, index, tensors) for index in range(4))
                    fake = PerRowNative()
                    fake.discovery_groups_v2 = (targets,)
                    row_scalar = npu.input("row_scalar", shape=(4, 1), dtype=dtype)
                    matrix = npu.input("matrix", shape=(4, 8), dtype=dtype)
                    inputs = (row_scalar, matrix) if scalar_index == 0 else (matrix, row_scalar)
                    output = npu.custom(*inputs, source=b"per-row kernel", carrier="Subtract")

                    prepared = npu.prepare(
                        npu.Graph([row_scalar, matrix], [output]),
                        native_dir="unused",
                        libraries=fake,  # type: ignore[arg-type]
                    )

                    self.assertEqual(output.shape, (4, 8))
                    self.assertEqual(prepared.mappings[0].status, "automatic")
                    self.assertEqual(prepared.groups[0].scalar_input_indices, (scalar_index,))
                    prepared.build(movi_dll_dir="movi")
                    patched_targets = fake.patch_args[2]
                    if dtype in {"f16", "f32"} and scalar_index == 1:
                        self.assertTrue(all(isinstance(target, npu.PatchTarget) for target in patched_targets))
                        self.assertTrue(all(target.contract_flags & 0x40 for target in patched_targets))
                    else:
                        self.assertEqual(patched_targets, targets)

        unsupported_tensors = (
            npu.PatchTensorContract("input", 0, "u32", 1, 4, 0x07),
            npu.PatchTensorContract("input", 1, "u32", 8, 32, 0x07),
            npu.PatchTensorContract("output", 0, "u32", 8, 32, 0x0F),
        )
        unsupported_targets = tuple(npu.PatchTargetV2(index, index, unsupported_tensors) for index in range(4))
        unsupported_fake = PerRowNative()
        unsupported_fake.discovery_groups_v2 = (unsupported_targets,)
        untested_scalar = npu.input("untested_scalar", shape=(2, 1), dtype="u32")
        untested_matrix = npu.input("untested_matrix", shape=(2, 16), dtype="u32")
        untested_output = npu.custom(
            untested_scalar,
            untested_matrix,
            source=b"untested broadcast",
            carrier="Subtract",
        )
        unsupported = npu.prepare(
            npu.Graph([untested_scalar, untested_matrix], [untested_output]),
            native_dir="unused",
            libraries=unsupported_fake,  # type: ignore[arg-type]
        )
        self.assertEqual(unsupported.mappings[0].status, "explicit-required")
        self.assertIn("scalar input positions", unsupported.mappings[0].reason)

    def test_exact_fp16_five_input_multi_scalar_group_maps_automatically(self) -> None:
        class MultiScalarNative(FakeNative):
            def discover_patch_targets(self, graph_blob: bytes) -> tuple[tuple[npu.PatchTarget, ...], ...]:
                raise npu.NativeError("patchblob discovery", 5, "unsupported", b"requires version 2")

        tensors = (
            npu.PatchTensorContract("input", 0, "f16", 16, 32, 0x07),
            npu.PatchTensorContract("input", 1, "f16", 1, 2, 0x07),
            npu.PatchTensorContract("input", 2, "f16", 1, 2, 0x07),
            npu.PatchTensorContract("input", 3, "f16", 1, 2, 0x07),
            npu.PatchTensorContract("input", 4, "f16", 1, 2, 0x07),
            npu.PatchTensorContract("output", 0, "f16", 16, 32, 0x0F),
        )
        targets = tuple(npu.PatchTargetV2(index, index, tensors) for index in range(4))
        fake = MultiScalarNative()
        fake.discovery_groups_v2 = (targets,)
        data = npu.input("data", shape=(1, 32), dtype="f16")
        scalars = tuple(npu.input(f"scalar_{index}", shape=(1, 1), dtype="f16") for index in range(4))
        output = npu.custom(
            data,
            *scalars,
            source=b"multi-scalar kernel",
            carrier="FakeQuantize",
            levels=16,
        )

        prepared = npu.prepare(
            npu.Graph([data, *scalars], [output]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        self.assertEqual(output.shape, (1, 32))
        self.assertEqual(prepared.mappings[0].status, "automatic")
        self.assertEqual(prepared.groups[0].input_count, 5)
        self.assertEqual(prepared.groups[0].scalar_input_indices, (1, 2, 3, 4))
        prepared.build(movi_dll_dir="movi")
        self.assertEqual(fake.patch_args[2], targets)

        other_data = npu.input("other_data", shape=(2, 16), dtype="f16")
        other_scalars = tuple(npu.input(f"other_scalar_{index}", shape=(1, 1), dtype="f16") for index in range(4))
        other_output = npu.custom(
            other_data,
            *other_scalars,
            source=b"untested multi-scalar kernel",
            carrier="FakeQuantize",
            levels=16,
        )
        unsupported = npu.prepare(
            npu.Graph([other_data, *other_scalars], [other_output]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )
        self.assertEqual(unsupported.mappings[0].status, "explicit-required")
        self.assertIn("scalar input positions", unsupported.mappings[0].reason)

    def test_exact_fp16_five_input_per_row_multi_scalar_group_maps_automatically(self) -> None:
        class MultiScalarNative(FakeNative):
            def discover_patch_targets(self, graph_blob: bytes) -> tuple[tuple[npu.PatchTarget, ...], ...]:
                raise npu.NativeError("patchblob discovery", 5, "unsupported", b"requires version 2")

        tensors = (
            npu.PatchTensorContract("input", 0, "f16", 8, 16, 0x07),
            npu.PatchTensorContract("input", 1, "f16", 1, 2, 0x07),
            npu.PatchTensorContract("input", 2, "f16", 1, 2, 0x07),
            npu.PatchTensorContract("input", 3, "f16", 1, 2, 0x07),
            npu.PatchTensorContract("input", 4, "f16", 1, 2, 0x07),
            npu.PatchTensorContract("output", 0, "f16", 8, 16, 0x0F),
        )
        targets = tuple(npu.PatchTargetV2(index, index, tensors) for index in range(4))
        fake = MultiScalarNative()
        fake.discovery_groups_v2 = (targets,)
        data = npu.input("data", shape=(4, 8), dtype="f16")
        parameters = tuple(npu.input(f"parameter_{index}", shape=(4, 1), dtype="f16") for index in range(4))
        output = npu.custom(
            data,
            *parameters,
            source=b"per-row multi-scalar kernel",
            carrier="FakeQuantize",
            levels=16,
        )

        prepared = npu.prepare(
            npu.Graph([data, *parameters], [output]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        self.assertEqual(prepared.mappings[0].status, "automatic")
        self.assertEqual(prepared.groups[0].per_invocation_input_element_counts, (8, 1, 1, 1, 1))
        prepared.build(movi_dll_dir="movi")
        self.assertEqual(fake.patch_args[2], targets)

        other_parameters = tuple(npu.input(f"other_parameter_{index}", shape=(2, 1), dtype="f16") for index in range(4))
        other_output = npu.custom(
            data,
            *other_parameters,
            source=b"untested per-row multi-scalar kernel",
            carrier="FakeQuantize",
            levels=16,
        )
        unsupported = npu.prepare(
            npu.Graph([data, *other_parameters], [other_output]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )
        self.assertEqual(unsupported.mappings[0].status, "explicit-required")
        self.assertIn("scalar input positions", unsupported.mappings[0].reason)

    def test_exact_fp16_hybrid_five_input_groups_map_automatically(self) -> None:
        class HybridNative(FakeNative):
            def discover_patch_targets(self, graph_blob: bytes) -> tuple[tuple[npu.PatchTarget, ...], ...]:
                raise npu.NativeError("patchblob discovery", 5, "unsupported", b"requires version 2")

        tensors = (
            npu.PatchTensorContract("input", 0, "f16", 4, 8, 0x07),
            npu.PatchTensorContract("input", 1, "f16", 4, 8, 0x07),
            npu.PatchTensorContract("input", 2, "f16", 1, 2, 0x07),
            npu.PatchTensorContract("input", 3, "f16", 1, 2, 0x07),
            npu.PatchTensorContract("input", 4, "f16", 1, 2, 0x07),
            npu.PatchTensorContract("output", 0, "f16", 4, 8, 0x0F),
        )
        targets = tuple(npu.PatchTargetV2(index, index, tensors) for index in range(16))
        fake = HybridNative()
        fake.discovery_groups_v2 = tuple(tuple(targets[group * 4 : group * 4 + 4]) for group in range(4))
        data = npu.input("data", shape=(4, 8), dtype="f16")
        auxiliary = npu.input("auxiliary", shape=(4, 8), dtype="f16")
        parameters = tuple(npu.input(f"parameter_{index}", shape=(4, 1), dtype="f16") for index in range(3))
        output = npu.custom(
            data,
            auxiliary,
            *parameters,
            source=b"hybrid five-input kernel",
            carrier="FakeQuantize",
            levels=16,
        )
        prepared = npu.prepare(
            npu.Graph([data, auxiliary, *parameters], [output]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        self.assertEqual(prepared.mappings[0].status, "automatic")
        self.assertEqual(prepared.mappings[0].group_indices, (0, 1, 2, 3))
        self.assertEqual(prepared.groups[0].scalar_input_indices, (2, 3, 4))
        prepared.build(movi_dll_dir="movi")
        self.assertEqual(fake.patch_args[2], targets)

        with self.assertRaisesRegex(ValueError, "compatible validated multi-group mapping"):
            prepared.plan(bindings={output: prepared.groups[:2]})

    def test_exact_fp16_graph_global_hybrid_five_input_maps_automatically(self) -> None:
        class HybridNative(FakeNative):
            def discover_patch_targets(self, graph_blob: bytes) -> tuple[tuple[npu.PatchTarget, ...], ...]:
                raise npu.NativeError("patchblob discovery", 5, "unsupported", b"requires version 2")

        tensors = (
            npu.PatchTensorContract("input", 0, "f16", 16, 32, 0x07),
            npu.PatchTensorContract("input", 1, "f16", 16, 32, 0x07),
            npu.PatchTensorContract("input", 2, "f16", 1, 2, 0x07),
            npu.PatchTensorContract("input", 3, "f16", 1, 2, 0x07),
            npu.PatchTensorContract("input", 4, "f16", 1, 2, 0x07),
            npu.PatchTensorContract("output", 0, "f16", 16, 32, 0x0F),
        )
        targets = tuple(npu.PatchTargetV2(index, index, tensors) for index in range(4))
        fake = HybridNative()
        fake.discovery_groups_v2 = (targets,)
        data = npu.input("data", shape=(1, 32), dtype="f16")
        auxiliary = npu.input("auxiliary", shape=(1, 32), dtype="f16")
        parameters = tuple(npu.input(f"parameter_{index}", shape=(1, 1), dtype="f16") for index in range(3))
        output = npu.custom(
            data,
            auxiliary,
            *parameters,
            source=b"graph-global hybrid five-input kernel",
            carrier="FakeQuantize",
            levels=16,
        )
        prepared = npu.prepare(
            npu.Graph([data, auxiliary, *parameters], [output]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )

        self.assertEqual(prepared.mappings[0].status, "automatic")
        self.assertEqual(prepared.mappings[0].group_indices, (0,))
        self.assertEqual(prepared.groups[0].scalar_input_indices, (2, 3, 4))
        prepared.build(movi_dll_dir="movi")
        self.assertEqual(fake.patch_args[2], targets)

        other_data = npu.input("other_data", shape=(2, 16), dtype="f16")
        other_auxiliary = npu.input("other_auxiliary", shape=(2, 16), dtype="f16")
        other_output = npu.custom(
            other_data,
            other_auxiliary,
            *parameters,
            source=b"untested graph-global hybrid five-input kernel",
            carrier="FakeQuantize",
            levels=16,
        )
        unsupported = npu.prepare(
            npu.Graph([other_data, other_auxiliary, *parameters], [other_output]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )
        self.assertEqual(unsupported.mappings[0].status, "explicit-required")
        self.assertIn("scalar input positions", unsupported.mappings[0].reason)

    def test_mixed_precision_plan_uses_v1_for_losslessly_representable_group(self) -> None:
        class MixedNative(FakeNative):
            def discover_patch_targets(self, graph_blob: bytes) -> tuple[tuple[npu.PatchTarget, ...], ...]:
                raise npu.NativeError("patchblob discovery", 5, "unsupported", b"mixed precision")

        def tensors(input_dtypes: tuple[str, ...], output_dtype: str) -> tuple[npu.PatchTensorContract, ...]:
            element_count = 16
            values = tuple(
                npu.PatchTensorContract(
                    "input",
                    index,
                    dtype,
                    element_count,
                    element_count * (2 if dtype == "f16" else 4),
                    0x07,
                )
                for index, dtype in enumerate(input_dtypes)
            )
            return values + (
                npu.PatchTensorContract(
                    "output",
                    0,
                    output_dtype,
                    element_count,
                    element_count * (2 if output_dtype == "f16" else 4),
                    0x0F,
                ),
            )

        fake = MixedNative()
        fake.discovery_groups_v2 = (
            (npu.PatchTargetV2(0, 0, tensors(("f32",), "f16"), 1),),
            (npu.PatchTargetV2(1, 1, tensors(("f16", "f16"), "f16"), 0),),
            (npu.PatchTargetV2(2, 2, tensors(("f16",), "f32"), 1),),
        )
        data = npu.input("data", shape=(1, 16), dtype="f32")
        peer = npu.input("peer", shape=(1, 16), dtype="f16")
        half = npu.custom(data, source=b"to-f16", carrier="Convert", _dtype="f16", _name="half")
        mixed = npu.custom(half, peer, source=b"blend", carrier="Maximum", _name="mixed")
        output = npu.custom(mixed, source=b"to-f32", carrier="Convert", _dtype="f32", _name="output")
        prepared = npu.prepare(
            npu.Graph([data, peer], [output]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )
        plan = prepared.plan(bindings={half: prepared.groups[0], mixed: prepared.groups[1], output: prepared.groups[2]})

        self.assertIsInstance(plan._target_groups[0][0], npu.PatchTargetV2)
        self.assertIsInstance(plan._target_groups[1][0], npu.PatchTarget)
        self.assertEqual(plan._target_groups[1], (npu.PatchTarget(1, 1, 2, 16, 32, 0x1F),))
        self.assertIsInstance(plan._target_groups[2][0], npu.PatchTargetV2)
        program = plan.build(movi_dll_dir="movi")
        self.assertEqual(len(fake.patch_calls), 3)
        self.assertEqual(program.graph_blob, b"patched3")

    def test_partition_groups_must_exactly_cover_custom_output(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f32")
        y = npu.custom(x, source=b"kernel", carrier="Abs")
        fake = FakeNative()
        fake.discovery_groups = (
            (npu.PatchTarget(0, 0, 1, 8, 32, 0x3B),),
            (npu.PatchTarget(1, 1, 1, 8, 32, 0x3B),),
        )

        with self.assertRaisesRegex(ValueError, "unique validated multi-group mapping"):
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
        ordinary_group = (npu.PatchTarget(0, 0, 1, 32, 64),)
        custom_group = (npu.PatchTarget(1, 1, 1, 32, 64),)
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
        unary_zero = (npu.PatchTarget(0, 0, 1, 32, 64),)
        unary_one = (npu.PatchTarget(1, 1, 1, 32, 64),)
        binary = (npu.PatchTarget(2, 2, 2, 32, 64),)
        unary_three = (npu.PatchTarget(3, 3, 1, 32, 64),)
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
        group_zero = (npu.PatchTarget(0, 0, 1, 32, 64),)
        group_one = (npu.PatchTarget(1, 1, 1, 32, 64),)
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

    def test_program_run_one_supports_copied_and_shared_tensors(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.Abs(x, _name="y")
        fake = FakeNative()
        program = npu.compile(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        values = np.arange(32, dtype=np.float16).reshape(1, 32)

        copied = program.run_one(values)
        shared_input = program.shared_array(x.shape, x.dtype)
        shared_output = program.shared_array(y.shape, y.dtype)
        np.copyto(shared_input, values)
        shared = program.run_one(shared_input, output=shared_output)

        np.testing.assert_array_equal(copied, values + np.float16(1))
        self.assertIs(shared, shared_output)
        np.testing.assert_array_equal(shared_output, values + np.float16(1))

    def test_program_single_shared_tensor_helpers(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.Abs(x, _name="y")
        fake = FakeNative()
        program = npu.compile(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]

        shared_input = program.shared_input()
        shared_output = program.shared_output()

        self.assertEqual(shared_input.shape, x.shape)
        self.assertEqual(shared_input.dtype, np.dtype("float16"))
        self.assertEqual(shared_output.shape, y.shape)
        self.assertEqual(shared_output.dtype, np.dtype("float16"))
        self.assertIs(shared_input._allocation._session, fake.shared_session)
        self.assertIs(shared_output._allocation._session, fake.shared_session)

    def test_program_i32_copied_and_shared_inference(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="i32")
        y = npu.Abs(x, _name="y")
        fake = FakeNative()
        fake.infer_dtype = "i32"
        program = npu.compile(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        values = np.arange(32, dtype=np.int32).reshape(1, 32)

        copied = program.run_one(values)
        shared_input = program.shared_input()
        shared_output = program.shared_output()
        np.copyto(shared_input, values)
        shared = program.run_one(shared_input, output=shared_output)

        np.testing.assert_array_equal(copied, values + np.int32(1))
        self.assertIs(shared, shared_output)
        self.assertEqual(shared_input.dtype, np.dtype("int32"))
        np.testing.assert_array_equal(shared_output, values + np.int32(1))

        bundle = ROOT / "build" / "python-api-i32-program.npunlock"
        try:
            program.export(bundle)
            loaded = npu.load(bundle, libraries=fake)  # type: ignore[arg-type]
        finally:
            bundle.unlink(missing_ok=True)
        self.assertEqual(loaded.input_contracts, (npu.TensorContract("x", (1, 32), "i32"),))
        self.assertEqual(loaded.output_contracts, (npu.TensorContract("y", (1, 32), "i32"),))
        np.testing.assert_array_equal(loaded.run_one(values), values + np.int32(1))

    def test_program_u8_copied_and_shared_inference(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="u8")
        y = npu.Abs(x, _name="y")
        fake = FakeNative()
        fake.infer_dtype = "u8"
        program = npu.compile(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        values = np.arange(32, dtype=np.uint8).reshape(1, 32)

        copied = program.run_one(values)
        shared_input = program.shared_input()
        shared_output = program.shared_output()
        np.copyto(shared_input, values)
        shared = program.run_one(shared_input, output=shared_output)

        np.testing.assert_array_equal(copied, values + np.uint8(1))
        self.assertIs(shared, shared_output)
        self.assertEqual(shared_input.dtype, np.dtype("uint8"))
        np.testing.assert_array_equal(shared_output, values + np.uint8(1))

        bundle = ROOT / "build" / "python-api-u8-program.npunlock"
        try:
            program.export(bundle)
            loaded = npu.load(bundle, libraries=fake)  # type: ignore[arg-type]
        finally:
            bundle.unlink(missing_ok=True)
        self.assertEqual(loaded.input_contracts, (npu.TensorContract("x", (1, 32), "u8"),))
        self.assertEqual(loaded.output_contracts, (npu.TensorContract("y", (1, 32), "u8"),))
        np.testing.assert_array_equal(loaded.run_one(values), values + np.uint8(1))

    def test_program_i8_copied_and_shared_inference(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="i8")
        y = npu.Abs(x, _name="y")
        fake = FakeNative()
        fake.infer_dtype = "i8"
        program = npu.compile(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        values = np.arange(-16, 16, dtype=np.int8).reshape(1, 32)

        copied = program.run_one(values)
        shared_input = program.shared_input()
        shared_output = program.shared_output()
        np.copyto(shared_input, values)
        shared = program.run_one(shared_input, output=shared_output)

        np.testing.assert_array_equal(copied, values + np.int8(1))
        self.assertIs(shared, shared_output)
        self.assertEqual(shared_input.dtype, np.dtype("int8"))
        np.testing.assert_array_equal(shared_output, values + np.int8(1))

        bundle = ROOT / "build" / "python-api-i8-program.npunlock"
        try:
            program.export(bundle)
            loaded = npu.load(bundle, libraries=fake)  # type: ignore[arg-type]
        finally:
            bundle.unlink(missing_ok=True)
        self.assertEqual(loaded.input_contracts, (npu.TensorContract("x", (1, 32), "i8"),))
        self.assertEqual(loaded.output_contracts, (npu.TensorContract("y", (1, 32), "i8"),))
        np.testing.assert_array_equal(loaded.run_one(values), values + np.int8(1))

    def test_program_i16_copied_and_shared_inference(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="i16")
        y = npu.Abs(x, _name="y")
        fake = FakeNative()
        fake.infer_dtype = "i16"
        program = npu.compile(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        values = np.arange(-16, 16, dtype=np.int16).reshape(1, 32)

        copied = program.run_one(values)
        shared_input = program.shared_input()
        shared_output = program.shared_output()
        np.copyto(shared_input, values)
        shared = program.run_one(shared_input, output=shared_output)

        np.testing.assert_array_equal(copied, values + np.int16(1))
        self.assertIs(shared, shared_output)
        self.assertEqual(shared_input.dtype, np.dtype("int16"))
        np.testing.assert_array_equal(shared_output, values + np.int16(1))

        bundle = ROOT / "build" / "python-api-i16-program.npunlock"
        try:
            program.export(bundle)
            loaded = npu.load(bundle, libraries=fake)  # type: ignore[arg-type]
        finally:
            bundle.unlink(missing_ok=True)
        self.assertEqual(loaded.input_contracts, (npu.TensorContract("x", (1, 32), "i16"),))
        self.assertEqual(loaded.output_contracts, (npu.TensorContract("y", (1, 32), "i16"),))
        np.testing.assert_array_equal(loaded.run_one(values), values + np.int16(1))

    def test_program_u16_copied_and_shared_inference(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="u16")
        y = npu.Abs(x, _name="y")
        fake = FakeNative()
        fake.infer_dtype = "u16"
        program = npu.compile(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        values = np.arange(32, dtype=np.uint16).reshape(1, 32)

        copied = program.run_one(values)
        shared_input = program.shared_input()
        shared_output = program.shared_output()
        np.copyto(shared_input, values)
        shared = program.run_one(shared_input, output=shared_output)

        np.testing.assert_array_equal(copied, values + np.uint16(1))
        self.assertIs(shared, shared_output)
        self.assertEqual(shared_input.dtype, np.dtype("uint16"))
        np.testing.assert_array_equal(shared_output, values + np.uint16(1))

        bundle = ROOT / "build" / "python-api-u16-program.npunlock"
        try:
            program.export(bundle)
            loaded = npu.load(bundle, libraries=fake)  # type: ignore[arg-type]
        finally:
            bundle.unlink(missing_ok=True)
        self.assertEqual(loaded.input_contracts, (npu.TensorContract("x", (1, 32), "u16"),))
        self.assertEqual(loaded.output_contracts, (npu.TensorContract("y", (1, 32), "u16"),))
        np.testing.assert_array_equal(loaded.run_one(values), values + np.uint16(1))

    def test_program_u32_copied_and_shared_inference(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="u32")
        y = npu.Abs(x, _name="y")
        fake = FakeNative()
        fake.infer_dtype = "u32"
        program = npu.compile(npu.Graph([x], [y]), native_dir="unused", libraries=fake)  # type: ignore[arg-type]
        values = np.arange(32, dtype=np.uint32).reshape(1, 32)

        copied = program.run_one(values)
        shared_input = program.shared_input()
        shared_output = program.shared_output()
        np.copyto(shared_input, values)
        shared = program.run_one(shared_input, output=shared_output)

        np.testing.assert_array_equal(copied, values + np.uint32(1))
        self.assertIs(shared, shared_output)
        self.assertEqual(shared_input.dtype, np.dtype("uint32"))
        np.testing.assert_array_equal(shared_output, values + np.uint32(1))

        bundle = ROOT / "build" / "python-api-u32-program.npunlock"
        try:
            program.export(bundle)
            loaded = npu.load(bundle, libraries=fake)  # type: ignore[arg-type]
        finally:
            bundle.unlink(missing_ok=True)
        self.assertEqual(loaded.input_contracts, (npu.TensorContract("x", (1, 32), "u32"),))
        self.assertEqual(loaded.output_contracts, (npu.TensorContract("y", (1, 32), "u32"),))
        np.testing.assert_array_equal(loaded.run_one(values), values + np.uint32(1))

    def test_program_i64_gather_index_contract(self) -> None:
        data = npu.input("data", shape=(1, 32), dtype="u8")
        indices = npu.input("indices", shape=(32,), dtype="i64")
        axis = npu.constant(np.asarray(1, dtype=np.int64))
        y = npu.Gather(data, indices, axis, _name="y")
        fake = FakeNative()
        fake.infer_dtype = "u8"
        program = npu.compile(
            npu.Graph([data, indices], [y]), native_dir="unused", libraries=fake  # type: ignore[arg-type]
        )
        values = np.arange(32, dtype=np.uint8).reshape(1, 32)
        index_values = np.arange(31, -1, -1, dtype=np.int64)

        copied = program.run({"data": values, "indices": index_values})["y"]
        shared_inputs = program.shared_inputs()
        shared_output = program.shared_output()
        np.copyto(shared_inputs["data"], values)
        np.copyto(shared_inputs["indices"], index_values)
        shared = program.run(shared_inputs, outputs={"y": shared_output})["y"]

        np.testing.assert_array_equal(copied, values + np.uint8(1))
        self.assertIs(shared, shared_output)
        self.assertEqual(shared_inputs["indices"].dtype, np.dtype("int64"))
        np.testing.assert_array_equal(shared_output, values + np.uint8(1))

        bundle = ROOT / "build" / "python-api-i64-index-program.npunlock"
        try:
            program.export(bundle)
            loaded = npu.load(bundle, libraries=fake)  # type: ignore[arg-type]
        finally:
            bundle.unlink(missing_ok=True)
        self.assertEqual(
            loaded.input_contracts,
            (npu.TensorContract("data", (1, 32), "u8"), npu.TensorContract("indices", (32,), "i64")),
        )
        np.testing.assert_array_equal(loaded.run({"data": values, "indices": index_values})["y"], values + np.uint8(1))

        i64_data = npu.input("i64_data", shape=(1, 32), dtype="i64")
        unsupported_output = npu.compile(
            npu.Graph([i64_data], [npu.Abs(i64_data)]),
            native_dir="unused",
            libraries=FakeNative(),  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(ValueError, "I64 graph outputs are not supported"):
            unsupported_output.shared_output()

    def test_program_single_shared_tensor_helpers_reject_ambiguous_contracts(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        peer = npu.input("peer", shape=(1, 32), dtype="f16")
        two_inputs = npu.compile(
            npu.Graph([x, peer], [npu.Add(x, peer)]),
            native_dir="unused",
            libraries=FakeNative(),  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(ValueError, "requires exactly one input; program has 2 inputs"):
            two_inputs.shared_input()

        left, right = npu.Split(
            x,
            axis=1,
            _outputs=(npu.TensorSpec((1, 16), "f16"), npu.TensorSpec((1, 16), "f16")),
        )
        two_outputs = npu.compile(
            npu.Graph([x], [left, right]),
            native_dir="unused",
            libraries=FakeNative(),  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(ValueError, "requires exactly one output; program has 2 outputs"):
            two_outputs.shared_output()

    def test_multi_output_contracts_follow_native_source_names(self) -> None:
        x = npu.input("x", shape=(1, 6), dtype="i16")
        axis = npu.constant(np.asarray(1, dtype=np.int64))
        unnamed = npu.compile(
            npu.Graph([x], list(npu.Split(x, axis, num_splits=3))),
            native_dir="unused",
            libraries=FakeNative(),  # type: ignore[arg-type]
        )
        named_outputs = npu.Split(x, axis, num_splits=3, _name="parts")
        named = npu.compile(
            npu.Graph([x], list(named_outputs)),
            native_dir="unused",
            libraries=FakeNative(),  # type: ignore[arg-type]
        )

        self.assertEqual(
            tuple(value.name for value in unnamed.output_contracts), ("Split_2.0", "Split_2.1", "Split_2.2")
        )
        self.assertEqual(tuple(value.name for value in named.output_contracts), ("parts.0", "parts.1", "parts.2"))

        bundle = ROOT / "build" / "python-api-multi-output-program.npunlock"
        try:
            named.export(bundle)
            loaded = npu.load(bundle, libraries=FakeNative())  # type: ignore[arg-type]
        finally:
            bundle.unlink(missing_ok=True)
        self.assertEqual(tuple(value.name for value in loaded.output_contracts), ("parts.0", "parts.1", "parts.2"))

    def test_program_run_one_rejects_multi_tensor_contracts(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        peer = npu.input("peer", shape=(1, 32), dtype="f16")
        fake = FakeNative()
        two_inputs = npu.compile(
            npu.Graph([x, peer], [npu.Add(x, peer)]),
            native_dir="unused",
            libraries=fake,  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(ValueError, "program has 2 inputs and 1 output"):
            two_inputs.run_one(np.zeros((1, 32), dtype=np.float16))

        left, right = npu.Split(
            x,
            axis=1,
            _outputs=(npu.TensorSpec((1, 16), "f16"), npu.TensorSpec((1, 16), "f16")),
        )
        two_outputs = npu.compile(
            npu.Graph([x], [left, right]),
            native_dir="unused",
            libraries=FakeNative(),  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(ValueError, "program has 1 input and 2 outputs"):
            two_outputs.run_one(np.zeros((1, 32), dtype=np.float16))

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

    def test_compare_native_graphs(self) -> None:
        fake = FakeNative()

        comparison = npu.compare_native_graphs(
            bytearray(b"baseline"), memoryview(b"candidate"), libraries=fake  # type: ignore[arg-type]
        )

        self.assertIs(comparison, fake.comparison)
        self.assertEqual(fake.compare_args, (b"baseline", b"candidate"))
        self.assertTrue(comparison.accepted)
        self.assertEqual(comparison.changed_group_indices, (1,))
        self.assertEqual(comparison.mismatch_names, ())

    def test_graph_comparison_rejection_reasons(self) -> None:
        comparison = npu.GraphComparison(
            (1 << 2) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10),
            (0, 1),
            False,
            False,
        )

        self.assertFalse(comparison.accepted)
        self.assertEqual(
            comparison.mismatch_names,
            (
                "output_routing",
                "partial_group_code_change",
                "dma",
                "act_dma_relations",
                "act_dma_dpu_relations",
                "act_dma_act_relations",
            ),
        )

    def test_compare_native_graphs_validates_blobs(self) -> None:
        fake = FakeNative()
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            npu.compare_native_graphs(b"", b"candidate", libraries=fake)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "bytes-like"):
            npu.compare_native_graphs(b"baseline", "candidate", libraries=fake)  # type: ignore[arg-type]

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


class ContractNative(FakeNative):
    def __init__(self) -> None:
        super().__init__()
        self.discovery_groups = ((npu.PatchTarget(0, 0, 1, 32, 64, 0x1F),),)

    def patch_graph(self, graph_blob: bytes, shave_elf: bytes, targets: object) -> npu.PatchResult:
        target_values = tuple(targets)  # type: ignore[arg-type]
        self.patch_args = (graph_blob, shave_elf, target_values)
        self.patch_calls.append(self.patch_args)
        patched = b"strict-patched-" + str(len(self.patch_calls)).encode("ascii")
        report = {
            "schema": "npunlock.patchblob.v1",
            "input_graph_sha256": hashlib.sha256(graph_blob).hexdigest(),
            "output_graph_sha256": hashlib.sha256(patched).hexdigest(),
            "shave_elf_sha256": hashlib.sha256(shave_elf).hexdigest(),
            "graph_contract": {
                "compiler": "8.3",
                "npu": "3720",
                "tiles": 1,
            },
            "targets": [
                {
                    "invocation_index": target.invocation_index,
                    "range_index": target.range_index,
                }
                for target in target_values
            ],
            "append": {"size": len(shave_elf)},
            "validation": {
                "section_contents_preserved": True,
                "mutation_set_preserved": True,
            },
        }
        return npu.PatchResult(patched, json.dumps(report, sort_keys=True).encode("utf-8"))


class ContribFoundationTests(unittest.TestCase):
    @staticmethod
    def _reviewed_binding(serialized: npu.SerializedIR) -> npu.ReviewedPatchBinding:
        payload = {
            "schema": "npunlock.reviewed-patch-binding.v1",
            "metadata": {"purpose": "offline test"},
            "serialized_ir": {
                "xml_sha256": hashlib.sha256(serialized.xml).hexdigest(),
                "weights_sha256": hashlib.sha256(serialized.weights).hexdigest(),
            },
            "build_flags": "NPU_TILES=1",
            "native_graph_sha256": hashlib.sha256(b"native").hexdigest(),
            "provenance": {
                "driver_index": 0,
                "device_index": 0,
                "driver_version": 1,
                "vendor_id": 0x8086,
                "device_id": 0x7D1D,
                "graph_extension_version": 0x10012,
                "compiler_version": [8, 3],
            },
            "inputs": [{"name": "x", "shape": [1, 32], "dtype": "f16"}],
            "outputs": [{"name": "y", "shape": [1, 32], "dtype": "f16"}],
            "targets": [
                {
                    "invocation_index": 0,
                    "range_index": 0,
                    "target_flags": 0,
                    "tensors": [
                        {
                            "role": "input",
                            "index": 0,
                            "dtype": "f16",
                            "element_count": 32,
                            "span_bytes": 64,
                            "observed_flags": 7,
                        },
                        {
                            "role": "output",
                            "index": 0,
                            "dtype": "f16",
                            "element_count": 32,
                            "span_bytes": 64,
                            "observed_flags": 15,
                        },
                    ],
                }
            ],
        }
        return npu.ReviewedPatchBinding.from_bytes(json.dumps(payload).encode("utf-8"))

    def test_carrier_and_program_contracts_replace_manual_contrib_checks(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs", _name="y")
        fake = ContractNative()
        prepared = npu.prepare(npu.Graph([x], [y]), libraries=fake)  # type: ignore[arg-type]
        carrier = npu.CarrierContract(
            (npu.ActGroupContract(("f16",), "f16", (32,), 32, 1),),
            mapping_indices=((0,),),
        )
        self.assertIs(prepared.require(carrier), prepared)
        program = prepared.build(movi_dll_dir="movi").require(
            npu.ProgramContract(
                inputs=(("x", (1, 32), "f16"),),
                outputs=(("y", (1, 32), "f16"),),
                patch_count=1,
                targets_per_patch=(1,),
            )
        )
        self.assertEqual(program.patch_evidence[0].graph_contract["tiles"], 1)
        report = program.patch_evidence[0]
        self.assertEqual(report.summary()["patched_target_count"], 1)
        self.assertNotIn("targets", report.summary())
        self.assertEqual(
            report.to_json()["targets"],
            [{"invocation_index": 0, "range_index": 0}],
        )
        v2_document = report.to_json()
        v2_document["schema"] = "npunlock.patchblob.v2"
        v2_document["section_table"] = {"old_offset": 100, "new_offset": 132}
        v2_report = npu.PatchReport.parse(json.dumps(v2_document).encode("utf-8"))
        self.assertEqual(v2_report.schema, "npunlock.patchblob.v2")
        self.assertEqual(v2_report.to_json(), v2_document)
        self.assertNotIn("section_table", v2_report.summary())
        v2_document["schema"] = "npunlock.patchblob.v3"
        with self.assertRaisesRegex(ValueError, "schema is unsupported"):
            npu.PatchReport.parse(json.dumps(v2_document).encode("utf-8"))
        v2_document["schema"] = []
        with self.assertRaisesRegex(ValueError, "schema is unsupported"):
            npu.PatchReport.parse(json.dumps(v2_document).encode("utf-8"))
        with self.assertRaisesRegex(ValueError, "target count"):
            prepared.require(npu.CarrierContract((npu.ActGroupContract(("f16",), "f16", (32,), 32, 2),)))

    def test_reviewed_external_ir_is_exact_and_builds_a_normal_program(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        serialized = npu.serialize_ir(npu.Graph([x], [npu.Abs(x, _name="y")]))
        binding = self._reviewed_binding(serialized)
        fake = ContractNative()
        prepared = npu.prepare_ir(
            serialized.xml,
            serialized.weights,
            binding=binding,
            libraries=fake,  # type: ignore[arg-type]
        )
        program = prepared.build(npu.KernelSpec(b"kernel"), movi_dll_dir="movi")
        self.assertEqual(program.input_contracts[0].name, "x")
        self.assertEqual(program.output_contracts[0].name, "y")
        self.assertEqual(fake.compile_ir_kwargs["build_flags"], "NPU_TILES=1")
        with self.assertRaisesRegex(ValueError, "XML"):
            npu.prepare_ir(
                serialized.xml + b"corrupt",
                serialized.weights,
                binding=binding,
                libraries=fake,  # type: ignore[arg-type]
            )
        self.assertEqual(fake.compile_ir_count, 1)

        class MisreportingNative(ContractNative):
            def patch_graph(self, graph_blob: bytes, shave_elf: bytes, targets: object) -> npu.PatchResult:
                result = super().patch_graph(graph_blob, shave_elf, targets)
                report = json.loads(result.report_json)
                report["targets"][0]["invocation_index"] = 99
                return npu.PatchResult(
                    result.graph_blob,
                    json.dumps(report, sort_keys=True).encode("utf-8"),
                )

        misreporting = MisreportingNative()
        exact = npu.prepare_ir(
            serialized.xml,
            serialized.weights,
            binding=binding,
            libraries=misreporting,  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(RuntimeError, "selectors"):
            exact.build(npu.KernelSpec(b"kernel"), movi_dll_dir="movi")

    def test_program_cache_rejects_corruption_and_reloads_valid_bundle(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs", _name="y")
        fake = ContractNative()
        program = npu.compile(
            npu.Graph([x], [y]),
            movi_dll_dir="movi",
            libraries=fake,  # type: ignore[arg-type]
        )
        contract = npu.ProgramContract(
            inputs=(("x", (1, 32), "f16"),),
            outputs=(("y", (1, 32), "f16"),),
            patch_count=1,
            targets_per_patch=(1,),
        )
        request = {
            "schema": "test-request.v1",
            "provenance": program.provenance.to_json(),  # type: ignore[union-attr]
        }
        directory = ROOT / "build" / "python-program-cache-test"
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.iterdir():
            path.unlink()
        cache = npu.ProgramCache(directory)
        cache.store(request, program)
        self.assertEqual(cache.status.state, "stored")
        loaded = cache.load(request, libraries=fake, timeout_ms=20_000, contract=contract)
        self.assertIsNotNone(loaded)
        self.assertEqual(cache.status.state, "hit")
        metadata = next(directory.glob("request-*.json"))
        value = json.loads(metadata.read_text(encoding="utf-8"))
        (directory / value["program"]["path"]).write_bytes(b"corrupt")
        self.assertIsNone(cache.load(request, libraries=fake, timeout_ms=20_000, contract=contract))
        self.assertEqual(cache.status.state, "invalid")
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()

    def test_compile_cached_owns_dependency_identity_and_cache_hit(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs", _name="y")
        graph = npu.Graph([x], [y], name="cached")
        carrier = npu.CarrierContract(
            (npu.ActGroupContract(("f16",), "f16", (32,), 32, 1),),
            mapping_indices=((0,),),
        )
        program_contract = npu.ProgramContract(
            inputs=(("x", (1, 32), "f16"),),
            outputs=(("y", (1, 32), "f16"),),
            patch_count=1,
            targets_per_patch=(1,),
        )
        root = ROOT / "build" / "python-compile-cache-test"
        movi = root / "movi"
        native = root / "native"
        cache = root / "cache"
        ir_worker = root / "workers" / "ir-worker.exe"
        movi_worker = root / "workers" / "movi-worker.exe"
        dependencies = {
            movi / "bin" / "moviCompile64.dll": b"compile",
            movi / "bin" / "moviAsm64.dll": b"assemble",
            movi / "bin" / "moviLLD64.dll": b"link",
            movi / "lib" / "mlibm.a": b"math",
            native / "npunlock.dll": b"native",
            native / "npunlock_worker.exe": b"worker",
            ir_worker: b"ir-worker-v1",
            movi_worker: b"movi-worker-v1",
        }
        for path, contents in dependencies.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        fake = ContractNative()
        try:
            first = npu.compile_cached(
                graph,
                cache_dir=cache,
                carrier_contract=carrier,
                program_contract=program_contract,
                movi_dll_dir=movi,
                native_dir=native,
                ir_worker=str(ir_worker),
                movi_worker=str(movi_worker),
                libraries=fake,  # type: ignore[arg-type]
            )
            second = npu.compile_cached(
                graph,
                cache_dir=cache,
                carrier_contract=carrier,
                program_contract=program_contract,
                movi_dll_dir=movi,
                native_dir=native,
                ir_worker=str(ir_worker),
                movi_worker=str(movi_worker),
                libraries=fake,  # type: ignore[arg-type]
            )
            ir_worker.write_bytes(b"ir-worker-v2")
            third = npu.compile_cached(
                graph,
                cache_dir=cache,
                carrier_contract=carrier,
                program_contract=program_contract,
                movi_dll_dir=movi,
                native_dir=native,
                ir_worker=str(ir_worker),
                movi_worker=str(movi_worker),
                libraries=fake,  # type: ignore[arg-type]
            )
            self.assertEqual(first.cache.state, "stored")
            self.assertEqual(second.cache.state, "hit")
            self.assertEqual(third.cache.state, "stored")
            self.assertEqual(fake.compile_ir_count, 2)
            self.assertEqual(len(fake.shave_calls), 2)
        finally:
            first.program.close() if "first" in locals() else None
            second.program.close() if "second" in locals() else None
            third.program.close() if "third" in locals() else None
            for path in sorted(root.rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
            root.rmdir()

    def test_compile_cached_rejects_dependency_drift_before_publication(self) -> None:
        x = npu.input("x", shape=(1, 32), dtype="f16")
        y = npu.custom(x, source=b"kernel", carrier="Abs", _name="y")
        graph = npu.Graph([x], [y], name="drifting-cache")
        carrier = npu.CarrierContract(
            (npu.ActGroupContract(("f16",), "f16", (32,), 32, 1),),
            mapping_indices=((0,),),
        )
        contract = npu.ProgramContract(
            inputs=(("x", (1, 32), "f16"),),
            outputs=(("y", (1, 32), "f16"),),
            patch_count=1,
            targets_per_patch=(1,),
        )
        root = ROOT / "build" / "python-compile-cache-drift-test"
        movi = root / "movi"
        native = root / "native"
        worker = root / "custom-worker.exe"
        dependencies = {
            movi / "bin" / "moviCompile64.dll": b"compile",
            movi / "bin" / "moviAsm64.dll": b"assemble",
            movi / "bin" / "moviLLD64.dll": b"link",
            movi / "lib" / "mlibm.a": b"math",
            native / "npunlock.dll": b"native",
            native / "npunlock_worker.exe": b"worker",
            worker: b"custom-worker-v1",
        }
        for path, contents in dependencies.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)

        class MutatingNative(ContractNative):
            def compile_shave(self, source: bytes, **kwargs: object) -> bytes:
                result = super().compile_shave(source, **kwargs)
                worker.write_bytes(b"custom-worker-v2")
                return result

        try:
            fake = MutatingNative()
            with self.assertRaisesRegex(RuntimeError, "dependencies changed"):
                npu.compile_cached(
                    graph,
                    cache_dir=root / "cache",
                    carrier_contract=carrier,
                    program_contract=contract,
                    movi_dll_dir=movi,
                    native_dir=native,
                    ir_worker=str(worker),
                    movi_worker=str(worker),
                    libraries=fake,  # type: ignore[arg-type]
                )
            self.assertFalse(any((root / "cache").glob("request-*.json")))
        finally:
            for path in sorted(root.rglob("*"), reverse=True):
                path.unlink() if path.is_file() else path.rmdir()
            root.rmdir()


if __name__ == "__main__":
    unittest.main()
