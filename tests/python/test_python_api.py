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

        source_free = npu.custom(x, carrier="Abs", _name="compiled_later")
        self.assertIsNone(source_free.producer.metadata["_kernel"])
        self.assertEqual(source_free.spec, x.spec)

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
        self.discovery_groups = ((npu.PatchTarget(0, 0, 1, 32, 64),),)
        self.infer_dtype = "f16"
        self.sessions: list[FakeInferenceSession] = []

    def compile_ir(self, xml: bytes, weights: bytes, **kwargs: object) -> npu.IrCompileResult:
        self.compile_ir_count += 1
        self.xml = xml
        self.weights = weights
        self.compile_ir_kwargs = kwargs
        return npu.IrCompileResult(b"native", 0, 0, 1, 0x8086, 0x7D1D, 0x10012, (8, 3))

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
        try:
            prepared.export(path, plan=plan)
            with zipfile.ZipFile(path, "r") as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["schema"], "npunlock.prepared.v1")
                self.assertNotIn(b"old source", archive.read("manifest.json"))
                self.assertEqual(manifest["bindings"][0]["group_indices"], [1])
                self.assertEqual(manifest["bindings"][1]["group_indices"], [0])

            reloaded_graph, _, _ = make_graph(b"new source")
            loaded = npu.load_prepared(path, graph=reloaded_graph, libraries=fake)  # type: ignore[arg-type]
        finally:
            path.unlink(missing_ok=True)

        self.assertEqual(fake.compile_ir_count, 1)
        self.assertEqual(tuple(mapping.status for mapping in loaded.mappings), ("saved", "saved"))
        loaded_plan = loaded.plan()
        self.assertEqual(loaded_plan.group_indices, ((1,), (0,)))
        self.assertEqual(tuple(kernel.source for kernel in loaded_plan.kernels), (b"mix", b"new source"))
        loaded_plan.build(movi_dll_dir="movi")
        self.assertEqual(fake.compile_ir_count, 1)

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
