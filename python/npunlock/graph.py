from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .kernel import CompiledKernel, KernelSpec
from .tensor import Tensor, TensorSpec, normalize_dtype

_RESERVED = {
    "_shape",
    "_dtype",
    "_name",
    "_outputs",
    "_kernel",
    "_carrier",
    "_patch_targets",
}

_SAME_SPEC_UNARY = {
    "Abs",
    "Exp",
    "Log",
    "Relu",
    "Sigmoid",
    "Sqrt",
    "Tanh",
}

_SAME_SPEC_BINARY = {
    "Add",
    "Divide",
    "Maximum",
    "Minimum",
    "Multiply",
    "Subtract",
}


def _broadcast_shape(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[int, ...]:
    """Return the NumPy-style broadcast shape for two static tensor shapes."""

    result: list[int] = []
    for offset in range(1, max(len(left), len(right)) + 1):
        left_dimension = left[-offset] if offset <= len(left) else 1
        right_dimension = right[-offset] if offset <= len(right) else 1
        if left_dimension != right_dimension and left_dimension != 1 and right_dimension != 1:
            raise ValueError(
                f"custom() cannot infer a broadcast output shape from {left!r} and {right!r}; pass _shape explicitly"
            )
        result.append(max(left_dimension, right_dimension))
    return tuple(reversed(result))


@dataclass(slots=True, eq=False)
class Node:
    op: str
    inputs: tuple[Tensor, ...]
    attrs: dict[str, object]
    output_specs: tuple[TensorSpec, ...]
    name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    outputs: tuple[Tensor, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.outputs = tuple(
            Tensor(self, index, spec, self.name if len(self.output_specs) == 1 else None)
            for index, spec in enumerate(self.output_specs)
        )


def _validate_operator_name(name: object) -> str:
    if not isinstance(name, str) or not name or any(character.isspace() for character in name):
        raise ValueError("operator name must be a non-empty string without whitespace")
    return name


def _constant_integer_vector(tensor: Tensor, operator: str, role: str) -> tuple[int, ...]:
    if tensor.producer.op != "Const":
        raise ValueError(f"automatic {operator} inference requires a constant {role} tensor")
    array = tensor.producer.metadata.get("value")
    if array is None:
        raise ValueError(f"{operator} {role} constant has no value")
    if getattr(array, "ndim", None) != 1 or getattr(getattr(array, "dtype", None), "kind", None) not in {
        "i",
        "u",
    }:
        raise ValueError(f"{operator} {role} must be a one-dimensional integer constant")
    return tuple(int(value) for value in array.reshape(-1))  # type: ignore[union-attr]


def _constant_integer_scalar(tensor: Tensor, operator: str, role: str) -> int:
    if tensor.producer.op != "Const":
        raise ValueError(f"automatic {operator} inference requires a constant {role} tensor")
    array = tensor.producer.metadata.get("value")
    if array is None:
        raise ValueError(f"{operator} {role} constant has no value")
    if getattr(array, "ndim", None) != 0 or getattr(getattr(array, "dtype", None), "kind", None) not in {
        "i",
        "u",
    }:
        raise ValueError(f"{operator} {role} must be a scalar integer constant")
    return int(array.item())  # type: ignore[union-attr]


def _output_specs(
    operator: str,
    inputs: tuple[Tensor, ...],
    metadata: Mapping[str, object],
    attributes: Mapping[str, object],
) -> tuple[TensorSpec, ...]:
    outputs = metadata.get("_outputs")
    if outputs is not None:
        if metadata.get("_shape") is not None or metadata.get("_dtype") is not None:
            raise ValueError("_outputs cannot be combined with _shape or _dtype")
        try:
            specs = tuple(
                item if isinstance(item, TensorSpec) else TensorSpec(item[0], item[1])  # type: ignore[index]
                for item in outputs  # type: ignore[union-attr]
            )
        except (TypeError, IndexError) as exc:
            raise TypeError("_outputs must contain TensorSpec or (shape, dtype) pairs") from exc
        if not specs:
            raise ValueError("_outputs must not be empty")
        return specs
    has_shape = "_shape" in metadata
    has_dtype = "_dtype" in metadata
    if has_shape or has_dtype:
        if not has_shape or not has_dtype:
            raise ValueError("ordinary operator overrides require both _shape and _dtype")
        return (TensorSpec(metadata["_shape"], metadata["_dtype"]),)
    if operator in _SAME_SPEC_UNARY and len(inputs) == 1:
        return (inputs[0].spec,)
    if operator in _SAME_SPEC_BINARY and len(inputs) == 2 and inputs[0].spec == inputs[1].spec:
        return (inputs[0].spec,)
    if operator == "MatMul" and len(inputs) == 2:
        left, right = inputs
        if len(left.shape) != 2 or len(right.shape) != 2:
            raise ValueError("automatic MatMul inference is limited to two rank-2 tensors")
        if left.dtype != right.dtype:
            raise ValueError("automatic MatMul inference requires matching input dtypes")
        transpose_a = attributes["transpose_a"]
        transpose_b = attributes["transpose_b"]
        left_rows, left_inner = (left.shape[1], left.shape[0]) if transpose_a else left.shape
        right_inner, right_columns = (right.shape[1], right.shape[0]) if transpose_b else right.shape
        if left_inner != right_inner:
            raise ValueError(f"MatMul inner dimensions do not match after transpose: {left_inner} != {right_inner}")
        return (TensorSpec((left_rows, right_columns), left.dtype),)
    if operator == "Convert" and len(inputs) == 1:
        if "destination_type" not in attributes:
            raise ValueError("automatic Convert inference requires destination_type")
        return (TensorSpec(inputs[0].shape, normalize_dtype(attributes["destination_type"])),)
    if operator == "Concat" and inputs:
        axis = attributes.get("axis")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("automatic Concat inference requires an integer axis")
        rank = len(inputs[0].shape)
        normalized_axis = axis + rank if axis < 0 else axis
        if not 0 <= normalized_axis < rank:
            raise ValueError(f"Concat axis {axis} is out of range for rank {rank}")
        if any(value.dtype != inputs[0].dtype or len(value.shape) != rank for value in inputs[1:]):
            raise ValueError("automatic Concat inference requires matching input dtypes and ranks")
        shape = list(inputs[0].shape)
        shape[normalized_axis] = 0
        for value in inputs:
            for dimension, (actual, expected) in enumerate(zip(value.shape, inputs[0].shape)):
                if dimension != normalized_axis and actual != expected:
                    raise ValueError("automatic Concat inference requires equal non-axis dimensions")
            shape[normalized_axis] += value.shape[normalized_axis]
        return (TensorSpec(shape, inputs[0].dtype),)
    if operator == "Transpose" and len(inputs) == 2:
        data, order_tensor = inputs
        order = _constant_integer_vector(order_tensor, "Transpose", "order")
        if len(order) != len(data.shape) or set(order) != set(range(len(data.shape))):
            raise ValueError("Transpose order must be a permutation of every input axis")
        return (TensorSpec(tuple(data.shape[axis] for axis in order), data.dtype),)
    if operator == "Reshape":
        if len(inputs) != 2:
            raise ValueError("automatic Reshape inference requires data and pattern tensors")
        data, pattern_tensor = inputs
        pattern = _constant_integer_vector(pattern_tensor, "Reshape", "pattern")
        special_zero = attributes["special_zero"]
        inferred_index: int | None = None
        output_shape: list[int] = []
        known_elements = 1
        for index, dimension in enumerate(pattern):
            if dimension == -1:
                if inferred_index is not None:
                    raise ValueError("Reshape pattern may contain at most one inferred dimension")
                inferred_index = index
                output_shape.append(1)
                continue
            if dimension == 0:
                if not special_zero:
                    raise ValueError("Reshape zero dimensions require special_zero=True")
                if index >= len(data.shape):
                    raise ValueError("Reshape special-zero dimension exceeds the input rank")
                dimension = data.shape[index]
            elif dimension < 0:
                raise ValueError("Reshape pattern dimensions must be positive, zero, or -1")
            output_shape.append(dimension)
            known_elements *= dimension
        input_elements = prod(data.shape)
        if inferred_index is not None:
            if known_elements == 0 or input_elements % known_elements != 0:
                raise ValueError("Reshape inferred dimension does not divide the input element count")
            output_shape[inferred_index] = input_elements // known_elements
        elif known_elements != input_elements:
            raise ValueError("Reshape input and output element counts do not match")
        return (TensorSpec(output_shape, data.dtype),)
    if operator == "Squeeze":
        if len(inputs) != 2:
            raise ValueError("automatic Squeeze inference requires data and axes tensors")
        data, axes_tensor = inputs
        axes = _constant_integer_vector(axes_tensor, "Squeeze", "axes")
        if not axes:
            raise ValueError("automatic Squeeze inference requires at least one axis")
        rank = len(data.shape)
        normalized_axes = tuple(axis + rank if axis < 0 else axis for axis in axes)
        if any(axis < 0 or axis >= rank for axis in normalized_axes):
            raise ValueError("Squeeze axis is out of range")
        if len(set(normalized_axes)) != len(normalized_axes):
            raise ValueError("Squeeze axes must be unique")
        if any(data.shape[axis] != 1 for axis in normalized_axes):
            raise ValueError("Squeeze axes must select dimensions of size one")
        output_shape = tuple(dimension for index, dimension in enumerate(data.shape) if index not in normalized_axes)
        if not output_shape:
            raise ValueError("automatic Squeeze inference does not produce scalar tensors")
        return (TensorSpec(output_shape, data.dtype),)
    if operator == "Unsqueeze":
        if len(inputs) != 2:
            raise ValueError("automatic Unsqueeze inference requires data and axes tensors")
        data, axes_tensor = inputs
        axes = _constant_integer_vector(axes_tensor, "Unsqueeze", "axes")
        if not axes:
            raise ValueError("automatic Unsqueeze inference requires at least one axis")
        output_rank = len(data.shape) + len(axes)
        normalized_axes = tuple(axis + output_rank if axis < 0 else axis for axis in axes)
        if any(axis < 0 or axis >= output_rank for axis in normalized_axes):
            raise ValueError("Unsqueeze axis is out of range")
        if len(set(normalized_axes)) != len(normalized_axes):
            raise ValueError("Unsqueeze axes must be unique")
        inserted = set(normalized_axes)
        source_dimensions = iter(data.shape)
        output_shape = tuple(1 if axis in inserted else next(source_dimensions) for axis in range(output_rank))
        return (TensorSpec(output_shape, data.dtype),)
    if operator == "Split":
        if len(inputs) != 2:
            raise ValueError("automatic Split inference requires data and axis tensors")
        data, axis_tensor = inputs
        axis = _constant_integer_scalar(axis_tensor, "Split", "axis")
        rank = len(data.shape)
        axis = axis + rank if axis < 0 else axis
        if axis < 0 or axis >= rank:
            raise ValueError("Split axis is out of range")
        num_splits = attributes.get("num_splits")
        if isinstance(num_splits, bool) or not isinstance(num_splits, int) or num_splits <= 0:
            raise ValueError("automatic Split inference requires a positive integer num_splits")
        if data.shape[axis] % num_splits != 0:
            raise ValueError("Split axis dimension must be divisible by num_splits")
        output_shape = list(data.shape)
        output_shape[axis] //= num_splits
        return tuple(TensorSpec(output_shape, data.dtype) for _ in range(num_splits))
    if operator == "VariadicSplit":
        if len(inputs) != 3:
            raise ValueError("automatic VariadicSplit inference requires data, axis, and split-length tensors")
        data, axis_tensor, lengths_tensor = inputs
        axis = _constant_integer_scalar(axis_tensor, "VariadicSplit", "axis")
        rank = len(data.shape)
        axis = axis + rank if axis < 0 else axis
        if axis < 0 or axis >= rank:
            raise ValueError("VariadicSplit axis is out of range")
        lengths = list(_constant_integer_vector(lengths_tensor, "VariadicSplit", "split lengths"))
        if not lengths:
            raise ValueError("VariadicSplit requires at least one split length")
        inferred = [index for index, length in enumerate(lengths) if length == -1]
        if len(inferred) > 1:
            raise ValueError("VariadicSplit permits at most one inferred split length")
        if any(length == 0 or length < -1 for length in lengths):
            raise ValueError("VariadicSplit lengths must be positive or one -1")
        known = sum(length for length in lengths if length > 0)
        axis_size = data.shape[axis]
        if inferred:
            remainder = axis_size - known
            if remainder <= 0:
                raise ValueError("VariadicSplit inferred length must be positive")
            lengths[inferred[0]] = remainder
        elif known != axis_size:
            raise ValueError("VariadicSplit lengths must sum to the selected dimension")
        return tuple(
            TensorSpec(data.shape[:axis] + (length,) + data.shape[axis + 1 :], data.dtype) for length in lengths
        )
    if operator == "Pad":
        if len(inputs) != 4:
            raise ValueError("automatic Pad inference requires data, pads_begin, pads_end, and pad_value tensors")
        data, pads_begin_tensor, pads_end_tensor, pad_value = inputs
        if attributes["mode"] != "constant":
            raise ValueError("automatic Pad inference supports only constant mode")
        pads_begin = _constant_integer_vector(pads_begin_tensor, "Pad", "pads_begin")
        pads_end = _constant_integer_vector(pads_end_tensor, "Pad", "pads_end")
        if len(pads_begin) != len(data.shape) or len(pads_end) != len(data.shape):
            raise ValueError("Pad begin/end vectors must contain one value per data axis")
        if any(value < 0 for value in pads_begin + pads_end):
            raise ValueError("automatic Pad inference supports only nonnegative pads")
        if pad_value.producer.op != "Const" or pad_value.shape:
            raise ValueError("automatic Pad inference requires a scalar constant pad_value")
        if pad_value.dtype != data.dtype:
            raise ValueError("Pad data and pad_value dtypes must match")
        output_shape = tuple(dimension + begin + end for dimension, begin, end in zip(data.shape, pads_begin, pads_end))
        return (TensorSpec(output_shape, data.dtype),)
    if operator == "Slice":
        if len(inputs) != 5:
            raise ValueError("automatic Slice inference requires data, starts, stops, steps, and axes tensors")
        data, starts_tensor, stops_tensor, steps_tensor, axes_tensor = inputs
        starts = _constant_integer_vector(starts_tensor, "Slice", "starts")
        stops = _constant_integer_vector(stops_tensor, "Slice", "stops")
        steps = _constant_integer_vector(steps_tensor, "Slice", "steps")
        axes = _constant_integer_vector(axes_tensor, "Slice", "axes")
        if not starts or not (len(starts) == len(stops) == len(steps) == len(axes)):
            raise ValueError("Slice starts, stops, steps, and axes must have the same nonzero length")
        if any(step <= 0 for step in steps):
            raise ValueError("automatic Slice inference supports only positive steps")
        rank = len(data.shape)
        normalized_axes = tuple(axis + rank if axis < 0 else axis for axis in axes)
        if any(axis < 0 or axis >= rank for axis in normalized_axes):
            raise ValueError("Slice axis is out of range")
        if len(set(normalized_axes)) != len(normalized_axes):
            raise ValueError("Slice axes must be unique")
        output_shape = list(data.shape)
        for start, stop, step, axis in zip(starts, stops, steps, normalized_axes):
            normalized_start, normalized_stop, normalized_step = slice(start, stop, step).indices(data.shape[axis])
            dimension = len(range(normalized_start, normalized_stop, normalized_step))
            if dimension == 0:
                raise ValueError("automatic Slice inference does not produce zero-sized dimensions")
            output_shape[axis] = dimension
        return (TensorSpec(output_shape, data.dtype),)
    if operator == "Gather":
        if len(inputs) != 3:
            raise ValueError("automatic Gather inference requires data, indices, and axis tensors")
        data, indices, axis_tensor = inputs
        if indices.dtype not in {"i32", "i64"}:
            raise ValueError("automatic Gather inference requires I32 or I64 indices")
        if attributes["batch_dims"] != 0:
            raise ValueError("automatic Gather inference supports only batch_dims=0")
        axis = _constant_integer_scalar(axis_tensor, "Gather", "axis")
        rank = len(data.shape)
        axis = axis + rank if axis < 0 else axis
        if axis < 0 or axis >= rank:
            raise ValueError("Gather axis is out of range")
        output_shape = data.shape[:axis] + indices.shape + data.shape[axis + 1 :]
        return (TensorSpec(output_shape, data.dtype),)
    raise ValueError(
        f"operator {operator!r} requires _shape and _dtype, or _outputs; "
        "automatic inference is limited to documented same-spec operators"
    )


def op(name: str, *inputs: Tensor, **attributes: object) -> Tensor | tuple[Tensor, ...]:
    operator = _validate_operator_name(name)
    if any(not isinstance(value, Tensor) for value in inputs):
        raise TypeError("operator positional arguments must be symbolic Tensor objects")
    metadata = {key: attributes.pop(key) for key in tuple(attributes) if key.startswith("_")}
    unknown_metadata = set(metadata) - _RESERVED
    if unknown_metadata:
        raise TypeError(f"unknown npunlock metadata: {sorted(unknown_metadata)!r}")
    if operator == "MatMul":
        for attribute in ("transpose_a", "transpose_b"):
            value = attributes.setdefault(attribute, False)
            if not isinstance(value, bool):
                raise TypeError(f"MatMul {attribute} must be bool")
    if operator == "Convert" and "destination_type" in attributes:
        attributes["destination_type"] = normalize_dtype(attributes["destination_type"])
    if operator == "Reshape":
        special_zero = attributes.setdefault("special_zero", False)
        if not isinstance(special_zero, bool):
            raise TypeError("Reshape special_zero must be bool")
    if operator == "Gather":
        batch_dims = attributes.setdefault("batch_dims", 0)
        if isinstance(batch_dims, bool) or not isinstance(batch_dims, int):
            raise TypeError("Gather batch_dims must be an integer")
    if operator == "Pad":
        mode = attributes.setdefault("mode", "constant")
        if not isinstance(mode, str):
            raise TypeError("Pad mode must be a string")
    specs = _output_specs(operator, tuple(inputs), metadata, attributes)
    node_name = metadata.get("_name")
    if node_name is not None and not isinstance(node_name, str):
        raise TypeError("_name must be a string or None")
    node = Node(operator, tuple(inputs), dict(attributes), specs, node_name, metadata)
    return node.outputs[0] if len(node.outputs) == 1 else node.outputs


def input(name: str, *, shape: object, dtype: object) -> Tensor:
    if not isinstance(name, str) or not name:
        raise ValueError("input name must be a non-empty string")
    spec = TensorSpec(shape, dtype)
    return Node("Parameter", (), {}, (spec,), name, {"kind": "input"}).outputs[0]


def constant(value: object, *, name: str | None = None) -> Tensor:
    import numpy as np

    array = np.asarray(value)
    dtype = normalize_numpy_dtype(array.dtype)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    spec = TensorSpec(array.shape, dtype)
    return Node("Const", (), {}, (spec,), name, {"kind": "constant", "value": array}).outputs[0]


def normalize_numpy_dtype(dtype: object) -> str:
    import numpy as np

    mapping = {
        np.dtype("bool"): "boolean",
        np.dtype("float16"): "f16",
        np.dtype("float32"): "f32",
        np.dtype("float64"): "f64",
        np.dtype("int8"): "i8",
        np.dtype("int16"): "i16",
        np.dtype("int32"): "i32",
        np.dtype("int64"): "i64",
        np.dtype("uint8"): "u8",
        np.dtype("uint16"): "u16",
        np.dtype("uint32"): "u32",
        np.dtype("uint64"): "u64",
    }
    try:
        return mapping[np.dtype(dtype)]
    except KeyError as exc:
        raise ValueError(f"NumPy dtype {dtype!r} cannot be serialized to IR") from exc


def custom(
    *inputs: Tensor,
    source: str | bytes | Path | None = None,
    kernel: KernelSpec | CompiledKernel | None = None,
    carrier: str,
    _shape: object | None = None,
    _dtype: object | None = None,
    _outputs: Sequence[TensorSpec] | None = None,
    name: str | None = None,
    _name: str | None = None,
    _patch_targets: Sequence[object] | None = None,
    **attributes: object,
) -> Tensor | tuple[Tensor, ...]:
    """Create a custom node, inferring one output from its carrier when possible.

    Source may be omitted when ``kernel`` is attached directly or an override
    will be supplied to the prepared graph's build plan.
    """

    if source is not None and kernel is not None:
        raise ValueError("custom() accepts source or kernel, not both")
    if kernel is not None and not isinstance(kernel, (KernelSpec, CompiledKernel)):
        raise TypeError("custom() kernel must be a KernelSpec or CompiledKernel")
    if name is not None and _name is not None:
        raise ValueError("custom() accepts name or _name, not both")
    if name is not None and not isinstance(name, str):
        raise TypeError("custom() name must be a string or None")

    carrier_name = _validate_operator_name(carrier)
    if any(not isinstance(value, Tensor) for value in inputs):
        raise TypeError("custom() inputs must be symbolic Tensor objects")
    metadata: dict[str, object] = {
        "_kernel": kernel if kernel is not None else source,
        "_carrier": carrier_name,
        "_name": name if name is not None else _name,
    }
    if _outputs is not None:
        metadata["_outputs"] = _outputs
    else:
        if _shape is None or _dtype is None:
            if not inputs:
                raise ValueError("custom() requires an input to infer omitted _shape or _dtype")
            if _shape is None:
                _shape = (
                    _broadcast_shape(inputs[0].shape, inputs[1].shape)
                    if carrier_name in _SAME_SPEC_BINARY and len(inputs) == 2
                    else inputs[0].shape
                )
            if _dtype is None:
                _dtype = (
                    normalize_dtype(attributes["destination_type"])
                    if carrier_name == "Convert" and "destination_type" in attributes
                    else inputs[0].dtype
                )
        metadata["_shape"] = _shape
        metadata["_dtype"] = _dtype
    if _patch_targets is not None:
        metadata["_patch_targets"] = tuple(_patch_targets)
    return op("Custom", *inputs, **attributes, **metadata)


@dataclass(frozen=True, slots=True)
class Graph:
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]
    name: str = "npunlock_graph"

    def __init__(
        self,
        inputs: Iterable[Tensor],
        outputs: Iterable[Tensor],
        name: str = "npunlock_graph",
    ):
        input_values = tuple(inputs)
        output_values = tuple(outputs)
        if not input_values or not output_values:
            raise ValueError("Graph requires at least one input and one output")
        if any(value.producer.op != "Parameter" for value in input_values):
            raise ValueError("Graph inputs must come from input()")
        if len(set(input_values)) != len(input_values):
            raise ValueError("Graph inputs must be unique")
        if not isinstance(name, str) or not name:
            raise ValueError("Graph name must be a non-empty string")
        nodes = _topological_nodes(output_values)
        reachable = {tensor for node in nodes for tensor in node.inputs}
        reachable.update(output_values)
        missing = [value.name or "<unnamed>" for value in input_values if value not in reachable]
        if missing:
            raise ValueError(f"Graph inputs are not connected to an output: {missing!r}")
        declared = set(input_values)
        undeclared = [
            node.name or "<unnamed>" for node in nodes if node.op == "Parameter" and node.outputs[0] not in declared
        ]
        if undeclared:
            raise ValueError(f"Graph uses undeclared inputs: {undeclared!r}")
        object.__setattr__(self, "inputs", input_values)
        object.__setattr__(self, "outputs", output_values)
        object.__setattr__(self, "name", name)

    @property
    def nodes(self) -> tuple[Node, ...]:
        return _topological_nodes(self.outputs)


def _topological_nodes(outputs: Iterable[Tensor]) -> tuple[Node, ...]:
    ordered: list[Node] = []
    active: set[Node] = set()
    complete: set[Node] = set()

    def visit(node: Node) -> None:
        if node in complete:
            return
        if node in active:
            raise ValueError("symbolic graph contains a cycle")
        active.add(node)
        for value in node.inputs:
            visit(value.producer)
        active.remove(node)
        complete.add(node)
        ordered.append(node)

    for output in outputs:
        if not isinstance(output, Tensor):
            raise TypeError("Graph outputs must be symbolic Tensor objects")
        visit(output.producer)
    return tuple(ordered)
