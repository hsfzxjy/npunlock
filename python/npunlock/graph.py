from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .tensor import Tensor, TensorSpec

_RESERVED = {
    "_shape",
    "_dtype",
    "_name",
    "_outputs",
    "_kernel",
    "_carrier",
    "_patch_targets",
}


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


def _output_specs(metadata: Mapping[str, object]) -> tuple[TensorSpec, ...]:
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
    if "_shape" not in metadata or "_dtype" not in metadata:
        raise ValueError("dynamic operators require _shape and _dtype, or _outputs")
    return (TensorSpec(metadata["_shape"], metadata["_dtype"]),)


def op(name: str, *inputs: Tensor, **attributes: object) -> Tensor | tuple[Tensor, ...]:
    operator = _validate_operator_name(name)
    if any(not isinstance(value, Tensor) for value in inputs):
        raise TypeError("operator positional arguments must be symbolic Tensor objects")
    metadata = {key: attributes.pop(key) for key in tuple(attributes) if key.startswith("_")}
    unknown_metadata = set(metadata) - _RESERVED
    if unknown_metadata:
        raise TypeError(f"unknown npunlock metadata: {sorted(unknown_metadata)!r}")
    specs = _output_specs(metadata)
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
    source: str | bytes | Path,
    carrier: str,
    _shape: object | None = None,
    _dtype: object | None = None,
    _outputs: Sequence[TensorSpec] | None = None,
    _name: str | None = None,
    _patch_targets: Sequence[object] | None = None,
    **attributes: object,
) -> Tensor | tuple[Tensor, ...]:
    """Create a custom node, inheriting a single output from its first input."""

    metadata: dict[str, object] = {
        "_kernel": source,
        "_carrier": _validate_operator_name(carrier),
        "_name": _name,
    }
    if _outputs is not None:
        metadata["_outputs"] = _outputs
    else:
        if _shape is None or _dtype is None:
            if not inputs:
                raise ValueError("custom() requires an input to infer omitted _shape or _dtype")
            if not isinstance(inputs[0], Tensor):
                raise TypeError("custom() inputs must be symbolic Tensor objects")
            if _shape is None:
                _shape = inputs[0].shape
            if _dtype is None:
                _dtype = inputs[0].dtype
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
