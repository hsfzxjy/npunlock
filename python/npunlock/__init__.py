from __future__ import annotations

import os
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Any, Callable, Mapping

from ._native import (
    InferenceInput,
    InferenceOutput,
    InferenceResult,
    IrCompileResult,
    NativeError,
    NativeLibraries,
    PatchResult,
    PatchTarget,
)
from .graph import Graph, Node, constant, custom, input, op
from .ir import SerializedIR, serialize_ir
from .tensor import DType, Shape, Tensor, TensorSpec

__all__ = [
    "DType",
    "Graph",
    "InferenceInput",
    "InferenceOutput",
    "InferenceResult",
    "IrCompileResult",
    "NativeError",
    "NativeLibraries",
    "Node",
    "PatchResult",
    "PatchTarget",
    "Program",
    "SerializedIR",
    "Shape",
    "Tensor",
    "TensorSpec",
    "compile",
    "configure",
    "constant",
    "custom",
    "input",
    "load_native",
    "load_native_file",
    "op",
    "serialize_ir",
]

_configured_movi_dll_dir: str | None = None


def configure(*, movi_dll_dir: str | Path | None) -> None:
    """Set the process-local MVC_DEPEND root used by custom compilation.

    Passing ``None`` clears the process-local override so that
    ``NPUNLOCK_MOVITOOLS_DIR`` is consulted again.
    """

    global _configured_movi_dll_dir
    if movi_dll_dir is None:
        _configured_movi_dll_dir = None
        return
    value = os.fspath(movi_dll_dir)
    if not value:
        raise ValueError("movi_dll_dir must not be empty")
    _configured_movi_dll_dir = value


def _resolve_movi_dll_dir(explicit: str | Path | None) -> str | Path | None:
    if explicit is not None:
        return explicit
    if _configured_movi_dll_dir is not None:
        return _configured_movi_dll_dir
    return os.environ.get("NPUNLOCK_MOVITOOLS_DIR") or None


class OpFactory:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, *inputs: Tensor, **attributes: object) -> Tensor | tuple[Tensor, ...]:
        return op(self.name, *inputs, **attributes)

    def __repr__(self) -> str:
        return f"OpFactory({self.name!r})"


def __getattr__(name: str) -> OpFactory:
    if name.startswith("_"):
        raise AttributeError(name)
    return OpFactory(name)


@dataclass(frozen=True, slots=True)
class Program:
    graph_blob: bytes
    graph: Graph
    serialized_ir: SerializedIR | None
    ir_provenance: IrCompileResult | None
    patch_reports: tuple[bytes, ...]
    _libraries: NativeLibraries = field(repr=False, compare=False)
    _timeout_ms: int = field(repr=False, compare=False)
    _infer_worker: str | None = field(repr=False, compare=False)

    def to_bytes(self) -> bytes:
        """Return the complete native graph blob."""

        return self.graph_blob

    def save(self, destination: str | Path) -> None:
        """Write the complete native graph blob to *destination*."""

        Path(destination).write_bytes(self.graph_blob)

    def run(self, inputs: Mapping[str, object]) -> Mapping[str, object]:
        import numpy as np

        numpy_dtypes = {
            "f16": np.dtype("float16"),
            "f32": np.dtype("float32"),
        }
        if not isinstance(inputs, Mapping):
            raise TypeError("Program.run() inputs must be a mapping")
        expected_names = tuple(value.name for value in self.graph.inputs)
        if any(name is None for name in expected_names):
            raise ValueError("all graph inputs must have names")
        if set(inputs) != set(expected_names):
            raise ValueError(
                f"input names must exactly match {list(expected_names)!r}; got {list(inputs)!r}"
            )
        native_inputs: list[InferenceInput] = []
        for tensor, name in zip(self.graph.inputs, expected_names):
            assert name is not None
            expected_dtype = numpy_dtypes.get(tensor.dtype)
            if expected_dtype is None:
                raise ValueError("graphinfer currently supports only static FP16/FP32 tensors")
            array = np.asarray(inputs[name])
            if array.dtype != expected_dtype or tuple(array.shape) != tensor.shape:
                raise ValueError(
                    f"input {name!r} requires shape {tensor.shape!r} and dtype {expected_dtype}"
                )
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            native_inputs.append(InferenceInput(name, array.tobytes(order="C")))
        inferred = self._libraries.infer_graph(
            self.graph_blob,
            native_inputs,
            timeout_ms=self._timeout_ms,
            worker=self._infer_worker,
        )
        if len(inferred.outputs) != len(self.graph.outputs):
            raise RuntimeError(
                f"graph returned {len(inferred.outputs)} outputs; expected {len(self.graph.outputs)}"
            )
        values: dict[str, object] = {}
        for index, (tensor, output) in enumerate(zip(self.graph.outputs, inferred.outputs)):
            name = tensor.name or f"Result_{index}"
            expected_dtype = numpy_dtypes.get(tensor.dtype)
            if expected_dtype is None:
                raise RuntimeError(
                    f"output {name!r} uses unsupported symbolic dtype {tensor.dtype!r}"
                )
            if output.dtype != tensor.dtype or output.shape != tensor.shape:
                raise RuntimeError(
                    f"output {name!r} returned shape {output.shape!r} and dtype {output.dtype}"
                )
            expected_size = int(np.prod(tensor.shape, dtype=np.int64)) * expected_dtype.itemsize
            if len(output.data) != expected_size:
                raise RuntimeError(
                    f"output {name!r} returned {len(output.data)} bytes; expected {expected_size}"
                )
            values[name] = np.frombuffer(output.data, dtype=expected_dtype).copy().reshape(tensor.shape)
        return values


def _native_blob_bytes(value: object) -> bytes:
    if isinstance(value, bytes):
        blob = value
    elif isinstance(value, (bytearray, memoryview)):
        blob = bytes(value)
    else:
        raise TypeError("native graph blob must be bytes-like")
    if not blob:
        raise ValueError("native graph blob must not be empty")
    return blob


def load_native(
    blob: bytes | bytearray | memoryview,
    *,
    graph: Graph,
    native_dir: str | Path | None = None,
    timeout_ms: int = 20_000,
    infer_worker: str | None = None,
    libraries: NativeLibraries | None = None,
) -> Program:
    """Create an executable program from native graph bytes.

    ``graph`` supplies the symbolic input and output contract used by
    :meth:`Program.run`; loading does not compile IR or custom C again.
    """

    if not isinstance(graph, Graph):
        raise TypeError("load_native() requires a Graph")
    native = libraries or NativeLibraries(native_dir)
    return Program(
        _native_blob_bytes(blob),
        graph,
        None,
        None,
        (),
        native,
        timeout_ms,
        infer_worker,
    )


def load_native_file(
    source: str | Path,
    *,
    graph: Graph,
    native_dir: str | Path | None = None,
    timeout_ms: int = 20_000,
    infer_worker: str | None = None,
    libraries: NativeLibraries | None = None,
) -> Program:
    """Create an executable program from a native graph file."""

    return load_native(
        Path(source).read_bytes(),
        graph=graph,
        native_dir=native_dir,
        timeout_ms=timeout_ms,
        infer_worker=infer_worker,
        libraries=libraries,
    )


def _kernel_source(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (str, Path)):
        return Path(value).read_bytes()
    raise TypeError("custom _kernel must be source bytes or a filesystem path")


def _combined_custom_targets(
    node: Node,
    groups: tuple[tuple[PatchTarget, ...], ...],
) -> tuple[PatchTarget, ...] | None:
    """Validate a compiler-partitioned custom operation and flatten its targets."""

    if len(groups) < 2 or len(node.outputs) != 1:
        return None
    item_sizes = {"f16": 2, "f32": 4}
    output = node.outputs[0]
    item_size = item_sizes.get(output.dtype)
    if item_size is None:
        return None
    targets = tuple(target for group in groups for target in group)
    if not targets:
        return None
    input_count = len(node.inputs)
    flags = targets[0].contract_flags
    if any(
        target.input_count != input_count
        or target.span_bytes != target.element_count * item_size
        or target.contract_flags != flags
        for target in targets
    ):
        return None
    first_invocation = targets[0].invocation_index
    if tuple(target.invocation_index for target in targets) != tuple(
        range(first_invocation, first_invocation + len(targets))
    ):
        return None
    if len({(target.invocation_index, target.range_index) for target in targets}) != len(
        targets
    ):
        return None
    if sum(target.element_count for target in targets) != prod(output.shape):
        return None
    return targets


def _automatic_target_groups(
    graph: Graph,
    custom_nodes: tuple[Node, ...],
    discovered_groups: tuple[tuple[PatchTarget, ...], ...],
) -> tuple[tuple[PatchTarget, ...], ...]:
    computational_nodes = tuple(
        node for node in graph.nodes if node.op not in {"Parameter", "Const"}
    )

    # Keep the established positional contract unchanged when it applies.
    if len(discovered_groups) == len(computational_nodes):
        node_groups = dict(zip(computational_nodes, discovered_groups))
        selected = tuple(node_groups[node] for node in custom_nodes)
        for node, targets in zip(custom_nodes, selected):
            if any(target.input_count != len(node.inputs) for target in targets):
                raise ValueError(
                    f"discovered ACT group for custom node {node.name or '<unnamed>'!r} "
                    "does not match its input arity"
                )
        return selected

    solutions: list[dict[Node, tuple[PatchTarget, ...]]] = []

    def visit(
        node_index: int,
        group_index: int,
        selected: dict[Node, tuple[PatchTarget, ...]],
    ) -> None:
        if len(solutions) > 1:
            return
        if node_index == len(computational_nodes):
            if group_index == len(discovered_groups):
                solutions.append(dict(selected))
            return

        remaining_nodes = len(computational_nodes) - node_index - 1
        maximum_take = len(discovered_groups) - group_index - remaining_nodes
        if maximum_take < 1:
            return
        node = computational_nodes[node_index]
        take_values = range(1, maximum_take + 1) if node in custom_nodes else (1,)
        for take in take_values:
            groups = discovered_groups[group_index : group_index + take]
            if take == 1:
                targets = groups[0]
                if not targets:
                    continue
                if node in custom_nodes and any(
                    target.input_count != len(node.inputs) for target in targets
                ):
                    continue
            else:
                targets = _combined_custom_targets(node, groups)
                if targets is None:
                    continue
            if node in custom_nodes:
                selected[node] = targets
            visit(node_index + 1, group_index + take, selected)
            selected.pop(node, None)

    visit(0, 0, {})
    if len(solutions) != 1:
        reason = "no" if not solutions else "multiple"
        raise ValueError(
            f"native graph has {len(discovered_groups)} positional ACT groups for "
            f"{len(computational_nodes)} computational nodes and {reason} valid mapping; "
            "automatic selection requires a one-to-one positional mapping or one unique "
            "exact-cover partition mapping, otherwise provide explicit _patch_targets"
        )
    return tuple(solutions[0][node] for node in custom_nodes)


def compile(
    graph: Graph,
    *,
    native_dir: str | Path | None = None,
    movi_dll_dir: str | Path | None = None,
    linker_script: str | Path | bytes | None = None,
    build_flags: str = "",
    definitions: tuple[str, ...] = (),
    timeout_ms: int = 20_000,
    ir_worker: str | None = None,
    movi_worker: str | None = None,
    infer_worker: str | None = None,
    libraries: NativeLibraries | None = None,
) -> Program:
    if not isinstance(graph, Graph):
        raise TypeError("compile() requires a Graph")
    serialized = serialize_ir(graph)
    preserves_fp32_custom = any(
        any(output.dtype == "f32" for output in node.outputs)
        for node in serialized.custom_nodes
    )
    if preserves_fp32_custom:
        accuracy_flag = 'EXECUTION_MODE_HINT="ACCURACY"'
        if not build_flags:
            build_flags = f"--config {accuracy_flag}"
        elif accuracy_flag not in build_flags:
            raise ValueError(
                "FP32 custom kernels require build_flags containing "
                "EXECUTION_MODE_HINT=\"ACCURACY\" to prevent FP16 lowering"
            )
    resolved_movi_dll_dir = _resolve_movi_dll_dir(movi_dll_dir)
    explicit_target_groups: tuple[tuple[PatchTarget, ...], ...] | None = None
    if serialized.custom_nodes:
        if resolved_movi_dll_dir is None:
            raise ValueError(
                "custom kernels require an MVC_DEPEND root from movi_dll_dir, "
                "configure(), or NPUNLOCK_MOVITOOLS_DIR"
            )
        supplied = tuple(node.metadata.get("_patch_targets") for node in serialized.custom_nodes)
        if any(value is not None for value in supplied):
            if not all(
                isinstance(value, tuple)
                and value
                and all(isinstance(target, PatchTarget) for target in value)
                for value in supplied
            ):
                raise ValueError(
                    "either omit _patch_targets for every custom node or provide a non-empty "
                    "PatchTarget sequence for every custom node"
                )
            explicit_target_groups = supplied  # type: ignore[assignment]
        else:
            if any(len(node.outputs) != 1 for node in serialized.custom_nodes):
                raise ValueError(
                    "automatic patch selection currently requires one output per custom node"
                )
    native = libraries or NativeLibraries(native_dir)
    ir_result = native.compile_ir(
        serialized.xml,
        serialized.weights,
        build_flags=build_flags,
        timeout_ms=timeout_ms,
        worker=ir_worker,
    )
    blob = ir_result.graph_blob
    reports: list[bytes] = []
    if serialized.custom_nodes:
        assert resolved_movi_dll_dir is not None
        script = (
            linker_script
            if isinstance(linker_script, bytes) or linker_script is None
            else Path(linker_script).read_bytes()
        )
        target_groups = explicit_target_groups
        if target_groups is None:
            target_groups = _automatic_target_groups(
                graph,
                serialized.custom_nodes,
                native.discover_patch_targets(blob),
            )
        for node, target_values in zip(serialized.custom_nodes, target_groups):
            elf = native.compile_shave(
                _kernel_source(node.metadata.get("_kernel")),
                movi_dll_dir=resolved_movi_dll_dir,
                linker_script=script,
                definitions=definitions,
                timeout_ms=timeout_ms,
                worker=movi_worker,
            )
            patched = native.patch_graph(blob, elf, target_values)
            blob = patched.graph_blob
            reports.append(patched.report_json)
    return Program(
        blob,
        graph,
        serialized,
        ir_result,
        tuple(reports),
        native,
        timeout_ms,
        infer_worker,
    )
