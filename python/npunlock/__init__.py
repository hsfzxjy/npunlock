from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ._native import (
    InferenceInput,
    InferenceOutput,
    InferenceResult,
    IrCompileResult,
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
    "op",
    "serialize_ir",
]

_configured_movi_dll_dir: str | None = None


def configure(*, movi_dll_dir: str | Path | None) -> None:
    """Set the process-local MoviTools directory used by custom compilation.

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
    serialized_ir: SerializedIR
    ir_provenance: IrCompileResult
    patch_reports: tuple[bytes, ...]
    _libraries: NativeLibraries = field(repr=False, compare=False)
    _timeout_ms: int = field(repr=False, compare=False)
    _infer_worker: str | None = field(repr=False, compare=False)

    def run(self, inputs: Mapping[str, object]) -> Mapping[str, object]:
        import numpy as np

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
            if tensor.dtype != "f16":
                raise ValueError("graphinfer currently supports only static FP16 tensors")
            array = np.asarray(inputs[name])
            if array.dtype != np.dtype("float16") or tuple(array.shape) != tensor.shape:
                raise ValueError(
                    f"input {name!r} requires shape {tensor.shape!r} and dtype float16"
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
            if output.dtype != "f16" or output.shape != tensor.shape:
                raise RuntimeError(
                    f"output {name!r} returned shape {output.shape!r} and dtype {output.dtype}"
                )
            expected_size = int(np.prod(tensor.shape, dtype=np.int64)) * np.dtype("float16").itemsize
            if len(output.data) != expected_size:
                raise RuntimeError(
                    f"output {name!r} returned {len(output.data)} bytes; expected {expected_size}"
                )
            values[name] = np.frombuffer(output.data, dtype=np.float16).copy().reshape(tensor.shape)
        return values


def _kernel_source(value: object) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (str, Path)):
        return Path(value).read_bytes()
    raise TypeError("custom _kernel must be source bytes or a filesystem path")


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
    resolved_movi_dll_dir = _resolve_movi_dll_dir(movi_dll_dir)
    if serialized.custom_nodes:
        if resolved_movi_dll_dir is None or linker_script is None:
            raise ValueError(
                "custom kernels require a MoviTools directory from movi_dll_dir, "
                "configure(), or NPUNLOCK_MOVITOOLS_DIR, plus linker_script"
            )
        for node in serialized.custom_nodes:
            target_values = node.metadata.get("_patch_targets")
            if not isinstance(target_values, tuple) or not target_values or not all(
                isinstance(target, PatchTarget) for target in target_values
            ):
                raise ValueError(
                    f"custom node {node.name or '<unnamed>'!r} requires explicit "
                    "_patch_targets made of PatchTarget values"
                )
    if libraries is None and native_dir is None:
        raise ValueError("native_dir is required when libraries is not supplied")
    native = libraries or NativeLibraries(native_dir)  # type: ignore[arg-type]
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
        assert resolved_movi_dll_dir is not None and linker_script is not None
        script = linker_script if isinstance(linker_script, bytes) else Path(linker_script).read_bytes()
        for node in serialized.custom_nodes:
            target_values = node.metadata.get("_patch_targets")
            assert isinstance(target_values, tuple)
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
