from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ._native import IrCompileResult, NativeLibraries, PatchResult, PatchTarget
from .graph import Graph, Node, constant, custom, input, op
from .ir import SerializedIR, serialize_ir
from .tensor import DType, Shape, Tensor, TensorSpec

__all__ = [
    "DType",
    "Graph",
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
    "constant",
    "custom",
    "input",
    "op",
    "serialize_ir",
]


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

    def run(self, inputs: Mapping[str, object]) -> Mapping[str, object]:
        del inputs
        raise NotImplementedError(
            "general Python execution awaits a public C tensor-I/O API; "
            "the current C runner intentionally supports only its fixed add-one proof"
        )


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
    libraries: NativeLibraries | None = None,
) -> Program:
    if not isinstance(graph, Graph):
        raise TypeError("compile() requires a Graph")
    serialized = serialize_ir(graph)
    if serialized.custom_nodes:
        if movi_dll_dir is None or linker_script is None:
            raise ValueError("custom kernels require movi_dll_dir and linker_script")
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
        assert movi_dll_dir is not None and linker_script is not None
        script = linker_script if isinstance(linker_script, bytes) else Path(linker_script).read_bytes()
        for node in serialized.custom_nodes:
            target_values = node.metadata.get("_patch_targets")
            assert isinstance(target_values, tuple)
            elf = native.compile_shave(
                _kernel_source(node.metadata.get("_kernel")),
                movi_dll_dir=movi_dll_dir,
                linker_script=script,
                definitions=definitions,
                timeout_ms=timeout_ms,
                worker=movi_worker,
            )
            patched = native.patch_graph(blob, elf, target_values)
            blob = patched.graph_blob
            reports.append(patched.report_json)
    return Program(blob, graph, serialized, ir_result, tuple(reports))
