from os import PathLike
from typing import Any, Iterable, Mapping, Sequence, TypeAlias

Shape: TypeAlias = tuple[int, ...]
DType: TypeAlias = str

class TensorSpec:
    shape: Shape
    dtype: DType
    def __init__(self, shape: object, dtype: object) -> None: ...

class Tensor:
    @property
    def shape(self) -> Shape: ...
    @property
    def dtype(self) -> DType: ...
    def __add__(self, other: Tensor) -> Tensor: ...
    def __mul__(self, other: Tensor) -> Tensor: ...
    def __matmul__(self, other: Tensor) -> Tensor: ...

class Graph:
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]
    name: str
    def __init__(self, inputs: Iterable[Tensor], outputs: Iterable[Tensor], name: str = ...) -> None: ...

class PatchTarget:
    def __init__(self, invocation_index: int, range_index: int, input_count: int,
                 element_count: int, span_bytes: int, contract_flags: int = ...) -> None: ...

class InferenceInput:
    selector: int | str
    data: bytes
    def __init__(self, selector: int | str, data: bytes) -> None: ...

class InferenceOutput:
    argument_index: int
    argument_name: str
    shape: Shape
    dtype: DType
    data: bytes

class InferenceResult:
    outputs: tuple[InferenceOutput, ...]
    driver_index: int
    device_index: int
    driver_version: int
    vendor_id: int
    device_id: int

class Program:
    graph_blob: bytes
    def run(self, inputs: Mapping[str, object]) -> Mapping[str, object]: ...

def input(name: str, *, shape: object, dtype: object) -> Tensor: ...
def constant(value: object, *, name: str | None = ...) -> Tensor: ...
def op(name: str, *inputs: Tensor, **attributes: object) -> Tensor | tuple[Tensor, ...]: ...
def custom(*inputs: Tensor, source: str | bytes | PathLike[str], carrier: str,
           _shape: object | None = ..., _dtype: object | None = ...,
           _outputs: Sequence[TensorSpec] | None = ..., _name: str | None = ...,
           _patch_targets: Sequence[PatchTarget] | None = ...,
           **attributes: object) -> Tensor | tuple[Tensor, ...]: ...
def compile(graph: Graph, *, native_dir: str | PathLike[str] | None = ...,
            movi_dll_dir: str | PathLike[str] | None = ...,
            linker_script: str | PathLike[str] | bytes | None = ...,
            build_flags: str = ..., definitions: tuple[str, ...] = ...,
            timeout_ms: int = ..., ir_worker: str | None = ...,
            movi_worker: str | None = ..., infer_worker: str | None = ...,
            libraries: Any = ...) -> Program: ...

def Abs(x: Tensor, *, _shape: object, _dtype: object, _name: str | None = ...) -> Tensor: ...
def Add(a: Tensor, b: Tensor, *, _shape: object, _dtype: object,
        _name: str | None = ..., **attributes: object) -> Tensor: ...
def MatMul(a: Tensor, b: Tensor, *, transpose_a: bool = ..., transpose_b: bool = ...,
           _shape: object, _dtype: object, _name: str | None = ...) -> Tensor: ...
def Multiply(a: Tensor, b: Tensor, *, _shape: object, _dtype: object,
             _name: str | None = ..., **attributes: object) -> Tensor: ...
