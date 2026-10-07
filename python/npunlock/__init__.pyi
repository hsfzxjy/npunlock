from os import PathLike
from typing import Any, Iterable, Mapping, Sequence, TypeAlias

import numpy as np

Shape: TypeAlias = tuple[int, ...]
DType: TypeAlias = str

class NativeError(RuntimeError):
    stage: str
    status: int
    status_name: str
    diagnostic: bytes
    stdout_log: bytes
    stderr_log: bytes

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

class TensorContract:
    name: str
    spec: TensorSpec
    def __init__(self, name: str, shape: object, dtype: object) -> None: ...
    @property
    def shape(self) -> Shape: ...
    @property
    def dtype(self) -> DType: ...

class Graph:
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]
    name: str
    def __init__(self, inputs: Iterable[Tensor], outputs: Iterable[Tensor], name: str = ...) -> None: ...

class PatchTarget:
    def __init__(
        self,
        invocation_index: int,
        range_index: int,
        input_count: int,
        element_count: int,
        span_bytes: int,
        contract_flags: int = ...,
    ) -> None: ...

class PatchTensorContract:
    role: str
    index: int
    dtype: str
    element_count: int
    span_bytes: int
    observed_flags: int
    def __init__(
        self,
        role: str,
        index: int,
        dtype: str,
        element_count: int,
        span_bytes: int,
        observed_flags: int,
    ) -> None: ...

class PatchTargetV2:
    invocation_index: int
    range_index: int
    tensors: tuple[PatchTensorContract, ...]
    target_flags: int
    def __init__(
        self,
        invocation_index: int,
        range_index: int,
        tensors: tuple[PatchTensorContract, ...],
        target_flags: int = ...,
    ) -> None: ...
    @property
    def input_count(self) -> int: ...
    @property
    def output(self) -> PatchTensorContract: ...

class ActGroup:
    index: int
    targets: tuple[PatchTarget | PatchTargetV2, ...]
    @property
    def input_count(self) -> int | None: ...
    @property
    def dtype(self) -> str | None: ...
    @property
    def element_count(self) -> int: ...
    @property
    def span_bytes(self) -> int: ...
    @property
    def contract_flags(self) -> int | None: ...
    @property
    def invocation_indices(self) -> tuple[int, ...]: ...
    @property
    def range_indices(self) -> tuple[int, ...]: ...
    @property
    def input_1_scalar(self) -> bool: ...
    @property
    def input_dtypes(self) -> tuple[str, ...] | None: ...
    @property
    def output_dtype(self) -> str | None: ...
    @property
    def unary_conversion(self) -> bool: ...

class CustomMapping:
    output: Tensor
    name: str
    status: str
    group_indices: tuple[int, ...]
    reason: str

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

class IrCompileResult:
    graph_blob: bytes
    driver_index: int
    device_index: int
    driver_version: int
    vendor_id: int
    device_id: int
    graph_extension_version: int
    compiler_version: tuple[int, int]

class SerializedIR:
    xml: bytes
    weights: bytes

class NativeLibraries:
    def __init__(self, directory: str | PathLike[str] | None = ...) -> None: ...
    def discover_patch_targets_v2(self, graph_blob: bytes) -> tuple[tuple[PatchTargetV2, ...], ...]: ...
    def query_ir_provenance(
        self,
        *,
        timeout_ms: int = ...,
        worker: str | None = ...,
    ) -> IrCompileResult: ...

class SharedArray(np.ndarray): ...

class KernelSpec:
    source: bytes
    definitions: tuple[str, ...]
    linker_script: bytes | None
    source_name: str
    def __init__(
        self,
        source: bytes | str | PathLike[str],
        *,
        definitions: Sequence[str] = ...,
        linker_script: bytes | str | PathLike[str] | None = ...,
    ) -> None: ...
    @property
    def source_sha256(self) -> str: ...
    @property
    def linker_script_sha256(self) -> str | None: ...

class Program:
    graph_blob: bytes
    graph: Graph | None
    serialized_ir: SerializedIR | None
    ir_provenance: IrCompileResult | None
    patch_reports: tuple[bytes, ...]
    artifact_manifest: bytes | None
    @property
    def input_contracts(self) -> tuple[TensorContract, ...]: ...
    @property
    def output_contracts(self) -> tuple[TensorContract, ...]: ...
    @property
    def closed(self) -> bool: ...
    def to_bytes(self) -> bytes: ...
    def save(self, destination: str | PathLike[str]) -> None: ...
    def export(self, destination: str | PathLike[str]) -> None: ...
    def shared_array(self, shape: object, dtype: object) -> SharedArray: ...
    def shared_inputs(self) -> dict[str, SharedArray]: ...
    def shared_outputs(self) -> dict[str, SharedArray]: ...
    def close(self) -> None: ...
    def __enter__(self) -> Program: ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...
    def run(
        self,
        inputs: Mapping[str, object],
        *,
        outputs: Mapping[str, object] | None = ...,
    ) -> Mapping[str, object]: ...

class BuildPlan:
    group_indices: tuple[tuple[int, ...], ...]
    kernels: tuple[KernelSpec, ...]
    def explain(self) -> str: ...
    def build(
        self,
        *,
        movi_dll_dir: str | PathLike[str] | None = ...,
        movi_worker: str | None = ...,
    ) -> Program: ...

class PreparedGraph:
    graph: Graph
    serialized_ir: SerializedIR
    ir_provenance: IrCompileResult
    groups: tuple[ActGroup, ...]
    mappings: tuple[CustomMapping, ...]
    def export(self, destination: str | PathLike[str], *, plan: BuildPlan | None = ...) -> None: ...
    def find_groups(
        self,
        *,
        input_count: int | None = ...,
        dtype: str | None = ...,
        input_dtypes: tuple[str, ...] | None = ...,
        output_dtype: str | None = ...,
        element_count: int | None = ...,
        input_1_scalar: bool | None = ...,
    ) -> tuple[ActGroup, ...]: ...
    def find_group(
        self,
        *,
        input_count: int | None = ...,
        dtype: str | None = ...,
        input_dtypes: tuple[str, ...] | None = ...,
        output_dtype: str | None = ...,
        element_count: int | None = ...,
        input_1_scalar: bool | None = ...,
    ) -> ActGroup: ...
    def explain(self) -> str: ...
    def plan(
        self,
        *,
        bindings: Mapping[Tensor, object] | None = ...,
        kernels: Mapping[Tensor, KernelSpec] | None = ...,
        linker_script: str | PathLike[str] | bytes | None = ...,
        definitions: Sequence[str] = ...,
    ) -> BuildPlan: ...
    def build(
        self,
        *,
        bindings: Mapping[Tensor, object] | None = ...,
        movi_dll_dir: str | PathLike[str] | None = ...,
        linker_script: str | PathLike[str] | bytes | None = ...,
        definitions: tuple[str, ...] = ...,
        movi_worker: str | None = ...,
    ) -> Program: ...

def input(name: str, *, shape: object, dtype: object) -> Tensor: ...
def configure(*, movi_dll_dir: str | PathLike[str] | None) -> None: ...
def constant(value: object, *, name: str | None = ...) -> Tensor: ...
def op(name: str, *inputs: Tensor, **attributes: object) -> Tensor | tuple[Tensor, ...]: ...
def custom(
    *inputs: Tensor,
    source: str | bytes | PathLike[str],
    carrier: str,
    _shape: object | None = ...,
    _dtype: object | None = ...,
    _outputs: Sequence[TensorSpec] | None = ...,
    _name: str | None = ...,
    _patch_targets: Sequence[PatchTarget | PatchTargetV2] | None = ...,
    **attributes: object,
) -> Tensor | tuple[Tensor, ...]: ...
def prepare(
    graph: Graph,
    *,
    native_dir: str | PathLike[str] | None = ...,
    build_flags: str = ...,
    timeout_ms: int = ...,
    ir_worker: str | None = ...,
    libraries: Any = ...,
) -> PreparedGraph: ...
def load_prepared(
    source: str | PathLike[str],
    *,
    graph: Graph,
    native_dir: str | PathLike[str] | None = ...,
    timeout_ms: int = ...,
    ir_worker: str | None = ...,
    libraries: Any = ...,
) -> PreparedGraph: ...
def compile(
    graph: Graph,
    *,
    native_dir: str | PathLike[str] | None = ...,
    movi_dll_dir: str | PathLike[str] | None = ...,
    linker_script: str | PathLike[str] | bytes | None = ...,
    build_flags: str = ...,
    definitions: tuple[str, ...] = ...,
    timeout_ms: int = ...,
    ir_worker: str | None = ...,
    movi_worker: str | None = ...,
    libraries: Any = ...,
) -> Program: ...
def load_native(
    blob: bytes | bytearray | memoryview,
    *,
    graph: Graph,
    native_dir: str | PathLike[str] | None = ...,
    timeout_ms: int = ...,
    libraries: Any = ...,
) -> Program: ...
def load_native_file(
    source: str | PathLike[str],
    *,
    graph: Graph,
    native_dir: str | PathLike[str] | None = ...,
    timeout_ms: int = ...,
    libraries: Any = ...,
) -> Program: ...
def load(
    source: str | PathLike[str],
    *,
    native_dir: str | PathLike[str] | None = ...,
    timeout_ms: int = ...,
    libraries: Any = ...,
) -> Program: ...
def Abs(x: Tensor, *, _shape: object = ..., _dtype: object = ..., _name: str | None = ...) -> Tensor: ...
def Add(
    a: Tensor,
    b: Tensor,
    *,
    _shape: object = ...,
    _dtype: object = ...,
    _name: str | None = ...,
    **attributes: object,
) -> Tensor: ...
def MatMul(
    a: Tensor,
    b: Tensor,
    *,
    transpose_a: bool = ...,
    transpose_b: bool = ...,
    _shape: object = ...,
    _dtype: object = ...,
    _name: str | None = ...,
) -> Tensor: ...
def Multiply(
    a: Tensor,
    b: Tensor,
    *,
    _shape: object = ...,
    _dtype: object = ...,
    _name: str | None = ...,
    **attributes: object,
) -> Tensor: ...
