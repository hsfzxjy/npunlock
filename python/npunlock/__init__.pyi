from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, TypeAlias, overload

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
    producer: Node
    output_index: int
    spec: TensorSpec
    name: str | None
    def __init__(self, producer: Node, output_index: int, spec: TensorSpec, name: str | None = ...) -> None: ...
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

class Node:
    op: str
    inputs: tuple[Tensor, ...]
    attrs: dict[str, object]
    output_specs: tuple[TensorSpec, ...]
    name: str | None
    metadata: dict[str, object]
    outputs: tuple[Tensor, ...]
    def __init__(
        self,
        op: str,
        inputs: tuple[Tensor, ...],
        attrs: dict[str, object],
        output_specs: tuple[TensorSpec, ...],
        name: str | None = ...,
        metadata: dict[str, object] = ...,
    ) -> None: ...

class Graph:
    inputs: tuple[Tensor, ...]
    outputs: tuple[Tensor, ...]
    name: str
    def __init__(self, inputs: Iterable[Tensor], outputs: Iterable[Tensor], name: str = ...) -> None: ...
    @property
    def nodes(self) -> tuple[Node, ...]: ...

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

class PatchResult:
    graph_blob: bytes
    report_json: bytes
    def __init__(self, graph_blob: bytes, report_json: bytes) -> None: ...

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
    def scalar_input_indices(self) -> tuple[int, ...] | None: ...
    @property
    def input_dtypes(self) -> tuple[str, ...] | None: ...
    @property
    def per_invocation_input_element_counts(self) -> tuple[int, ...] | None: ...
    @property
    def per_invocation_output_element_count(self) -> int | None: ...
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

class GraphComparison:
    mismatch_flags: int
    changed_group_indices: tuple[int, ...]
    contract_stable: bool
    unique_changed_group: bool
    @property
    def accepted(self) -> bool: ...
    @property
    def mismatch_names(self) -> tuple[str, ...]: ...

class ActGroupCompleteness:
    group_index: int
    state: str
    output_symbol_raw: int
    invocation_count: int
    routed_invocation_count: int
    replica_invocation_count: int
    unresolved_invocation_count: int
    unresolved_flags: int
    @property
    def observed(self) -> bool: ...
    @property
    def complete(self) -> bool: ...
    @property
    def unresolved_names(self) -> tuple[str, ...]: ...

class ActGroupCoverage:
    group_index: int
    invocation_count: int
    direct_output_state: str
    act_consumer_state: str
    dpu_consumer_state: str
    direct_output_unresolved_flags: int
    act_consumer_unresolved_flags: int
    dpu_consumer_unresolved_flags: int
    direct_routed_invocation_count: int
    direct_replica_invocation_count: int
    act_matched_invocation_count: int
    act_consumer_group_count: int
    act_consumer_input_count: int
    dpu_matched_invocation_count: int
    dpu_input_count: int
    dma_dpu_consumer_state: str
    dma_dpu_consumer_unresolved_flags: int
    dma_dpu_tile_count_expected: int
    dma_dpu_translated_tile_count: int
    dma_dpu_matched_input_count: int
    dma_act_consumer_state: str
    dma_act_consumer_unresolved_flags: int
    dma_act_routed_invocation_count: int
    dma_act_replica_invocation_count: int
    dma_act_consumer_group_count: int
    dma_act_consumer_input_count: int
    dma_act_ddr_span_bytes: int
    @property
    def observed_kinds(self) -> tuple[str, ...]: ...
    @property
    def complete_kinds(self) -> tuple[str, ...]: ...
    @property
    def direct_output_unresolved_names(self) -> tuple[str, ...]: ...
    @property
    def act_consumer_unresolved_names(self) -> tuple[str, ...]: ...
    @property
    def dpu_consumer_unresolved_names(self) -> tuple[str, ...]: ...
    @property
    def dma_dpu_consumer_unresolved_names(self) -> tuple[str, ...]: ...
    @property
    def dma_act_consumer_unresolved_names(self) -> tuple[str, ...]: ...

class CarrierProbe:
    output: Tensor
    baseline_carrier: str
    candidate_carrier: str
    comparison: GraphComparison
    baseline_ir_sha256: str
    candidate_ir_sha256: str
    baseline_graph_blob_sha256: str
    candidate_graph_blob_sha256: str
    candidate_ir: SerializedIR
    candidate_provenance: IrCompileResult
    @property
    def accepted(self) -> bool: ...
    @property
    def group_index(self) -> int | None: ...

class CarrierMappingEvidence:
    output: Tensor
    probes: tuple[CarrierProbe, ...]
    @property
    def accepted_probes(self) -> tuple[CarrierProbe, ...]: ...
    @property
    def status(self) -> str: ...
    @property
    def group_index(self) -> int | None: ...
    def explain(self) -> str: ...

class CarrierProbeFailure:
    candidate_carrier: str
    error_type: str
    message: str
    stage: str | None
    status: int | None
    diagnostic: bytes
    stdout_log: bytes
    stderr_log: bytes

class CarrierProbeReport:
    output: Tensor
    probes: tuple[CarrierProbe, ...]
    failures: tuple[CarrierProbeFailure, ...]
    @property
    def evidence(self) -> CarrierMappingEvidence: ...
    @property
    def status(self) -> str: ...
    @property
    def group_index(self) -> int | None: ...
    def explain(self) -> str: ...

class ReviewedBinding:
    output: Tensor
    group_index: int
    coverage_kind: str
    baseline_graph_blob_sha256: str
    evidence: CarrierMappingEvidence
    coverage: ActGroupCoverage
    def explain(self) -> str: ...

class InferenceInput:
    selector: int | str
    data: bytes | bytearray | memoryview
    def __init__(self, selector: int | str, data: bytes | bytearray | memoryview) -> None: ...

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

class NpuProvenance:
    driver_index: int
    device_index: int
    driver_version: int
    vendor_id: int
    device_id: int
    graph_extension_version: int
    compiler_version: tuple[int, int]
    def __init__(
        self,
        driver_index: int,
        device_index: int,
        driver_version: int,
        vendor_id: int,
        device_id: int,
        graph_extension_version: int,
        compiler_version: tuple[int, int],
    ) -> None: ...
    @classmethod
    def from_ir_result(cls, value: IrCompileResult) -> NpuProvenance: ...
    @classmethod
    def from_json(cls, value: object) -> NpuProvenance: ...
    def to_json(self) -> dict[str, object]: ...

class ArtifactIdentity:
    size: int
    sha256: str
    def __init__(self, size: int, sha256: str) -> None: ...
    @classmethod
    def from_bytes(cls, value: bytes) -> ArtifactIdentity: ...
    def to_json(self) -> dict[str, object]: ...

class PatchReport:
    raw: bytes
    schema: str
    input_graph_sha256: str
    output_graph_sha256: str
    shave_elf_sha256: str
    compiler: str
    npu: str
    graph_contract: Mapping[str, object]
    targets: tuple[Mapping[str, object], ...]
    append: Mapping[str, object]
    validation: Mapping[str, object]
    @classmethod
    def parse(cls, payload: bytes) -> PatchReport: ...
    def to_json(self) -> dict[str, object]: ...
    def summary(self) -> dict[str, object]: ...

class ActGroupContract:
    input_dtypes: tuple[DType, ...]
    output_dtype: DType
    input_element_counts: tuple[int, ...]
    output_element_count: int
    target_count: int
    invocation_indices: tuple[int, ...] | None
    range_indices: tuple[int, ...] | None
    def __init__(
        self,
        input_dtypes: tuple[DType, ...],
        output_dtype: DType,
        input_element_counts: tuple[int, ...],
        output_element_count: int,
        target_count: int,
        invocation_indices: tuple[int, ...] | None = ...,
        range_indices: tuple[int, ...] | None = ...,
    ) -> None: ...
    def mismatch(self, group: ActGroup) -> str | None: ...
    def to_json(self) -> dict[str, object]: ...

class CarrierContract:
    groups: tuple[ActGroupContract, ...]
    mapping_indices: tuple[tuple[int, ...], ...] | None
    def __init__(
        self,
        groups: tuple[ActGroupContract, ...],
        mapping_indices: tuple[tuple[int, ...], ...] | None = ...,
    ) -> None: ...
    def validate(self, prepared: PreparedGraph) -> None: ...
    def to_json(self) -> dict[str, object]: ...

class ProgramContract:
    inputs: tuple[tuple[str, Shape, DType], ...]
    outputs: tuple[tuple[str, Shape, DType], ...]
    patch_count: int
    compiler: str
    npu: str
    targets_per_patch: tuple[int, ...] | None
    def __init__(
        self,
        inputs: tuple[tuple[str, Shape, DType], ...],
        outputs: tuple[tuple[str, Shape, DType], ...],
        patch_count: int,
        compiler: str = ...,
        npu: str = ...,
        targets_per_patch: tuple[int, ...] | None = ...,
    ) -> None: ...
    def validate(self, program: Program) -> None: ...
    def to_json(self) -> dict[str, object]: ...

class ReviewedPatchBinding:
    xml_sha256: str
    weights_sha256: str
    build_flags: str
    native_graph_sha256: str
    provenance: NpuProvenance
    targets: tuple[PatchTargetV2, ...]
    inputs: tuple[tuple[str, Shape, DType], ...]
    outputs: tuple[tuple[str, Shape, DType], ...]
    metadata: Mapping[str, object]
    @classmethod
    def from_bytes(cls, payload: bytes) -> ReviewedPatchBinding: ...
    @classmethod
    def load(cls, source: str | PathLike[str]) -> ReviewedPatchBinding: ...
    def validate_ir(self, xml: bytes, weights: bytes) -> None: ...

class SerializedIR:
    xml: bytes
    weights: bytes
    custom_nodes: tuple[Node, ...]

class NativeLibraries:
    def __init__(self, directory: str | PathLike[str] | None = ...) -> None: ...
    def discover_patch_targets_v2(self, graph_blob: bytes) -> tuple[tuple[PatchTargetV2, ...], ...]: ...
    def compare_graphs(self, baseline_graph_blob: bytes, candidate_graph_blob: bytes) -> GraphComparison: ...
    def inspect_group_completeness(self, graph_blob: bytes) -> tuple[ActGroupCompleteness, ...]: ...
    def inspect_group_coverage(self, graph_blob: bytes) -> tuple[ActGroupCoverage, ...]: ...
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

class CompiledKernel:
    elf: bytes
    elf_name: str
    def __init__(self, elf: bytes | str | PathLike[str]) -> None: ...
    @property
    def elf_sha256(self) -> str: ...
    def save(self, destination: str | PathLike[str]) -> None: ...

class Program:
    graph_blob: bytes
    graph: Graph | None
    serialized_ir: SerializedIR | None
    ir_provenance: IrCompileResult | None
    patch_reports: tuple[bytes, ...]
    artifact_manifest: bytes | None
    @property
    def provenance(self) -> NpuProvenance | None: ...
    @property
    def patch_evidence(self) -> tuple[PatchReport, ...]: ...
    @property
    def manifest(self) -> dict[str, object]: ...
    def require(self, contract: ProgramContract) -> Program: ...
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
    def shared_input(self) -> SharedArray: ...
    def shared_outputs(self) -> dict[str, SharedArray]: ...
    def shared_output(self) -> SharedArray: ...
    def close(self) -> None: ...
    def __enter__(self) -> Program: ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...
    def run(
        self,
        inputs: Mapping[str, object],
        *,
        outputs: Mapping[str, object] | None = ...,
    ) -> Mapping[str, object]: ...
    def run_one(self, input_value: object, *, output: object | None = ...) -> object: ...

class BuildPlan:
    group_indices: tuple[tuple[int, ...], ...]
    kernels: tuple[KernelSpec | CompiledKernel, ...]
    reviewed_bindings: tuple[ReviewedBinding | None, ...]
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
    def require(self, contract: CarrierContract) -> PreparedGraph: ...
    def export(self, destination: str | PathLike[str], *, plan: BuildPlan | None = ...) -> None: ...
    def find_groups(
        self,
        *,
        input_count: int | None = ...,
        dtype: str | None = ...,
        input_dtypes: tuple[str, ...] | None = ...,
        output_dtype: str | None = ...,
        element_count: int | None = ...,
        scalar_input_indices: tuple[int, ...] | None = ...,
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
        scalar_input_indices: tuple[int, ...] | None = ...,
        input_1_scalar: bool | None = ...,
    ) -> ActGroup: ...
    def explain(self) -> str: ...
    def probe_carrier(
        self,
        output: Tensor,
        carrier: str,
        *,
        ir_worker: str | None = ...,
    ) -> CarrierProbe: ...
    def probe_mapping(
        self,
        output: Tensor,
        carriers: Sequence[str],
        *,
        ir_worker: str | None = ...,
    ) -> CarrierMappingEvidence: ...
    def probe_candidates(
        self,
        output: Tensor,
        carriers: Sequence[str],
        *,
        ir_worker: str | None = ...,
    ) -> CarrierProbeReport: ...
    def inspect_group_completeness(self) -> tuple[ActGroupCompleteness, ...]: ...
    def inspect_group_coverage(self) -> tuple[ActGroupCoverage, ...]: ...
    def probe_binding(
        self,
        output: Tensor,
        carriers: Sequence[str] | CarrierProbeReport,
        *,
        coverage_kind: str | None = ...,
        ir_worker: str | None = ...,
    ) -> ReviewedBinding: ...
    def review_binding(
        self,
        output: Tensor,
        evidence: CarrierMappingEvidence,
        *,
        coverage_kind: str,
    ) -> ReviewedBinding: ...
    def plan(
        self,
        *,
        bindings: Mapping[Tensor, object] | None = ...,
        kernels: Mapping[Tensor, KernelSpec | CompiledKernel] | None = ...,
        linker_script: str | PathLike[str] | bytes | None = ...,
        definitions: Sequence[str] = ...,
    ) -> BuildPlan: ...
    def build(
        self,
        *,
        bindings: Mapping[Tensor, object] | None = ...,
        kernels: Mapping[Tensor, KernelSpec | CompiledKernel] | None = ...,
        movi_dll_dir: str | PathLike[str] | None = ...,
        linker_script: str | PathLike[str] | bytes | None = ...,
        definitions: tuple[str, ...] = ...,
        movi_worker: str | None = ...,
    ) -> Program: ...

class PreparedIr:
    xml: bytes
    weights: bytes
    binding: ReviewedPatchBinding
    ir_provenance: IrCompileResult
    @property
    def provenance(self) -> NpuProvenance: ...
    def build(
        self,
        kernel: KernelSpec | CompiledKernel | bytes | str | PathLike[str],
        *,
        movi_dll_dir: str | PathLike[str] | None = ...,
        definitions: Sequence[str] = ...,
        linker_script: bytes | str | PathLike[str] | None = ...,
        movi_worker: str | None = ...,
    ) -> Program: ...

class CacheStatus:
    state: str
    diagnostic: str | None
    load_ms: float
    def __init__(self, state: str, diagnostic: str | None = ..., load_ms: float = ...) -> None: ...

class CachedCompileResult:
    program: Program
    cache: CacheStatus
    build_ms: float
    explanation: str
    def __init__(self, program: Program, cache: CacheStatus, build_ms: float, explanation: str) -> None: ...

class ProgramCache:
    directory: Path
    status: CacheStatus
    def __init__(self, directory: str | PathLike[str]) -> None: ...
    def load(
        self,
        request: Mapping[str, object],
        *,
        libraries: NativeLibraries,
        timeout_ms: int,
        contract: ProgramContract,
    ) -> Program | None: ...
    def store(self, request: Mapping[str, object], program: Program) -> None: ...

def input(name: str, *, shape: object, dtype: object) -> Tensor: ...
def serialize_ir(
    graph: Graph,
    *,
    _carrier_overrides: Mapping[Node, str] | None = ...,
) -> SerializedIR: ...
def configure(*, movi_dll_dir: str | PathLike[str] | None) -> None: ...
def compile_kernel(
    kernel: KernelSpec | bytes | str | PathLike[str],
    *,
    definitions: Sequence[str] = ...,
    linker_script: bytes | str | PathLike[str] | None = ...,
    native_dir: str | PathLike[str] | None = ...,
    movi_dll_dir: str | PathLike[str] | None = ...,
    timeout_ms: int = ...,
    movi_worker: str | None = ...,
    libraries: Any = ...,
) -> CompiledKernel: ...
def constant(value: object, *, name: str | None = ...) -> Tensor: ...
def op(name: str, *inputs: Tensor, **attributes: object) -> Tensor | tuple[Tensor, ...]: ...
def custom(
    *inputs: Tensor,
    source: str | bytes | PathLike[str] | None = ...,
    kernel: KernelSpec | CompiledKernel | None = ...,
    carrier: str,
    _shape: object | None = ...,
    _dtype: object | None = ...,
    _outputs: Sequence[TensorSpec] | None = ...,
    name: str | None = ...,
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
def prepare_ir(
    xml: bytes | bytearray | memoryview,
    weights: bytes | bytearray | memoryview,
    *,
    binding: ReviewedPatchBinding,
    native_dir: str | PathLike[str] | None = ...,
    timeout_ms: int = ...,
    ir_worker: str | None = ...,
    libraries: Any = ...,
) -> PreparedIr: ...
def compile_cached(
    graph: Graph,
    *,
    cache_dir: str | PathLike[str] | None,
    carrier_contract: CarrierContract,
    program_contract: ProgramContract,
    native_dir: str | PathLike[str] | None = ...,
    movi_dll_dir: str | PathLike[str] | None = ...,
    kernels: Mapping[Tensor, KernelSpec | CompiledKernel] | None = ...,
    linker_script: str | PathLike[str] | bytes | None = ...,
    build_flags: str = ...,
    definitions: Sequence[str] = ...,
    timeout_ms: int = ...,
    ir_worker: str | None = ...,
    movi_worker: str | None = ...,
    libraries: Any = ...,
) -> CachedCompileResult: ...
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
    kernels: Mapping[Tensor, KernelSpec | CompiledKernel] | None = ...,
    linker_script: str | PathLike[str] | bytes | None = ...,
    build_flags: str = ...,
    definitions: tuple[str, ...] = ...,
    timeout_ms: int = ...,
    ir_worker: str | None = ...,
    movi_worker: str | None = ...,
    libraries: Any = ...,
) -> Program: ...
def compare_native_graphs(
    baseline: bytes | bytearray | memoryview,
    candidate: bytes | bytearray | memoryview,
    *,
    native_dir: str | PathLike[str] | None = ...,
    libraries: Any = ...,
) -> GraphComparison: ...
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
def Convert(
    data: Tensor,
    *,
    destination_type: object,
    _shape: object = ...,
    _dtype: object = ...,
    _name: str | None = ...,
) -> Tensor: ...
def Concat(
    *inputs: Tensor,
    axis: int,
    _shape: object = ...,
    _dtype: object = ...,
    _name: str | None = ...,
) -> Tensor: ...
def Transpose(
    data: Tensor,
    order: Tensor,
    *,
    _shape: object = ...,
    _dtype: object = ...,
    _name: str | None = ...,
) -> Tensor: ...
def Reshape(
    data: Tensor,
    pattern: Tensor,
    *,
    special_zero: bool = ...,
    _shape: object = ...,
    _dtype: object = ...,
    _name: str | None = ...,
) -> Tensor: ...
def Squeeze(
    data: Tensor,
    axes: Tensor,
    *,
    _shape: object = ...,
    _dtype: object = ...,
    _name: str | None = ...,
) -> Tensor: ...
def Unsqueeze(
    data: Tensor,
    axes: Tensor,
    *,
    _shape: object = ...,
    _dtype: object = ...,
    _name: str | None = ...,
) -> Tensor: ...
@overload
def Split(
    data: Tensor,
    axis: Tensor,
    *,
    num_splits: int,
    _outputs: Sequence[TensorSpec] | None = ...,
    _name: str | None = ...,
) -> tuple[Tensor, ...]: ...
@overload
def Split(
    data: Tensor,
    *,
    axis: int,
    _outputs: Sequence[TensorSpec],
    _name: str | None = ...,
    **attributes: object,
) -> tuple[Tensor, ...]: ...
def Gather(
    data: Tensor,
    indices: Tensor,
    axis: Tensor,
    *,
    batch_dims: int = ...,
    _shape: object = ...,
    _dtype: object = ...,
    _name: str | None = ...,
) -> Tensor: ...
def VariadicSplit(
    data: Tensor,
    axis: Tensor,
    split_lengths: Tensor,
    *,
    _outputs: Sequence[TensorSpec] | None = ...,
    _name: str | None = ...,
) -> tuple[Tensor, ...]: ...
def Slice(
    data: Tensor,
    starts: Tensor,
    stops: Tensor,
    steps: Tensor,
    axes: Tensor,
    *,
    _shape: object = ...,
    _dtype: object = ...,
    _name: str | None = ...,
) -> Tensor: ...
def Pad(
    data: Tensor,
    pads_begin: Tensor,
    pads_end: Tensor,
    pad_value: Tensor,
    *,
    mode: str = ...,
    _shape: object = ...,
    _dtype: object = ...,
    _name: str | None = ...,
) -> Tensor: ...
