from __future__ import annotations

import ctypes
import hashlib
import io
import json
import os
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ._native import (
    InferenceInput,
    InferenceOutput,
    InferenceResult,
    InferenceSession,
    IrCompileResult,
    NativeError,
    NativeLibraries,
    NativeSharedBuffer,
    PatchResult,
    PatchTarget,
)
from .graph import Graph, Node, constant, custom, input, op
from .ir import SerializedIR, serialize_ir
from .tensor import DType, Shape, Tensor, TensorSpec

__all__ = [
    "ActGroup",
    "CustomMapping",
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
    "PreparedGraph",
    "Program",
    "SerializedIR",
    "Shape",
    "SharedArray",
    "Tensor",
    "TensorContract",
    "TensorSpec",
    "compile",
    "configure",
    "constant",
    "custom",
    "input",
    "load_native",
    "load_native_file",
    "load",
    "op",
    "prepare",
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


class SharedArray(np.ndarray):
    """A NumPy array backed by host/NPU shared Level Zero memory."""

    _allocation: NativeSharedBuffer | None

    def __new__(
        cls,
        allocation: NativeSharedBuffer,
        shape: Shape,
        dtype: np.dtype[Any],
    ) -> SharedArray:
        raw_type = ctypes.c_uint8 * allocation.size
        raw = raw_type.from_address(allocation.address)
        value = np.ctypeslib.as_array(raw).view(dtype).reshape(shape).view(cls)
        value._allocation = allocation
        return value

    def __array_finalize__(self, source: object) -> None:
        allocation = getattr(source, "_allocation", None)
        if allocation is None:
            self._allocation = None
            return
        address = self.__array_interface__["data"][0]
        self._allocation = allocation if allocation.address <= address < allocation.address + allocation.size else None


class _ProgramSharedState:
    def __init__(self) -> None:
        self.session: InferenceSession | None = None


@dataclass(frozen=True, slots=True)
class ActGroup:
    """One validated positional ACT group discovered in a native carrier."""

    index: int
    targets: tuple[PatchTarget, ...]

    @property
    def input_count(self) -> int | None:
        values = {target.input_count for target in self.targets}
        return next(iter(values)) if len(values) == 1 else None

    @property
    def dtype(self) -> str | None:
        values = {
            (
                "f16"
                if target.contract_flags & 0x04 and not target.contract_flags & 0x20
                else "f32" if target.contract_flags & 0x20 and not target.contract_flags & 0x04 else None
            )
            for target in self.targets
        }
        return next(iter(values)) if len(values) == 1 else None

    @property
    def element_count(self) -> int:
        return sum(target.element_count for target in self.targets)

    @property
    def span_bytes(self) -> int:
        return sum(target.span_bytes for target in self.targets)

    @property
    def contract_flags(self) -> int | None:
        values = {target.contract_flags for target in self.targets}
        return next(iter(values)) if len(values) == 1 else None

    @property
    def invocation_indices(self) -> tuple[int, ...]:
        return tuple(target.invocation_index for target in self.targets)

    @property
    def range_indices(self) -> tuple[int, ...]:
        return tuple(target.range_index for target in self.targets)

    @property
    def input_1_scalar(self) -> bool:
        flags = self.contract_flags
        return flags is not None and bool(flags & 0x40)


@dataclass(frozen=True, slots=True)
class CustomMapping:
    """The prepared carrier's current mapping decision for one custom node."""

    output: Tensor
    name: str
    status: str
    group_indices: tuple[int, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class TensorContract:
    """A named static tensor at a saved program boundary."""

    name: str
    spec: TensorSpec

    def __init__(self, name: str, shape: object, dtype: object):
        if not isinstance(name, str) or not name:
            raise ValueError("tensor contract name must be a non-empty string")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "spec", TensorSpec(shape, dtype))

    @property
    def shape(self) -> Shape:
        return self.spec.shape

    @property
    def dtype(self) -> DType:
        return self.spec.dtype


def _contract_json(contract: TensorContract) -> dict[str, object]:
    return {"name": contract.name, "shape": list(contract.shape), "dtype": contract.dtype}


def _write_zip_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


@dataclass(frozen=True, slots=True)
class Program:
    graph_blob: bytes
    graph: Graph | None
    serialized_ir: SerializedIR | None
    ir_provenance: IrCompileResult | None
    patch_reports: tuple[bytes, ...]
    _libraries: NativeLibraries = field(repr=False, compare=False)
    _timeout_ms: int = field(repr=False, compare=False)
    _shared_state: _ProgramSharedState = field(default_factory=_ProgramSharedState, repr=False, compare=False)
    _saved_inputs: tuple[TensorContract, ...] = field(default=(), repr=False, compare=False)
    _saved_outputs: tuple[TensorContract, ...] = field(default=(), repr=False, compare=False)
    artifact_manifest: bytes | None = field(default=None, repr=False, compare=False)

    @property
    def input_contracts(self) -> tuple[TensorContract, ...]:
        if self._saved_inputs:
            return self._saved_inputs
        if self.graph is None:
            raise RuntimeError("program has no input contract")
        values: list[TensorContract] = []
        for tensor in self.graph.inputs:
            if tensor.name is None:
                raise ValueError("all graph inputs must have names")
            values.append(TensorContract(tensor.name, tensor.shape, tensor.dtype))
        return tuple(values)

    @property
    def output_contracts(self) -> tuple[TensorContract, ...]:
        if self._saved_outputs:
            return self._saved_outputs
        if self.graph is None:
            raise RuntimeError("program has no output contract")
        return tuple(
            TensorContract(tensor.name or f"Result_{index}", tensor.shape, tensor.dtype)
            for index, tensor in enumerate(self.graph.outputs)
        )

    def to_bytes(self) -> bytes:
        """Return the complete native graph blob."""

        return self.graph_blob

    def save(self, destination: str | Path) -> None:
        """Write the complete native graph blob to *destination*."""

        Path(destination).write_bytes(self.graph_blob)

    def export(self, destination: str | Path) -> None:
        """Write a self-describing, redistributable npunlock program bundle."""

        input_contracts = self.input_contracts
        output_contracts = self.output_contracts
        patch_entries = tuple(
            {
                "path": f"patch-reports/{index:03d}.json",
                "size": len(report),
                "sha256": hashlib.sha256(report).hexdigest(),
            }
            for index, report in enumerate(self.patch_reports)
        )
        provenance = None
        if self.ir_provenance is not None:
            provenance = {
                "driver_index": self.ir_provenance.driver_index,
                "device_index": self.ir_provenance.device_index,
                "driver_version": self.ir_provenance.driver_version,
                "vendor_id": self.ir_provenance.vendor_id,
                "device_id": self.ir_provenance.device_id,
                "graph_extension_version": self.ir_provenance.graph_extension_version,
                "compiler_version": list(self.ir_provenance.compiler_version),
            }
        manifest = {
            "schema": "npunlock.program.v1",
            "graph": {
                "path": "graph.blob",
                "size": len(self.graph_blob),
                "sha256": hashlib.sha256(self.graph_blob).hexdigest(),
            },
            "inputs": [_contract_json(contract) for contract in input_contracts],
            "outputs": [_contract_json(contract) for contract in output_contracts],
            "provenance": provenance,
            "patch_reports": list(patch_entries),
        }
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as archive:
            _write_zip_member(archive, "manifest.json", manifest_bytes)
            _write_zip_member(archive, "graph.blob", self.graph_blob)
            for entry, report in zip(patch_entries, self.patch_reports):
                _write_zip_member(archive, entry["path"], report)
        Path(destination).write_bytes(archive_bytes.getvalue())

    def shared_array(self, shape: object, dtype: object) -> SharedArray:
        """Allocate a NumPy-compatible host/NPU shared tensor buffer."""

        spec = TensorSpec(shape, dtype)
        numpy_dtype = {"f16": np.dtype("float16"), "f32": np.dtype("float32")}.get(spec.dtype)
        if numpy_dtype is None:
            raise ValueError("shared arrays currently support only f16 and f32")
        size = prod(spec.shape) * numpy_dtype.itemsize
        session = self._shared_state.session
        if session is None:
            session = self._libraries.create_inference_session(self.graph_blob, timeout_ms=self._timeout_ms)
            self._shared_state.session = session
        return SharedArray(session.create_buffer(size), spec.shape, numpy_dtype)

    @staticmethod
    def _validate_shared_array(
        value: object,
        tensor: TensorContract,
        name: str,
        session: InferenceSession,
    ) -> SharedArray:
        expected_dtype = {"f16": np.dtype("float16"), "f32": np.dtype("float32")}.get(tensor.dtype)
        if not isinstance(value, SharedArray) or value._allocation is None:
            raise TypeError(f"shared tensor {name!r} must be a live SharedArray")
        if (
            value._allocation._session is not session
            or tuple(value.shape) != tensor.shape
            or value.dtype != expected_dtype
            or not value.flags.c_contiguous
            or value.ctypes.data != value._allocation.address
            or value.nbytes != value._allocation.size
        ):
            raise ValueError(f"shared tensor {name!r} has the wrong session, shape, dtype, or memory span")
        return value

    def run(
        self,
        inputs: Mapping[str, object],
        *,
        outputs: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        numpy_dtypes = {
            "f16": np.dtype("float16"),
            "f32": np.dtype("float32"),
        }
        if not isinstance(inputs, Mapping):
            raise TypeError("Program.run() inputs must be a mapping")
        input_contracts = self.input_contracts
        output_contracts = self.output_contracts
        expected_names = tuple(value.name for value in input_contracts)
        if set(inputs) != set(expected_names):
            raise ValueError(f"input names must exactly match {list(expected_names)!r}; got {list(inputs)!r}")
        if outputs is not None:
            if not isinstance(outputs, Mapping):
                raise TypeError("Program.run() outputs must be a mapping")
            expected_outputs = tuple(tensor.name for tensor in output_contracts)
            if set(outputs) != set(expected_outputs):
                raise ValueError(
                    f"output names must exactly match {list(expected_outputs)!r}; " f"got {list(outputs)!r}"
                )
            session = self._shared_state.session
            if session is None:
                raise ValueError("allocate shared tensors with this Program before shared inference")
            shared_inputs = tuple(
                (
                    name,
                    self._validate_shared_array(inputs[name], tensor, name, session)._allocation,
                )
                for tensor, name in zip(input_contracts, expected_names)
            )
            shared_outputs = tuple(
                (
                    name,
                    self._validate_shared_array(outputs[name], tensor, name, session)._allocation,
                )
                for tensor, name in zip(output_contracts, expected_outputs)
            )
            session.infer(shared_inputs, shared_outputs)  # type: ignore[arg-type]
            return dict(outputs)
        native_inputs: list[InferenceInput] = []
        for tensor, name in zip(input_contracts, expected_names):
            expected_dtype = numpy_dtypes.get(tensor.dtype)
            if expected_dtype is None:
                raise ValueError("graphinfer currently supports only static FP16/FP32 tensors")
            array = np.asarray(inputs[name])
            if array.dtype != expected_dtype or tuple(array.shape) != tensor.shape:
                raise ValueError(f"input {name!r} requires shape {tensor.shape!r} and dtype {expected_dtype}")
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            native_inputs.append(InferenceInput(name, array.tobytes(order="C")))
        inferred = self._libraries.infer_graph(
            self.graph_blob,
            native_inputs,
            timeout_ms=self._timeout_ms,
        )
        if len(inferred.outputs) != len(output_contracts):
            raise RuntimeError(f"graph returned {len(inferred.outputs)} outputs; expected {len(output_contracts)}")
        values: dict[str, object] = {}
        for tensor, output in zip(output_contracts, inferred.outputs):
            name = tensor.name
            expected_dtype = numpy_dtypes.get(tensor.dtype)
            if expected_dtype is None:
                raise RuntimeError(f"output {name!r} uses unsupported symbolic dtype {tensor.dtype!r}")
            if output.dtype != tensor.dtype or output.shape != tensor.shape:
                raise RuntimeError(f"output {name!r} returned shape {output.shape!r} and dtype {output.dtype}")
            expected_size = int(np.prod(tensor.shape, dtype=np.int64)) * expected_dtype.itemsize
            if len(output.data) != expected_size:
                raise RuntimeError(f"output {name!r} returned {len(output.data)} bytes; expected {expected_size}")
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
    )


def load_native_file(
    source: str | Path,
    *,
    graph: Graph,
    native_dir: str | Path | None = None,
    timeout_ms: int = 20_000,
    libraries: NativeLibraries | None = None,
) -> Program:
    """Create an executable program from a native graph file."""

    return load_native(
        Path(source).read_bytes(),
        graph=graph,
        native_dir=native_dir,
        timeout_ms=timeout_ms,
        libraries=libraries,
    )


def _bundle_member(
    archive: zipfile.ZipFile,
    name: str,
    *,
    maximum_size: int,
) -> bytes:
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise ValueError(f"program bundle is missing {name!r}") from exc
    if info.flag_bits & 1:
        raise ValueError(f"program bundle member {name!r} must not be encrypted")
    if info.file_size > maximum_size:
        raise ValueError(f"program bundle member {name!r} exceeds the supported size")
    data = archive.read(info)
    if len(data) != info.file_size:
        raise ValueError(f"program bundle member {name!r} is truncated")
    return data


def _manifest_contracts(value: object, label: str) -> tuple[TensorContract, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"program bundle {label} must be a non-empty list")
    contracts: list[TensorContract] = []
    names: set[str] = set()
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"name", "shape", "dtype"}:
            raise ValueError(f"program bundle {label} contains an invalid tensor contract")
        name = entry["name"]
        shape = entry["shape"]
        dtype = entry["dtype"]
        if not isinstance(name, str) or name in names:
            raise ValueError(f"program bundle {label} contains an empty or duplicate tensor name")
        contract = TensorContract(name, shape, dtype)
        if contract.dtype not in {"f16", "f32"} or len(contract.shape) > 5:
            raise ValueError(f"program bundle tensor {name!r} is outside the graphinfer contract")
        names.add(name)
        contracts.append(contract)
    return tuple(contracts)


def _manifest_blob_entry(value: object, label: str) -> tuple[str, int, str]:
    if not isinstance(value, dict) or set(value) != {"path", "size", "sha256"}:
        raise ValueError(f"program bundle {label} entry is invalid")
    path = value["path"]
    size = value["size"]
    digest = value["sha256"]
    if not isinstance(path, str) or not path or path.startswith(("/", "\\")) or ".." in Path(path).parts:
        raise ValueError(f"program bundle {label} path is invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"program bundle {label} size is invalid")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"program bundle {label} hash is invalid")
    return path, size, digest


def load(
    source: str | Path,
    *,
    native_dir: str | Path | None = None,
    timeout_ms: int = 20_000,
    libraries: NativeLibraries | None = None,
) -> Program:
    """Load a self-describing program bundle without recompilation or MoviTools."""

    source_path = Path(source)
    try:
        with zipfile.ZipFile(source_path, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("program bundle contains duplicate member names")
            manifest_bytes = _bundle_member(archive, "manifest.json", maximum_size=1024 * 1024)
            try:
                manifest = json.loads(manifest_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("program bundle manifest is not valid UTF-8 JSON") from exc
            if not isinstance(manifest, dict) or manifest.get("schema") != "npunlock.program.v1":
                raise ValueError("program bundle schema is unsupported")
            required = {"schema", "graph", "inputs", "outputs", "provenance", "patch_reports"}
            if set(manifest) != required:
                raise ValueError("program bundle manifest fields are invalid")
            if manifest["provenance"] is not None and not isinstance(manifest["provenance"], dict):
                raise ValueError("program bundle provenance is invalid")
            graph_path, graph_size, graph_hash = _manifest_blob_entry(manifest["graph"], "graph")
            if graph_path != "graph.blob" or graph_size == 0:
                raise ValueError("program bundle graph entry is invalid")
            graph_blob = _bundle_member(archive, graph_path, maximum_size=2 * 1024 * 1024 * 1024)
            if len(graph_blob) != graph_size or hashlib.sha256(graph_blob).hexdigest() != graph_hash:
                raise ValueError("program bundle graph size or SHA-256 does not match its manifest")

            inputs = _manifest_contracts(manifest["inputs"], "inputs")
            outputs = _manifest_contracts(manifest["outputs"], "outputs")
            report_values = manifest["patch_reports"]
            if not isinstance(report_values, list) or len(report_values) > 1024:
                raise ValueError("program bundle patch_reports is invalid")
            reports: list[bytes] = []
            report_paths: list[str] = []
            total_report_size = 0
            for index, entry in enumerate(report_values):
                report_path, report_size, report_hash = _manifest_blob_entry(entry, f"patch report {index}")
                if report_path != f"patch-reports/{index:03d}.json" or report_size > 16 * 1024 * 1024:
                    raise ValueError(f"program bundle patch report {index} entry is invalid")
                total_report_size += report_size
                if total_report_size > 64 * 1024 * 1024:
                    raise ValueError("program bundle patch reports exceed the supported total size")
                report = _bundle_member(archive, report_path, maximum_size=16 * 1024 * 1024)
                if len(report) != report_size or hashlib.sha256(report).hexdigest() != report_hash:
                    raise ValueError(f"program bundle patch report {index} does not match its manifest")
                report_paths.append(report_path)
                reports.append(report)
            expected_names = {"manifest.json", graph_path, *report_paths}
            if set(names) != expected_names:
                raise ValueError("program bundle contains undeclared members")
    except zipfile.BadZipFile as exc:
        raise ValueError("program bundle is not a valid ZIP archive") from exc

    native = libraries or NativeLibraries(native_dir)
    return Program(
        graph_blob,
        None,
        None,
        None,
        tuple(reports),
        native,
        timeout_ms,
        _saved_inputs=inputs,
        _saved_outputs=outputs,
        artifact_manifest=manifest_bytes,
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
    if len({(target.invocation_index, target.range_index) for target in targets}) != len(targets):
        return None
    if sum(target.element_count for target in targets) != prod(output.shape):
        return None
    return targets


def _automatic_group_indices(
    graph: Graph,
    custom_nodes: tuple[Node, ...],
    discovered_groups: tuple[tuple[PatchTarget, ...], ...],
) -> tuple[tuple[tuple[int, ...], ...] | None, str]:
    computational_nodes = tuple(node for node in graph.nodes if node.op not in {"Parameter", "Const"})

    # Keep the established positional contract unchanged when it applies.
    if len(discovered_groups) == len(computational_nodes):
        node_groups = {node: index for index, node in enumerate(computational_nodes)}
        selected = tuple((node_groups[node],) for node in custom_nodes)
        for node, indices in zip(custom_nodes, selected):
            targets = discovered_groups[indices[0]]
            if any(target.input_count != len(node.inputs) for target in targets):
                return (
                    None,
                    f"discovered ACT group {indices[0]} for custom node "
                    f"{node.name or '<unnamed>'!r} does not match its input arity",
                )
        return selected, "one-to-one positional mapping"

    solutions: list[dict[Node, tuple[int, ...]]] = []

    def visit(
        node_index: int,
        group_index: int,
        selected: dict[Node, tuple[int, ...]],
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
                if node in custom_nodes and any(target.input_count != len(node.inputs) for target in targets):
                    continue
            else:
                targets = _combined_custom_targets(node, groups)
                if targets is None:
                    continue
            if node in custom_nodes:
                selected[node] = tuple(range(group_index, group_index + take))
            visit(node_index + 1, group_index + take, selected)
            selected.pop(node, None)

    visit(0, 0, {})
    if len(solutions) != 1:
        reason = "no" if not solutions else "multiple"
        return None, (
            f"native graph has {len(discovered_groups)} positional ACT groups for "
            f"{len(computational_nodes)} computational nodes and {reason} valid mapping; "
            "automatic selection requires a one-to-one positional mapping or one unique "
            "exact-cover partition mapping"
        )
    return tuple(solutions[0][node] for node in custom_nodes), "unique exact-cover partition mapping"


def _embedded_target_groups(custom_nodes: tuple[Node, ...]) -> tuple[tuple[PatchTarget, ...], ...] | None:
    if not custom_nodes:
        return None
    supplied = tuple(node.metadata.get("_patch_targets") for node in custom_nodes)
    if not any(value is not None for value in supplied):
        return None
    if not all(
        isinstance(value, tuple) and value and all(isinstance(target, PatchTarget) for target in value)
        for value in supplied
    ):
        raise ValueError(
            "either omit _patch_targets for every custom node or provide a non-empty "
            "PatchTarget sequence for every custom node"
        )
    return supplied  # type: ignore[return-value]


def _effective_build_flags(serialized: SerializedIR, build_flags: str) -> str:
    preserves_fp32_custom = any(
        any(output.dtype == "f32" for output in node.outputs) for node in serialized.custom_nodes
    )
    if not preserves_fp32_custom:
        return build_flags
    accuracy_flag = 'EXECUTION_MODE_HINT="ACCURACY"'
    if not build_flags:
        return f"--config {accuracy_flag}"
    if accuracy_flag not in build_flags:
        raise ValueError(
            "FP32 custom kernels require build_flags containing "
            'EXECUTION_MODE_HINT="ACCURACY" to prevent FP16 lowering'
        )
    return build_flags


@dataclass(frozen=True, slots=True)
class PreparedGraph:
    """A carrier compiled once and ready for custom-kernel target binding."""

    graph: Graph
    serialized_ir: SerializedIR
    ir_provenance: IrCompileResult
    groups: tuple[ActGroup, ...]
    mappings: tuple[CustomMapping, ...]
    _libraries: NativeLibraries = field(repr=False, compare=False)
    _timeout_ms: int = field(repr=False, compare=False)
    _automatic_indices: tuple[tuple[int, ...], ...] | None = field(repr=False, compare=False)
    _embedded_targets: tuple[tuple[PatchTarget, ...], ...] | None = field(repr=False, compare=False)

    def find_groups(
        self,
        *,
        input_count: int | None = None,
        dtype: str | None = None,
        element_count: int | None = None,
        input_1_scalar: bool | None = None,
    ) -> tuple[ActGroup, ...]:
        """Return groups matching caller-specified validated contract fields."""

        if input_count is not None and (
            isinstance(input_count, bool) or not isinstance(input_count, int) or input_count <= 0
        ):
            raise ValueError("input_count must be a positive integer or None")
        if dtype is not None and dtype not in {"f16", "f32"}:
            raise ValueError("dtype must be 'f16', 'f32', or None")
        if element_count is not None and (
            isinstance(element_count, bool) or not isinstance(element_count, int) or element_count <= 0
        ):
            raise ValueError("element_count must be a positive integer or None")
        if input_1_scalar is not None and not isinstance(input_1_scalar, bool):
            raise TypeError("input_1_scalar must be bool or None")
        return tuple(
            group
            for group in self.groups
            if (input_count is None or group.input_count == input_count)
            and (dtype is None or group.dtype == dtype)
            and (element_count is None or group.element_count == element_count)
            and (input_1_scalar is None or group.input_1_scalar == input_1_scalar)
        )

    def explain(self) -> str:
        """Render the discovered groups and mapping decisions deterministically."""

        lines = [
            f"graph {self.graph.name!r}: {len(self.groups)} ACT groups, "
            f"{len(self.serialized_ir.custom_nodes)} custom nodes",
            "ACT groups:",
        ]
        if not self.groups:
            lines.append("  (none discovered or raw explicit targets are in use)")
        for group in self.groups:
            flags = "mixed" if group.contract_flags is None else f"0x{group.contract_flags:08x}"
            dtype = group.dtype or "unknown"
            lines.append(
                f"  [{group.index}] inputs={group.input_count} dtype={dtype} "
                f"elements={group.element_count} span={group.span_bytes} "
                f"targets={len(group.targets)} flags={flags}"
            )
        lines.append("Custom mappings:")
        if not self.mappings:
            lines.append("  (no custom nodes)")
        for mapping in self.mappings:
            groups = ",".join(str(index) for index in mapping.group_indices) or "none"
            lines.append(f"  {mapping.name}: {mapping.status}; groups={groups}; {mapping.reason}")
        return "\n".join(lines)

    def _normalize_binding(self, value: object) -> tuple[int, ...]:
        values: tuple[object, ...]
        if isinstance(value, (int, ActGroup)) and not isinstance(value, bool):
            values = (value,)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            values = tuple(value)
        else:
            raise TypeError("a prepared binding must be an ActGroup, group index, or sequence of them")
        if not values:
            raise ValueError("a prepared binding must select at least one ACT group")
        indices: list[int] = []
        for item in values:
            if isinstance(item, bool):
                raise TypeError("ACT group indices must be integers")
            if isinstance(item, ActGroup):
                index = item.index
                if not 0 <= index < len(self.groups) or self.groups[index] != item:
                    raise ValueError("ActGroup does not belong to this PreparedGraph")
            elif isinstance(item, int):
                index = item
                if not 0 <= index < len(self.groups):
                    raise ValueError(f"ACT group index {index} is out of range")
            else:
                raise TypeError("prepared binding sequences may contain only ActGroup or int values")
            if index in indices:
                raise ValueError(f"ACT group {index} is selected more than once for one custom node")
            indices.append(index)
        return tuple(indices)

    def build(
        self,
        *,
        bindings: Mapping[Tensor, object] | None = None,
        movi_dll_dir: str | Path | None = None,
        linker_script: str | Path | bytes | None = None,
        definitions: tuple[str, ...] = (),
        movi_worker: str | None = None,
    ) -> Program:
        """Compile and patch custom kernels into this prepared carrier."""

        custom_nodes = self.serialized_ir.custom_nodes
        resolved_movi_dll_dir = _resolve_movi_dll_dir(movi_dll_dir)
        if custom_nodes and resolved_movi_dll_dir is None:
            raise ValueError(
                "custom kernels require an MVC_DEPEND root from movi_dll_dir, " "configure(), or NPUNLOCK_MOVITOOLS_DIR"
            )
        if bindings is not None and not isinstance(bindings, Mapping):
            raise TypeError("bindings must be a mapping from custom output tensors to ACT groups")
        supplied_bindings = {} if bindings is None else dict(bindings)
        if self._embedded_targets is not None and supplied_bindings:
            raise ValueError("prepared bindings cannot be combined with embedded _patch_targets")

        target_groups: tuple[tuple[PatchTarget, ...], ...]
        if self._embedded_targets is not None:
            target_groups = self._embedded_targets
        else:
            bound_indices: dict[Node, tuple[int, ...]] = {}
            for output, value in supplied_bindings.items():
                if not isinstance(output, Tensor) or output.producer not in custom_nodes:
                    raise ValueError("prepared binding keys must be outputs of custom nodes in this graph")
                node = output.producer
                if node in bound_indices:
                    raise ValueError(f"custom node {node.name or '<unnamed>'!r} is bound more than once")
                if len(node.outputs) != 1:
                    raise ValueError("prepared group bindings currently require one output per custom node")
                bound_indices[node] = self._normalize_binding(value)

            automatic = dict(zip(custom_nodes, self._automatic_indices)) if self._automatic_indices is not None else {}
            missing = [node for node in custom_nodes if node not in bound_indices and node not in automatic]
            if missing:
                names = ", ".join(repr(node.name or "<unnamed>") for node in missing)
                reason = self.mappings[0].reason if self.mappings else "no automatic mapping"
                raise ValueError(
                    f"prepared graph needs explicit bindings for custom nodes {names}: {reason}; "
                    "pass bindings={custom_output: prepared_group}"
                )

            used: dict[int, Node] = {}
            selected_targets: list[tuple[PatchTarget, ...]] = []
            for node in custom_nodes:
                indices = bound_indices[node] if node in bound_indices else automatic[node]
                for index in indices:
                    previous = used.get(index)
                    if previous is not None:
                        raise ValueError(
                            f"ACT group {index} is assigned to both "
                            f"{previous.name or '<unnamed>'!r} and {node.name or '<unnamed>'!r}"
                        )
                    used[index] = node
                groups = tuple(self.groups[index].targets for index in indices)
                if len(groups) == 1:
                    targets = groups[0]
                    if any(target.input_count != len(node.inputs) for target in targets):
                        raise ValueError(
                            f"ACT group {indices[0]} does not match custom node "
                            f"{node.name or '<unnamed>'!r} input arity"
                        )
                else:
                    combined = _combined_custom_targets(node, groups)
                    if combined is None:
                        raise ValueError(
                            f"ACT groups {indices!r} do not form a compatible exact cover for "
                            f"custom node {node.name or '<unnamed>'!r}"
                        )
                    targets = combined
                selected_targets.append(targets)
            target_groups = tuple(selected_targets)

        blob = self.ir_provenance.graph_blob
        reports: list[bytes] = []
        if custom_nodes:
            assert resolved_movi_dll_dir is not None
            script = (
                linker_script
                if isinstance(linker_script, bytes) or linker_script is None
                else Path(linker_script).read_bytes()
            )
            for node, target_values in zip(custom_nodes, target_groups):
                elf = self._libraries.compile_shave(
                    _kernel_source(node.metadata.get("_kernel")),
                    movi_dll_dir=resolved_movi_dll_dir,
                    linker_script=script,
                    definitions=definitions,
                    timeout_ms=self._timeout_ms,
                    worker=movi_worker,
                )
                patched = self._libraries.patch_graph(blob, elf, target_values)
                blob = patched.graph_blob
                reports.append(patched.report_json)
        return Program(
            blob,
            self.graph,
            self.serialized_ir,
            self.ir_provenance,
            tuple(reports),
            self._libraries,
            self._timeout_ms,
        )


def prepare(
    graph: Graph,
    *,
    native_dir: str | Path | None = None,
    build_flags: str = "",
    timeout_ms: int = 20_000,
    ir_worker: str | None = None,
    libraries: NativeLibraries | None = None,
) -> PreparedGraph:
    """Compile a carrier graph once and expose its validated ACT groups."""

    if not isinstance(graph, Graph):
        raise TypeError("prepare() requires a Graph")
    serialized = serialize_ir(graph)
    build_flags = _effective_build_flags(serialized, build_flags)
    embedded_targets = _embedded_target_groups(serialized.custom_nodes)
    if embedded_targets is None and any(len(node.outputs) != 1 for node in serialized.custom_nodes):
        raise ValueError("automatic and prepared patch selection currently require one output per custom node")
    native = libraries or NativeLibraries(native_dir)
    ir_result = native.compile_ir(
        serialized.xml,
        serialized.weights,
        build_flags=build_flags,
        timeout_ms=timeout_ms,
        worker=ir_worker,
    )

    groups: tuple[ActGroup, ...] = ()
    automatic_indices: tuple[tuple[int, ...], ...] | None = None
    mappings: tuple[CustomMapping, ...] = ()
    if serialized.custom_nodes:
        if embedded_targets is not None:
            mappings = tuple(
                CustomMapping(
                    node.outputs[0],
                    node.name or "<unnamed>",
                    "raw-explicit",
                    (),
                    "using caller-supplied _patch_targets; positional discovery was skipped",
                )
                for node in serialized.custom_nodes
            )
        else:
            discovered = native.discover_patch_targets(ir_result.graph_blob)
            groups = tuple(ActGroup(index, targets) for index, targets in enumerate(discovered))
            automatic_indices, reason = _automatic_group_indices(graph, serialized.custom_nodes, discovered)
            mappings = tuple(
                CustomMapping(
                    node.outputs[0],
                    node.name or "<unnamed>",
                    "automatic" if automatic_indices is not None else "explicit-required",
                    automatic_indices[index] if automatic_indices is not None else (),
                    reason,
                )
                for index, node in enumerate(serialized.custom_nodes)
            )
    return PreparedGraph(
        graph,
        serialized,
        ir_result,
        groups,
        mappings,
        native,
        timeout_ms,
        automatic_indices,
        embedded_targets,
    )


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
    prepared = prepare(
        graph,
        native_dir=native_dir,
        build_flags=build_flags,
        timeout_ms=timeout_ms,
        ir_worker=ir_worker,
        libraries=libraries,
    )
    return prepared.build(
        movi_dll_dir=movi_dll_dir,
        linker_script=linker_script,
        definitions=definitions,
        movi_worker=movi_worker,
    )
