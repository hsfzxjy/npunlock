from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

from .tensor import DType, Shape, TensorSpec

if TYPE_CHECKING:
    from . import ActGroup, PreparedGraph, Program, TensorContract
    from ._native import IrCompileResult


def _uint(value: object, label: str, maximum: int = 0xFFFFFFFF) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"{label} must be an unsigned integer")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class NpuProvenance:
    """Stable driver/compiler identity independent of a graph payload."""

    driver_index: int
    device_index: int
    driver_version: int
    vendor_id: int
    device_id: int
    graph_extension_version: int
    compiler_version: tuple[int, int]

    def __post_init__(self) -> None:
        _uint(self.driver_index, "driver_index")
        _uint(self.device_index, "device_index")
        _uint(self.driver_version, "driver_version")
        _uint(self.vendor_id, "vendor_id")
        _uint(self.device_id, "device_id")
        _uint(self.graph_extension_version, "graph_extension_version")
        if (
            not isinstance(self.compiler_version, tuple)
            or len(self.compiler_version) != 2
            or any(_uint(value, "compiler_version", 0xFFFF) != value for value in self.compiler_version)
        ):
            raise ValueError("compiler_version must contain two unsigned 16-bit integers")

    @classmethod
    def from_ir_result(cls, value: IrCompileResult) -> NpuProvenance:
        return cls(
            value.driver_index,
            value.device_index,
            value.driver_version,
            value.vendor_id,
            value.device_id,
            value.graph_extension_version,
            value.compiler_version,
        )

    @classmethod
    def from_json(cls, value: object) -> NpuProvenance:
        if not isinstance(value, Mapping):
            raise ValueError("provenance must be an object")
        required = {
            "driver_index",
            "device_index",
            "driver_version",
            "vendor_id",
            "device_id",
            "graph_extension_version",
            "compiler_version",
        }
        if set(value) != required:
            raise ValueError("provenance fields are invalid")
        compiler = value["compiler_version"]
        if not isinstance(compiler, list) or len(compiler) != 2:
            raise ValueError("compiler_version must contain two integers")
        return cls(
            _uint(value["driver_index"], "driver_index"),
            _uint(value["device_index"], "device_index"),
            _uint(value["driver_version"], "driver_version"),
            _uint(value["vendor_id"], "vendor_id"),
            _uint(value["device_id"], "device_id"),
            _uint(value["graph_extension_version"], "graph_extension_version"),
            (
                _uint(compiler[0], "compiler_version", 0xFFFF),
                _uint(compiler[1], "compiler_version", 0xFFFF),
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            "driver_index": self.driver_index,
            "device_index": self.device_index,
            "driver_version": self.driver_version,
            "vendor_id": self.vendor_id,
            "device_id": self.device_id,
            "graph_extension_version": self.graph_extension_version,
            "compiler_version": list(self.compiler_version),
        }


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    size: int
    sha256: str

    def __post_init__(self) -> None:
        _uint(self.size, "artifact size", (1 << 63) - 1)
        _sha256(self.sha256, "artifact sha256")

    @classmethod
    def from_bytes(cls, value: bytes) -> ArtifactIdentity:
        return cls(len(value), hashlib.sha256(value).hexdigest())

    def to_json(self) -> dict[str, object]:
        return {"size": self.size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class PatchReport:
    """Validated, typed view of one patchblob preservation report."""

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
    def parse(cls, payload: bytes) -> PatchReport:
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("patch report is not valid UTF-8 JSON") from exc
        schema = value.get("schema") if isinstance(value, dict) else None
        if not isinstance(schema, str) or schema not in (
            "npunlock.patchblob.v1",
            "npunlock.patchblob.v2",
        ):
            raise ValueError("patch report schema is unsupported")
        graph_contract = value.get("graph_contract")
        targets = value.get("targets")
        append = value.get("append")
        validation = value.get("validation")
        if (
            not isinstance(graph_contract, dict)
            or not isinstance(targets, list)
            or not all(isinstance(target, dict) for target in targets)
            or not isinstance(append, dict)
            or not isinstance(validation, dict)
        ):
            raise ValueError("patch report structure is invalid")
        compiler = graph_contract.get("compiler")
        npu = graph_contract.get("npu")
        if not isinstance(compiler, str) or not isinstance(npu, str):
            raise ValueError("patch report graph contract is invalid")
        if (
            validation.get("section_contents_preserved") is not True
            or validation.get("mutation_set_preserved") is not True
        ):
            raise ValueError("patch report does not prove preservation")
        return cls(
            bytes(payload),
            schema,
            _sha256(value.get("input_graph_sha256"), "input graph sha256"),
            _sha256(value.get("output_graph_sha256"), "output graph sha256"),
            _sha256(value.get("shave_elf_sha256"), "SHAVE ELF sha256"),
            compiler,
            npu,
            graph_contract,
            tuple(targets),
            append,
            validation,
        )

    def to_json(self) -> dict[str, object]:
        """Return the complete validated report as JSON-compatible values."""

        value = json.loads(self.raw.decode("utf-8"))
        if not isinstance(value, dict):  # The constructor is intentionally public.
            raise ValueError("patch report structure is invalid")
        return value

    def summary(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "input_graph_sha256": self.input_graph_sha256,
            "shave_elf_sha256": self.shave_elf_sha256,
            "output_graph_sha256": self.output_graph_sha256,
            "graph_contract": dict(self.graph_contract),
            "append": dict(self.append),
            "validation": dict(self.validation),
            "patched_target_count": len(self.targets),
        }


@dataclass(frozen=True, slots=True)
class ActGroupContract:
    """Expected stable invocation-local contract for one positional ACT group."""

    input_dtypes: tuple[DType, ...]
    output_dtype: DType
    input_element_counts: tuple[int, ...]
    output_element_count: int
    target_count: int
    invocation_indices: tuple[int, ...] | None = None
    range_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.input_dtypes:
            raise ValueError("ActGroupContract requires at least one input")
        if len(self.input_dtypes) != len(self.input_element_counts):
            raise ValueError("input dtype and element-count lengths differ")
        for dtype in (*self.input_dtypes, self.output_dtype):
            TensorSpec((1,), dtype)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.input_element_counts
        ):
            raise ValueError("input element counts must be positive integers")
        if (
            isinstance(self.output_element_count, bool)
            or not isinstance(self.output_element_count, int)
            or self.output_element_count <= 0
        ):
            raise ValueError("output element count must be a positive integer")
        if isinstance(self.target_count, bool) or not isinstance(self.target_count, int) or self.target_count <= 0:
            raise ValueError("target count must be a positive integer")
        for label, values in (("invocation", self.invocation_indices), ("range", self.range_indices)):
            if values is not None:
                if len(values) != self.target_count or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
                ):
                    raise ValueError(f"{label} indices must contain one unsigned integer per target")

    def mismatch(self, group: ActGroup) -> str | None:
        if group.input_dtypes != self.input_dtypes:
            return f"input dtypes {group.input_dtypes!r} != {self.input_dtypes!r}"
        if group.output_dtype != self.output_dtype:
            return f"output dtype {group.output_dtype!r} != {self.output_dtype!r}"
        if group.per_invocation_input_element_counts != self.input_element_counts:
            return (
                "input element counts "
                f"{group.per_invocation_input_element_counts!r} != {self.input_element_counts!r}"
            )
        if group.per_invocation_output_element_count != self.output_element_count:
            return (
                "output element count "
                f"{group.per_invocation_output_element_count!r} != {self.output_element_count!r}"
            )
        if len(group.targets) != self.target_count:
            return f"target count {len(group.targets)} != {self.target_count}"
        if self.invocation_indices is not None and group.invocation_indices != self.invocation_indices:
            return f"invocation indices {group.invocation_indices!r} != {self.invocation_indices!r}"
        if self.range_indices is not None and group.range_indices != self.range_indices:
            return f"range indices {group.range_indices!r} != {self.range_indices!r}"
        return None

    def to_json(self) -> dict[str, object]:
        return {
            "input_dtypes": list(self.input_dtypes),
            "output_dtype": self.output_dtype,
            "input_element_counts": list(self.input_element_counts),
            "output_element_count": self.output_element_count,
            "target_count": self.target_count,
            "invocation_indices": None if self.invocation_indices is None else list(self.invocation_indices),
            "range_indices": None if self.range_indices is None else list(self.range_indices),
        }


@dataclass(frozen=True, slots=True)
class CarrierContract:
    groups: tuple[ActGroupContract, ...]
    mapping_indices: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        if not self.groups:
            raise ValueError("CarrierContract requires at least one ACT group")
        if self.mapping_indices is not None:
            for mapping in self.mapping_indices:
                if not mapping or any(
                    isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.groups)
                    for index in mapping
                ):
                    raise ValueError("carrier mapping indices are invalid")

    def validate(self, prepared: PreparedGraph) -> None:
        if len(prepared.groups) != len(self.groups):
            raise ValueError(f"carrier exposed {len(prepared.groups)} ACT groups; expected {len(self.groups)}")
        for index, (actual, expected) in enumerate(zip(prepared.groups, self.groups)):
            mismatch = expected.mismatch(actual)
            if mismatch is not None:
                raise ValueError(f"carrier ACT group {index} contract changed: {mismatch}")
        if self.mapping_indices is not None:
            actual = tuple(
                mapping.group_indices if mapping.status == "automatic" else () for mapping in prepared.mappings
            )
            if actual != self.mapping_indices:
                raise ValueError(f"carrier mapping changed: {actual!r} != {self.mapping_indices!r}")

    def to_json(self) -> dict[str, object]:
        return {
            "groups": [group.to_json() for group in self.groups],
            "mapping_indices": (
                None if self.mapping_indices is None else [list(indices) for indices in self.mapping_indices]
            ),
        }


@dataclass(frozen=True, slots=True)
class ProgramContract:
    inputs: tuple[tuple[str, Shape, DType], ...]
    outputs: tuple[tuple[str, Shape, DType], ...]
    patch_count: int
    compiler: str = "8.3"
    npu: str = "3720"
    targets_per_patch: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        for label, values in (("input", self.inputs), ("output", self.outputs)):
            names: set[str] = set()
            for name, shape, dtype in values:
                if not isinstance(name, str) or not name or name in names:
                    raise ValueError(f"{label} tensor names must be non-empty and unique")
                TensorSpec(shape, dtype)
                names.add(name)
        if isinstance(self.patch_count, bool) or not isinstance(self.patch_count, int) or self.patch_count < 0:
            raise ValueError("patch_count must be a nonnegative integer")
        if not isinstance(self.compiler, str) or not self.compiler or not isinstance(self.npu, str) or not self.npu:
            raise ValueError("compiler and NPU identities must be non-empty strings")
        if self.targets_per_patch is not None and (
            len(self.targets_per_patch) != self.patch_count
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in self.targets_per_patch
            )
        ):
            raise ValueError("targets_per_patch must contain one positive count per patch")

    @staticmethod
    def _contracts(values: Sequence[TensorContract]) -> tuple[tuple[str, Shape, DType], ...]:
        return tuple((value.name, value.shape, value.dtype) for value in values)

    def validate(self, program: Program) -> None:
        actual_inputs = self._contracts(program.input_contracts)
        actual_outputs = self._contracts(program.output_contracts)
        if actual_inputs != self.inputs:
            raise ValueError(f"program input contract changed: {actual_inputs!r} != {self.inputs!r}")
        if actual_outputs != self.outputs:
            raise ValueError(f"program output contract changed: {actual_outputs!r} != {self.outputs!r}")
        if len(program.patch_reports) != self.patch_count:
            raise ValueError(f"program has {len(program.patch_reports)} patch reports; expected {self.patch_count}")
        reports = tuple(PatchReport.parse(payload) for payload in program.patch_reports)
        for index, report in enumerate(reports):
            if report.compiler != self.compiler or report.npu != self.npu:
                raise ValueError(
                    f"patch report {index} graph contract changed: " f"compiler={report.compiler!r}, npu={report.npu!r}"
                )
            if self.targets_per_patch is not None and len(report.targets) != self.targets_per_patch[index]:
                raise ValueError(
                    f"patch report {index} has {len(report.targets)} targets; "
                    f"expected {self.targets_per_patch[index]}"
                )
        for previous, current in zip(reports, reports[1:]):
            if previous.output_graph_sha256 != current.input_graph_sha256:
                raise ValueError("program patch-report graph hash chain is broken")
        if reports and reports[-1].output_graph_sha256 != hashlib.sha256(program.graph_blob).hexdigest():
            raise ValueError("program graph hash does not match its final patch report")

    def to_json(self) -> dict[str, object]:
        return {
            "inputs": [{"name": name, "shape": list(shape), "dtype": dtype} for name, shape, dtype in self.inputs],
            "outputs": [{"name": name, "shape": list(shape), "dtype": dtype} for name, shape, dtype in self.outputs],
            "patch_count": self.patch_count,
            "compiler": self.compiler,
            "npu": self.npu,
            "targets_per_patch": (None if self.targets_per_patch is None else list(self.targets_per_patch)),
        }
