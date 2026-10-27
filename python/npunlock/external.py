from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ._native import PatchTargetV2, PatchTensorContract
from .contracts import NpuProvenance
from .tensor import TensorSpec


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _tensor_contracts(value: object, label: str) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"reviewed binding {label} must be a non-empty list")
    result: list[tuple[str, tuple[int, ...], str]] = []
    names: set[str] = set()
    for item in value:
        if not isinstance(item, dict) or set(item) != {"name", "shape", "dtype"}:
            raise ValueError(f"reviewed binding {label} contains an invalid tensor contract")
        name = item["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"reviewed binding {label} tensor names must be non-empty and unique")
        spec = TensorSpec(item["shape"], item["dtype"])
        result.append((name, spec.shape, spec.dtype))
        names.add(name)
    return tuple(result)


def _targets(value: object) -> tuple[PatchTargetV2, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("reviewed binding targets must be a non-empty list")
    result: list[PatchTargetV2] = []
    for item in value:
        if not isinstance(item, dict) or set(item) not in (
            {"invocation_index", "range_index", "target_flags", "tensors"},
            {"invocation_index", "range_index", "tensors"},
        ):
            raise ValueError("reviewed binding target fields are invalid")
        tensors_value = item["tensors"]
        if not isinstance(tensors_value, list) or not tensors_value:
            raise ValueError("reviewed binding target tensors must be a non-empty list")
        tensors: list[PatchTensorContract] = []
        for tensor in tensors_value:
            if not isinstance(tensor, dict) or set(tensor) != {
                "role",
                "index",
                "dtype",
                "element_count",
                "span_bytes",
                "observed_flags",
            }:
                raise ValueError("reviewed binding tensor fields are invalid")
            tensors.append(
                PatchTensorContract(
                    tensor["role"],
                    tensor["index"],
                    tensor["dtype"],
                    tensor["element_count"],
                    tensor["span_bytes"],
                    tensor["observed_flags"],
                )
            )
        result.append(
            PatchTargetV2(
                item["invocation_index"],
                item["range_index"],
                tuple(tensors),
                item.get("target_flags", 0),
            )
        )
    if len({(target.invocation_index, target.range_index) for target in result}) != len(result):
        raise ValueError("reviewed binding contains duplicate invocation/range selectors")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ReviewedPatchBinding:
    """Exact, provenance-bound patch contract for caller-produced serialized IR."""

    xml_sha256: str
    weights_sha256: str
    build_flags: str
    native_graph_sha256: str
    provenance: NpuProvenance
    targets: tuple[PatchTargetV2, ...]
    inputs: tuple[tuple[str, tuple[int, ...], str], ...]
    outputs: tuple[tuple[str, tuple[int, ...], str], ...]
    metadata: Mapping[str, object]

    @classmethod
    def from_bytes(cls, payload: bytes) -> ReviewedPatchBinding:
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("reviewed binding is not valid UTF-8 JSON") from exc
        if not isinstance(value, dict) or value.get("schema") != "npunlock.reviewed-patch-binding.v1":
            raise ValueError("reviewed patch binding schema is unsupported")
        if set(value) != {
            "schema",
            "metadata",
            "serialized_ir",
            "build_flags",
            "native_graph_sha256",
            "provenance",
            "inputs",
            "outputs",
            "targets",
        }:
            raise ValueError("reviewed patch binding fields are invalid")
        serialized = value["serialized_ir"]
        metadata = value["metadata"]
        build_flags = value["build_flags"]
        if not isinstance(serialized, dict) or set(serialized) != {"xml_sha256", "weights_sha256"}:
            raise ValueError("reviewed binding serialized_ir identity is invalid")
        if not isinstance(metadata, dict):
            raise ValueError("reviewed binding metadata must be an object")
        if not isinstance(build_flags, str) or "\x00" in build_flags:
            raise ValueError("reviewed binding build_flags must be a NUL-free string")
        return cls(
            _digest(serialized["xml_sha256"], "IR XML sha256"),
            _digest(serialized["weights_sha256"], "IR weights sha256"),
            build_flags,
            _digest(value["native_graph_sha256"], "native graph sha256"),
            NpuProvenance.from_json(value["provenance"]),
            _targets(value["targets"]),
            _tensor_contracts(value["inputs"], "inputs"),
            _tensor_contracts(value["outputs"], "outputs"),
            metadata,
        )

    @classmethod
    def load(cls, source: str | Path) -> ReviewedPatchBinding:
        return cls.from_bytes(Path(source).read_bytes())

    def validate_ir(self, xml: bytes, weights: bytes) -> None:
        if hashlib.sha256(xml).hexdigest() != self.xml_sha256:
            raise ValueError("serialized IR XML does not match the reviewed binding")
        if hashlib.sha256(weights).hexdigest() != self.weights_sha256:
            raise ValueError("serialized IR weights do not match the reviewed binding")
