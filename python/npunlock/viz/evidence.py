"""Read-only projection of validated npunlock core evidence.

This module deliberately produces and validates plain mappings.  The core does
not import the visualizer, and the generic artifact model does not import core
types.  The projection is diagnostic evidence only; it is never a patch
selection or authorization input.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from types import MappingProxyType
from typing import Any


EVIDENCE_FORMAT = "npunlock.viz-evidence"
EVIDENCE_VERSION = 1
SOURCE_CONTRACTS = MappingProxyType(
    {
        "act_groups": "npunlock.prepared-act-groups.v1",
        "group_coverage": "patchblob.group-coverage.v1",
        "exact_relations": "patchblob.exact-relations.v1",
        "graph_io": "patchblob.graph-io-symbols.v1",
        "topology": "npunlock.native-topology-proposal.v1",
    }
)
_ENDPOINT_KINDS = {"act_group", "dpu_invariant", "graph_input", "graph_output"}
_RELATION_KINDS = {
    "act_to_output",
    "act_to_act",
    "act_to_dpu",
    "act_to_dpu_via_dma",
    "act_to_act_via_dma",
    "dpu_to_act",
    "graph_input_to_act_via_dma",
}
_COVERAGE_KINDS = {
    "direct_output",
    "act_consumer",
    "dpu_consumer",
    "dma_dpu_consumer",
    "dma_act_consumer",
}
_COVERAGE_STATES = {"not_observed", "incomplete", "complete"}


def _endpoint(value: Any) -> dict[str, Any]:
    return {
        "kind": value.kind,
        "primary_index": value.primary_index,
        "secondary_index": value.secondary_index,
    }


def _coverage(value: Any) -> dict[str, Any]:
    if not is_dataclass(value):
        raise TypeError("group coverage records must be dataclass instances")
    properties = asdict(value)
    kinds = {}
    for name, prefix in (
        ("direct_output", "direct_output"),
        ("act_consumer", "act_consumer"),
        ("dpu_consumer", "dpu_consumer"),
        ("dma_dpu_consumer", "dma_dpu_consumer"),
        ("dma_act_consumer", "dma_act_consumer"),
    ):
        kinds[name] = {
            "state": getattr(value, f"{prefix}_state"),
            "unresolved": list(getattr(value, f"{prefix}_unresolved_names")),
        }
    return {
        "group_index": value.group_index,
        "invocation_count": value.invocation_count,
        "kinds": kinds,
        "properties": properties,
    }


def project_prepared_evidence(
    prepared: Any,
    *,
    artifact_id: str = "native",
    review: Any | None = None,
) -> dict[str, Any]:
    """Project current core inspection results into a versioned plain mapping.

    ``prepared`` is intentionally duck-typed to keep this adapter from importing
    the execution package.  Its inspection methods remain the source of truth.
    An optional topology review must belong to the current prepared graph.
    """

    if not isinstance(artifact_id, str) or not artifact_id:
        raise ValueError("evidence artifact_id must be a non-empty string")
    blob = prepared.ir_provenance.graph_blob
    if not isinstance(blob, bytes) or not blob:
        raise ValueError("prepared evidence requires a non-empty native graph blob")
    digest = hashlib.sha256(blob).hexdigest()

    groups = []
    for group in prepared.groups:
        groups.append(
            {
                "index": group.index,
                "invocation_indices": list(group.invocation_indices),
                "range_indices": list(group.range_indices),
                "input_count": group.input_count,
                "input_dtypes": list(group.input_dtypes) if group.input_dtypes is not None else None,
                "output_dtype": group.output_dtype,
                "input_element_counts": (
                    list(group.per_invocation_input_element_counts)
                    if group.per_invocation_input_element_counts is not None
                    else None
                ),
                "output_element_count": group.per_invocation_output_element_count,
                "scalar_input_indices": (
                    list(group.scalar_input_indices) if group.scalar_input_indices is not None else None
                ),
            }
        )

    coverage = [_coverage(value) for value in prepared.inspect_group_coverage()]
    relations = [
        {
            "kind": relation.kind,
            "source": _endpoint(relation.source),
            "destination": _endpoint(relation.destination),
            "evidence_count": relation.evidence_count,
        }
        for relation in prepared.inspect_exact_relations()
    ]
    graph_io = [
        {
            "kind": symbol.kind,
            "symbol_table_index": symbol.symbol_table_index,
            "symbol_index": symbol.symbol_index,
            "size_bytes": symbol.size_bytes,
            "name": symbol.name,
        }
        for symbol in prepared.inspect_graph_io_symbols()
    ]
    proposal = prepared.propose_native_topology_mapping()
    topology: dict[str, Any] = {
        "status": proposal.status,
        "reason": proposal.reason,
        "names": list(proposal.names),
        "contract_candidates": [list(values) for values in proposal.contract_candidates],
        "viable_candidates": [list(values) for values in proposal.viable_candidates],
        "assignments": [list(values) for values in proposal.assignments],
        "truncated": proposal.truncated,
        "review": None,
    }
    if review is not None:
        if review.baseline_graph_blob_sha256 != digest:
            raise ValueError("topology review belongs to a different native graph blob")
        if (
            review.proposal != proposal
            or len(review.outputs) != len(proposal.outputs)
            or any(left is not right for left, right in zip(review.outputs, proposal.outputs))
        ):
            raise ValueError("topology review is stale or belongs to a different symbolic graph")
        topology["review"] = {
            "status": "complete",
            "group_indices": list(review.group_indices),
            "coverage_kinds": list(review.coverage_kinds),
        }

    projection = {
        "format": EVIDENCE_FORMAT,
        "version": EVIDENCE_VERSION,
        "artifact": {"id": artifact_id, "size": len(blob), "sha256": digest},
        "sources": dict(SOURCE_CONTRACTS),
        "groups": groups,
        "group_coverage": coverage,
        "exact_relations": relations,
        "graph_io": graph_io,
        "topology": topology,
    }
    validate_evidence(projection)
    return projection


def _integer(value: Any, what: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{what} must be an integer >= {minimum}")
    return value


def _text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string")
    return value


def _sequence(value: Any, what: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{what} must be an array")
    return value


def _indices(value: Any, what: str) -> list[int]:
    return [_integer(item, f"{what} item") for item in _sequence(value, what)]


def _validate_endpoint(value: Any, what: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"kind", "primary_index", "secondary_index"}:
        raise ValueError(f"{what} has an invalid endpoint shape")
    if value["kind"] not in _ENDPOINT_KINDS:
        raise ValueError(f"{what} has unsupported endpoint kind {value['kind']!r}")
    _integer(value["primary_index"], f"{what} primary_index")
    if value["secondary_index"] is not None:
        _integer(value["secondary_index"], f"{what} secondary_index")


def validate_evidence(value: Any) -> Mapping[str, Any]:
    """Validate a projection's identity, version, contracts, and bounded shape."""

    required = {
        "format",
        "version",
        "artifact",
        "sources",
        "groups",
        "group_coverage",
        "exact_relations",
        "graph_io",
        "topology",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("npunlock visualization evidence has an invalid top-level shape")
    if value["format"] != EVIDENCE_FORMAT or value["version"] != EVIDENCE_VERSION:
        raise ValueError("unsupported npunlock visualization evidence format or version")
    artifact = value["artifact"]
    if not isinstance(artifact, Mapping) or set(artifact) != {"id", "size", "sha256"}:
        raise ValueError("evidence artifact identity is invalid")
    _text(artifact["id"], "evidence artifact id")
    _integer(artifact["size"], "evidence artifact size", minimum=1)
    digest = _text(artifact["sha256"], "evidence artifact sha256")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("evidence artifact sha256 must be lowercase hexadecimal")
    if value["sources"] != SOURCE_CONTRACTS:
        raise ValueError("evidence source contracts are unsupported or incomplete")

    group_indices: set[int] = set()
    group_keys = {
        "index",
        "invocation_indices",
        "range_indices",
        "input_count",
        "input_dtypes",
        "output_dtype",
        "input_element_counts",
        "output_element_count",
        "scalar_input_indices",
    }
    for group in _sequence(value["groups"], "evidence groups"):
        if not isinstance(group, Mapping) or set(group) != group_keys:
            raise ValueError("evidence group has an invalid shape")
        index = _integer(group["index"], "evidence group index")
        if index in group_indices:
            raise ValueError("evidence group indices must be unique")
        if index != len(group_indices):
            raise ValueError("evidence groups must be ordered contiguously from zero")
        group_indices.add(index)
        invocation_indices = _indices(group["invocation_indices"], "evidence group invocation_indices")
        range_indices = _indices(group["range_indices"], "evidence group range_indices")
        if not invocation_indices or len(invocation_indices) != len(range_indices):
            raise ValueError("evidence group invocation and range indices must be non-empty and aligned")
        for key in ("input_count", "output_element_count"):
            if group[key] is not None:
                _integer(group[key], f"evidence group {key}", minimum=1)
        for key in ("input_dtypes", "input_element_counts", "scalar_input_indices"):
            if group[key] is not None:
                values = _sequence(group[key], f"evidence group {key}")
                for item in values:
                    if key == "input_dtypes":
                        _text(item, f"evidence group {key} item")
                    else:
                        _integer(item, f"evidence group {key} item")
        if group["output_dtype"] is not None:
            _text(group["output_dtype"], "evidence group output_dtype")
    if group_indices != set(range(len(group_indices))):
        raise ValueError("evidence group indices must be contiguous from zero")

    relation_endpoints: list[Mapping[str, Any]] = []
    for relation in _sequence(value["exact_relations"], "exact relations"):
        if not isinstance(relation, Mapping) or set(relation) != {
            "kind",
            "source",
            "destination",
            "evidence_count",
        }:
            raise ValueError("exact relation has an invalid shape")
        if relation["kind"] not in _RELATION_KINDS:
            raise ValueError(f"exact relation kind is unsupported: {relation['kind']!r}")
        _validate_endpoint(relation["source"], "exact relation source")
        _validate_endpoint(relation["destination"], "exact relation destination")
        relation_endpoints.extend((relation["source"], relation["destination"]))
        _integer(relation["evidence_count"], "exact relation evidence_count", minimum=1)

    io_endpoints: set[tuple[str, int, int | None]] = set()
    for symbol in _sequence(value["graph_io"], "graph I/O symbols"):
        if not isinstance(symbol, Mapping) or set(symbol) != {
            "kind",
            "symbol_table_index",
            "symbol_index",
            "size_bytes",
            "name",
        }:
            raise ValueError("graph I/O symbol has an invalid shape")
        if symbol["kind"] not in {"input", "output"}:
            raise ValueError("graph I/O symbol kind must be input or output")
        _integer(symbol["symbol_table_index"], "graph I/O symbol table index")
        _integer(symbol["symbol_index"], "graph I/O symbol index")
        _integer(symbol["size_bytes"], "graph I/O symbol size", minimum=1)
        _text(symbol["name"], "graph I/O symbol name")
        endpoint = (
            f"graph_{symbol['kind']}",
            symbol["symbol_index"],
            symbol["symbol_table_index"] if symbol["kind"] == "input" else None,
        )
        if endpoint in io_endpoints:
            raise ValueError("graph I/O endpoint identities must be unique")
        io_endpoints.add(endpoint)
    for endpoint in relation_endpoints:
        kind = endpoint["kind"]
        if kind == "act_group" and (
            endpoint["primary_index"] not in group_indices or endpoint["secondary_index"] is not None
        ):
            raise ValueError("exact relation references an unknown or malformed ACT group")
        if kind == "dpu_invariant" and endpoint["secondary_index"] is not None:
            raise ValueError("DPU relation endpoints cannot have a secondary index")
        if (
            kind.startswith("graph_")
            and (
                kind,
                endpoint["primary_index"],
                endpoint["secondary_index"],
            )
            not in io_endpoints
        ):
            raise ValueError("exact relation references an unknown graph I/O endpoint")

    coverage_indices = set()
    for coverage in _sequence(value["group_coverage"], "group coverage"):
        if not isinstance(coverage, Mapping) or set(coverage) != {
            "group_index",
            "invocation_count",
            "kinds",
            "properties",
        }:
            raise ValueError("group coverage has an invalid shape")
        index = _integer(coverage["group_index"], "group coverage index")
        if index in coverage_indices:
            raise ValueError("group coverage indices must be unique")
        coverage_indices.add(index)
        if index not in group_indices:
            raise ValueError("group coverage references an unknown ACT group")
        _integer(coverage["invocation_count"], "group coverage invocation count")
        if not isinstance(coverage["kinds"], Mapping) or set(coverage["kinds"]) != _COVERAGE_KINDS:
            raise ValueError("group coverage kinds are incomplete or unsupported")
        for kind, state in coverage["kinds"].items():
            if not isinstance(state, Mapping) or set(state) != {"state", "unresolved"}:
                raise ValueError(f"group coverage kind {kind!r} has an invalid shape")
            if state["state"] not in _COVERAGE_STATES:
                raise ValueError(f"group coverage kind {kind!r} has an invalid state")
            for unresolved in _sequence(state["unresolved"], f"group coverage {kind} unresolved"):
                _text(unresolved, f"group coverage {kind} unresolved item")
        if not isinstance(coverage["properties"], Mapping):
            raise ValueError("group coverage kinds and properties must be objects")
        group = value["groups"][index]
        if coverage["invocation_count"] != len(group["invocation_indices"]):
            raise ValueError("group coverage invocation count does not match the ACT group")
    if coverage_indices != group_indices:
        raise ValueError("group coverage must contain every and only declared ACT group")

    topology = value["topology"]
    topology_keys = {
        "status",
        "reason",
        "names",
        "contract_candidates",
        "viable_candidates",
        "assignments",
        "truncated",
        "review",
    }
    if not isinstance(topology, Mapping) or set(topology) != topology_keys:
        raise ValueError("topology evidence has an invalid shape")
    if topology["status"] not in {"none", "unique", "ambiguous"}:
        raise ValueError("topology evidence has an invalid status")
    _text(topology["reason"], "topology reason")
    names = [_text(item, "topology name") for item in _sequence(topology["names"], "topology names")]
    topology_indices: dict[str, list[list[int]]] = {}
    for key in ("contract_candidates", "viable_candidates", "assignments"):
        topology_indices[key] = []
        for values in _sequence(topology[key], f"topology {key}"):
            indices = _indices(values, f"topology {key} entry")
            topology_indices[key].append(indices)
            if any(index not in group_indices for index in indices):
                raise ValueError(f"topology {key} references an unknown ACT group")
    if len(topology["contract_candidates"]) != len(names) or len(topology["viable_candidates"]) != len(names):
        raise ValueError("topology candidate lists must align with names")
    if not isinstance(topology["truncated"], bool):
        raise ValueError("topology truncated must be boolean")
    assignments = topology_indices["assignments"]
    if any(len(assignment) != len(names) for assignment in assignments):
        raise ValueError("topology assignments must align with names")
    expected_status = (
        "none"
        if not assignments
        else ("unique" if len(assignments) == 1 and not topology["truncated"] else "ambiguous")
    )
    if topology["status"] != expected_status:
        raise ValueError("topology status does not match its assignments")
    review = topology["review"]
    if review is not None:
        if not isinstance(review, Mapping) or set(review) != {"status", "group_indices", "coverage_kinds"}:
            raise ValueError("topology review has an invalid shape")
        if review["status"] != "complete":
            raise ValueError("topology review must be complete")
        reviewed = _indices(review["group_indices"], "topology review group_indices")
        kinds = [_text(item, "topology review coverage kind") for item in review["coverage_kinds"]]
        if (
            len(reviewed) != len(names)
            or len(kinds) != len(names)
            or any(index not in group_indices for index in reviewed)
        ):
            raise ValueError("topology review must align with names and known groups")
        if (
            topology["status"] != "unique"
            or reviewed != assignments[0]
            or any(kind not in _COVERAGE_KINDS for kind in kinds)
        ):
            raise ValueError("topology review must match the unique proposal and supported coverage kinds")
    return value


__all__ = [
    "EVIDENCE_FORMAT",
    "EVIDENCE_VERSION",
    "SOURCE_CONTRACTS",
    "project_prepared_evidence",
    "validate_evidence",
]
