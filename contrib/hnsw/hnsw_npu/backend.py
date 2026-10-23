from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import npunlock as npu

from .index import HnswIndex, SearchStats, search_index
from .packing import (
    CHUNK_ELEMENTS,
    LOGICAL_ELEMENTS,
    REPLICAS,
    pack_graph,
    pack_query,
    unpack_result,
    validate_capacity,
)


KERNEL_PATH = Path(__file__).resolve().parents[1] / "kernels" / "hnsw_search.c"


@dataclass(frozen=True, slots=True)
class SearchResult:
    ids: np.ndarray
    distances: np.ndarray
    stats: SearchStats
    timings: dict[str, float]


class NpuHnsw:
    """Execute bounded, complete HNSW query traversal on ACT-SHAVE."""

    def __init__(
        self,
        index: HnswIndex,
        *,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None = None,
        timeout_ms: int = 20_000,
    ):
        validate_capacity(index)
        self.index = index
        self.movi_dll_dir = Path(movi_dll_dir).resolve()
        self.source = KERNEL_PATH.read_bytes()
        self.packed_graph = pack_graph(index)
        npu.configure(movi_dll_dir=movi_dll_dir)
        query = npu.input("hnsw_query_vectors", shape=(1, LOGICAL_ELEMENTS), dtype="f32")
        graph = npu.input("hnsw_adjacency", shape=(1, LOGICAL_ELEMENTS), dtype="f32")
        output = npu.custom(
            query,
            graph,
            source=self.source,
            carrier="Subtract",
            _name="hnsw_results",
        )
        prepared = npu.prepare(
            npu.Graph([query, graph], [output], name="hnsw_l2_search"),
            native_dir=native_dir,
            timeout_ms=timeout_ms,
        )
        if len(prepared.groups) != 1:
            raise RuntimeError(f"HNSW carrier exposed {len(prepared.groups)} ACT groups, expected one")
        group = prepared.groups[0]
        if group.input_dtypes != ("f32", "f32") or group.output_dtype != "f32":
            raise RuntimeError(
                f"HNSW carrier dtype changed: inputs={group.input_dtypes!r}, output={group.output_dtype!r}"
            )
        if (
            group.per_invocation_input_element_counts != (CHUNK_ELEMENTS, CHUNK_ELEMENTS)
            or group.per_invocation_output_element_count != CHUNK_ELEMENTS
            or len(group.targets) != REPLICAS
        ):
            raise RuntimeError(
                "HNSW carrier contract changed: "
                f"inputs={group.per_invocation_input_element_counts!r}, "
                f"output={group.per_invocation_output_element_count!r}, targets={len(group.targets)}"
            )
        started = perf_counter()
        self.program = prepared.build(movi_dll_dir=movi_dll_dir)
        self.build_ms = (perf_counter() - started) * 1000.0
        self.carrier_explanation = prepared.explain()
        self.shared_inputs = self.program.shared_inputs()
        self.shared_output = self.program.shared_output()
        np.copyto(self.shared_inputs["hnsw_adjacency"], self.packed_graph)

    @property
    def patch_evidence(self) -> dict[str, object]:
        report = json.loads(self.program.patch_reports[0].decode("utf-8"))
        return {
            "schema": report["schema"],
            "input_graph_sha256": report["input_graph_sha256"],
            "shave_elf_sha256": report["shave_elf_sha256"],
            "output_graph_sha256": report["output_graph_sha256"],
            "graph_contract": report["graph_contract"],
            "append": report["append"],
            "validation": report["validation"],
            "patched_target_count": len(report["targets"]),
        }

    @property
    def build_manifest(self) -> dict[str, object] | None:
        payload = self.program.artifact_manifest
        return None if payload is None else json.loads(payload.decode("utf-8"))

    @property
    def runtime_provenance(self) -> dict[str, object] | None:
        value = self.program.ir_provenance
        if value is None:
            return None
        return {
            "driver_index": value.driver_index,
            "device_index": value.device_index,
            "driver_version": value.driver_version,
            "vendor_id": value.vendor_id,
            "device_id": value.device_id,
            "graph_extension_version": value.graph_extension_version,
            "compiler_version": list(value.compiler_version),
        }

    @property
    def movi_toolchain(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for relative in (
            "bin/moviCompile64.dll",
            "bin/moviAsm64.dll",
            "bin/moviLLD64.dll",
            "lib/mlibm.a",
        ):
            data = (self.movi_dll_dir / Path(relative)).read_bytes()
            result[relative] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return result

    def search(self, query: object, *, k: int = 5, ef_search: int = 16) -> SearchResult:
        started = perf_counter()
        packed_query = pack_query(self.index, query, k=k, ef_search=ef_search)
        pack_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        np.copyto(self.shared_inputs["hnsw_query_vectors"], packed_query)
        upload_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        outputs = self.program.run(
            {
                "hnsw_query_vectors": self.shared_inputs["hnsw_query_vectors"],
                "hnsw_adjacency": self.shared_inputs["hnsw_adjacency"],
            },
            outputs={"hnsw_results": self.shared_output},
        )
        graphinfer_wall_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        copied = np.array(outputs["hnsw_results"], copy=True)
        download_ms = (perf_counter() - started) * 1000.0
        packed_result = unpack_result(copied, index=self.index, k=k, ef_search=ef_search)
        reference_ids, reference_distances, reference_stats = search_index(self.index, query, k=k, ef_search=ef_search)
        if not np.array_equal(packed_result.ids, reference_ids) or not np.array_equal(
            packed_result.distances.view("V4"), reference_distances.view("V4")
        ):
            raise RuntimeError("NPU HNSW result does not match the independent host traversal")
        if packed_result.stats != reference_stats:
            raise RuntimeError(
                f"NPU HNSW traversal counters {packed_result.stats!r} do not match host {reference_stats!r}"
            )
        return SearchResult(
            ids=packed_result.ids,
            distances=packed_result.distances,
            stats=packed_result.stats,
            timings={
                "pack_ms": pack_ms,
                "upload_ms": upload_ms,
                "graphinfer_wall_ms": graphinfer_wall_ms,
                "download_ms": download_ms,
            },
        )

    def close(self) -> None:
        self.program.close()

    def __enter__(self) -> "NpuHnsw":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
