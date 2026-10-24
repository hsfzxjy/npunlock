from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import npunlock as npu

from .cache import ProgramCache, provenance_identity
from .index import HnswIndex, SearchStats, search_index
from .packing import (
    CHUNK_ELEMENTS,
    MAX_BATCH_QUERIES,
    REPLICAS,
    _make_query_buffer,
    _unpack_result_arrays,
    _update_query_buffer,
    pack_graph,
    validate_capacity,
)


KERNEL_PATH = Path(__file__).resolve().parents[1] / "kernels" / "hnsw_search.c"


@dataclass(frozen=True, slots=True)
class SearchResult:
    ids: np.ndarray
    distances: np.ndarray
    stats: SearchStats
    timings: dict[str, float]


@dataclass(frozen=True, slots=True)
class SearchBatch:
    ids: np.ndarray
    distances: np.ndarray
    stats: tuple[SearchStats, ...]
    timings: dict[str, float]
    _results: tuple[SearchResult, ...] | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def results(self) -> tuple[SearchResult, ...]:
        """Materialize backward-compatible per-query wrappers on first access."""

        results = self._results
        if results is None:
            results = tuple(
                SearchResult(self.ids[row], self.distances[row], self.stats[row], self.timings)
                for row in range(self.ids.shape[0])
            )
            object.__setattr__(self, "_results", results)
        return results


def _validate_program(program: npu.Program, *, logical_elements: int) -> None:
    expected_inputs = (
        ("hnsw_query_vectors", (1, logical_elements), "f32"),
        ("hnsw_adjacency", (1, logical_elements), "f32"),
    )
    actual_inputs = tuple((contract.name, contract.shape, contract.dtype) for contract in program.input_contracts)
    if actual_inputs != expected_inputs:
        raise RuntimeError(f"HNSW program input contract changed: {actual_inputs!r}")
    expected_outputs = (("hnsw_results", (1, logical_elements), "f32"),)
    actual_outputs = tuple((contract.name, contract.shape, contract.dtype) for contract in program.output_contracts)
    if actual_outputs != expected_outputs:
        raise RuntimeError(f"HNSW program output contract changed: {actual_outputs!r}")
    if len(program.patch_reports) != 1:
        raise RuntimeError(f"HNSW program has {len(program.patch_reports)} patch reports, expected one")
    try:
        report = json.loads(program.patch_reports[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("HNSW patch report is invalid") from error
    graph_contract = report.get("graph_contract")
    targets = report.get("targets")
    validation = report.get("validation")
    if (
        report.get("schema") != "npunlock.patchblob.v1"
        or not isinstance(graph_contract, dict)
        or graph_contract.get("compiler") != "8.3"
        or graph_contract.get("npu") != "3720"
        or not isinstance(targets, list)
        or len(targets) != REPLICAS
        or not isinstance(validation, dict)
        or validation.get("section_contents_preserved") is not True
        or validation.get("mutation_set_preserved") is not True
    ):
        raise RuntimeError("HNSW patch report contract changed")
    for index, target in enumerate(targets):
        invocation_elements = logical_elements // REPLICAS
        if (
            not isinstance(target, dict)
            or target.get("invocation_index") != index
            or target.get("range_index") != index
            or target.get("input_count") != 2
            or target.get("element_count") != invocation_elements
            or target.get("span_bytes") != invocation_elements * 4
            or target.get("contract_flags") != 0x3B
        ):
            raise RuntimeError(f"HNSW patch target {index} contract changed")
    if report.get("output_graph_sha256") != hashlib.sha256(program.graph_blob).hexdigest():
        raise RuntimeError("HNSW final graph hash does not match its patch report")


class NpuHnsw:
    """Execute bounded, complete HNSW query traversal on ACT-SHAVE."""

    def __init__(
        self,
        index: HnswIndex,
        *,
        movi_dll_dir: str | Path,
        native_dir: str | Path | None = None,
        timeout_ms: int = 20_000,
        cache_dir: str | Path | None = None,
        batch_capacity: int = REPLICAS,
    ):
        validate_capacity(index)
        if (
            not isinstance(batch_capacity, int)
            or batch_capacity < REPLICAS
            or batch_capacity > MAX_BATCH_QUERIES
            or batch_capacity % REPLICAS != 0
        ):
            raise ValueError(
                f"batch_capacity must be a multiple of {REPLICAS} in " f"[{REPLICAS}, {MAX_BATCH_QUERIES}]"
            )
        self.index = index
        self.batch_capacity = batch_capacity
        self.logical_elements = batch_capacity * CHUNK_ELEMENTS
        self.invocation_elements = self.logical_elements // REPLICAS
        self.movi_dll_dir = Path(movi_dll_dir).resolve()
        self.source = KERNEL_PATH.read_bytes()
        self.packed_graph = pack_graph(index, query_capacity=batch_capacity)
        npu.configure(movi_dll_dir=movi_dll_dir)
        query = npu.input("hnsw_query_vectors", shape=(1, self.logical_elements), dtype="f32")
        graph = npu.input("hnsw_adjacency", shape=(1, self.logical_elements), dtype="f32")
        output = npu.custom(
            query,
            graph,
            source=self.source,
            carrier="Subtract",
            _name="hnsw_results",
        )
        graph_definition = npu.Graph([query, graph], [output], name=f"hnsw_l2_search_{batch_capacity}q")
        cache_started = perf_counter()
        cache = ProgramCache(
            cache_dir,
            source=self.source,
            logical_elements=self.logical_elements,
            movi_dll_dir=movi_dll_dir,
            native_dir=native_dir,
            timeout_ms=timeout_ms,
        )
        self.program = cache.load()
        if self.program is not None:
            try:
                _validate_program(self.program, logical_elements=self.logical_elements)
            except Exception as error:
                self.program.close()
                self.program = None
                cache.status = "invalid"
                cache.diagnostic = str(error)
        self.cache_load_ms = (perf_counter() - cache_started) * 1000.0 if cache.enabled else 0.0
        if self.program is None:
            hash_task = cache.start_dependency_hashes()
            try:
                prepared = npu.prepare(
                    graph_definition,
                    native_dir=native_dir,
                    timeout_ms=timeout_ms,
                    libraries=cache.libraries,
                )
                if len(prepared.groups) != 1:
                    raise RuntimeError(f"HNSW carrier exposed {len(prepared.groups)} ACT groups, expected one")
                group = prepared.groups[0]
                if group.input_dtypes != ("f32", "f32") or group.output_dtype != "f32":
                    raise RuntimeError(
                        "HNSW carrier dtype changed: " f"inputs={group.input_dtypes!r}, output={group.output_dtype!r}"
                    )
                if (
                    group.per_invocation_input_element_counts != (self.invocation_elements, self.invocation_elements)
                    or group.per_invocation_output_element_count != self.invocation_elements
                    or len(group.targets) != REPLICAS
                ):
                    raise RuntimeError(
                        "HNSW carrier contract changed: "
                        f"inputs={group.per_invocation_input_element_counts!r}, "
                        f"output={group.per_invocation_output_element_count!r}, "
                        f"targets={len(group.targets)}"
                    )
                started = perf_counter()
                self.program = prepared.build(movi_dll_dir=movi_dll_dir)
                self.build_ms = (perf_counter() - started) * 1000.0
                _validate_program(self.program, logical_elements=self.logical_elements)
                self.carrier_explanation = prepared.explain()
                cache.store(self.program, hash_task)
            except BaseException:
                if hash_task is not None:
                    hash_task[0].shutdown(wait=True)
                if self.program is not None:
                    self.program.close()
                raise
        else:
            self.build_ms = 0.0
            self.carrier_explanation = "loaded validated HNSW program from cache"
        self.cache_status = cache.status
        self.cache_diagnostic = cache.diagnostic
        self._runtime_provenance = (
            cache.provenance
            if cache.provenance is not None
            else provenance_identity(self.program.ir_provenance) if self.program.ir_provenance is not None else None
        )
        self.shared_inputs = self.program.shared_inputs()
        self.shared_output = self.program.shared_output()
        started = perf_counter()
        np.copyto(
            self.shared_inputs["hnsw_query_vectors"],
            _make_query_buffer(index, query_capacity=batch_capacity),
        )
        np.copyto(self.shared_inputs["hnsw_adjacency"], self.packed_graph)
        self.static_input_upload_ms = (perf_counter() - started) * 1000.0

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
            "targets": report["targets"],
            "validation": report["validation"],
            "patched_target_count": len(report["targets"]),
        }

    @property
    def build_manifest(self) -> dict[str, object] | None:
        payload = self.program.artifact_manifest
        return None if payload is None else json.loads(payload.decode("utf-8"))

    @property
    def runtime_provenance(self) -> dict[str, object] | None:
        return self._runtime_provenance

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

    def search_many(
        self,
        queries: object,
        *,
        k: int = 5,
        ef_search: int = 16,
        verify: bool = True,
    ) -> SearchBatch:
        pack_ms = 0.0
        started = perf_counter()
        query_matrix, query_count = _update_query_buffer(
            self.shared_inputs["hnsw_query_vectors"],
            self.index,
            queries,
            k=k,
            ef_search=ef_search,
        )
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
        download_ms = 0.0
        started = perf_counter()
        result_ids, result_distances, result_stats = _unpack_result_arrays(
            outputs["hnsw_results"],
            index=self.index,
            k=k,
            ef_search=ef_search,
            query_count=query_count,
        )
        unpack_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        if verify:
            for query, ids, distances, stats in zip(
                query_matrix,
                result_ids,
                result_distances,
                result_stats,
            ):
                reference_ids, reference_distances, reference_stats = search_index(
                    self.index, query, k=k, ef_search=ef_search
                )
                if not np.array_equal(ids, reference_ids) or not np.array_equal(
                    distances.view("V4"), reference_distances.view("V4")
                ):
                    raise RuntimeError("NPU HNSW result does not match the independent host traversal")
                if stats != reference_stats:
                    raise RuntimeError(
                        "NPU HNSW traversal counters " f"{stats!r} do not match host {reference_stats!r}"
                    )
        verify_ms = (perf_counter() - started) * 1000.0
        timings = {
            "pack_ms": pack_ms,
            "upload_ms": upload_ms,
            "graphinfer_wall_ms": graphinfer_wall_ms,
            "download_ms": download_ms,
            "unpack_ms": unpack_ms,
            "host_verify_ms": verify_ms,
        }
        return SearchBatch(
            ids=result_ids,
            distances=result_distances,
            stats=result_stats,
            timings=timings,
        )

    def search(
        self,
        query: object,
        *,
        k: int = 5,
        ef_search: int = 16,
        verify: bool = True,
    ) -> SearchResult:
        return self.search_many(query, k=k, ef_search=ef_search, verify=verify).results[0]

    def close(self) -> None:
        self.program.close()

    def __enter__(self) -> "NpuHnsw":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
