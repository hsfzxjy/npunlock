from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import npunlock as npu

from .index import HnswIndex, SearchStats, search_index
from .packing import (
    CHUNK_ELEMENTS,
    MAX_BATCH_QUERIES,
    REPLICAS,
    _make_query_buffer,
    _stats_from_counter_rows,
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
    timings: dict[str, float]
    _counter_rows: np.ndarray = field(repr=False, compare=False)
    _stats: tuple[SearchStats, ...] | None = field(default=None, repr=False, compare=False)
    _results: tuple[SearchResult, ...] | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def stats(self) -> tuple[SearchStats, ...]:
        """Materialize traversal-counter objects on first access."""

        stats = self._stats
        if stats is None:
            stats = _stats_from_counter_rows(self._counter_rows)
            object.__setattr__(self, "_stats", stats)
        return stats

    @property
    def results(self) -> tuple[SearchResult, ...]:
        """Materialize backward-compatible per-query wrappers on first access."""

        results = self._results
        if results is None:
            stats = self.stats
            results = tuple(
                SearchResult(self.ids[row], self.distances[row], stats[row], self.timings)
                for row in range(self.ids.shape[0])
            )
            object.__setattr__(self, "_results", results)
        return results


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
        result = npu.compile_cached(
            graph_definition,
            cache_dir=cache_dir,
            carrier_contract=npu.CarrierContract(
                (
                    npu.ActGroupContract(
                        ("f32", "f32"),
                        "f32",
                        (self.invocation_elements, self.invocation_elements),
                        self.invocation_elements,
                        REPLICAS,
                        invocation_indices=tuple(range(REPLICAS)),
                        range_indices=tuple(range(REPLICAS)),
                    ),
                ),
                mapping_indices=((0,),),
            ),
            program_contract=npu.ProgramContract(
                inputs=(
                    ("hnsw_query_vectors", (1, self.logical_elements), "f32"),
                    ("hnsw_adjacency", (1, self.logical_elements), "f32"),
                ),
                outputs=(("hnsw_results", (1, self.logical_elements), "f32"),),
                patch_count=1,
                targets_per_patch=(REPLICAS,),
            ),
            native_dir=native_dir,
            movi_dll_dir=movi_dll_dir,
            timeout_ms=timeout_ms,
        )
        self.program = result.program
        self.build_ms = result.build_ms
        self.carrier_explanation = result.explanation
        self.cache_load_ms = result.cache.load_ms
        self.cache_status = result.cache.state
        self.cache_diagnostic = result.cache.diagnostic
        provenance = self.program.provenance
        self._runtime_provenance = None if provenance is None else provenance.to_json()
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
        return self.program.patch_evidence[0].to_json()

    @property
    def build_manifest(self) -> dict[str, object]:
        return self.program.manifest

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
        result_ids, result_distances, counter_rows = _unpack_result_arrays(
            outputs["hnsw_results"],
            index=self.index,
            k=k,
            ef_search=ef_search,
            query_count=query_count,
        )
        unpack_ms = (perf_counter() - started) * 1000.0
        started = perf_counter()
        result_stats = None
        if verify:
            result_stats = _stats_from_counter_rows(counter_rows)
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
            timings=timings,
            _counter_rows=counter_rows,
            _stats=result_stats,
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
