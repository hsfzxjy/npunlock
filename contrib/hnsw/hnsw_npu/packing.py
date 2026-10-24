from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .index import HnswIndex, SearchStats


CHUNK_ELEMENTS = 1024
REPLICAS = 4
LOGICAL_ELEMENTS = REPLICAS * CHUNK_ELEMENTS
MAX_CHUNKS_PER_INVOCATION = 6
MAX_BATCH_QUERIES = REPLICAS * MAX_CHUNKS_PER_INVOCATION
MAX_NODES = 48
MAX_DIMENSIONS = 32
MAX_LEVELS = 4
MAX_M = 4
MAX_EF = 24
MAX_K = 16

QUERY_MAGIC = 372021
GRAPH_MAGIC = 372022
OUTPUT_MAGIC = 372023
FORMAT_VERSION = 3
QUERY_HEADER_ELEMENTS = 8
GRAPH_HEADER_ELEMENTS = 8
OUTPUT_HEADER_ELEMENTS = 16
OUTPUT_ELEMENTS = OUTPUT_HEADER_ELEMENTS + 2 * MAX_K
NODE_GRAPH_ELEMENTS = 1 + MAX_LEVELS * (1 + MAX_M)

_COUNTER_LOWER_BOUNDS = np.asarray((1, 1, 0, 0, 0, 1, 1), dtype=np.uint32)
_COUNTER_LOWER_BOUNDS.flags.writeable = False


@dataclass(frozen=True, slots=True)
class PackedResult:
    ids: np.ndarray
    distances: np.ndarray
    stats: SearchStats


def validate_capacity(index: HnswIndex) -> None:
    node_count, dimensions = index.vectors.shape
    if node_count > MAX_NODES:
        raise ValueError(f"index has {node_count} nodes, maximum is {MAX_NODES}")
    if dimensions > MAX_DIMENSIONS:
        raise ValueError(f"index dimension is {dimensions}, maximum is {MAX_DIMENSIONS}")
    if index.m > MAX_M:
        raise ValueError(f"index M is {index.m}, maximum is {MAX_M}")
    if index.max_level >= MAX_LEVELS:
        raise ValueError(f"index has level {index.max_level}, maximum is {MAX_LEVELS - 1}")
    if QUERY_HEADER_ELEMENTS + dimensions * (node_count + 1) > CHUNK_ELEMENTS:
        raise ValueError("query and vector payload do not fit one HNSW chunk")
    if GRAPH_HEADER_ELEMENTS + node_count * NODE_GRAPH_ELEMENTS > CHUNK_ELEMENTS:
        raise ValueError("adjacency payload does not fit one HNSW chunk")


def _validate_query_capacity(query_capacity: int) -> None:
    if (
        not isinstance(query_capacity, int)
        or query_capacity < REPLICAS
        or query_capacity > MAX_BATCH_QUERIES
        or query_capacity % REPLICAS != 0
    ):
        raise ValueError(f"query_capacity must be a multiple of {REPLICAS} in " f"[{REPLICAS}, {MAX_BATCH_QUERIES}]")


def _replicate(chunk: np.ndarray, copies: int = REPLICAS) -> np.ndarray:
    return np.tile(chunk, copies).reshape(1, copies * CHUNK_ELEMENTS)


def pack_graph(index: HnswIndex, *, query_capacity: int = REPLICAS) -> np.ndarray:
    validate_capacity(index)
    _validate_query_capacity(query_capacity)
    node_count, dimensions = index.vectors.shape
    chunk = np.zeros(CHUNK_ELEMENTS, dtype=np.float32)
    words = chunk.view(np.uint32)
    words[:GRAPH_HEADER_ELEMENTS] = (
        GRAPH_MAGIC,
        FORMAT_VERSION,
        node_count,
        dimensions,
        index.entry_point,
        index.max_level,
        MAX_LEVELS,
        index.m,
    )
    cursor = GRAPH_HEADER_ELEMENTS
    for node in range(node_count):
        words[cursor] = index.levels[node]
        cursor += 1
        for layer in range(MAX_LEVELS):
            links = index.neighbors[node][layer] if layer <= index.levels[node] else ()
            words[cursor] = len(links)
            cursor += 1
            words[cursor : cursor + len(links)] = links
            cursor += MAX_M
    return _replicate(chunk, query_capacity)


def _validate_search_parameters(index: HnswIndex, *, k: int, ef_search: int) -> None:
    if not isinstance(ef_search, int) or not 1 <= ef_search <= min(MAX_EF, index.vectors.shape[0]):
        raise ValueError(f"ef_search must be in [1, {min(MAX_EF, index.vectors.shape[0])}]")
    if not isinstance(k, int) or not 1 <= k <= min(MAX_K, ef_search):
        raise ValueError(f"k must be in [1, {min(MAX_K, ef_search)}]")


def _normalize_queries(index: HnswIndex, queries: object, *, maximum: int = MAX_BATCH_QUERIES) -> np.ndarray:
    validate_capacity(index)
    matrix = np.asarray(queries, dtype=np.float32)
    dimensions = index.vectors.shape[1]
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != dimensions:
        raise ValueError(f"queries must have shape (count, {dimensions})")
    if not 1 <= matrix.shape[0] <= maximum:
        raise ValueError(f"query count must be in [1, {maximum}]")
    if not np.isfinite(matrix).all():
        raise ValueError("queries must contain only finite values")
    return np.ascontiguousarray(matrix)


def _make_query_buffer(index: HnswIndex, *, query_capacity: int = REPLICAS) -> np.ndarray:
    """Create the static query/vector payload shared by later submissions."""

    validate_capacity(index)
    _validate_query_capacity(query_capacity)
    node_count, dimensions = index.vectors.shape
    chunks = np.zeros((query_capacity, CHUNK_ELEMENTS), dtype=np.float32)
    vectors_begin = QUERY_HEADER_ELEMENTS + dimensions
    for chunk in chunks:
        chunk.view(np.uint32)[:QUERY_HEADER_ELEMENTS] = (
            QUERY_MAGIC,
            FORMAT_VERSION,
            node_count,
            dimensions,
            index.entry_point,
            index.max_level,
            1,
            1,
        )
        chunk[vectors_begin : vectors_begin + index.vectors.size] = index.vectors.reshape(-1)
    return chunks.reshape(1, query_capacity * CHUNK_ELEMENTS)


def _update_query_buffer(
    target: object,
    index: HnswIndex,
    queries: object,
    *,
    k: int,
    ef_search: int,
) -> tuple[np.ndarray, int]:
    """Update only dynamic query fields in a prepared packed buffer."""

    packed = np.asarray(target)
    if packed.size % CHUNK_ELEMENTS != 0:
        raise ValueError("target element count must be a multiple of the HNSW chunk size")
    query_capacity = packed.size // CHUNK_ELEMENTS
    _validate_query_capacity(query_capacity)
    matrix = _normalize_queries(index, queries, maximum=query_capacity)
    _validate_search_parameters(index, k=k, ef_search=ef_search)
    if packed.dtype != np.float32 or not packed.flags.c_contiguous or not packed.flags.writeable:
        raise ValueError("target must be a writable contiguous float32 buffer")
    chunks = packed.reshape(query_capacity, CHUNK_ELEMENTS)
    active_count = matrix.shape[0]
    chunks.view(np.uint32)[:, :QUERY_HEADER_ELEMENTS] = (
        QUERY_MAGIC,
        FORMAT_VERSION,
        index.vectors.shape[0],
        index.vectors.shape[1],
        index.entry_point,
        index.max_level,
        ef_search,
        k,
    )
    query_end = QUERY_HEADER_ELEMENTS + index.vectors.shape[1]
    chunks[:active_count, QUERY_HEADER_ELEMENTS:query_end] = matrix
    if active_count < query_capacity:
        chunks[active_count:, QUERY_HEADER_ELEMENTS:query_end] = matrix[0]
    return matrix, active_count


def pack_queries(
    index: HnswIndex,
    queries: object,
    *,
    k: int,
    ef_search: int,
    query_capacity: int | None = None,
) -> tuple[np.ndarray, int]:
    """Pack one to 24 queries into independent 1,024-element payloads.

    Unused chunks repeat the first query. This keeps every chunk valid and
    retains a byte-for-byte execution check for a single-query submission.
    """

    matrix = _normalize_queries(index, queries)
    if query_capacity is None:
        query_capacity = max(REPLICAS, ((matrix.shape[0] + REPLICAS - 1) // REPLICAS) * REPLICAS)
    _validate_query_capacity(query_capacity)
    if matrix.shape[0] > query_capacity:
        raise ValueError(f"query count {matrix.shape[0]} exceeds capacity {query_capacity}")
    packed = _make_query_buffer(index, query_capacity=query_capacity)
    _, active_count = _update_query_buffer(packed, index, matrix, k=k, ef_search=ef_search)
    return packed, active_count


def pack_query(index: HnswIndex, query: object, *, k: int, ef_search: int) -> np.ndarray:
    packed, _ = pack_queries(index, query, k=k, ef_search=ef_search)
    return packed


@lru_cache(maxsize=128)
def _result_validation_arrays(
    node_count: int,
    dimensions: int,
    entry_point: int,
    max_level: int,
    ef_search: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cache immutable lower and upper bounds for the output header."""

    expected_header = np.asarray(
        (
            OUTPUT_MAGIC,
            0,
            node_count,
            dimensions,
            entry_point,
            max_level,
            ef_search,
            k,
        ),
        dtype=np.uint32,
    )
    counter_upper_bounds = np.asarray(
        (
            node_count,
            MAX_NODES * (MAX_LEVELS * MAX_M + 1),
            node_count * max_level,
            node_count,
            1,
            MAX_NODES,
            ef_search,
        ),
        dtype=np.uint32,
    )
    header_lower_bounds = np.concatenate(
        (expected_header, _COUNTER_LOWER_BOUNDS, np.asarray((k,), dtype=np.uint32))
    )
    header_upper_bounds = np.concatenate(
        (expected_header, counter_upper_bounds, np.asarray((k,), dtype=np.uint32))
    )
    header_lower_bounds.flags.writeable = False
    header_upper_bounds.flags.writeable = False
    return header_lower_bounds, header_upper_bounds


def _unpack_result_arrays(
    output: object,
    *,
    index: HnswIndex,
    k: int,
    ef_search: int,
    query_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = np.asarray(output, dtype=np.float32).reshape(-1)
    if flat.size % CHUNK_ELEMENTS != 0:
        raise ValueError("kernel output element count is not a multiple of the HNSW chunk size")
    query_capacity = flat.size // CHUNK_ELEMENTS
    _validate_query_capacity(query_capacity)
    if not isinstance(query_count, int) or not 1 <= query_count <= query_capacity:
        raise ValueError(f"query_count must be in [1, {query_capacity}]")
    chunks = flat.reshape(query_capacity, CHUNK_ELEMENTS)
    if query_count < query_capacity:
        inactive = chunks[query_count:, :OUTPUT_ELEMENTS].view(np.uint32)
        first = chunks[0, :OUTPUT_ELEMENTS].view(np.uint32)
        if not np.array_equal(inactive, np.broadcast_to(first, inactive.shape)):
            raise ValueError("unused replicated HNSW chunks produced different output bytes")

    header_words = chunks[:query_count, :OUTPUT_HEADER_ELEMENTS].view(np.uint32)
    records = chunks[:query_count, OUTPUT_HEADER_ELEMENTS : OUTPUT_HEADER_ELEMENTS + 2 * k].reshape(query_count, k, 2)
    ids = records[:, :, 0].view(np.int32).copy()
    distances = records[:, :, 1].copy()
    header_lower_bounds, header_upper_bounds = _result_validation_arrays(
        index.vectors.shape[0],
        index.vectors.shape[1],
        index.entry_point,
        index.max_level,
        ef_search,
        k,
    )
    counters = header_words[:, 8:15]
    if ((header_words < header_lower_bounds) | (header_words > header_upper_bounds)).any():
        # Keep precise malformed-output diagnostics off the successful hot path.
        if not np.all(header_words[:, :8] == header_lower_bounds[:8]):
            raise ValueError("kernel output header does not match the requested index and search")
        if not np.all(header_words[:, 15] == k):
            raise ValueError("kernel returned an unexpected result count")
        termination_flags = header_words[:, 12]
        if ((termination_flags != 0) & (termination_flags != 1)).any():
            raise ValueError("kernel returned an invalid termination flag")
        if (
            (counters < header_lower_bounds[8:15]) | (counters > header_upper_bounds[8:15])
        ).any():
            raise ValueError("kernel returned an out-of-range traversal counter")
        raise ValueError("kernel returned invalid output metadata")

    if (
        (ids < 0)
        | (ids >= index.vectors.shape[0])
        | ~np.isfinite(distances)
        | (distances < 0)
    ).any():
        # Preserve specific corruption diagnostics without multiple reductions
        # on every successful decode.
        if (ids < 0).any() or (ids >= index.vectors.shape[0]).any():
            raise ValueError("kernel returned an invalid node id")
        if not np.isfinite(distances).all() or (distances < 0).any():
            raise ValueError("kernel returned an invalid distance")
        raise ValueError("kernel returned invalid result records")
    if (np.diff(np.sort(ids, axis=1), axis=1) == 0).any():
        raise ValueError("kernel returned duplicate node ids")
    if k > 1:
        previous_distances = distances[:, :-1]
        next_distances = distances[:, 1:]
        if (
            (next_distances < previous_distances)
            | ((next_distances == previous_distances) & (ids[:, 1:] < ids[:, :-1]))
        ).any():
            raise ValueError("kernel results are not ordered by distance and node id")

    return ids, distances, counters.copy()


def _stats_from_counter_rows(counters: np.ndarray) -> tuple[SearchStats, ...]:
    return tuple(
        SearchStats(
            visited_count=row[0],
            distance_count=row[1],
            upper_hops=row[2],
            base_expansions=row[3],
            terminated_by_bound=bool(row[4]),
            candidate_peak=row[5],
            top_peak=row[6],
        )
        for row in counters.tolist()
    )


def unpack_results(
    output: object,
    *,
    index: HnswIndex,
    k: int,
    ef_search: int,
    query_count: int,
) -> tuple[PackedResult, ...]:
    ids, distances, counters = _unpack_result_arrays(
        output,
        index=index,
        k=k,
        ef_search=ef_search,
        query_count=query_count,
    )
    stats = _stats_from_counter_rows(counters)
    return tuple(PackedResult(ids=ids[row], distances=distances[row], stats=stats[row]) for row in range(query_count))


def unpack_result(output: object, *, index: HnswIndex, k: int, ef_search: int) -> PackedResult:
    return unpack_results(output, index=index, k=k, ef_search=ef_search, query_count=1)[0]
