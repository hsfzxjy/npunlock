from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .index import HnswIndex, SearchStats


CHUNK_ELEMENTS = 1024
LOGICAL_ELEMENTS = 4096
REPLICAS = 4
MAX_NODES = 48
MAX_DIMENSIONS = 32
MAX_LEVELS = 4
MAX_M = 4
MAX_EF = 24
MAX_K = 16

QUERY_MAGIC = 372021
GRAPH_MAGIC = 372022
OUTPUT_MAGIC = 372023
FORMAT_VERSION = 1
QUERY_HEADER_ELEMENTS = 8
GRAPH_HEADER_ELEMENTS = 8
OUTPUT_HEADER_ELEMENTS = 16
OUTPUT_ELEMENTS = OUTPUT_HEADER_ELEMENTS + 2 * MAX_K
NODE_GRAPH_ELEMENTS = 1 + MAX_LEVELS * (1 + MAX_M)


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
        raise ValueError("query and vector payload do not fit one validated ACT invocation")
    if GRAPH_HEADER_ELEMENTS + node_count * NODE_GRAPH_ELEMENTS > CHUNK_ELEMENTS:
        raise ValueError("adjacency payload does not fit one validated ACT invocation")


def _replicate(chunk: np.ndarray) -> np.ndarray:
    return np.tile(chunk, REPLICAS).reshape(1, LOGICAL_ELEMENTS)


def pack_graph(index: HnswIndex) -> np.ndarray:
    validate_capacity(index)
    node_count, dimensions = index.vectors.shape
    chunk = np.zeros(CHUNK_ELEMENTS, dtype=np.float32)
    chunk[:GRAPH_HEADER_ELEMENTS] = (
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
        chunk[cursor] = index.levels[node]
        cursor += 1
        for layer in range(MAX_LEVELS):
            links = index.neighbors[node][layer] if layer <= index.levels[node] else ()
            chunk[cursor] = len(links)
            cursor += 1
            for slot in range(MAX_M):
                chunk[cursor + slot] = links[slot] if slot < len(links) else -1.0
            cursor += MAX_M
    return _replicate(chunk)


def _validate_search_parameters(index: HnswIndex, *, k: int, ef_search: int) -> None:
    if not isinstance(ef_search, int) or not 1 <= ef_search <= min(MAX_EF, index.vectors.shape[0]):
        raise ValueError(f"ef_search must be in [1, {min(MAX_EF, index.vectors.shape[0])}]")
    if not isinstance(k, int) or not 1 <= k <= min(MAX_K, ef_search):
        raise ValueError(f"k must be in [1, {min(MAX_K, ef_search)}]")


def _normalize_queries(index: HnswIndex, queries: object) -> np.ndarray:
    validate_capacity(index)
    matrix = np.asarray(queries, dtype=np.float32)
    dimensions = index.vectors.shape[1]
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != dimensions:
        raise ValueError(f"queries must have shape (count, {dimensions})")
    if not 1 <= matrix.shape[0] <= REPLICAS:
        raise ValueError(f"query count must be in [1, {REPLICAS}]")
    if not np.isfinite(matrix).all():
        raise ValueError("queries must contain only finite values")
    return np.ascontiguousarray(matrix)


def _make_query_buffer(index: HnswIndex) -> np.ndarray:
    """Create the static query/vector payload shared by later submissions."""

    validate_capacity(index)
    node_count, dimensions = index.vectors.shape
    chunks = np.zeros((REPLICAS, CHUNK_ELEMENTS), dtype=np.float32)
    vectors_begin = QUERY_HEADER_ELEMENTS + dimensions
    for chunk in chunks:
        chunk[:QUERY_HEADER_ELEMENTS] = (
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
    return chunks.reshape(1, LOGICAL_ELEMENTS)


def _update_query_buffer(
    target: object,
    index: HnswIndex,
    queries: object,
    *,
    k: int,
    ef_search: int,
) -> tuple[np.ndarray, int]:
    """Update only dynamic query fields in a prepared packed buffer."""

    matrix = _normalize_queries(index, queries)
    _validate_search_parameters(index, k=k, ef_search=ef_search)
    packed = np.asarray(target)
    if (
        packed.dtype != np.float32
        or packed.size != LOGICAL_ELEMENTS
        or not packed.flags.c_contiguous
        or not packed.flags.writeable
    ):
        raise ValueError(f"target must be a writable contiguous float32 buffer with {LOGICAL_ELEMENTS} elements")
    chunks = packed.reshape(REPLICAS, CHUNK_ELEMENTS)
    active_count = matrix.shape[0]
    for slot, chunk in enumerate(chunks):
        query = matrix[slot if slot < active_count else 0]
        chunk[:QUERY_HEADER_ELEMENTS] = (
            QUERY_MAGIC,
            FORMAT_VERSION,
            index.vectors.shape[0],
            index.vectors.shape[1],
            index.entry_point,
            index.max_level,
            ef_search,
            k,
        )
        chunk[QUERY_HEADER_ELEMENTS : QUERY_HEADER_ELEMENTS + index.vectors.shape[1]] = query
    return matrix, active_count


def pack_queries(index: HnswIndex, queries: object, *, k: int, ef_search: int) -> tuple[np.ndarray, int]:
    """Pack one to four queries into independent ACT invocation payloads.

    Unused invocations repeat the first query. This keeps every invocation valid
    and retains a byte-for-byte execution check for a single-query submission.
    """

    packed = _make_query_buffer(index)
    _, active_count = _update_query_buffer(packed, index, queries, k=k, ef_search=ef_search)
    return packed, active_count


def pack_query(index: HnswIndex, query: object, *, k: int, ef_search: int) -> np.ndarray:
    packed, _ = pack_queries(index, query, k=k, ef_search=ef_search)
    return packed


def _unpack_chunk(chunk: np.ndarray, *, index: HnswIndex, k: int, ef_search: int) -> PackedResult:
    header = tuple(int(value) for value in chunk[:OUTPUT_HEADER_ELEMENTS])
    expected = (
        OUTPUT_MAGIC,
        0,
        index.vectors.shape[0],
        index.vectors.shape[1],
        index.entry_point,
        index.max_level,
        ef_search,
        k,
    )
    if header[:8] != expected:
        raise ValueError(f"kernel output header {header[:8]!r} does not match {expected!r}")
    if header[15] != k:
        raise ValueError(f"kernel returned {header[15]} results, expected {k}")
    records = chunk[OUTPUT_HEADER_ELEMENTS : OUTPUT_HEADER_ELEMENTS + 2 * k].reshape(k, 2)
    ids = records[:, 0].astype(np.int32)
    distances = records[:, 1].copy()
    if not np.array_equal(records[:, 0], ids.astype(np.float32)):
        raise ValueError("kernel returned a non-integral node id")
    if (ids < 0).any() or (ids >= index.vectors.shape[0]).any() or len(set(ids.tolist())) != k:
        raise ValueError("kernel returned invalid or duplicate node ids")
    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("kernel returned an invalid distance")
    order = list(zip(distances.tolist(), ids.tolist()))
    if order != sorted(order):
        raise ValueError("kernel results are not ordered by distance and node id")
    return PackedResult(
        ids=ids,
        distances=distances,
        stats=SearchStats(
            visited_count=header[8],
            distance_count=header[9],
            upper_hops=header[10],
            base_expansions=header[11],
            terminated_by_bound=bool(header[12]),
            candidate_peak=header[13],
            top_peak=header[14],
        ),
    )


def unpack_results(
    output: object,
    *,
    index: HnswIndex,
    k: int,
    ef_search: int,
    query_count: int,
) -> tuple[PackedResult, ...]:
    flat = np.asarray(output, dtype=np.float32).reshape(-1)
    if flat.size != LOGICAL_ELEMENTS:
        raise ValueError(f"kernel output has {flat.size} elements, expected {LOGICAL_ELEMENTS}")
    if not isinstance(query_count, int) or not 1 <= query_count <= REPLICAS:
        raise ValueError(f"query_count must be in [1, {REPLICAS}]")
    chunks = flat.reshape(REPLICAS, CHUNK_ELEMENTS)
    for replica in range(query_count, REPLICAS):
        if not np.array_equal(
            chunks[0, :OUTPUT_ELEMENTS].view("V4"),
            chunks[replica, :OUTPUT_ELEMENTS].view("V4"),
        ):
            raise ValueError("unused replicated ACT invocations produced different output bytes")
    return tuple(_unpack_chunk(chunks[query], index=index, k=k, ef_search=ef_search) for query in range(query_count))


def unpack_result(output: object, *, index: HnswIndex, k: int, ef_search: int) -> PackedResult:
    return unpack_results(output, index=index, k=k, ef_search=ef_search, query_count=1)[0]
