"""Bounded HNSW search for npunlock ACT-SHAVE kernels."""

from .backend import NpuHnsw, SearchResult
from .index import HnswIndex, SearchStats, build_hnsw, load_index, save_index, search_index
from .packing import (
    CHUNK_ELEMENTS,
    LOGICAL_ELEMENTS,
    MAX_EF,
    MAX_K,
    MAX_LEVELS,
    MAX_M,
    MAX_NODES,
    pack_graph,
    pack_query,
    unpack_result,
)

__all__ = [
    "CHUNK_ELEMENTS",
    "HnswIndex",
    "LOGICAL_ELEMENTS",
    "MAX_EF",
    "MAX_K",
    "MAX_LEVELS",
    "MAX_M",
    "MAX_NODES",
    "NpuHnsw",
    "SearchResult",
    "SearchStats",
    "build_hnsw",
    "load_index",
    "pack_graph",
    "pack_query",
    "search_index",
    "save_index",
    "unpack_result",
]
