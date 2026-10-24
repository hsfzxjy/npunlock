from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class SearchStats:
    visited_count: int
    distance_count: int
    upper_hops: int
    base_expansions: int
    terminated_by_bound: bool
    candidate_peak: int
    top_peak: int


@dataclass(frozen=True, slots=True)
class HnswIndex:
    """A small, explicit HNSW graph suitable for the bounded NPU carrier."""

    vectors: np.ndarray
    levels: tuple[int, ...]
    neighbors: tuple[tuple[tuple[int, ...], ...], ...]
    entry_point: int
    max_level: int
    m: int

    def __post_init__(self) -> None:
        vectors = np.asarray(self.vectors)
        if vectors.dtype != np.float32 or vectors.ndim != 2 or not vectors.flags.c_contiguous:
            raise ValueError("vectors must be a contiguous float32 matrix")
        if vectors.shape[0] == 0 or vectors.shape[1] == 0:
            raise ValueError("vectors must be non-empty")
        if not np.isfinite(vectors).all():
            raise ValueError("vectors must contain only finite values")
        if len(self.levels) != vectors.shape[0] or len(self.neighbors) != vectors.shape[0]:
            raise ValueError("levels and neighbors must describe every vector")
        if not 0 <= self.entry_point < vectors.shape[0]:
            raise ValueError("entry_point is out of range")
        if self.max_level != max(self.levels) or self.levels[self.entry_point] != self.max_level:
            raise ValueError("entry_point must exist at max_level")
        if self.m <= 0:
            raise ValueError("m must be positive")
        for node, (level, layers) in enumerate(zip(self.levels, self.neighbors)):
            if level < 0 or len(layers) != level + 1:
                raise ValueError(f"node {node} has an invalid level layout")
            for layer, links in enumerate(layers):
                if len(links) > self.m or len(set(links)) != len(links):
                    raise ValueError(f"node {node} layer {layer} has invalid degree")
                for neighbor in links:
                    if neighbor == node or not 0 <= neighbor < vectors.shape[0]:
                        raise ValueError(f"node {node} layer {layer} has an invalid neighbor")
                    if self.levels[neighbor] < layer:
                        raise ValueError(f"node {node} layer {layer} links below the neighbor's level")


def _distance(a: np.ndarray, b: np.ndarray) -> np.float32:
    total = np.float32(0.0)
    for left, right in zip(a, b):
        delta = np.float32(left - right)
        total = np.float32(total + np.float32(delta * delta))
    return total


def _search_distance(a: np.ndarray, b: np.ndarray) -> np.float32:
    if a.size == 16:
        delta = np.subtract(a, b, dtype=np.float32)
        squared = np.multiply(delta, delta, dtype=np.float32)
        sums = np.add(
            np.add(squared[0:4], squared[4:8], dtype=np.float32),
            np.add(squared[8:12], squared[12:16], dtype=np.float32),
            dtype=np.float32,
        )
        return np.float32(
            np.float32(sums[0] + sums[1]) + np.float32(sums[2] + sums[3])
        )
    return _distance(a, b)


def _ordered(distance: np.float32, node: int) -> tuple[float, int]:
    return float(distance), node


def _search_layer_build(
    vectors: np.ndarray,
    neighbors: list[list[list[int]]],
    query: np.ndarray,
    entry_points: list[int],
    ef: int,
    layer: int,
) -> list[tuple[np.float32, int]]:
    visited = set(entry_points)
    candidates = sorted((_distance(query, vectors[node]), node) for node in entry_points)
    top = list(candidates)
    while candidates:
        candidate_distance, candidate = candidates.pop(0)
        worst_distance, worst_node = max(top, key=lambda item: _ordered(item[0], item[1]))
        if len(top) >= ef and _ordered(candidate_distance, candidate) > _ordered(worst_distance, worst_node):
            break
        for neighbor in neighbors[candidate][layer]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            distance = _distance(query, vectors[neighbor])
            worst_distance, worst_node = max(top, key=lambda item: _ordered(item[0], item[1]))
            if len(top) < ef or _ordered(distance, neighbor) < _ordered(worst_distance, worst_node):
                candidates.append((distance, neighbor))
                candidates.sort(key=lambda item: _ordered(item[0], item[1]))
                top.append((distance, neighbor))
                if len(top) > ef:
                    top.remove(max(top, key=lambda item: _ordered(item[0], item[1])))
    return sorted(top, key=lambda item: _ordered(item[0], item[1]))


def _prune(vectors: np.ndarray, owner: int, links: list[int], m: int) -> list[int]:
    unique = set(links)
    unique.discard(owner)
    return sorted(unique, key=lambda node: _ordered(_distance(vectors[owner], vectors[node]), node))[:m]


def build_hnsw(
    vectors: object,
    *,
    m: int = 4,
    ef_construction: int = 24,
    seed: int = 1729,
    max_levels: int = 4,
) -> HnswIndex:
    """Construct HNSW with simple-neighbor selection and bounded levels.

    This follows the paper's incremental insertion algorithm: greedy descent
    above a new node's level, efConstruction search on participating layers,
    bidirectional linking, and degree pruning by nearest distance.
    """

    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("vectors must be a non-empty rank-2 matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must contain only finite values")
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    if not isinstance(ef_construction, int) or ef_construction < m:
        raise ValueError("ef_construction must be an integer at least m")
    if not isinstance(max_levels, int) or not 2 <= max_levels <= 16:
        raise ValueError("max_levels must be an integer in [2, 16]")

    rng = np.random.default_rng(seed)
    level_scale = 1.0 / math.log(float(max(2, m)))
    sampled_levels = [
        min(max_levels - 1, int(-math.log(max(float(rng.random()), np.finfo(float).tiny)) * level_scale))
        for _ in range(matrix.shape[0])
    ]
    neighbors: list[list[list[int]]] = []
    entry_point = 0
    current_max_level = sampled_levels[0]
    neighbors.append([[] for _ in range(sampled_levels[0] + 1)])

    for node in range(1, matrix.shape[0]):
        node_level = sampled_levels[node]
        neighbors.append([[] for _ in range(node_level + 1)])
        current = entry_point
        current_distance = _distance(matrix[node], matrix[current])

        for layer in range(current_max_level, node_level, -1):
            changed = True
            while changed:
                changed = False
                for candidate in neighbors[current][layer]:
                    distance = _distance(matrix[node], matrix[candidate])
                    if _ordered(distance, candidate) < _ordered(current_distance, current):
                        current = candidate
                        current_distance = distance
                        changed = True

        for layer in range(min(node_level, current_max_level), -1, -1):
            candidates = _search_layer_build(
                matrix,
                neighbors,
                matrix[node],
                [current],
                ef_construction,
                layer,
            )
            selected = [candidate for _, candidate in candidates[:m]]
            neighbors[node][layer] = selected
            for candidate in selected:
                neighbors[candidate][layer] = _prune(
                    matrix,
                    candidate,
                    neighbors[candidate][layer] + [node],
                    m,
                )
            if candidates:
                current = candidates[0][1]

        if node_level > current_max_level:
            entry_point = node
            current_max_level = node_level

    frozen_neighbors = tuple(tuple(tuple(layer) for layer in node) for node in neighbors)
    return HnswIndex(
        vectors=matrix,
        levels=tuple(sampled_levels),
        neighbors=frozen_neighbors,
        entry_point=entry_point,
        max_level=current_max_level,
        m=m,
    )


def search_index(
    index: HnswIndex,
    query: object,
    *,
    k: int,
    ef_search: int,
) -> tuple[np.ndarray, np.ndarray, SearchStats]:
    """Execute the same hierarchical/best-first search used by the NPU kernel."""

    vector = np.ascontiguousarray(query, dtype=np.float32).reshape(-1)
    if vector.shape != (index.vectors.shape[1],) or not np.isfinite(vector).all():
        raise ValueError(f"query must be a finite vector of dimension {index.vectors.shape[1]}")
    if not isinstance(ef_search, int) or not 1 <= ef_search <= index.vectors.shape[0]:
        raise ValueError("ef_search must be a positive integer no larger than the index size")
    if not isinstance(k, int) or not 0 < k <= ef_search:
        raise ValueError("k must be a positive integer no larger than ef_search")

    distance_count = 1
    upper_hops = 0
    current = index.entry_point
    current_distance = _search_distance(vector, index.vectors[current])
    for layer in range(index.max_level, 0, -1):
        for _ in range(index.vectors.shape[0]):
            best = current
            best_distance = current_distance
            for candidate in index.neighbors[current][layer]:
                distance = _search_distance(vector, index.vectors[candidate])
                distance_count += 1
                if _ordered(distance, candidate) < _ordered(best_distance, best):
                    best = candidate
                    best_distance = distance
            if best == current:
                break
            current = best
            current_distance = best_distance
            upper_hops += 1
        else:
            raise RuntimeError("greedy descent exceeded its node bound")

    visited = {current}
    candidates: list[tuple[np.float32, int]] = [(current_distance, current)]
    top: list[tuple[np.float32, int]] = [(current_distance, current)]
    candidate_peak = 1
    top_peak = 1
    base_expansions = 0
    terminated_by_bound = False
    while candidates:
        candidates.sort(key=lambda item: _ordered(item[0], item[1]))
        candidate_distance, candidate = candidates.pop(0)
        worst_distance, worst_node = max(top, key=lambda item: _ordered(item[0], item[1]))
        if len(top) >= ef_search and _ordered(candidate_distance, candidate) > _ordered(worst_distance, worst_node):
            terminated_by_bound = True
            break
        base_expansions += 1
        for neighbor in index.neighbors[candidate][0]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            distance = _search_distance(vector, index.vectors[neighbor])
            distance_count += 1
            worst_distance, worst_node = max(top, key=lambda item: _ordered(item[0], item[1]))
            if len(top) < ef_search or _ordered(distance, neighbor) < _ordered(worst_distance, worst_node):
                candidates.append((distance, neighbor))
                top.append((distance, neighbor))
                if len(top) > ef_search:
                    top.remove(max(top, key=lambda item: _ordered(item[0], item[1])))
                candidate_peak = max(candidate_peak, len(candidates))
                top_peak = max(top_peak, len(top))

    ordered = sorted(top, key=lambda item: _ordered(item[0], item[1]))[:k]
    return (
        np.asarray([node for _, node in ordered], dtype=np.int32),
        np.asarray([distance for distance, _ in ordered], dtype=np.float32),
        SearchStats(
            visited_count=len(visited),
            distance_count=distance_count,
            upper_hops=upper_hops,
            base_expansions=base_expansions,
            terminated_by_bound=terminated_by_bound,
            candidate_peak=candidate_peak,
            top_peak=top_peak,
        ),
    )


def save_index(index: HnswIndex, path: str | Path) -> None:
    """Serialize the bounded graph without pickle or implementation pointers."""

    max_levels = index.max_level + 1
    adjacency = np.full(
        (index.vectors.shape[0], max_levels, index.m),
        -1,
        dtype=np.int32,
    )
    counts = np.zeros((index.vectors.shape[0], max_levels), dtype=np.int32)
    for node, layers in enumerate(index.neighbors):
        for layer, links in enumerate(layers):
            counts[node, layer] = len(links)
            adjacency[node, layer, : len(links)] = links
    np.savez_compressed(
        Path(path),
        format=np.asarray(["npunlock-hnsw-v1"]),
        vectors=index.vectors,
        levels=np.asarray(index.levels, dtype=np.int32),
        adjacency=adjacency,
        counts=counts,
        entry_point=np.asarray([index.entry_point], dtype=np.int32),
        max_level=np.asarray([index.max_level], dtype=np.int32),
        m=np.asarray([index.m], dtype=np.int32),
    )


def load_index(path: str | Path) -> HnswIndex:
    """Load and fully validate an index written by :func:`save_index`."""

    with np.load(Path(path), allow_pickle=False) as archive:
        required = {
            "format",
            "vectors",
            "levels",
            "adjacency",
            "counts",
            "entry_point",
            "max_level",
            "m",
        }
        if set(archive.files) != required:
            raise ValueError("serialized index has an unexpected field set")
        marker = archive["format"]
        if marker.shape != (1,) or marker[0] != "npunlock-hnsw-v1":
            raise ValueError("serialized index has an unsupported format")
        vectors = np.ascontiguousarray(archive["vectors"], dtype=np.float32)
        levels_array = archive["levels"]
        adjacency = archive["adjacency"]
        counts = archive["counts"]
        scalars = [archive[name] for name in ("entry_point", "max_level", "m")]
        if any(value.dtype != np.int32 or value.shape != (1,) for value in scalars):
            raise ValueError("serialized index scalar metadata is malformed")
        entry_point, max_level, m = (int(value[0]) for value in scalars)
        if levels_array.dtype != np.int32 or levels_array.shape != (vectors.shape[0],):
            raise ValueError("serialized index levels are malformed")
        if (
            adjacency.dtype != np.int32
            or counts.dtype != np.int32
            or adjacency.shape != (vectors.shape[0], max_level + 1, m)
            or counts.shape != (vectors.shape[0], max_level + 1)
        ):
            raise ValueError("serialized index adjacency is malformed")
        levels = tuple(int(value) for value in levels_array)
        neighbors: list[tuple[tuple[int, ...], ...]] = []
        for node, level in enumerate(levels):
            layers: list[tuple[int, ...]] = []
            for layer in range(level + 1):
                count = int(counts[node, layer])
                if not 0 <= count <= m or np.any(adjacency[node, layer, count:] != -1):
                    raise ValueError("serialized index contains invalid adjacency padding")
                layers.append(tuple(int(value) for value in adjacency[node, layer, :count]))
            if np.any(counts[node, level + 1 :] != 0) or np.any(adjacency[node, level + 1 :] != -1):
                raise ValueError("serialized index contains links above a node level")
            neighbors.append(tuple(layers))
    return HnswIndex(
        vectors=vectors,
        levels=levels,
        neighbors=tuple(neighbors),
        entry_point=entry_point,
        max_level=max_level,
        m=m,
    )
