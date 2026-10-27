from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from hnsw_npu import (
    CHUNK_ELEMENTS,
    MAX_BATCH_QUERIES,
    build_hnsw,
    load_index,
    pack_queries,
    save_index,
    search_index,
)


def vectors() -> np.ndarray:
    return np.random.default_rng(739).normal(size=(48, 16)).astype(np.float32)


class HnswTests(unittest.TestCase):
    def test_build_and_search_are_deterministic(self) -> None:
        first = build_hnsw(vectors(), seed=1729)
        second = build_hnsw(vectors(), seed=1729)
        self.assertEqual(first.levels, second.levels)
        self.assertEqual(first.neighbors, second.neighbors)

        ids, distances, stats = search_index(
            first,
            first.vectors[17],
            k=5,
            ef_search=16,
        )
        self.assertEqual(ids[0], 17)
        self.assertEqual(distances[0], np.float32(0.0))
        self.assertEqual(len(set(ids.tolist())), 5)
        self.assertGreater(stats.distance_count, 1)

    def test_query_batch_preserves_query_count(self) -> None:
        index = build_hnsw(vectors(), seed=1729)
        packed, count = pack_queries(
            index,
            index.vectors[:3],
            k=5,
            ef_search=16,
        )
        self.assertEqual(count, 3)
        self.assertEqual(packed.dtype, np.float32)
        self.assertEqual(packed.ndim, 2)

        repetitions = (MAX_BATCH_QUERIES + len(index.vectors) - 1) // len(index.vectors)
        maximum_batch = np.tile(index.vectors, (repetitions, 1))[:MAX_BATCH_QUERIES]
        packed, count = pack_queries(index, maximum_batch, k=5, ef_search=16)
        self.assertEqual(count, MAX_BATCH_QUERIES)
        self.assertEqual(packed.shape, (1, MAX_BATCH_QUERIES * CHUNK_ELEMENTS))

    def test_index_round_trip(self) -> None:
        index = build_hnsw(vectors(), seed=1729)
        path = Path(__file__).resolve().parents[3] / "build" / "hnsw-index-round-trip-test.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            save_index(index, path)
            loaded = load_index(path)
        finally:
            path.unlink(missing_ok=True)
        np.testing.assert_array_equal(loaded.vectors, index.vectors)
        self.assertEqual(loaded.levels, index.levels)
        self.assertEqual(loaded.neighbors, index.neighbors)


if __name__ == "__main__":
    unittest.main()
