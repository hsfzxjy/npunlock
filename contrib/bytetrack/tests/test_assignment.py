from __future__ import annotations

import itertools
import importlib.util
import unittest

import numpy as np

from bytetrack_npu.assignment import scipy_assignment
from bytetrack_npu.assignment import CpuLapjvAssignment


def brute_force_objective(costs: np.ndarray, threshold: float, forbidden: np.ndarray) -> float:
    rows, columns = costs.shape
    best = float(rows + columns) * threshold * 0.5
    for count in range(1, min(rows, columns) + 1):
        for selected_rows in itertools.combinations(range(rows), count):
            for selected_columns in itertools.permutations(range(columns), count):
                if any(forbidden[row, column] for row, column in zip(selected_rows, selected_columns)):
                    continue
                match_cost = sum(float(costs[row, column]) for row, column in zip(selected_rows, selected_columns))
                objective = match_cost + (rows + columns - 2 * count) * threshold * 0.5
                best = min(best, objective)
    return best


class AssignmentTests(unittest.TestCase):
    def assert_optimal(self, costs: object, threshold: float, forbidden: object | None = None) -> None:
        matrix = np.asarray(costs, dtype=np.float32)
        mask = np.zeros(matrix.shape, dtype=np.bool_) if forbidden is None else np.asarray(forbidden, dtype=np.bool_)
        effective = mask | ~np.isfinite(matrix) | (matrix > np.float32(threshold))
        result = scipy_assignment(matrix, threshold, forbidden=mask)
        self.assertAlmostEqual(
            result.augmented_cost,
            brute_force_objective(matrix, threshold, effective),
            places=6,
        )

    def test_empty_shapes(self) -> None:
        for shape in ((0, 0), (0, 3), (4, 0)):
            result = scipy_assignment(np.empty(shape, dtype=np.float32), 0.5)
            self.assertEqual(result.matches.shape, (0, 2))
            self.assertEqual(len(result.unmatched_rows), shape[0])
            self.assertEqual(len(result.unmatched_columns), shape[1])

    def test_single_and_rectangular(self) -> None:
        self.assert_optimal([[0.25]], 0.5)
        self.assert_optimal([[0.8]], 0.5)
        self.assert_optimal([[0.2, 0.7, 0.1], [0.3, 0.4, 0.8]], 0.6)
        self.assert_optimal([[0.2, 0.3], [0.9, 0.1], [0.4, 0.8]], 0.6)

    def test_threshold_is_part_of_optimization(self) -> None:
        costs = np.asarray([[0.0, 0.49], [0.49, 2.0]], dtype=np.float32)
        result = scipy_assignment(costs, 0.5)
        self.assertEqual(result.matches.tolist(), [[0, 0]])
        self.assertAlmostEqual(result.augmented_cost, 0.5)
        self.assert_optimal(costs, 0.5)

    def test_forbidden_and_nonfinite_costs(self) -> None:
        costs = np.asarray([[0.1, np.nan, 0.2], [np.inf, 0.1, 0.3]], dtype=np.float32)
        forbidden = np.zeros_like(costs, dtype=np.bool_)
        forbidden[0, 0] = True
        result = scipy_assignment(costs, 0.5, forbidden=forbidden)
        self.assertEqual(result.matches.tolist(), [[0, 2], [1, 1]])
        self.assert_optimal(costs, 0.5, forbidden)

    def test_equal_cost_tie_is_valid_and_deterministic(self) -> None:
        costs = np.full((3, 3), 0.25, dtype=np.float32)
        first = scipy_assignment(costs, 0.5)
        second = scipy_assignment(costs, 0.5)
        self.assertTrue(np.array_equal(first.matches, second.matches))
        self.assertEqual(len(first.matches), 3)
        self.assert_optimal(costs, 0.5)

    def test_dense_ambiguous_and_multiple_augmentations(self) -> None:
        costs = np.asarray(
            [
                [0.10, 0.11, 0.90, 0.90],
                [0.10, 0.90, 0.90, 0.90],
                [0.90, 0.10, 0.11, 0.90],
                [0.90, 0.90, 0.10, 0.11],
            ],
            dtype=np.float32,
        )
        self.assert_optimal(costs, 0.5)

    def test_random_small_matrices(self) -> None:
        generator = np.random.default_rng(20261026)
        for rows, columns in ((2, 3), (3, 2), (3, 3), (4, 3)):
            for _ in range(10):
                costs = generator.uniform(0.0, 1.0, size=(rows, columns)).astype(np.float32)
                forbidden = generator.random((rows, columns)) < 0.15
                self.assert_optimal(costs, 0.65, forbidden)

    def test_maximum_capacity(self) -> None:
        costs = np.full((16, 16), 0.8, dtype=np.float32)
        np.fill_diagonal(costs, np.linspace(0.01, 0.16, 16, dtype=np.float32))
        result = scipy_assignment(costs, 0.5)
        self.assertEqual(result.matches.tolist(), [[index, index] for index in range(16)])

    @unittest.skipUnless(importlib.util.find_spec("lap") is not None, "lap is not installed")
    def test_lapjv_baseline_agrees_with_scipy(self) -> None:
        generator = np.random.default_rng(17)
        backend = CpuLapjvAssignment()
        for shape in ((2, 4), (4, 2), (8, 8), (16, 16)):
            costs = generator.uniform(0.0, 1.0, size=shape).astype(np.float32)
            scipy_result = scipy_assignment(costs, 0.75)
            lap_result = backend.assign(costs, 0.75)
            self.assertAlmostEqual(scipy_result.augmented_cost, lap_result.augmented_cost, places=6)

    def test_rejects_invalid_contracts(self) -> None:
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            scipy_assignment(np.zeros(3, dtype=np.float32), 0.5)
        with self.assertRaisesRegex(ValueError, "capacity"):
            scipy_assignment(np.zeros((17, 1), dtype=np.float32), 0.5)
        with self.assertRaisesRegex(ValueError, "threshold"):
            scipy_assignment(np.zeros((1, 1), dtype=np.float32), np.inf)
        with self.assertRaisesRegex(ValueError, "forbidden mask"):
            scipy_assignment(np.zeros((2, 2), dtype=np.float32), 0.5, forbidden=np.zeros((2, 1)))


if __name__ == "__main__":
    unittest.main()
