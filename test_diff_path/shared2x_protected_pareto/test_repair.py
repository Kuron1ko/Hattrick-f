from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from repair import (
    solve_medium_low_pareto,
    solve_medium_low_per_od_pareto,
    solve_medium_with_edge_reservation,
)


class ProtectedParetoRepairTests(unittest.TestCase):
    def test_reserves_low_and_maximizes_medium(self):
        # Two OD pairs, two paths each, and two independent edges.
        path_to_edge = csr_matrix(
            np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )
        )
        demand = np.asarray([4.0, 4.0, 3.0, 3.0])
        policy, diagnostics = solve_medium_with_edge_reservation(
            path_to_edge,
            demand,
            np.ones(4, dtype=bool),
            np.asarray([5.0, 5.0]),
            np.asarray([1.0, 2.0]),
            2,
            1.0,
        )
        load = np.asarray(path_to_edge.T @ (policy * demand)).reshape(-1)
        self.assertTrue(np.all(load <= np.asarray([4.0, 3.0]) + 1e-6))
        self.assertTrue(np.all(policy.reshape(2, 2).sum(axis=1) <= 1.0 + 1e-7))
        self.assertAlmostEqual(float((policy * demand).sum()), 7.0, places=5)
        self.assertEqual(diagnostics["status"], 0)

    def test_mask_and_reserve_factor(self):
        path_to_edge = csr_matrix(np.eye(2, dtype=np.float64))
        policy, _ = solve_medium_with_edge_reservation(
            path_to_edge,
            np.asarray([2.0, 2.0]),
            np.asarray([True, False]),
            np.asarray([3.0, 3.0]),
            np.asarray([1.0, 1.0]),
            2,
            1.5,
        )
        self.assertEqual(float(policy[1]), 0.0)
        self.assertLessEqual(float(policy[0] * 2.0), 1.5 + 1e-6)

    def test_joint_pareto_keeps_low_floor(self):
        path_to_edge = csr_matrix(
            np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )
        )
        medium, low, diagnostics = solve_medium_low_pareto(
            path_to_edge,
            np.asarray([4.0, 4.0, 3.0, 3.0]),
            np.asarray([2.0, 2.0, 1.0, 1.0]),
            np.ones(4, dtype=bool),
            np.ones(4, dtype=bool),
            np.asarray([5.0, 5.0]),
            baseline_low_admitted=2.5,
            paths_per_pair=2,
            low_floor_factor=1.0,
        )
        medium_total = float(np.dot(medium, np.asarray([4.0, 4.0, 3.0, 3.0])))
        low_total = float(np.dot(low, np.asarray([2.0, 2.0, 1.0, 1.0])))
        load = np.asarray(
            path_to_edge.T
            @ (
                medium * np.asarray([4.0, 4.0, 3.0, 3.0])
                + low * np.asarray([2.0, 2.0, 1.0, 1.0])
            )
        ).reshape(-1)
        self.assertGreaterEqual(low_total, 2.5 - 2e-6)
        self.assertGreaterEqual(medium_total, 7.0 - 2e-6)
        self.assertTrue(np.all(load <= 5.0 + 1e-6))
        self.assertEqual(diagnostics["status"], 0)

    def test_per_od_floor_preserves_each_pair(self):
        path_to_edge = csr_matrix(
            np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )
        )
        low_baseline = np.asarray([0.8, 0.2, 0.3, 0.7])
        medium, low, diagnostics = solve_medium_low_per_od_pareto(
            path_to_edge,
            np.asarray([4.0, 4.0, 3.0, 3.0]),
            np.asarray([2.0, 2.0, 1.0, 1.0]),
            low_baseline,
            np.ones(4, dtype=bool),
            np.ones(4, dtype=bool),
            np.asarray([5.0, 5.0]),
            paths_per_pair=2,
        )
        low_flow = (low * np.asarray([2.0, 2.0, 1.0, 1.0])).reshape(2, 2).sum(axis=1)
        baseline_pair = low_baseline.reshape(2, 2).sum(axis=1)
        self.assertTrue(np.all(low_flow >= baseline_pair - 2e-6))
        self.assertGreater(float((medium * np.asarray([4.0, 4.0, 3.0, 3.0])).sum()), 0.0)
        self.assertGreaterEqual(diagnostics["minimum_low_pair_floor_slack"], -1e-6)


if __name__ == "__main__":
    unittest.main()
