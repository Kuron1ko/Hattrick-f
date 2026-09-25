from __future__ import annotations

import unittest

import torch

from objectives import (
    CONTROL_NAMES,
    EPSILON_NAMES,
    SWAP_NAMES,
    build_objectives,
    high_epsilon_guard,
    validate_objective_names,
)


class HighGuardTests(unittest.TestCase):
    def test_positive_and_zero_regions_are_per_snapshot(self):
        admitted = torch.tensor([[9.96, 0.0], [9.99, 0.0]], requires_grad=True)
        oracle = torch.tensor([10.0, 10.0], requires_grad=True)
        result = high_epsilon_guard(admitted, oracle, 0.005)
        self.assertAlmostEqual(result.training_threshold, 0.9975)
        self.assertAlmostEqual(float(result.violation[0]), 0.0015, places=6)
        self.assertEqual(float(result.violation[1]), 0.0)
        self.assertAlmostEqual(float(result.active_fraction), 0.5)

    def test_oracle_is_detached_but_high_has_gradient(self):
        admitted = torch.tensor([[9.90, 0.0]], requires_grad=True)
        oracle = torch.tensor([10.0], requires_grad=True)
        result = high_epsilon_guard(admitted, oracle, 0.01)
        result.loss.backward()
        self.assertIsNotNone(admitted.grad)
        self.assertLess(float(admitted.grad[0, 0]), 0.0)
        self.assertIsNone(oracle.grad)

    def test_invalid_oracle_is_rejected(self):
        with self.assertRaises(ValueError):
            high_epsilon_guard(torch.ones(1, 2), torch.zeros(1), 0.01)

    def test_objective_orders_are_exact(self):
        validate_objective_names("control", CONTROL_NAMES)
        validate_objective_names("swap", SWAP_NAMES)
        validate_objective_names("epsilon", EPSILON_NAMES)
        with self.assertRaises(AssertionError):
            validate_objective_names("swap", CONTROL_NAMES)

    def test_epsilon_uses_medium_increment_and_last_high_mlu(self):
        admitted_high = torch.tensor([[0.99, 0.0]], requires_grad=True)
        admitted_medium = torch.tensor([[0.4, 0.0]], requires_grad=True)
        edge = torch.tensor([[0.8, 0.4]], requires_grad=True)
        losses, reported, names, guard = build_objectives(
            "epsilon",
            0.01,
            edges_high=edge,
            edges_high_medium=edge + 0.1,
            edges_all=edge + 0.2,
            admitted_high=admitted_high,
            admitted_medium=admitted_medium,
            all_traffic=admitted_high + admitted_medium,
            opt1=torch.tensor([1.0]),
            opt2=torch.tensor([1.0]),
            opt3=torch.tensor([1.0]),
            opt1_mf=torch.tensor([1.0]),
            opt2_mf=torch.tensor([1.5]),
            opt3_mf=torch.tensor([2.0]),
        )
        self.assertEqual(names, EPSILON_NAMES)
        self.assertIsNotNone(guard)
        self.assertEqual(len(losses), 6)
        self.assertAlmostEqual(reported[1], 0.8, places=6)


if __name__ == "__main__":
    unittest.main()
