from __future__ import annotations

import unittest

import torch

from penalties import priority_order_penalty


class PriorityOrderPenaltyTests(unittest.TestCase):
    def test_hinge_zero_and_positive_regions(self) -> None:
        medium = torch.tensor([[0.8], [0.4]], requires_grad=True)
        low = torch.tensor([[0.4], [0.9]], requires_grad=True)
        oracle = torch.ones(2)
        result = priority_order_penalty(medium, low, oracle, oracle, "hinge")
        self.assertAlmostEqual(float(result.loss), 0.25, places=7)
        self.assertTrue(torch.allclose(result.positive_gap, torch.tensor([0.0, 0.5])))
        self.assertAlmostEqual(float(result.active_fraction), 0.5, places=7)

    def test_oracles_are_detached(self) -> None:
        medium = torch.tensor([[0.4]], requires_grad=True)
        low = torch.tensor([[0.8]], requires_grad=True)
        medium_oracle = torch.tensor([1.0], requires_grad=True)
        low_oracle = torch.tensor([1.0], requires_grad=True)
        result = priority_order_penalty(
            medium, low, medium_oracle, low_oracle, "hinge"
        )
        result.loss.backward()
        self.assertIsNone(medium_oracle.grad)
        self.assertIsNone(low_oracle.grad)
        self.assertIsNotNone(medium.grad)

    def test_directional_hinge_has_no_direct_low_gradient(self) -> None:
        medium = torch.tensor([[0.4]], requires_grad=True)
        low = torch.tensor([[0.8]], requires_grad=True)
        result = priority_order_penalty(
            medium, low, torch.ones(1), torch.ones(1), "directional_hinge"
        )
        medium_grad, low_grad = torch.autograd.grad(
            result.loss, (medium, low), allow_unused=True
        )
        self.assertLess(float(medium_grad), 0.0)
        self.assertIsNone(low_grad)

    def test_tail_squared_uses_largest_quarter(self) -> None:
        medium = torch.zeros((4, 1), requires_grad=True)
        low = torch.tensor([[0.1], [0.2], [0.3], [0.4]], requires_grad=True)
        result = priority_order_penalty(
            medium,
            low,
            torch.ones(4),
            torch.ones(4),
            "tail_directional_squared",
        )
        self.assertAlmostEqual(float(result.loss), 0.16, places=7)

    def test_increasing_medium_is_a_descent_direction(self) -> None:
        medium = torch.tensor([[0.4]], dtype=torch.float64)
        low = torch.tensor([[0.8]], dtype=torch.float64)
        before = priority_order_penalty(
            medium, low, torch.ones(1), torch.ones(1), "hinge"
        ).loss
        after = priority_order_penalty(
            medium + 1e-4, low, torch.ones(1), torch.ones(1), "hinge"
        ).loss
        self.assertLess(float(after), float(before))

    def test_zero_weight_preserves_base_gradient(self) -> None:
        parameter = torch.tensor([0.3, -0.2], requires_grad=True)
        base = (parameter.square()).sum()
        penalty = torch.relu(parameter[0] - parameter[1])
        base_grad = torch.autograd.grad(base, parameter, retain_graph=True)[0]
        combined_grad = torch.autograd.grad(base + 0.0 * penalty, parameter)[0]
        self.assertTrue(torch.equal(base_grad, combined_grad))


if __name__ == "__main__":
    unittest.main()
