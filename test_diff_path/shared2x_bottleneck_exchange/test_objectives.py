from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import torch

from objectives import (
    BOTTLENECK_NAMES,
    CONTROL_NAMES,
    TAIL_ORDER_TRANSFER_NAMES,
    build_objectives,
    bottleneck_release_loss,
    high_tail_guard,
    order_loss,
    transfer_loss,
)


class TailOrderTransferTests(unittest.TestCase):
    @staticmethod
    def _path_to_edge() -> torch.Tensor:
        # path 0 -> edge 0; path 1 -> edges 0 and 1
        indices = torch.tensor([[0, 1, 1], [0, 0, 1]])
        values = torch.ones(3)
        return torch.sparse_coo_tensor(indices, values, (2, 2)).coalesce()

    def test_bottleneck_release_targets_low_without_freezing_it(self):
        high = torch.tensor([[4.0, 0.0]], requires_grad=True)
        medium = torch.tensor([[4.0, 0.0]], requires_grad=True)
        low = torch.tensor([[1.0, 2.0]], requires_grad=True)
        loss, values = bottleneck_release_loss(
            high,
            medium,
            low,
            self._path_to_edge(),
            torch.tensor([[10.0, 10.0]], requires_grad=True),
            torch.tensor([[10.0]], requires_grad=True),
            torch.tensor([[2.0]], requires_grad=True),
            utilization_threshold=0.5,
        )
        loss.backward()
        self.assertGreater(float(loss.item()), 0.0)
        self.assertIsNotNone(low.grad)
        self.assertGreater(float(low.grad.abs().sum().item()), 0.0)
        # Criticality and inversion gate are detached: Medium cannot reduce the
        # penalty by making its own bottleneck disappear.
        self.assertIsNone(medium.grad)
        self.assertIsNone(high.grad)
        self.assertGreater(
            float(values["bottleneck_weighted_low_overlap"].item()), 0.0
        )

    def test_bottleneck_release_is_zero_without_inversion(self):
        loss, _ = bottleneck_release_loss(
            torch.tensor([[4.0, 0.0]], requires_grad=True),
            torch.tensor([[8.0, 0.0]], requires_grad=True),
            torch.tensor([[0.1, 0.1]], requires_grad=True),
            self._path_to_edge(),
            torch.tensor([[10.0, 10.0]]),
            torch.tensor([[5.0]]),
            torch.tensor([[5.0]]),
            utilization_threshold=0.5,
        )
        self.assertEqual(float(loss.item()), 0.0)

    def test_bottleneck_build_order_and_multiplier(self):
        kwargs = self._components()
        losses, _, names, _, auxiliary = build_objectives(
            "bottleneck",
            0.005,
            reference_medium=torch.ones_like(kwargs["admitted_medium"]),
            reference_low=torch.ones_like(kwargs["admitted_low"]),
            path_to_edge=self._path_to_edge(),
            capacities=torch.full((2, 2), 4.0),
            bottleneck_multiplier=2.0,
            **kwargs,
        )
        self.assertEqual(names, BOTTLENECK_NAMES)
        self.assertEqual(len(losses), len(names))
        self.assertIn("bottleneck_per_snapshot", auxiliary)

    def test_tail_uses_largest_twenty_percent(self):
        admitted = torch.tensor([[9.0], [8.0], [7.0], [6.0], [5.0]])
        oracle = torch.full((5, 1), 10.0, requires_grad=True)
        guard = high_tail_guard(admitted, oracle, 0.005, tail_fraction=0.20)
        expected = torch.relu(torch.tensor(0.9975 - 0.5))
        self.assertEqual(guard.tail_count, 1)
        self.assertTrue(torch.allclose(guard.loss, expected))
        self.assertFalse(guard.normalized_high.requires_grad)

    def test_order_has_gradients_for_medium_and_low(self):
        medium = torch.tensor([[3.0]], requires_grad=True)
        low = torch.tensor([[4.0]], requires_grad=True)
        medium_oracle = torch.tensor([[5.0]], requires_grad=True)
        low_oracle = torch.tensor([[5.0]], requires_grad=True)
        loss, values = order_loss(medium, low, medium_oracle, low_oracle)
        loss.backward()
        self.assertGreater(float(values["order_violation"].item()), 0.0)
        self.assertLess(float(medium.grad.item()), 0.0)
        self.assertGreater(float(low.grad.item()), 0.0)
        self.assertIsNone(medium_oracle.grad)
        self.assertIsNone(low_oracle.grad)

    def test_transfer_restores_medium_and_does_not_freeze_low(self):
        medium = torch.tensor([[8.0]], requires_grad=True)
        low = torch.tensor([[8.0]], requires_grad=True)
        reference_medium = torch.tensor([[10.0]], requires_grad=True)
        reference_low = torch.tensor([[10.0]], requires_grad=True)
        loss, values = transfer_loss(
            medium, low, reference_medium, reference_low
        )
        loss.backward()
        self.assertAlmostEqual(float(loss.item()), 4.0)
        self.assertLess(float(medium.grad.item()), 0.0)
        self.assertLess(float(low.grad.item()), 0.0)
        self.assertIsNone(reference_medium.grad)
        self.assertIsNone(reference_low.grad)
        self.assertAlmostEqual(float(values["low_loss"].item()), 2.0)

    def test_transfer_zero_when_medium_gain_covers_low_loss(self):
        loss, _ = transfer_loss(
            torch.tensor([[12.0]], requires_grad=True),
            torch.tensor([[9.0]], requires_grad=True),
            torch.tensor([[10.0]]),
            torch.tensor([[10.0]]),
        )
        self.assertEqual(float(loss.item()), 0.0)

    def test_transfer_normalization_is_detached_and_dimensionless(self):
        normalizer = torch.tensor([[4.0]], requires_grad=True)
        loss, values = transfer_loss(
            torch.tensor([[8.0]], requires_grad=True),
            torch.tensor([[8.0]], requires_grad=True),
            torch.tensor([[10.0]]),
            torch.tensor([[10.0]]),
            normalizer=normalizer,
        )
        loss.backward()
        self.assertAlmostEqual(float(values["transfer_deficit_absolute"].item()), 4.0)
        self.assertAlmostEqual(float(loss.item()), 1.0)
        self.assertIsNone(normalizer.grad)

    def test_curriculum_disables_total_objectives_without_changing_names(self):
        kwargs = self._components()
        losses, _, names, _, _ = build_objectives(
            "tail_order_transfer",
            0.005,
            reference_medium=torch.ones_like(kwargs["admitted_medium"]),
            reference_low=torch.ones_like(kwargs["admitted_low"]),
            curriculum_scale=0.0,
            **kwargs,
        )
        self.assertEqual(names, TAIL_ORDER_TRANSFER_NAMES)
        self.assertEqual(float(losses[5].detach().item()), 0.0)
        self.assertEqual(float(losses[6].detach().item()), 0.0)

    def test_control_numerically_matches_previous_experiment(self):
        old_path = (
            Path(__file__).resolve().parent.parent
            / "shared2x_order_epsilon"
            / "objectives.py"
        )
        spec = importlib.util.spec_from_file_location("old_epsilon_objectives", old_path)
        old = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = old
        spec.loader.exec_module(old)
        kwargs = self._components()
        new_losses, new_values, new_names, _, _ = build_objectives(
            "control", None, **kwargs
        )
        old_kwargs = {key: value for key, value in kwargs.items() if key != "admitted_low"}
        old_losses, old_values, old_names, _ = old.build_objectives(
            "control", None, **old_kwargs
        )
        self.assertEqual(new_names, CONTROL_NAMES)
        self.assertEqual(new_names, old_names)
        for left, right in zip(new_losses, old_losses):
            self.assertTrue(torch.allclose(left, right))
        self.assertEqual(new_values, old_values)

    @staticmethod
    def _components() -> dict[str, torch.Tensor]:
        batch = 2
        return {
            "edges_high": torch.tensor([[0.2, 0.3], [0.3, 0.4]], requires_grad=True),
            "edges_high_medium": torch.tensor(
                [[0.4, 0.5], [0.5, 0.6]], requires_grad=True
            ),
            "edges_all": torch.tensor([[0.6, 0.7], [0.7, 0.8]], requires_grad=True),
            "admitted_high": torch.full((batch, 2), 2.0, requires_grad=True),
            "admitted_medium": torch.full((batch, 2), 1.5, requires_grad=True),
            "admitted_low": torch.full((batch, 2), 1.0, requires_grad=True),
            "all_traffic": torch.full((batch, 2), 4.5, requires_grad=True),
            "opt1": torch.ones((batch, 1)),
            "opt2": torch.ones((batch, 1)),
            "opt3": torch.ones((batch, 1)),
            "opt1_mf": torch.full((batch, 1), 4.0),
            "opt2_mf": torch.full((batch, 1), 8.0),
            "opt3_mf": torch.full((batch, 1), 12.0),
        }


if __name__ == "__main__":
    unittest.main()
