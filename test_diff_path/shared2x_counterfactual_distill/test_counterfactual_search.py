from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from counterfactual_search import (
    changed_od_distillation_loss,
    distillation_loss,
    guarded_changed_od_distillation_loss,
    masked_policy,
    search_counterfactual,
    search_counterfactual_batch,
)


class ToySimulator:
    def simulate(
        self, split_ratios, tms, capacities, pte_info, batch_size, props, rate_cap=False
    ):
        high, medium, low = [value.reshape(batch_size, -1) for value in split_ratios]
        # High remains exactly feasible. Moving High from path 0 to path 1
        # opens residual capacity for every Medium path.
        medium_factor = 0.5 + 0.5 * high[:, 1:2]
        return high, medium * medium_factor, low, None


class CounterfactualSearchTests(unittest.TestCase):
    def setUp(self):
        self.props = SimpleNamespace(num_paths_per_pair=2, rate_cap=False)
        indices = torch.tensor([[0, 1], [0, 1]])
        self.pte = torch.sparse_coo_tensor(
            indices, torch.ones(2), (2, 2)
        ).coalesce()
        self.masks = [torch.ones(2, dtype=torch.bool) for _ in range(3)]
        self.tms = [torch.ones((1, 2, 1)) for _ in range(3)]

    def test_search_finds_high_feasible_medium_improvement(self):
        policies = (
            torch.tensor([[[0.9], [0.1]]]),
            torch.tensor([[[0.5], [0.5]]]),
            torch.tensor([[[0.5], [0.5]]]),
        )
        result = search_counterfactual(
            ToySimulator(),
            self.props,
            policies,
            self.masks,
            self.tms,
            torch.ones((1, 2)),
            self.pte,
            torch.tensor([[1.0]]),
            rounds=4,
            step_grid=(2.0, 1.0, 0.5),
        )
        self.assertGreater(result.accepted_rounds, 0)
        self.assertGreater(
            float(result.target.totals[0, 1]),
            float(result.initial.totals[0, 1]),
        )
        self.assertGreaterEqual(float(result.target.totals[0, 0]), result.high_floor)
        self.assertGreater(result.policy_l1, 0.0)

    def test_masked_policy_never_uses_disabled_path(self):
        result = masked_policy(
            torch.tensor([[0.0, 100.0]]),
            torch.tensor([True, False]),
            2,
        )
        self.assertEqual(float(result[0, 1, 0]), 0.0)
        self.assertAlmostEqual(float(result[0, 0, 0]), 1.0)

    def test_batch_search_keeps_acceptance_independent(self):
        policies = (
            torch.tensor([[[0.9], [0.1]], [[0.8], [0.2]]]),
            torch.tensor([[[0.5], [0.5]], [[0.5], [0.5]]]),
            torch.tensor([[[0.5], [0.5]], [[0.5], [0.5]]]),
        )
        results = search_counterfactual_batch(
            ToySimulator(),
            self.props,
            policies,
            self.masks,
            [torch.ones((2, 2, 1)) for _ in range(3)],
            torch.ones((2, 2)),
            self.pte,
            torch.ones((2, 1)),
            coordinate_rounds=2,
            coordinate_candidate_limit=8,
        )
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertGreater(float(result.target.totals[0, 1]), float(result.initial.totals[0, 1]))
            self.assertGreaterEqual(float(result.target.totals[0, 0]), result.high_floor)

    def test_distillation_detaches_teacher_and_trains_prediction(self):
        prediction = [
            torch.tensor([[[0.8], [0.2]]], requires_grad=True)
            for _ in range(3)
        ]
        teacher = [
            torch.tensor([[[0.2], [0.8]]], requires_grad=True)
            for _ in range(3)
        ]
        loss = distillation_loss(prediction, teacher, self.masks)
        loss.backward()
        self.assertGreater(float(loss.item()), 0.0)
        self.assertTrue(all(value.grad is not None for value in prediction))
        self.assertTrue(all(value.grad is None for value in teacher))

    def test_changed_od_loss_ignores_unchanged_pairs(self):
        prediction = [
            torch.tensor(
                [[[0.7], [0.3], [0.1], [0.9]]], requires_grad=True
            )
            for _ in range(3)
        ]
        source = [
            torch.tensor([[[0.8], [0.2], [0.6], [0.4]]]) for _ in range(3)
        ]
        teacher = [
            torch.tensor([[[0.2], [0.8], [0.6], [0.4]]], requires_grad=True)
            for _ in range(3)
        ]
        masks = [torch.ones(4, dtype=torch.bool) for _ in range(3)]
        loss = changed_od_distillation_loss(
            prediction, teacher, source, masks, k=2
        )
        loss.backward()
        self.assertGreater(float(loss.item()), 0.0)
        self.assertTrue(all(torch.count_nonzero(value.grad[:, 2:]) == 0 for value in prediction))
        self.assertTrue(all(value.grad is None for value in teacher))

    def test_guarded_loss_anchors_unchanged_high_and_low(self):
        prediction = [
            torch.tensor([[[0.7], [0.3], [0.1], [0.9]]], requires_grad=True)
            for _ in range(3)
        ]
        source = [
            torch.tensor([[[0.8], [0.2], [0.6], [0.4]]]) for _ in range(3)
        ]
        teacher = [value.clone().requires_grad_(True) for value in source]
        masks = [torch.ones(4, dtype=torch.bool) for _ in range(3)]
        loss = guarded_changed_od_distillation_loss(
            prediction, teacher, source, masks, k=2
        )
        loss.backward()
        self.assertGreater(torch.count_nonzero(prediction[0].grad).item(), 0)
        self.assertEqual(torch.count_nonzero(prediction[1].grad).item(), 0)
        self.assertGreater(torch.count_nonzero(prediction[2].grad).item(), 0)
        self.assertTrue(all(value.grad is None for value in teacher))


if __name__ == "__main__":
    unittest.main()
