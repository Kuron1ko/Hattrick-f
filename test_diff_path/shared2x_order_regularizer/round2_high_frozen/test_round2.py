from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from hybrid_model import FrozenHighEnsemble, module_state_sha256
from penalty import full_bidirectional_hinge, order_hinge


class DummyPolicy(nn.Module):
    def __init__(self, policies):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(policies, dtype=torch.float32))

    def forward(self, props, *args):
        batch = args[4].shape[0]
        policies = tuple(
            torch.softmax(self.logits[index], dim=0).reshape(1, 2, 1).expand(batch, -1, -1)
            for index in range(3)
        )
        if props.research_return_policy:
            return policies
        raise AssertionError("DummyPolicy is only used through research_return_policy")

    def simulate(self, split_ratios, tms, capacities, pte_info, batch_size, props, rate_cap=False):
        return tuple(item.reshape(batch_size, -1) for item in split_ratios) + (None,)


def _dummy_call(model):
    props = SimpleNamespace(
        research_return_policy=False,
        research_return_admitted=True,
        sim_mf_mlu=0,
        mode="train",
        rate_cap=False,
    )
    batch = 1
    pte = torch.eye(2).to_sparse().coalesce()
    tm = torch.ones(batch, 2, 1)
    return model(
        props,
        torch.zeros(1, 1, 1),
        torch.zeros(2, 1, dtype=torch.long),
        torch.full((1, 2), 100.0),
        torch.zeros(1, 2, 1, dtype=torch.long),
        tm,
        tm,
        tm,
        tm,
        tm,
        tm,
        pte,
        {},
        {},
        None,
    )


class Round2Tests(unittest.TestCase):
    def test_hinge_zero_region(self) -> None:
        medium = torch.tensor([[2.0]], requires_grad=True)
        low = torch.tensor([[1.0]], requires_grad=True)
        result = full_bidirectional_hinge(
            medium, low, torch.tensor([1.0]), torch.tensor([1.0])
        )
        self.assertEqual(result.loss.item(), 0.0)
        result.loss.backward()
        self.assertEqual(medium.grad.item(), 0.0)
        self.assertEqual(low.grad.item(), 0.0)

    def test_hinge_keeps_both_medium_and_low_gradients(self) -> None:
        medium = torch.tensor([[1.0]], requires_grad=True)
        low = torch.tensor([[2.0]], requires_grad=True)
        result = full_bidirectional_hinge(
            medium, low, torch.tensor([1.0]), torch.tensor([1.0])
        )
        result.loss.backward()
        self.assertAlmostEqual(medium.grad.item(), -1.0)
        self.assertAlmostEqual(low.grad.item(), 1.0)

    def test_oracle_is_detached(self) -> None:
        medium = torch.tensor([[1.0]], requires_grad=True)
        low = torch.tensor([[2.0]], requires_grad=True)
        medium_oracle = torch.tensor([1.0], requires_grad=True)
        low_oracle = torch.tensor([1.0], requires_grad=True)
        result = full_bidirectional_hinge(medium, low, medium_oracle, low_oracle)
        result.loss.backward()
        self.assertIsNone(medium_oracle.grad)
        self.assertIsNone(low_oracle.grad)

    def test_flow_balanced_hinge_keeps_low_gradient_nonzero(self) -> None:
        medium = torch.tensor([[1.0]], requires_grad=True)
        low = torch.tensor([[2.0]], requires_grad=True)
        result = order_hinge(
            medium,
            low,
            torch.tensor([10.0]),
            torch.tensor([2.0]),
            kind="flow_balanced_hinge",
        )
        result.loss.backward()
        self.assertAlmostEqual(result.loss.item(), 0.9)
        self.assertAlmostEqual(medium.grad.item(), -0.1)
        self.assertAlmostEqual(low.grad.item(), 0.1)
        self.assertGreater(abs(low.grad.item()), 0.0)

    def test_tail_flow_balanced_squared_uses_top_quartile(self) -> None:
        medium = torch.zeros((4, 1), requires_grad=True)
        low = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=True)
        result = order_hinge(
            medium,
            low,
            torch.ones(4),
            torch.ones(4),
            kind="tail_flow_balanced_squared",
        )
        self.assertAlmostEqual(result.loss.item(), 16.0)
        result.loss.backward()
        self.assertEqual(torch.count_nonzero(low.grad).item(), 1)
        self.assertGreater(low.grad[-1].item(), 0.0)

    def test_hard_frozen_high_policy_is_immutable(self) -> None:
        teacher = DummyPolicy([[3.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        student = DummyPolicy([[0.0, 3.0], [1.0, 0.0], [1.0, 0.0]])
        model = FrozenHighEnsemble(teacher, student)
        teacher_hash = module_state_sha256(model.teacher)
        before = _dummy_call(model)[6].detach().clone()
        optimizer = torch.optim.SGD(model.student.parameters(), lr=0.2)
        output = _dummy_call(model)
        loss = output[7].sum() - output[8].sum()
        loss.backward()
        optimizer.step()
        after = _dummy_call(model)[6].detach().clone()
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(module_state_sha256(model.teacher), teacher_hash)
        model.assert_teacher_immutable()


if __name__ == "__main__":
    unittest.main()
