from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from frameworks.hattrick_system import prediction_gap_edge_util


def test_zero_multiplier_is_original_objective():
    actual = torch.tensor([[0.8, 1.1]], requires_grad=True)
    predicted = torch.tensor([[0.9, 0.7]], requires_grad=True)
    adjusted = prediction_gap_edge_util(actual, predicted, 0.0)
    assert torch.equal(adjusted, actual)


def test_only_esm_underprediction_adds_cost():
    actual = torch.tensor([[0.8, 1.1, 0.9]], requires_grad=True)
    predicted = torch.tensor([[0.9, 0.7, 0.9]], requires_grad=True)
    adjusted = prediction_gap_edge_util(actual, predicted, 0.5)
    torch.testing.assert_close(adjusted, torch.tensor([[0.8, 1.3, 0.9]]))


def test_gap_is_a_detached_edge_cost():
    actual = torch.tensor([[0.8, 1.1]], requires_grad=True)
    predicted = torch.tensor([[0.9, 0.7]], requires_grad=True)
    prediction_gap_edge_util(actual, predicted, 0.5).sum().backward()
    torch.testing.assert_close(actual.grad, torch.ones_like(actual))
    assert predicted.grad is None
