from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(ROOT))

from ordered_projection import ordered_project_gradients
from utils.robust_proj_utils import project_gradients_one_optimizer_robust


class OrderedProjectionTests(unittest.TestCase):
    def test_six_objectives_are_ordered_and_finite(self):
        parameter = torch.nn.Parameter(torch.tensor([0.7, -0.4, 0.2], dtype=torch.float32))
        model = torch.nn.ParameterList([parameter])
        vectors = (
            torch.tensor([1.0, 1.0, 0.0]),
            torch.tensor([1.0, 0.0, 1.0]),
            torch.tensor([0.0, 1.0, 1.0]),
            torch.tensor([2.0, -1.0, 1.0]),
            torch.tensor([-1.0, 2.0, 1.0]),
            torch.tensor([1.0, 2.0, -1.0]),
        )
        losses = tuple(torch.dot(parameter, vector) for vector in vectors)
        result = ordered_project_gradients(model, losses)
        self.assertTrue(torch.isfinite(result.final_gradient).all().item())
        self.assertTrue(torch.allclose(result.projected_gradients[0], vectors[0]))
        for lower in range(1, len(result.projected_gradients)):
            for higher in range(lower):
                dot = torch.dot(
                    result.projected_gradients[lower],
                    result.projected_gradients[higher],
                )
                self.assertLessEqual(abs(float(dot.item())), 2e-5)

    def test_zero_objective_is_skipped(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float32))
        model = torch.nn.ParameterList([parameter])
        zero = (parameter * 0.0).sum()
        first = parameter[0] + parameter[1]
        second = parameter[0] - parameter[1]
        result = ordered_project_gradients(model, (zero, first, second))
        self.assertTrue(torch.equal(result.projected_gradients[0], torch.zeros(2)))
        self.assertTrue(torch.allclose(result.projected_gradients[1], torch.tensor([1.0, 1.0])))
        self.assertLess(abs(float(torch.dot(
            result.projected_gradients[1], result.projected_gradients[2]
        ).item())), 1e-6)

    def test_four_objective_extension_matches_repository_direction(self):
        parameter = torch.nn.Parameter(torch.tensor([0.2, -0.7, 1.1, 0.3]))
        model = torch.nn.ParameterList([parameter])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        def losses():
            return (
                (parameter.square()).sum(),
                ((parameter - 1.0).square()).sum(),
                torch.sin(parameter).sum(),
                (parameter * torch.tensor([1.0, -2.0, 3.0, 0.5])).sum(),
            )

        repository, *_ = project_gradients_one_optimizer_robust(
            model, *losses(), optimizer
        )
        model.zero_grad(set_to_none=True)
        extended = ordered_project_gradients(model, losses()).final_gradient
        cosine = torch.nn.functional.cosine_similarity(repository, extended, dim=0)
        self.assertGreater(float(cosine.item()), 0.99999)


if __name__ == "__main__":
    unittest.main()
