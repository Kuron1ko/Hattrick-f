from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from utils.robust_proj_utils import flatten_grads


@dataclass
class ProjectionResult:
    final_gradient: torch.Tensor
    parameter_shapes: list[torch.Size]
    raw_gradients: list[torch.Tensor]
    projected_gradients: list[torch.Tensor]


def _project_away_from_basis(
    vector: torch.Tensor,
    orthonormal_basis: list[torch.Tensor],
) -> torch.Tensor:
    """Project ``vector`` into the null space of all earlier objectives."""
    projected = vector.to(dtype=torch.float64)
    # A second pass removes float round-off accumulated across objectives.
    for _ in range(2):
        for basis_vector in orthonormal_basis:
            projected = projected - torch.dot(projected, basis_vector) * basis_vector
    return projected


def ordered_project_gradients(
    model: torch.nn.Module,
    losses: Sequence[torch.Tensor],
    *,
    zero_tolerance: float = 1e-12,
) -> ProjectionResult:
    """Generalize Hattrick's ordered projection from four to N objectives.

    Each objective gradient is clipped exactly as in the repository helper,
    then projected against the already-projected higher-priority gradients.
    Zero gradients are skipped without changing the order of the remaining
    objectives.
    """
    if not losses:
        raise ValueError("At least one ordered objective is required")

    raw_gradients: list[torch.Tensor] = []
    shapes: list[torch.Size] | None = None
    for index, loss in enumerate(losses):
        if loss.ndim != 0:
            raise ValueError(f"Objective {index} must be scalar")
        gradient, current_shapes = flatten_grads(
            model,
            loss,
            retain_graph=index < len(losses) - 1,
            zero_grad=index < len(losses) - 1,
        )
        if shapes is None:
            shapes = current_shapes
        raw_gradients.append(gradient.detach().to(dtype=torch.float32))

    assert shapes is not None
    orthonormal_basis: list[torch.Tensor] = []
    projected_gradients: list[torch.Tensor] = []
    for raw in raw_gradients:
        work = raw.to(dtype=torch.float64)
        raw_norm = torch.linalg.vector_norm(work)
        if not torch.isfinite(raw_norm).item():
            raise RuntimeError("Non-finite raw objective gradient")
        if float(raw_norm.item()) <= zero_tolerance:
            projected_gradients.append(torch.zeros_like(raw))
            continue

        projected = _project_away_from_basis(work, orthonormal_basis)
        projected_norm = torch.linalg.vector_norm(projected)
        if not torch.isfinite(projected_norm).item():
            raise RuntimeError("Non-finite projected objective gradient")
        if float(projected_norm.item()) <= zero_tolerance:
            projected_gradients.append(torch.zeros_like(raw))
            continue

        basis_vector = projected / projected_norm
        orthonormal_basis.append(basis_vector)
        projected_gradients.append(projected.to(dtype=raw.dtype))

    final_gradient = torch.stack(projected_gradients, dim=0).sum(dim=0)
    if not torch.isfinite(final_gradient).all().item():
        raise RuntimeError("Non-finite final ordered gradient")
    return ProjectionResult(
        final_gradient=final_gradient,
        parameter_shapes=shapes,
        raw_gradients=raw_gradients,
        projected_gradients=projected_gradients,
    )


def projection_diagnostics(result: ProjectionResult, names: Sequence[str]) -> dict[str, float]:
    if len(names) != len(result.raw_gradients):
        raise ValueError("Objective names and gradients must have the same length")
    diagnostics: dict[str, float] = {}
    for name, raw, projected in zip(names, result.raw_gradients, result.projected_gradients):
        diagnostics[f"raw_gradient_norm_{name}"] = float(torch.linalg.vector_norm(raw).item())
        diagnostics[f"projected_gradient_norm_{name}"] = float(
            torch.linalg.vector_norm(projected).item()
        )
    for lower_index in range(1, len(result.projected_gradients)):
        lower = result.projected_gradients[lower_index].to(dtype=torch.float64)
        for higher_index in range(lower_index):
            higher = result.projected_gradients[higher_index].to(dtype=torch.float64)
            denominator = torch.linalg.vector_norm(lower) * torch.linalg.vector_norm(higher)
            cosine = 0.0
            if float(denominator.item()) > 1e-20:
                cosine = float((torch.dot(lower, higher) / denominator).item())
            diagnostics[f"projected_cosine_{names[lower_index]}_vs_{names[higher_index]}"] = cosine
    return diagnostics
