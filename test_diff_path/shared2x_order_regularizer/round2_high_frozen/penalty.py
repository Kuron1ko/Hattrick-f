from __future__ import annotations

from dataclasses import dataclass
import math

import torch

PENALTIES = ("full_hinge", "flow_balanced_hinge", "tail_flow_balanced_squared")


@dataclass(frozen=True)
class OrderPenalty:
    loss: torch.Tensor
    medium_norm: torch.Tensor
    low_norm: torch.Tensor
    raw_gap: torch.Tensor
    active_fraction: torch.Tensor
    low_gradient_scale: torch.Tensor


def _positive_detached(values: torch.Tensor, name: str) -> torch.Tensor:
    result = values.detach().reshape(-1)
    if not torch.isfinite(result).all().item():
        raise ValueError(f"{name} contains NaN or Inf")
    if not (result > 0).all().item():
        raise ValueError(f"{name} must be strictly positive")
    return result


def full_bidirectional_hinge(
    admitted_medium: torch.Tensor,
    admitted_low: torch.Tensor,
    medium_oracle_increment: torch.Tensor,
    low_oracle_increment: torch.Tensor,
) -> OrderPenalty:
    """Penalize Low NormFulFill above Medium without detaching either class.

    The oracle increments are labels and therefore detached.  In the active
    region, minimizing this loss raises Medium and lowers Low.  High does not
    appear in this function; round 2 protects it structurally with a frozen
    High policy rather than with stop-gradient inside this loss.
    """

    return order_hinge(
        admitted_medium,
        admitted_low,
        medium_oracle_increment,
        low_oracle_increment,
        kind="full_hinge",
    )


def order_hinge(
    admitted_medium: torch.Tensor,
    admitted_low: torch.Tensor,
    medium_oracle_increment: torch.Tensor,
    low_oracle_increment: torch.Tensor,
    kind: str,
    tail_fraction: float = 0.25,
) -> OrderPenalty:
    """Compute the exact hinge value with selectable, nonzero Low gradient.

    ``flow_balanced_hinge`` has exactly the same forward value as the full
    hinge.  Its Low derivative is multiplied by oracle_low/oracle_medium, so
    the aggregate admitted-flow derivatives on Medium and Low have equal
    magnitudes.  Low is not frozen: its derivative remains nonzero.
    """

    if kind not in PENALTIES:
        raise ValueError(f"Unknown penalty {kind!r}; expected one of {PENALTIES}")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must be in (0, 1]")
    medium_denominator = _positive_detached(
        medium_oracle_increment, "medium oracle increment"
    )
    low_denominator = _positive_detached(low_oracle_increment, "low oracle increment")
    medium_total = admitted_medium.reshape(admitted_medium.shape[0], -1).sum(dim=1)
    low_total = admitted_low.reshape(admitted_low.shape[0], -1).sum(dim=1)
    medium_norm = medium_total / medium_denominator
    low_norm = low_total / low_denominator
    raw_gap = low_norm - medium_norm
    if kind == "full_hinge":
        low_gradient_scale = torch.ones_like(low_norm)
    else:
        low_gradient_scale = (low_denominator / medium_denominator).clamp(max=1.0)
    # The value equals low_norm exactly; only its derivative is rescaled.
    low_surrogate = low_norm.detach() + low_gradient_scale * (low_norm - low_norm.detach())
    surrogate_gap = low_surrogate - medium_norm
    active = raw_gap > 0
    positive_surrogate = torch.relu(surrogate_gap)
    if kind == "tail_flow_balanced_squared":
        terms = positive_surrogate.square()
        k = max(1, int(math.ceil(terms.numel() * tail_fraction)))
        loss = torch.topk(terms, k=k, largest=True, sorted=False).values.mean()
    else:
        loss = positive_surrogate.mean()
    if not torch.isfinite(loss).item():
        raise ValueError("order hinge is not finite")
    return OrderPenalty(
        loss=loss,
        medium_norm=medium_norm,
        low_norm=low_norm,
        raw_gap=raw_gap,
        active_fraction=active.to(dtype=medium_norm.dtype).mean(),
        low_gradient_scale=low_gradient_scale,
    )
