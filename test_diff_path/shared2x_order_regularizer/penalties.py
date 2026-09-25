from __future__ import annotations

import math
from dataclasses import dataclass

import torch


PENALTIES = ("hinge", "directional_hinge", "tail_directional_squared")


@dataclass(frozen=True)
class PenaltyResult:
    loss: torch.Tensor
    medium_norm: torch.Tensor
    low_norm: torch.Tensor
    raw_gap: torch.Tensor
    positive_gap: torch.Tensor
    active_fraction: torch.Tensor


def _per_sample_total(admitted: torch.Tensor) -> torch.Tensor:
    if admitted.ndim < 2:
        raise ValueError(f"Expected admitted traffic with a batch dimension, got {tuple(admitted.shape)}")
    return admitted.reshape(admitted.shape[0], -1).sum(dim=1)


def _detached_positive_oracle(oracle: torch.Tensor, name: str) -> torch.Tensor:
    values = oracle.detach().reshape(-1)
    if not torch.isfinite(values).all().item():
        raise ValueError(f"{name} contains NaN or Inf")
    if not (values > 0).all().item():
        minimum = float(values.min().item())
        raise ValueError(f"{name} must be strictly positive; minimum={minimum}")
    return values


def priority_order_penalty(
    admitted_medium: torch.Tensor,
    admitted_low: torch.Tensor,
    medium_oracle: torch.Tensor,
    low_oracle: torch.Tensor,
    kind: str,
    tail_fraction: float = 0.25,
) -> PenaltyResult:
    """Compute the per-snapshot NormFulFill priority-order penalty.

    Oracle increments are detached labels.  The directional variants also detach
    the Low performance anchor so the regularizer can only raise Medium directly;
    the unchanged total-flow objective remains responsible for preserving Low.
    """

    if kind not in PENALTIES:
        raise ValueError(f"Unknown penalty {kind!r}; expected one of {PENALTIES}")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must be in (0, 1]")

    medium_denominator = _detached_positive_oracle(medium_oracle, "medium oracle increment")
    low_denominator = _detached_positive_oracle(low_oracle, "low oracle increment")
    medium_norm = _per_sample_total(admitted_medium) / medium_denominator
    low_norm = _per_sample_total(admitted_low) / low_denominator

    low_anchor = low_norm if kind == "hinge" else low_norm.detach()
    raw_gap = low_norm - medium_norm
    directional_gap = low_anchor - medium_norm
    positive_gap = torch.relu(raw_gap)

    if kind in ("hinge", "directional_hinge"):
        terms = torch.relu(directional_gap)
        loss = terms.mean()
    else:
        terms = torch.relu(directional_gap).square()
        k = max(1, int(math.ceil(terms.numel() * tail_fraction)))
        loss = torch.topk(terms, k=k, largest=True, sorted=False).values.mean()

    if not torch.isfinite(loss).item():
        raise ValueError("Priority-order penalty is not finite")
    active_fraction = (raw_gap > 0).to(dtype=medium_norm.dtype).mean()
    return PenaltyResult(
        loss=loss,
        medium_norm=medium_norm,
        low_norm=low_norm,
        raw_gap=raw_gap,
        positive_gap=positive_gap,
        active_fraction=active_fraction,
    )
