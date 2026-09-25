from __future__ import annotations

from dataclasses import dataclass
import math
import sys
from pathlib import Path
from typing import Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.training_utils import loss_mf, loss_mlu


APPROACHES = (
    "control",
    "tail",
    "tail_order",
    "tail_order_transfer",
    "tail_order_transfer_normalized",
)
EPSILONS = (0.005,)

CONTROL_NAMES = ("Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml")
TAIL_NAMES = ("HighTail", "Fm", "Uhm", "Fhml", "Uhml", "Uh")
TAIL_ORDER_NAMES = (
    "HighTail",
    "Order",
    "Fm",
    "Uhm",
    "Fhml",
    "Uhml",
    "Uh",
)
TAIL_ORDER_TRANSFER_NAMES = (
    "HighTail",
    "Order",
    "Transfer",
    "Fm",
    "Uhm",
    "FhmlCurriculum",
    "UhmlCurriculum",
    "Uh",
)
TAIL_ORDER_TRANSFER_NORMALIZED_NAMES = (
    "HighTail",
    "Order",
    "TransferNorm",
    "Fm",
    "Uhm",
    "FhmlCurriculum",
    "UhmlCurriculum",
    "Uh",
)


@dataclass(frozen=True)
class HighGuard:
    loss: torch.Tensor
    normalized_high: torch.Tensor
    slack: torch.Tensor
    violation: torch.Tensor
    active_fraction: torch.Tensor
    training_threshold: float
    tail_fraction: float
    tail_count: int


def _batch_totals(admitted: torch.Tensor) -> torch.Tensor:
    if admitted.ndim < 2:
        raise ValueError("admitted flow must have a batch and path dimension")
    return admitted.reshape(admitted.shape[0], -1).sum(dim=1)


def _positive_detached_denominator(value: torch.Tensor, name: str) -> torch.Tensor:
    result = value.detach().reshape(-1)
    if not torch.isfinite(result).all().item() or not (result > 0).all().item():
        raise ValueError(f"{name} must be finite and strictly positive")
    return result


def high_tail_guard(
    admitted_high: torch.Tensor,
    oracle_high: torch.Tensor,
    epsilon: float,
    *,
    tail_fraction: float = 0.20,
) -> HighGuard:
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("epsilon must lie strictly between zero and one")
    if tail_fraction <= 0 or tail_fraction > 1:
        raise ValueError("tail_fraction must lie in (0, 1]")
    oracle = _positive_detached_denominator(oracle_high, "oracle High flow")
    normalized = _batch_totals(admitted_high) / oracle
    if normalized.shape != oracle.shape:
        raise ValueError("oracle and admitted batch sizes differ")
    threshold = 1.0 - epsilon / 2.0
    slack = normalized - threshold
    violation = torch.relu(-slack)
    tail_count = max(1, int(math.ceil(violation.numel() * tail_fraction)))
    loss = torch.topk(violation, k=tail_count, largest=True).values.mean()
    return HighGuard(
        loss=loss,
        normalized_high=normalized,
        slack=slack,
        violation=violation,
        active_fraction=(violation > 0).to(dtype=normalized.dtype).mean(),
        training_threshold=threshold,
        tail_fraction=tail_fraction,
        tail_count=tail_count,
    )


def order_loss(
    admitted_medium: torch.Tensor,
    admitted_low: torch.Tensor,
    oracle_medium_increment: torch.Tensor,
    oracle_low_increment: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    medium_denominator = _positive_detached_denominator(
        oracle_medium_increment, "oracle Medium increment"
    )
    low_denominator = _positive_detached_denominator(
        oracle_low_increment, "oracle Low increment"
    )
    normalized_medium = _batch_totals(admitted_medium) / medium_denominator
    normalized_low = _batch_totals(admitted_low) / low_denominator
    gap = normalized_low - normalized_medium
    violation = torch.relu(gap)
    return violation.mean(), {
        "normalized_medium": normalized_medium,
        "normalized_low": normalized_low,
        "order_gap": gap,
        "order_violation": violation,
    }


def transfer_loss(
    admitted_medium: torch.Tensor,
    admitted_low: torch.Tensor,
    reference_medium: torch.Tensor,
    reference_low: torch.Tensor,
    normalizer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    current_medium = _batch_totals(admitted_medium)
    current_low = _batch_totals(admitted_low)
    baseline_medium = _batch_totals(reference_medium).detach()
    baseline_low = _batch_totals(reference_low).detach()
    # Keep the Medium delta signed so the constraint has a restoring gradient
    # when Medium falls below Phase A.  Rectifying this term would create a
    # dead zone exactly where Medium most needs to be increased.
    medium_gain = current_medium - baseline_medium
    low_loss = torch.relu(baseline_low - current_low)
    raw_deficit = torch.relu(low_loss - medium_gain)
    if normalizer is None:
        deficit = raw_deficit
    else:
        denominator = _positive_detached_denominator(
            normalizer, "transfer normalizer"
        )
        deficit = raw_deficit / denominator
    return deficit.mean(), {
        "medium_gain": medium_gain,
        "low_loss": low_loss,
        "transfer_deficit": deficit,
        "transfer_deficit_absolute": raw_deficit,
    }


def build_objectives(
    approach: str,
    epsilon: float | None,
    *,
    edges_high: torch.Tensor,
    edges_high_medium: torch.Tensor,
    edges_all: torch.Tensor,
    admitted_high: torch.Tensor,
    admitted_medium: torch.Tensor,
    admitted_low: torch.Tensor,
    all_traffic: torch.Tensor,
    opt1: torch.Tensor,
    opt2: torch.Tensor,
    opt3: torch.Tensor,
    opt1_mf: torch.Tensor,
    opt2_mf: torch.Tensor,
    opt3_mf: torch.Tensor,
    reference_medium: torch.Tensor | None = None,
    reference_low: torch.Tensor | None = None,
    curriculum_scale: float = 1.0,
) -> tuple[
    tuple[torch.Tensor, ...],
    tuple[float, ...],
    tuple[str, ...],
    HighGuard | None,
    dict[str, torch.Tensor],
]:
    if approach not in APPROACHES:
        raise ValueError(f"unknown approach: {approach}")
    loss_fh, value_fh = loss_mf(admitted_high, opt1_mf.detach())
    loss_uh, value_uh = loss_mlu(edges_high, opt1.detach())
    loss_fhm, value_fhm = loss_mf(
        admitted_high + admitted_medium, opt2_mf.detach()
    )
    loss_uhm, value_uhm = loss_mlu(edges_high_medium, opt2.detach())
    loss_fhml, value_fhml = loss_mf(all_traffic, opt3_mf.detach())
    loss_uhml, value_uhml = loss_mlu(edges_all, opt3.detach())
    if approach == "control":
        return (
            (loss_fh, loss_uh, loss_fhm, loss_uhm, loss_fhml, loss_uhml),
            (value_fh, value_uh, value_fhm, value_uhm, value_fhml, value_uhml),
            CONTROL_NAMES,
            None,
            {},
        )
    if epsilon is None or float(epsilon) not in EPSILONS:
        raise ValueError(f"tail approaches require one of {EPSILONS}")
    guard = high_tail_guard(admitted_high, opt1_mf, float(epsilon))
    loss_fm, value_fm = loss_mf(
        admitted_medium, (opt2_mf - opt1_mf).detach()
    )
    base_values = (
        float(guard.normalized_high.detach().mean().item()),
        value_fm,
        value_uhm,
        value_fhml,
        value_uhml,
        value_uh,
    )
    if approach == "tail":
        return (
            (guard.loss, loss_fm, loss_uhm, loss_fhml, loss_uhml, loss_uh),
            base_values,
            TAIL_NAMES,
            guard,
            {},
        )
    loss_order, order = order_loss(
        admitted_medium,
        admitted_low,
        opt2_mf - opt1_mf,
        opt3_mf - opt2_mf,
    )
    if approach == "tail_order":
        return (
            (
                guard.loss,
                loss_order,
                loss_fm,
                loss_uhm,
                loss_fhml,
                loss_uhml,
                loss_uh,
            ),
            (
                base_values[0],
                float(loss_order.detach().item()),
                *base_values[1:],
            ),
            TAIL_ORDER_NAMES,
            guard,
            order,
        )
    if reference_medium is None or reference_low is None:
        raise ValueError("transfer approaches require Phase-A reference admissions")
    normalize_transfer = approach == "tail_order_transfer_normalized"
    loss_transfer, transfer = transfer_loss(
        admitted_medium,
        admitted_low,
        reference_medium,
        reference_low,
        normalizer=(opt2_mf - opt1_mf) if normalize_transfer else None,
    )
    scale = float(curriculum_scale)
    aux = {**order, **transfer}
    return (
        (
            guard.loss,
            loss_order,
            loss_transfer,
            loss_fm,
            loss_uhm,
            loss_fhml * scale,
            loss_uhml * scale,
            loss_uh,
        ),
        (
            base_values[0],
            float(loss_order.detach().item()),
            float(loss_transfer.detach().item()),
            base_values[1],
            base_values[2],
            value_fhml * scale,
            value_uhml * scale,
            value_uh,
        ),
        (
            TAIL_ORDER_TRANSFER_NORMALIZED_NAMES
            if normalize_transfer
            else TAIL_ORDER_TRANSFER_NAMES
        ),
        guard,
        aux,
    )


def validate_objective_names(approach: str, names: Sequence[str]) -> None:
    expected = {
        "control": CONTROL_NAMES,
        "tail": TAIL_NAMES,
        "tail_order": TAIL_ORDER_NAMES,
        "tail_order_transfer": TAIL_ORDER_TRANSFER_NAMES,
        "tail_order_transfer_normalized": TAIL_ORDER_TRANSFER_NORMALIZED_NAMES,
    }[approach]
    if tuple(names) != expected:
        raise AssertionError(f"{approach} objectives {tuple(names)} != {expected}")
