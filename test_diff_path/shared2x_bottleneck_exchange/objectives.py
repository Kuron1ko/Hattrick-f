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
    "bottleneck",
    "bottleneck_order",
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
BOTTLENECK_NAMES = (
    "HighTail",
    "TransferNorm",
    "BottleneckRelease",
    "Fm",
    "Uhm",
    "FhmlCurriculum",
    "UhmlCurriculum",
    "Uh",
)
BOTTLENECK_ORDER_NAMES = (
    "HighTail",
    "TransferNorm",
    "Order",
    "BottleneckRelease",
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


def _admitted_link_load(
    admitted: torch.Tensor,
    path_to_edge: torch.Tensor,
) -> torch.Tensor:
    """Replay differentiable admitted path flow onto directed links."""
    batch = admitted.shape[0]
    flattened = admitted.reshape(batch, -1)
    matrix = path_to_edge.coalesce()
    if matrix.ndim != 2 or matrix.shape[0] != flattened.shape[1]:
        raise ValueError("path-to-edge matrix does not match admitted path flow")
    return torch.sparse.mm(matrix.t(), flattened.t()).t()


def bottleneck_release_loss(
    admitted_high: torch.Tensor,
    admitted_medium: torch.Tensor,
    admitted_low: torch.Tensor,
    path_to_edge: torch.Tensor,
    capacities: torch.Tensor,
    oracle_medium_increment: torch.Tensor,
    oracle_low_increment: torch.Tensor,
    *,
    utilization_threshold: float = 0.85,
    multiplier: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Penalize Low only on links that are tight for High+Medium.

    Link criticality and the per-snapshot inversion gate are detached.  The
    network therefore cannot make the penalty disappear by reducing Medium;
    the direct gradient is through Low admission/routing, while the separate
    Fm objective supplies the Medium gradient.  Low remains fully trainable.
    """
    if not 0.0 < utilization_threshold < 1.0:
        raise ValueError("utilization_threshold must lie in (0, 1)")
    if not math.isfinite(float(multiplier)) or float(multiplier) <= 0.0:
        raise ValueError("bottleneck multiplier must be finite and positive")
    medium_denominator = _positive_detached_denominator(
        oracle_medium_increment, "oracle Medium increment"
    )
    low_denominator = _positive_detached_denominator(
        oracle_low_increment, "oracle Low increment"
    )
    normalized_medium = _batch_totals(admitted_medium) / medium_denominator
    normalized_low = _batch_totals(admitted_low) / low_denominator
    inversion_gate = torch.relu(normalized_low - normalized_medium).detach()

    high_load = _admitted_link_load(admitted_high, path_to_edge)
    medium_load = _admitted_link_load(admitted_medium, path_to_edge)
    low_load = _admitted_link_load(admitted_low, path_to_edge)
    capacity = capacities.reshape(capacities.shape[0], -1)
    if capacity.shape != high_load.shape:
        raise ValueError("capacity tensor does not match replayed link loads")
    valid = capacity > 0
    if not valid.any(dim=1).all().item():
        raise ValueError("every snapshot must contain a positive-capacity link")
    safe_capacity = torch.where(valid, capacity, torch.ones_like(capacity))
    high_medium_util = (high_load + medium_load) / safe_capacity
    low_util = low_load / safe_capacity

    # A linear ramp is easier to audit than softmax: 0 below 85% utilization,
    # 1 at saturation.  Detaching prevents weight manipulation via Medium.
    pressure = torch.clamp(
        (high_medium_util.detach() - utilization_threshold)
        / (1.0 - utilization_threshold),
        min=0.0,
        max=1.0,
    ) * valid.to(dtype=high_medium_util.dtype)
    pressure_sum = pressure.sum(dim=1)
    weights = pressure / pressure_sum.clamp_min(1e-12).unsqueeze(1)
    weighted_low_overlap = (weights * low_util).sum(dim=1)
    active_pressure = (pressure_sum > 0).to(dtype=weighted_low_overlap.dtype)
    per_snapshot = inversion_gate * weighted_low_overlap * active_pressure
    loss = per_snapshot.mean() * float(multiplier)
    return loss, {
        "bottleneck_inversion_gate": inversion_gate,
        "bottleneck_high_medium_util_max": torch.where(
            valid, high_medium_util.detach(), torch.zeros_like(high_medium_util)
        ).max(dim=1).values,
        "bottleneck_pressure_link_fraction": (
            (pressure > 0).to(dtype=pressure.dtype).sum(dim=1)
            / valid.to(dtype=pressure.dtype).sum(dim=1)
        ),
        "bottleneck_weighted_low_overlap": weighted_low_overlap,
        "bottleneck_active_snapshot": active_pressure,
        "bottleneck_per_snapshot": per_snapshot,
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
    path_to_edge: torch.Tensor | None = None,
    capacities: torch.Tensor | None = None,
    bottleneck_multiplier: float = 1.0,
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
    normalize_transfer = approach in (
        "tail_order_transfer_normalized",
        "bottleneck",
        "bottleneck_order",
    )
    loss_transfer, transfer = transfer_loss(
        admitted_medium,
        admitted_low,
        reference_medium,
        reference_low,
        normalizer=(opt2_mf - opt1_mf) if normalize_transfer else None,
    )
    scale = float(curriculum_scale)
    aux = {**order, **transfer}
    if approach in ("bottleneck", "bottleneck_order"):
        if path_to_edge is None or capacities is None:
            raise ValueError("bottleneck approaches require link topology and capacities")
        loss_bottleneck, bottleneck = bottleneck_release_loss(
            admitted_high,
            admitted_medium,
            admitted_low,
            path_to_edge,
            capacities,
            opt2_mf - opt1_mf,
            opt3_mf - opt2_mf,
            multiplier=bottleneck_multiplier,
        )
        aux.update(bottleneck)
        common_losses = (
            guard.loss,
            loss_transfer,
            loss_bottleneck,
            loss_fm,
            loss_uhm,
            loss_fhml * scale,
            loss_uhml * scale,
            loss_uh,
        )
        common_values = (
            base_values[0],
            float(loss_transfer.detach().item()),
            float(loss_bottleneck.detach().item()),
            base_values[1],
            base_values[2],
            value_fhml * scale,
            value_uhml * scale,
            value_uh,
        )
        if approach == "bottleneck":
            return (
                common_losses,
                common_values,
                BOTTLENECK_NAMES,
                guard,
                aux,
            )
        return (
            (
                guard.loss,
                loss_transfer,
                loss_order,
                loss_bottleneck,
                loss_fm,
                loss_uhm,
                loss_fhml * scale,
                loss_uhml * scale,
                loss_uh,
            ),
            (
                base_values[0],
                float(loss_transfer.detach().item()),
                float(loss_order.detach().item()),
                float(loss_bottleneck.detach().item()),
                base_values[1],
                base_values[2],
                value_fhml * scale,
                value_uhml * scale,
                value_uh,
            ),
            BOTTLENECK_ORDER_NAMES,
            guard,
            aux,
        )
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
        "bottleneck": BOTTLENECK_NAMES,
        "bottleneck_order": BOTTLENECK_ORDER_NAMES,
    }[approach]
    if tuple(names) != expected:
        raise AssertionError(f"{approach} objectives {tuple(names)} != {expected}")
