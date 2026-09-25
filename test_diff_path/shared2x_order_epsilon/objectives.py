from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.training_utils import loss_mf, loss_mlu


APPROACHES = ("control", "swap", "epsilon")
EPSILONS = (0.0025, 0.005, 0.01)

CONTROL_NAMES = ("Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml")
SWAP_NAMES = ("Fh", "Fhm", "Uh", "Uhm", "Fhml", "Uhml")
EPSILON_NAMES = ("HighGuard", "Fm", "Uhm", "Fhml", "Uhml", "Uh")


@dataclass(frozen=True)
class HighGuard:
    loss: torch.Tensor
    normalized_high: torch.Tensor
    slack: torch.Tensor
    violation: torch.Tensor
    active_fraction: torch.Tensor
    training_threshold: float


def _batch_totals(admitted: torch.Tensor) -> torch.Tensor:
    if admitted.ndim < 2:
        raise ValueError("admitted flow must have a batch and path dimension")
    return admitted.reshape(admitted.shape[0], -1).sum(dim=1)


def high_epsilon_guard(
    admitted_high: torch.Tensor,
    oracle_high: torch.Tensor,
    epsilon: float,
) -> HighGuard:
    """Per-snapshot High guard with half-epsilon training safety margin."""
    if epsilon <= 0 or epsilon >= 1:
        raise ValueError("epsilon must lie strictly between zero and one")
    oracle = oracle_high.detach().reshape(-1)
    if oracle.shape[0] != admitted_high.shape[0]:
        raise ValueError("oracle and admitted batch sizes differ")
    if not torch.isfinite(oracle).all().item() or not (oracle > 0).all().item():
        raise ValueError("oracle High flow must be finite and strictly positive")
    normalized = _batch_totals(admitted_high) / oracle
    threshold = 1.0 - epsilon / 2.0
    slack = normalized - threshold
    violation = torch.relu(-slack)
    return HighGuard(
        loss=violation.mean(),
        normalized_high=normalized,
        slack=slack,
        violation=violation,
        active_fraction=(violation > 0).to(dtype=normalized.dtype).mean(),
        training_threshold=threshold,
    )


def build_objectives(
    approach: str,
    epsilon: float | None,
    *,
    edges_high: torch.Tensor,
    edges_high_medium: torch.Tensor,
    edges_all: torch.Tensor,
    admitted_high: torch.Tensor,
    admitted_medium: torch.Tensor,
    all_traffic: torch.Tensor,
    opt1: torch.Tensor,
    opt2: torch.Tensor,
    opt3: torch.Tensor,
    opt1_mf: torch.Tensor,
    opt2_mf: torch.Tensor,
    opt3_mf: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], tuple[float, ...], tuple[str, ...], HighGuard | None]:
    """Build the exact ordered Phase-B objectives from the experiment spec."""
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
        )
    if approach == "swap":
        return (
            (loss_fh, loss_fhm, loss_uh, loss_uhm, loss_fhml, loss_uhml),
            (value_fh, value_fhm, value_uh, value_uhm, value_fhml, value_uhml),
            SWAP_NAMES,
            None,
        )
    if epsilon is None or float(epsilon) not in EPSILONS:
        raise ValueError(f"epsilon approach requires one of {EPSILONS}")
    guard = high_epsilon_guard(admitted_high, opt1_mf, float(epsilon))
    loss_fm, value_fm = loss_mf(
        admitted_medium, (opt2_mf - opt1_mf).detach()
    )
    return (
        (guard.loss, loss_fm, loss_uhm, loss_fhml, loss_uhml, loss_uh),
        (
            float(guard.normalized_high.detach().mean().item()),
            value_fm,
            value_uhm,
            value_fhml,
            value_uhml,
            value_uh,
        ),
        EPSILON_NAMES,
        guard,
    )


def validate_objective_names(approach: str, names: Sequence[str]) -> None:
    expected = {
        "control": CONTROL_NAMES,
        "swap": SWAP_NAMES,
        "epsilon": EPSILON_NAMES,
    }[approach]
    if tuple(names) != expected:
        raise AssertionError(f"{approach} objectives {tuple(names)} != {expected}")
