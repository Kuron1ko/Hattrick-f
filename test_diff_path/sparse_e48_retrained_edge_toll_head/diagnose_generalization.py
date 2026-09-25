from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
SPARSE_DIR = TEST_DIR / "shared2x_sparse_path_cross_attention"
EDGE_DIR = TEST_DIR / "shared2x_edge_toll_head"
STRICT_EVALUATOR = SPARSE_DIR / "evaluate_strict_esm_sequential.py"
SPARSE_RUNNER = SPARSE_DIR / "run_experiment.py"
EDGE_RUNNER = EDGE_DIR / "run_experiment.py"
SPARSE_CHECKPOINT = (
    SPARSE_DIR
    / "checkpoint_selection_validation_only"
    / "archive"
    / "epoch_048.pt"
)
HEAD_CHECKPOINT = THIS_DIR / "artifacts" / "best_edge_toll_head.pt"
OUTPUT_DIR = ROOT / "output" / "analysis" / "sparse_e48_matched_head_shift"
K = 8
SPLITS = {
    "train": (0, 318),
    "safety": (318, 350),
    "validation": (350, 400),
    "test": (400, 500),
}
CLASSES = ("High", "Medium", "Low")
FEATURE_NAMES = (
    "pred_high_load",
    "pred_medium_load",
    "pred_low_load",
    "pred_high_medium_load",
    "pred_total_load",
    "pred_residual_after_high",
    "pred_residual_after_high_medium",
)


class TollStrengthAblation:
    """Diagnostic-only wrapper that changes toll strength, not its ordering."""

    def __init__(
        self,
        head,
        scale: float | tuple[float, float] = 1.0,
        force_gate_one: bool = False,
    ):
        self.head = head
        if isinstance(scale, tuple):
            self.scale = (float(scale[0]), float(scale[1]))
        else:
            self.scale = (float(scale), float(scale))
        self.force_gate_one = bool(force_gate_one)

    def __call__(self, edge_features, base_policies, pte):
        _, tolls, gates = self.head(edge_features, base_policies, pte)
        if self.force_gate_one:
            tolls = tolls / gates.clamp_min(1e-9).unsqueeze(1)
        class_scale = tolls.new_tensor(self.scale).reshape(1, 1, 2)
        tolls = tolls * class_scale
        batch = edge_features.shape[0]
        policies = []
        for class_index, base in enumerate(base_policies):
            path_cost = torch.sparse.mm(
                pte, tolls[:, :, class_index].transpose(0, 1)
            ).transpose(0, 1)
            grouped_base = base.reshape(batch, -1, K)
            grouped_cost = path_cost.reshape(batch, -1, K)
            valid = grouped_base > 0
            logits = torch.log(grouped_base.clamp_min(1e-12)) - grouped_cost
            logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
            routed = torch.softmax(logits, dim=-1) * grouped_base.sum(
                dim=-1, keepdim=True
            )
            policies.append(torch.where(valid, routed, torch.zeros_like(routed)).reshape_as(base))
        return policies, tolls, gates


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    keep = np.isfinite(left) & np.isfinite(right)
    left = left[keep]
    right = right[keep]
    if left.size < 3 or left.std() <= 1e-15 or right.std() <= 1e-15:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def ks_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(np.asarray(left, dtype=np.float64).reshape(-1))
    right = np.sort(np.asarray(right, dtype=np.float64).reshape(-1))
    grid = np.unique(np.concatenate([left, right]))
    left_cdf = np.searchsorted(left, grid, side="right") / len(left)
    right_cdf = np.searchsorted(right, grid, side="right") / len(right)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def rows_index(rows: list[dict]) -> dict[tuple[int, str], dict]:
    return {(int(row["snapshot"]), str(row["class"])): row for row in rows}


def class_summary(rows: list[dict], class_name: str) -> dict:
    selected = [row for row in rows if row["class"] == class_name]
    result = {"class": class_name, "n": len(selected)}
    for field in ("demand", "admitted_traffic", "norm_fulfill"):
        result[field] = describe(np.asarray([row[field] for row in selected]))
    return result


def evaluate_split(runtime, edge, model, props, cache, candidate_cache):
    baseline_rows, _ = runtime.evaluate_cache(
        model, props, cache, adapter=None, batch_size=20
    )
    adapter = edge.TwoPolicyAdapter(int(cache.policies[0].shape[1]))
    candidate_rows, _ = runtime.evaluate_cache(
        model, props, candidate_cache, adapter=adapter, batch_size=20
    )
    return baseline_rows, candidate_rows


def actual_edge_features(cache):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    capacities = cache.capacities.to(dtype=torch.float32).clamp_min(1e-9)
    loads = []
    for policy, demand in zip(cache.policies, cache.tms):
        flow = policy.squeeze(-1).to(torch.float32) * demand.squeeze(-1).to(torch.float32)
        loads.append(torch.sparse.mm(pte.t(), flow.t()).t() / capacities)
    high, medium, low = loads
    return torch.stack(
        [
            high,
            medium,
            low,
            high + medium,
            high + medium + low,
            torch.relu(1.0 - high),
            torch.relu(1.0 - high - medium),
        ],
        dim=-1,
    ).detach()


def paired_class_gain(baseline_rows, candidate_rows, class_name: str) -> dict:
    baseline = rows_index(baseline_rows)
    candidate = rows_index(candidate_rows)
    snapshots = sorted(snapshot for snapshot, name in baseline if name == class_name)
    baseline_values = np.asarray(
        [float(baseline[(snapshot, class_name)]["norm_fulfill"]) for snapshot in snapshots]
    )
    candidate_values = np.asarray(
        [float(candidate[(snapshot, class_name)]["norm_fulfill"]) for snapshot in snapshots]
    )
    delta = np.asarray(
        candidate_values - baseline_values
    )
    grid = np.unique(np.concatenate([baseline_values, candidate_values]))
    base_cdf = np.searchsorted(np.sort(baseline_values), grid, side="right") / len(
        baseline_values
    )
    candidate_cdf = np.searchsorted(
        np.sort(candidate_values), grid, side="right"
    ) / len(candidate_values)
    return {
        "warning": "offline diagnostic uses actual-TM edge features; invalid for deployment inference",
        "baseline_mean": float(baseline_values.mean()),
        "candidate_mean": float(candidate_values.mean()),
        "gain": describe(delta),
        "p1_gain": float(
            np.quantile(candidate_values, 0.01)
            - np.quantile(baseline_values, 0.01)
        ),
        "p10_gain": float(
            np.quantile(candidate_values, 0.10)
            - np.quantile(baseline_values, 0.10)
        ),
        "ecdf_max_upward_violation": float(np.max(candidate_cdf - base_cdf)),
        "positive_fraction": float(np.mean(delta > 1e-7)),
        "negative_count": int(np.sum(delta < -1e-7)),
    }


def split_diagnostics(edge, model, props, cache, candidate_cache, features, head, pte):
    batch = len(cache)
    od_count = int(cache.policies[0].shape[1] // K)
    capacities = cache.capacities.to(torch.float32).clamp_min(1e-9)
    base_policies = [value.squeeze(-1).to(torch.float32) for value in cache.policies]
    candidate_policies = [
        candidate_cache.path_features[:, : cache.policies[1].shape[1]].to(torch.float32),
        candidate_cache.path_features[:, cache.policies[1].shape[1] :].to(torch.float32),
    ]
    with torch.no_grad():
        _, tolls, gates = head(features, base_policies[1:], pte)

    per_snapshot: dict[str, np.ndarray] = {}
    demand_summary: dict[str, dict] = {}
    edge_error_summary: dict[str, dict] = {}
    demand_group_error = 0.0
    actual_edge_loads = []
    for class_index, class_name in enumerate(CLASSES):
        actual_path = cache.tms[class_index].squeeze(-1).to(torch.float32)
        predicted_path = cache.predicted_tms[class_index].squeeze(-1).to(torch.float32)
        actual_group = actual_path.reshape(batch, od_count, K)
        predicted_group = predicted_path.reshape(batch, od_count, K)
        demand_group_error = max(
            demand_group_error,
            float((actual_group - actual_group[:, :, :1]).abs().max().item()),
            float((predicted_group - predicted_group[:, :, :1]).abs().max().item()),
        )
        actual_od = actual_group[:, :, 0]
        predicted_od = predicted_group[:, :, 0]
        actual_total = actual_od.sum(dim=1)
        predicted_total = predicted_od.sum(dim=1)
        aggregate_error = predicted_total - actual_total
        per_snapshot[f"actual_{class_name.lower()}_demand"] = actual_total.cpu().numpy()
        per_snapshot[f"pred_{class_name.lower()}_demand"] = predicted_total.cpu().numpy()
        per_snapshot[f"pred_error_{class_name.lower()}_demand"] = aggregate_error.cpu().numpy()
        per_snapshot[f"pred_abs_error_{class_name.lower()}_demand"] = aggregate_error.abs().cpu().numpy()
        demand_summary[class_name] = {
            "actual_total": describe(actual_total.cpu().numpy()),
            "predicted_total": describe(predicted_total.cpu().numpy()),
            "aggregate_prediction_error": describe(aggregate_error.cpu().numpy()),
            "aggregate_absolute_prediction_error": describe(aggregate_error.abs().cpu().numpy()),
            "od_weighted_nmae": float(
                (predicted_od - actual_od).abs().sum().item()
                / max(actual_od.abs().sum().item(), 1e-12)
            ),
            "od_flat_correlation": correlation(
                actual_od.cpu().numpy(), predicted_od.cpu().numpy()
            ),
        }

        actual_flow = base_policies[class_index] * actual_path
        actual_edge = (
            torch.sparse.mm(pte.t(), actual_flow.t()).t() / capacities
        )
        actual_edge_loads.append(actual_edge)
        predicted_edge = features[:, :, class_index]
        edge_error = predicted_edge - actual_edge
        edge_abs_mean = edge_error.abs().mean(dim=1)
        edge_error_summary[class_name] = {
            "flat_mae": float(edge_error.abs().mean().item()),
            "relative_flat_l1": float(
                edge_error.abs().sum().item()
                / max(actual_edge.abs().sum().item(), 1e-12)
            ),
            "flat_correlation": correlation(
                actual_edge.cpu().numpy(), predicted_edge.cpu().numpy()
            ),
            "per_snapshot_edge_mae": describe(edge_abs_mean.cpu().numpy()),
        }
        per_snapshot[f"edge_pred_mae_{class_name.lower()}"] = edge_abs_mean.cpu().numpy()

    feature_summary: dict[str, dict] = {}
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        channel = features[:, :, feature_index]
        per_snapshot[f"feature_mean_{feature_name}"] = channel.mean(dim=1).cpu().numpy()
        per_snapshot[f"feature_max_{feature_name}"] = channel.amax(dim=1).cpu().numpy()
        feature_summary[feature_name] = {
            "flat": describe(channel.cpu().numpy()),
            "per_snapshot_mean": describe(channel.mean(dim=1).cpu().numpy()),
            "per_snapshot_max": describe(channel.amax(dim=1).cpu().numpy()),
        }

    high = features[:, :, 0]
    high_medium = features[:, :, 3]
    per_snapshot["fraction_edges_pred_high_ge_1"] = (high >= 1.0).float().mean(dim=1).cpu().numpy()
    per_snapshot["fraction_edges_pred_high_medium_ge_1"] = (
        (high_medium >= 1.0).float().mean(dim=1).cpu().numpy()
    )

    # Diagnostic cut-coverage proxy.  An escape path avoids every edge that is
    # already overloaded by the baseline H+M plan.  This is not used by the
    # deployed policy; it measures whether local toll rerouting has a genuinely
    # uncongested alternative available for each Medium OD.
    pte_dense = (pte.to_dense() > 0).to(torch.float32)
    medium_base_group = base_policies[1].reshape(batch, od_count, K)
    medium_candidate_group = candidate_policies[0].reshape(batch, od_count, K)
    medium_valid = medium_base_group > 0
    pred_medium_od = cache.predicted_tms[1].squeeze(-1).reshape(
        batch, od_count, K
    )[:, :, 0]
    actual_medium_od = cache.tms[1].squeeze(-1).reshape(batch, od_count, K)[:, :, 0]
    escape_summary = {}
    with torch.no_grad():
        admitted_ratios = model.simulate(
            cache.policies,
            list(cache.tms),
            cache.capacities,
            edge.pte_info(cache),
            batch,
            props,
            rate_cap=props.rate_cap,
        )[:3]
    admitted_high_path = (
        admitted_ratios[0].reshape(batch, -1)
        * cache.tms[0].squeeze(-1).to(torch.float32)
    )
    admitted_high_edge = torch.sparse.mm(
        pte.transpose(0, 1), admitted_high_path.transpose(0, 1)
    ).transpose(0, 1)
    residual_after_high = (capacities - admitted_high_edge).clamp_min(0.0)
    planned_medium_path = (
        base_policies[1] * cache.tms[1].squeeze(-1).to(torch.float32)
    )
    planned_medium_edge = torch.sparse.mm(
        pte.transpose(0, 1), planned_medium_path.transpose(0, 1)
    ).transpose(0, 1)
    exact_medium_overload = torch.where(
        residual_after_high > 0,
        planned_medium_edge / residual_after_high.clamp_min(1e-12) >= 1.0,
        planned_medium_edge > 0,
    )
    active_sets = {
        "predicted": high_medium >= 1.0,
        "actual": (actual_edge_loads[0] + actual_edge_loads[1]) >= 1.0,
        "exact_actual_residual": exact_medium_overload,
    }
    for source, active_edges in active_sets.items():
        active_edge_count_per_path = torch.matmul(
            active_edges.to(torch.float32), pte_dense.transpose(0, 1)
        ).reshape(batch, od_count, K)
        escape = medium_valid & (active_edge_count_per_path == 0)
        zero_escape = (~escape.any(dim=-1)).to(torch.float32)
        min_active_edges = torch.where(
            medium_valid,
            active_edge_count_per_path,
            torch.full_like(active_edge_count_per_path, 1e9),
        ).amin(dim=-1)
        base_escape_mass = (
            medium_base_group * escape.to(medium_base_group.dtype)
        ).sum(dim=-1) / medium_base_group.sum(dim=-1).clamp_min(1e-12)
        candidate_escape_mass = (
            medium_candidate_group * escape.to(medium_candidate_group.dtype)
        ).sum(dim=-1) / medium_candidate_group.sum(dim=-1).clamp_min(1e-12)

        def demand_weighted(value, demand):
            return (value * demand).sum(dim=1) / demand.sum(dim=1).clamp_min(1e-12)

        pred_zero_escape = demand_weighted(zero_escape, pred_medium_od)
        actual_zero_escape = demand_weighted(zero_escape, actual_medium_od)
        pred_min_active = demand_weighted(min_active_edges, pred_medium_od)
        pred_base_escape_mass = demand_weighted(base_escape_mass, pred_medium_od)
        pred_candidate_escape_mass = demand_weighted(
            candidate_escape_mass, pred_medium_od
        )
        pred_escape_mass_shift = pred_candidate_escape_mass - pred_base_escape_mass
        prefix = f"medium_{source}_active_set"
        per_snapshot[f"{prefix}_zero_escape_pred_weighted"] = (
            pred_zero_escape.cpu().numpy()
        )
        per_snapshot[f"{prefix}_zero_escape_actual_weighted"] = (
            actual_zero_escape.cpu().numpy()
        )
        per_snapshot[f"{prefix}_min_active_edges_pred_weighted"] = (
            pred_min_active.cpu().numpy()
        )
        per_snapshot[f"{prefix}_base_escape_mass_pred_weighted"] = (
            pred_base_escape_mass.cpu().numpy()
        )
        per_snapshot[f"{prefix}_candidate_escape_mass_pred_weighted"] = (
            pred_candidate_escape_mass.cpu().numpy()
        )
        per_snapshot[f"{prefix}_escape_mass_shift_pred_weighted"] = (
            pred_escape_mass_shift.cpu().numpy()
        )
        escape_summary[source] = {
            "zero_escape_od_predicted_demand_weighted": describe(
                pred_zero_escape.cpu().numpy()
            ),
            "zero_escape_od_actual_demand_weighted": describe(
                actual_zero_escape.cpu().numpy()
            ),
            "min_active_edges_predicted_demand_weighted": describe(
                pred_min_active.cpu().numpy()
            ),
            "base_escape_mass_predicted_demand_weighted": describe(
                pred_base_escape_mass.cpu().numpy()
            ),
            "candidate_escape_mass_predicted_demand_weighted": describe(
                pred_candidate_escape_mass.cpu().numpy()
            ),
            "escape_mass_shift_predicted_demand_weighted": describe(
                pred_escape_mass_shift.cpu().numpy()
            ),
        }

    policy_summary: dict[str, dict] = {}
    for offset, class_name in enumerate(("Medium", "Low")):
        base = base_policies[offset + 1].reshape(batch, od_count, K)
        candidate = candidate_policies[offset].reshape(batch, od_count, K)
        base_probability = base / base.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        candidate_probability = candidate / candidate.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        base_entropy = -(
            base_probability * base_probability.clamp_min(1e-12).log()
        ).sum(dim=-1)
        candidate_entropy = -(
            candidate_probability * candidate_probability.clamp_min(1e-12).log()
        ).sum(dim=-1)
        base_effective_paths = base_entropy.exp()
        candidate_effective_paths = candidate_entropy.exp()
        base_pmax = base_probability.amax(dim=-1)
        candidate_pmax = candidate_probability.amax(dim=-1)
        tv = 0.5 * (candidate - base).abs().sum(dim=-1)
        pred_demand = cache.predicted_tms[offset + 1].squeeze(-1).reshape(batch, od_count, K)[:, :, 0]
        actual_demand = cache.tms[offset + 1].squeeze(-1).reshape(batch, od_count, K)[:, :, 0]
        pred_weighted_tv = (tv * pred_demand).sum(dim=1) / pred_demand.sum(dim=1).clamp_min(1e-12)
        actual_weighted_tv = (tv * actual_demand).sum(dim=1) / actual_demand.sum(dim=1).clamp_min(1e-12)
        argmax_changed = (candidate.argmax(dim=-1) != base.argmax(dim=-1)).float()
        pred_weighted_flip = (argmax_changed * pred_demand).sum(dim=1) / pred_demand.sum(dim=1).clamp_min(1e-12)
        pred_weighted_base_effective = (
            base_effective_paths * pred_demand
        ).sum(dim=1) / pred_demand.sum(dim=1).clamp_min(1e-12)
        pred_weighted_candidate_effective = (
            candidate_effective_paths * pred_demand
        ).sum(dim=1) / pred_demand.sum(dim=1).clamp_min(1e-12)
        pred_weighted_base_pmax = (
            base_pmax * pred_demand
        ).sum(dim=1) / pred_demand.sum(dim=1).clamp_min(1e-12)
        pred_weighted_candidate_pmax = (
            candidate_pmax * pred_demand
        ).sum(dim=1) / pred_demand.sum(dim=1).clamp_min(1e-12)
        path_cost = torch.sparse.mm(
            pte, tolls[:, :, offset].transpose(0, 1)
        ).transpose(0, 1).reshape(batch, od_count, K)
        valid = base > 0
        valid_max = torch.where(valid, path_cost, torch.full_like(path_cost, -1e9)).amax(dim=-1)
        valid_min = torch.where(valid, path_cost, torch.full_like(path_cost, 1e9)).amin(dim=-1)
        path_cost_range = valid_max - valid_min
        pred_weighted_path_cost_range = (
            path_cost_range * pred_demand
        ).sum(dim=1) / pred_demand.sum(dim=1).clamp_min(1e-12)
        per_snapshot[f"{class_name.lower()}_policy_tv_pred_weighted"] = pred_weighted_tv.cpu().numpy()
        per_snapshot[f"{class_name.lower()}_policy_tv_actual_weighted"] = actual_weighted_tv.cpu().numpy()
        per_snapshot[f"{class_name.lower()}_policy_argmax_flip_pred_weighted"] = pred_weighted_flip.cpu().numpy()
        per_snapshot[f"{class_name.lower()}_base_effective_paths_pred_weighted"] = pred_weighted_base_effective.cpu().numpy()
        per_snapshot[f"{class_name.lower()}_candidate_effective_paths_pred_weighted"] = pred_weighted_candidate_effective.cpu().numpy()
        per_snapshot[f"{class_name.lower()}_base_pmax_pred_weighted"] = pred_weighted_base_pmax.cpu().numpy()
        per_snapshot[f"{class_name.lower()}_candidate_pmax_pred_weighted"] = pred_weighted_candidate_pmax.cpu().numpy()
        per_snapshot[f"{class_name.lower()}_path_cost_range_pred_weighted"] = pred_weighted_path_cost_range.cpu().numpy()
        policy_summary[class_name] = {
            "tv_unweighted": describe(tv.cpu().numpy()),
            "tv_predicted_demand_weighted": describe(pred_weighted_tv.cpu().numpy()),
            "tv_actual_demand_weighted": describe(actual_weighted_tv.cpu().numpy()),
            "argmax_flip_predicted_demand_weighted": describe(pred_weighted_flip.cpu().numpy()),
            "base_effective_paths_predicted_demand_weighted": describe(pred_weighted_base_effective.cpu().numpy()),
            "candidate_effective_paths_predicted_demand_weighted": describe(pred_weighted_candidate_effective.cpu().numpy()),
            "base_pmax_predicted_demand_weighted": describe(pred_weighted_base_pmax.cpu().numpy()),
            "candidate_pmax_predicted_demand_weighted": describe(pred_weighted_candidate_pmax.cpu().numpy()),
            "path_cost_range_predicted_demand_weighted": describe(pred_weighted_path_cost_range.cpu().numpy()),
            "max_abs_path_probability_delta": float((candidate - base).abs().max().item()),
        }

    for offset, class_name in enumerate(("medium", "low")):
        per_snapshot[f"{class_name}_gate"] = gates[:, offset].cpu().numpy()
        per_snapshot[f"{class_name}_toll_abs_mean"] = tolls[:, :, offset].abs().mean(dim=1).cpu().numpy()
        per_snapshot[f"{class_name}_toll_abs_max"] = tolls[:, :, offset].abs().amax(dim=1).cpu().numpy()

    return {
        "demand_group_repeat_max_error": demand_group_error,
        "demand": demand_summary,
        "actual_vs_predicted_edge_load": edge_error_summary,
        "features": feature_summary,
        "gates": {
            "Medium": describe(gates[:, 0].cpu().numpy()),
            "Low": describe(gates[:, 1].cpu().numpy()),
        },
        "tolls": {
            "Medium_abs": describe(tolls[:, :, 0].abs().cpu().numpy()),
            "Low_abs": describe(tolls[:, :, 1].abs().cpu().numpy()),
        },
        "policy_change": policy_summary,
        "medium_active_set_escape": escape_summary,
        "per_snapshot": per_snapshot,
    }


def attach_outcomes(split_name: str, start: int, stop: int, diagnostics: dict, baseline_rows, candidate_rows):
    base = rows_index(baseline_rows)
    candidate = rows_index(candidate_rows)
    per_snapshot = diagnostics["per_snapshot"]
    medium_base = np.asarray(
        [float(base[(snapshot, "Medium")]["norm_fulfill"]) for snapshot in range(start, stop)]
    )
    medium_candidate = np.asarray(
        [float(candidate[(snapshot, "Medium")]["norm_fulfill"]) for snapshot in range(start, stop)]
    )
    low_base = np.asarray(
        [float(base[(snapshot, "Low")]["norm_fulfill"]) for snapshot in range(start, stop)]
    )
    low_candidate = np.asarray(
        [float(candidate[(snapshot, "Low")]["norm_fulfill"]) for snapshot in range(start, stop)]
    )
    medium_gain = medium_candidate - medium_base
    low_gain = low_candidate - low_base
    headroom = 1.0 - medium_base
    per_snapshot["medium_base_norm"] = medium_base
    per_snapshot["medium_candidate_norm"] = medium_candidate
    per_snapshot["medium_gain"] = medium_gain
    per_snapshot["medium_headroom"] = headroom
    per_snapshot["low_base_norm"] = low_base
    per_snapshot["low_candidate_norm"] = low_candidate
    per_snapshot["low_gain"] = low_gain

    correlations = {}
    for key, values in per_snapshot.items():
        if key in {"medium_gain", "medium_candidate_norm"}:
            continue
        correlations[key] = correlation(medium_gain, values)

    positive_headroom = headroom > 1e-9
    outcome = {
        "baseline": {name: class_summary(baseline_rows, name) for name in CLASSES},
        "candidate": {name: class_summary(candidate_rows, name) for name in CLASSES},
        "medium_gain": describe(medium_gain),
        "medium_gain_positive_fraction": float(np.mean(medium_gain > 1e-7)),
        "medium_gain_negative_count": int(np.sum(medium_gain < -1e-7)),
        "medium_headroom": describe(headroom),
        "medium_gain_over_positive_headroom_mean": float(
            medium_gain[positive_headroom].sum() / headroom[positive_headroom].sum()
        ) if positive_headroom.any() else None,
        "low_gain": describe(low_gain),
        "medium_gain_correlations": correlations,
        "ten_snapshot_blocks": [],
    }
    for block_start in range(start, stop, 10):
        left = block_start - start
        right = min(stop, block_start + 10) - start
        outcome["ten_snapshot_blocks"].append(
            {
                "window": [block_start, min(stop, block_start + 10)],
                "medium_base_mean": float(medium_base[left:right].mean()),
                "medium_candidate_mean": float(medium_candidate[left:right].mean()),
                "medium_gain_mean": float(medium_gain[left:right].mean()),
                "low_gain_mean": float(low_gain[left:right].mean()),
                "actual_medium_demand_mean": float(per_snapshot["actual_medium_demand"][left:right].mean()),
                "pred_medium_demand_mean": float(per_snapshot["pred_medium_demand"][left:right].mean()),
                "medium_gate_mean": float(per_snapshot["medium_gate"][left:right].mean()),
                "medium_policy_tv_pred_weighted_mean": float(
                    per_snapshot["medium_policy_tv_pred_weighted"][left:right].mean()
                ),
                "medium_base_effective_paths_pred_weighted_mean": float(
                    per_snapshot["medium_base_effective_paths_pred_weighted"][left:right].mean()
                ),
                "medium_base_pmax_pred_weighted_mean": float(
                    per_snapshot["medium_base_pmax_pred_weighted"][left:right].mean()
                ),
                "medium_path_cost_range_pred_weighted_mean": float(
                    per_snapshot["medium_path_cost_range_pred_weighted"][left:right].mean()
                ),
            }
        )
    diagnostics["outcome"] = outcome
    diagnostics["split"] = split_name


def drift_report(reference: dict, target: dict) -> dict:
    result = {}
    ref = reference["per_snapshot"]
    tgt = target["per_snapshot"]
    keys = sorted(set(ref).intersection(tgt))
    for key in keys:
        if key in {
            "medium_candidate_norm", "medium_gain", "low_candidate_norm", "low_gain"
        }:
            continue
        left = np.asarray(ref[key], dtype=np.float64)
        right = np.asarray(tgt[key], dtype=np.float64)
        pooled = math.sqrt(max((left.var() + right.var()) / 2.0, 1e-18))
        result[key] = {
            "reference_mean": float(left.mean()),
            "target_mean": float(right.mean()),
            "mean_delta": float(right.mean() - left.mean()),
            "standardized_mean_delta": float((right.mean() - left.mean()) / pooled),
            "ks_distance": ks_distance(left, right),
            "target_outside_reference_range_fraction": float(
                np.mean((right < left.min()) | (right > left.max()))
            ),
        }
    return result


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        result = []
        for row in csv.DictReader(handle):
            result.append(
                {
                    key: value if key == "class" else int(value) if key == "snapshot" else float(value)
                    for key, value in row.items()
                }
            )
        return result


def external_native_comparison() -> dict:
    pairs = {
        "validation": (
            EDGE_DIR / "validation_baseline_rows.csv",
            EDGE_DIR / "validation_candidate_rows.csv",
        ),
        "test": (
            EDGE_DIR / "strict_esm_2x_hattrick_rows.csv",
            EDGE_DIR / "strict_esm_2x_candidate_rows.csv",
        ),
    }
    result = {}
    for split, (base_path, candidate_path) in pairs.items():
        base = rows_index(read_csv_rows(base_path))
        candidate = rows_index(read_csv_rows(candidate_path))
        snapshots = sorted(snapshot for snapshot, name in base if name == "Medium")
        base_values = np.asarray([base[(snapshot, "Medium")]["norm_fulfill"] for snapshot in snapshots])
        candidate_values = np.asarray([candidate[(snapshot, "Medium")]["norm_fulfill"] for snapshot in snapshots])
        gain = candidate_values - base_values
        result[split] = {
            "window": [min(snapshots), max(snapshots) + 1],
            "baseline_medium_mean": float(base_values.mean()),
            "candidate_medium_mean": float(candidate_values.mean()),
            "gain": describe(gain),
            "positive_fraction": float(np.mean(gain > 1e-7)),
            "negative_count": int(np.sum(gain < -1e-7)),
        }
    return result


def main() -> None:
    strict = load_module("matched_head_shift_strict", STRICT_EVALUATOR)
    edge = load_module("matched_head_shift_edge", EDGE_RUNNER)
    sparse = load_module("matched_head_shift_sparse", SPARSE_RUNNER)
    runtime = edge.runtime
    torch.set_num_threads(1)
    device = torch.device("cpu")
    props = runtime.build_props(4, device)

    sparse_payload = torch.load(SPARSE_CHECKPOINT, map_location=device, weights_only=False)
    model = sparse.SparsePathCrossAttentionHattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(sparse_payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    head_payload = torch.load(HEAD_CHECKPOINT, map_location=device, weights_only=False)
    head = edge.EdgeTollHead(
        int(head_payload["edge_count"]), int(head_payload["feature_count"])
    ).to(device)
    head.load_state_dict(head_payload["state_dict"], strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    split_results = {}
    snapshot_rows = []
    for split_name, (start, stop) in SPLITS.items():
        cache, information_audit = strict.build_strict_policy_cache(
            runtime, model, props, start, stop, 20, actual_input_mode="zero"
        )
        features = edge.edge_features(cache)
        candidate_cache, _ = edge.build_candidate_cache(head, cache, features, batch_size=20)
        oracle_features = actual_edge_features(cache)
        oracle_candidate_cache, _ = edge.build_candidate_cache(
            head, cache, oracle_features, batch_size=20
        )
        pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
        diagnostics = split_diagnostics(
            edge, model, props, cache, candidate_cache, features, head, pte
        )
        baseline_rows, candidate_rows = evaluate_split(
            runtime, edge, model, props, cache, candidate_cache
        )
        adapter = edge.TwoPolicyAdapter(int(cache.policies[0].shape[1]))
        oracle_candidate_rows, _ = runtime.evaluate_cache(
            model,
            props,
            oracle_candidate_cache,
            adapter=adapter,
            batch_size=20,
        )
        attach_outcomes(
            split_name, start, stop, diagnostics, baseline_rows, candidate_rows
        )
        diagnostics["information_audit"] = information_audit
        diagnostics["actual_feature_oracle_ablation"] = {
            "feature_max_abs_delta_from_esm": float(
                (oracle_features - features).abs().max().item()
            ),
            "Medium": paired_class_gain(
                baseline_rows, oracle_candidate_rows, "Medium"
            ),
            "Low": paired_class_gain(baseline_rows, oracle_candidate_rows, "Low"),
        }
        if split_name in ("safety", "validation", "test"):
            strength_ablation = {}
            settings = (
                ("scale_0.50", TollStrengthAblation(head, scale=0.50)),
                ("scale_1.25", TollStrengthAblation(head, scale=1.25)),
                ("scale_1.50", TollStrengthAblation(head, scale=1.50)),
                ("scale_2.00", TollStrengthAblation(head, scale=2.00)),
                (
                    "medium_1.25_low_1.00",
                    TollStrengthAblation(head, scale=(1.25, 1.00)),
                ),
                (
                    "medium_1.50_low_1.00",
                    TollStrengthAblation(head, scale=(1.50, 1.00)),
                ),
                (
                    "medium_1.25_low_0.50",
                    TollStrengthAblation(head, scale=(1.25, 0.50)),
                ),
                (
                    "medium_1.50_low_0.50",
                    TollStrengthAblation(head, scale=(1.50, 0.50)),
                ),
                ("gate_forced_one", TollStrengthAblation(head, force_gate_one=True)),
            )
            for label, ablated_head in settings:
                ablated_cache, _ = edge.build_candidate_cache(
                    ablated_head, cache, features, batch_size=20
                )
                ablated_rows, _ = runtime.evaluate_cache(
                    model,
                    props,
                    ablated_cache,
                    adapter=adapter,
                    batch_size=20,
                )
                strength_ablation[label] = {
                    "warning": "offline diagnostic only; not a selected deployment policy",
                    "Medium": paired_class_gain(
                        baseline_rows, ablated_rows, "Medium"
                    ),
                    "Low": paired_class_gain(baseline_rows, ablated_rows, "Low"),
                }
            diagnostics["toll_strength_ablation"] = strength_ablation
        per_snapshot = diagnostics.pop("per_snapshot")
        for offset, snapshot in enumerate(range(start, stop)):
            row = {"split": split_name, "snapshot": snapshot}
            row.update({key: finite_float(values[offset]) for key, values in per_snapshot.items()})
            snapshot_rows.append(row)
        split_results[split_name] = diagnostics

    drift = {
        "validation_to_test": drift_report(
            {"per_snapshot": {k: np.asarray([r[k] for r in snapshot_rows if r["split"] == "validation"], dtype=float)
                              for k in snapshot_rows[0] if k not in ("split", "snapshot")}},
            {"per_snapshot": {k: np.asarray([r[k] for r in snapshot_rows if r["split"] == "test"], dtype=float)
                              for k in snapshot_rows[0] if k not in ("split", "snapshot")}},
        )
    }

    report = {
        "analysis": "sparse-e48 matched edge-toll generalization diagnosis",
        "strict_esm": True,
        "splits": {name: list(bounds) for name, bounds in SPLITS.items()},
        "checkpoint_identity": {
            "sparse_epoch": int(sparse_payload["epoch"]),
            "sparse_sha256": sha256(SPARSE_CHECKPOINT),
            "head_sha256": sha256(HEAD_CHECKPOINT),
            "head_backbone_sha256": head_payload.get("backbone_sha256"),
            "head_test_data_read": head_payload.get("test_data_read"),
        },
        "split_results": split_results,
        "validation_to_test_drift": drift["validation_to_test"],
        "native_hattrick_edge_toll_reference": external_native_comparison(),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "diagnosis.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows_path = OUTPUT_DIR / "per_snapshot.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(snapshot_rows[0]))
        writer.writeheader()
        writer.writerows(snapshot_rows)
    print(json.dumps({
        "report": str(report_path),
        "rows": str(rows_path),
        "validation_medium_gain": split_results["validation"]["outcome"]["medium_gain"]["mean"],
        "test_medium_gain": split_results["test"]["outcome"]["medium_gain"]["mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
