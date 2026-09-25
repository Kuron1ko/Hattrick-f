from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
ROOT = TEST_DIR.parent
RUNTIME_DIR = TEST_DIR / "shared2x_medium_adapter"
CORRECTION_PATH = (
    TEST_DIR / "shared2x_active_set_router" / "probe_esm_self_correction.py"
)
for item in (str(ROOT), str(TEST_DIR), str(RUNTIME_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

spec = importlib.util.spec_from_file_location(
    "teacher_toll_correction_runtime", CORRECTION_PATH
)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {CORRECTION_PATH}")
correction = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = correction
spec.loader.exec_module(correction)
runtime = correction.runtime


K = 8
TEACHER_CONFIG = (24, 0.06, 0.25, 0.01)


class ProjectedPolicyAdapter:
    def __init__(self, path_count: int):
        self.path_count = int(path_count)

    def adapt_batch(self, policies, batch):
        projected = batch["path_features"]
        policies[1] = projected[:, : self.path_count].unsqueeze(-1)
        policies[2] = projected[:, self.path_count :].unsqueeze(-1)
        return policies


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def fit_edge_tolls(
    base: torch.Tensor,
    target: torch.Tensor,
    pte: torch.Tensor,
    ridge: float,
    demand: torch.Tensor,
    weighting: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Project an OD-centered log policy change onto path-summed edge tolls."""
    batch, path_count = base.shape
    base_grouped = base.reshape(batch, -1, K)
    target_grouped = target.reshape(batch, -1, K)
    valid = base_grouped > 1e-10
    log_ratio = torch.log(target_grouped.clamp_min(1e-12)) - torch.log(
        base_grouped.clamp_min(1e-12)
    )
    count = valid.sum(dim=2, keepdim=True).clamp_min(1)
    mean = (log_ratio * valid).sum(dim=2, keepdim=True) / count
    centered_ratio = torch.where(valid, log_ratio - mean, torch.zeros_like(log_ratio))
    response = centered_ratio.reshape(batch, path_count).to(dtype=torch.float64)
    design = pte.to(dtype=torch.float64)
    edge_count = design.shape[1]
    if weighting == "uniform":
        centered_design = design.reshape(-1, K, edge_count)
        centered_design = centered_design - centered_design.mean(dim=1, keepdim=True)
        centered_design = centered_design.reshape_as(design)
        gram = centered_design.transpose(0, 1) @ centered_design
        gram = gram + float(ridge) * torch.eye(
            edge_count, device=gram.device, dtype=gram.dtype
        )
        projection = torch.linalg.solve(gram, centered_design.transpose(0, 1))
        tolls = -(response @ projection.transpose(0, 1))
    elif weighting in {"target_route", "target_flow"}:
        design_grouped = design.reshape(-1, K, edge_count)
        target_route = target_grouped / target_grouped.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-12)
        toll_rows = []
        for sample in range(batch):
            weights = target_route[sample].to(dtype=torch.float64)
            if weighting == "target_flow":
                od_demand = demand[sample].reshape(-1, K)[:, :1].to(
                    dtype=torch.float64
                )
                weights = weights * od_demand
            weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            design_mean = (
                weights.unsqueeze(-1) * design_grouped
            ).sum(dim=1) / weight_sum
            centered_design = design_grouped - design_mean.unsqueeze(1)
            response_grouped = response[sample].reshape(-1, K)
            response_mean = (weights * response_grouped).sum(dim=1, keepdim=True) / weight_sum
            centered_response = response_grouped - response_mean
            square_root_weight = weights.clamp_min(0.0).sqrt()
            weighted_design = (
                centered_design * square_root_weight.unsqueeze(-1)
            ).reshape(path_count, edge_count)
            weighted_response = (
                centered_response * square_root_weight
            ).reshape(path_count)
            gram = weighted_design.transpose(0, 1) @ weighted_design
            gram = gram + float(ridge) * torch.eye(
                edge_count, device=gram.device, dtype=gram.dtype
            )
            rhs = weighted_design.transpose(0, 1) @ weighted_response
            toll_rows.append(-torch.linalg.solve(gram, rhs))
        tolls = torch.stack(toll_rows)
    else:
        raise ValueError(f"Unknown weighting: {weighting}")
    path_cost = tolls @ design.transpose(0, 1)
    logits = torch.log(base_grouped.clamp_min(1e-12)).to(dtype=torch.float64)
    logits = logits - path_cost.reshape_as(logits)
    logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
    reconstructed = torch.softmax(logits, dim=2) * base_grouped.sum(
        dim=2, keepdim=True
    ).to(dtype=torch.float64)
    reconstructed = torch.where(valid, reconstructed, torch.zeros_like(reconstructed))
    reconstructed = reconstructed.reshape(batch, path_count).to(dtype=base.dtype)

    target_route = target_grouped / target_grouped.sum(dim=2, keepdim=True).clamp_min(
        1e-12
    )
    reconstructed_grouped = reconstructed.reshape_as(target_grouped)
    reconstructed_route = reconstructed_grouped / reconstructed_grouped.sum(
        dim=2, keepdim=True
    ).clamp_min(1e-12)
    kl = (
        target_route.clamp_min(1e-12)
        * (
            target_route.clamp_min(1e-12).log()
            - reconstructed_route.clamp_min(1e-12).log()
        )
    ).sum(dim=2)
    stats = {
        "policy_mae": float((reconstructed - target).abs().mean().item()),
        "policy_max_abs": float((reconstructed - target).abs().max().item()),
        "route_kl_mean": float(kl.mean().item()),
        "toll_abs_mean": float(tolls.abs().mean().item()),
        "toll_abs_max": float(tolls.abs().max().item()),
    }
    return reconstructed, tolls.to(dtype=torch.float32), stats


def dense_path_edge(cache) -> torch.Tensor:
    return cache.dataset.pte.to_dense().to(device=cache.capacities.device)


def summary_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def compact(summary: list[dict]) -> dict[str, dict[str, float]]:
    return {
        class_name: {
            metric: float(row[metric])
            for metric in (
                "norm_fulfill_mean",
                "norm_fulfill_p1",
                "norm_fulfill_p10",
            )
        }
        for class_name, row in summary_index(summary).items()
    }


def gaps(candidate: list[dict], baseline: list[dict]) -> dict[str, float]:
    c = summary_index(candidate)
    b = summary_index(baseline)
    return {
        f"{class_name}.{metric}": float(c[class_name][metric] - b[class_name][metric])
        for class_name in ("High", "Medium", "Low")
        for metric in (
            "norm_fulfill_mean",
            "norm_fulfill_p1",
            "norm_fulfill_p10",
        )
    }


def project_cache(model, props, cache, ridge: float, weighting: str):
    teacher = correction.correct_cache(model, props, cache, *TEACHER_CONFIG)
    path_count = int(cache.policies[0].shape[1])
    teacher_medium = teacher.path_features[:, :path_count]
    teacher_low = teacher.path_features[:, path_count:]
    design = dense_path_edge(cache)
    medium, medium_tolls, medium_stats = fit_edge_tolls(
        cache.policies[1].squeeze(-1),
        teacher_medium,
        design,
        ridge,
        cache.predicted_tms[1].squeeze(-1),
        weighting,
    )
    low, low_tolls, low_stats = fit_edge_tolls(
        cache.policies[2].squeeze(-1),
        teacher_low,
        design,
        ridge,
        cache.predicted_tms[2].squeeze(-1),
        weighting,
    )
    projected = replace(cache, path_features=torch.cat([medium, low], dim=1))
    return teacher, projected, {
        "medium": medium_stats,
        "low": low_stats,
        "medium_tolls": medium_tolls,
        "low_tolls": low_tolls,
    }


def evaluate(model, props, cache, adapted=None):
    adapter = None
    if adapted is not None:
        adapter = ProjectedPolicyAdapter(int(cache.policies[0].shape[1]))
    _, summary = runtime.evaluate_cache(
        model, props, adapted if adapted is not None else cache, adapter, batch_size=16
    )
    return summary


def run_split(model, props, start: int, end: int, ridge: float, weighting: str) -> dict:
    cache = runtime.build_policy_cache(model, props, start, end, batch_size=16)
    baseline = evaluate(model, props, cache)
    started = time.perf_counter()
    teacher, projected, projection = project_cache(
        model, props, cache, ridge, weighting
    )
    teacher_summary = evaluate(model, props, cache, teacher)
    projected_summary = evaluate(model, props, cache, projected)
    result = {
        "range": [start, end],
        "ridge": ridge,
        "weighting": weighting,
        "seconds": time.perf_counter() - started,
        "baseline": compact(baseline),
        "teacher": compact(teacher_summary),
        "projected": compact(projected_summary),
        "teacher_minus_baseline": gaps(teacher_summary, baseline),
        "projected_minus_baseline": gaps(projected_summary, baseline),
        "projected_minus_teacher": gaps(projected_summary, teacher_summary),
        "projection_stats": {
            "medium": projection["medium"],
            "low": projection["low"],
        },
    }
    return result, projection


def safe(result: dict) -> bool:
    delta = result["projected_minus_baseline"]
    return (
        result["projected"]["High"]["norm_fulfill_mean"] >= 0.995
        and delta["Medium.norm_fulfill_mean"] > 0.0
        and delta["Medium.norm_fulfill_p1"] > 0.0
        and delta["Medium.norm_fulfill_p10"] > 0.0
        and delta["Low.norm_fulfill_mean"] >= -0.003
        and delta["Low.norm_fulfill_p10"] >= -0.01
    )


def main() -> None:
    torch.manual_seed(20260822)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(3, device)
    model, checkpoint = runtime.load_backbone(3, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    small_results = []
    for weighting, ridge in (
        ("uniform", 1e-3),
        ("target_route", 1e-4),
        ("target_route", 1e-3),
        ("target_route", 1e-2),
        ("target_flow", 1e-4),
        ("target_flow", 1e-3),
        ("target_flow", 1e-2),
    ):
        result, _ = run_split(model, props, 350, 358, ridge, weighting)
        result["safe"] = safe(result)
        small_results.append(result)
        print(
            f"[small] weighting={weighting} ridge={ridge:g} safe={result['safe']} "
            f"medium={result['projected_minus_baseline']['Medium.norm_fulfill_mean']:+.6f}",
            flush=True,
        )
    eligible = [row for row in small_results if row["safe"]]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: min(
                row["projected_minus_baseline"]["Medium.norm_fulfill_mean"],
                row["projected_minus_baseline"]["Medium.norm_fulfill_p1"],
                row["projected_minus_baseline"]["Medium.norm_fulfill_p10"],
            ),
        )
        selected_ridge = float(selected["ridge"])
        selected_weighting = str(selected["weighting"])
        holdout, _ = run_split(
            model, props, 358, 400, selected_ridge, selected_weighting
        )
        holdout["safe"] = safe(holdout)
    else:
        selected_ridge = None
        selected_weighting = None
        holdout = None

    payload = {
        "method": "ESM-SAR log-policy projection onto 72 edge tolls per class",
        "hypothesis": (
            "The 24-step per-path correction lies mostly in the path-summed edge-toll subspace"
        ),
        "teacher_config": {
            "steps": TEACHER_CONFIG[0],
            "learning_rate": TEACHER_CONFIG[1],
            "low_weight": TEACHER_CONFIG[2],
            "anchor_weight": TEACHER_CONFIG[3],
        },
        "strict_esm_teacher": True,
        "actual_tm_used_for_policy": False,
        "checkpoint": str(checkpoint),
        "small_results": small_results,
        "selected_ridge": selected_ridge,
        "selected_weighting": selected_weighting,
        "holdout": holdout,
        "representation_hypothesis_pass": bool(holdout and holdout["safe"]),
    }
    output = HERE / "artifacts" / "toll_projection_probe.json"
    write_json(output, payload)
    print(json.dumps({
        "selected_ridge": selected_ridge,
        "selected_weighting": selected_weighting,
        "holdout": holdout,
        "representation_hypothesis_pass": payload["representation_hypothesis_pass"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
