from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
RUNTIME_DIR = TEST_DIR / "shared2x_medium_adapter"
for item in (str(ROOT), str(TEST_DIR), str(RUNTIME_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

spec = importlib.util.spec_from_file_location(
    "shared2x_esm_self_correction_runtime", RUNTIME_DIR / "run_experiment.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load frozen Hattrick runtime")
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)


class CorrectedPolicyAdapter:
    def __init__(self, path_count: int):
        self.path_count = int(path_count)

    def adapt_batch(self, policies: list[torch.Tensor], batch: dict) -> list[torch.Tensor]:
        corrected = batch["path_features"]
        policies[1] = corrected[:, : self.path_count].unsqueeze(-1)
        policies[2] = corrected[:, self.path_count :].unsqueeze(-1)
        return policies


def routed_policy(logits: torch.Tensor, base: torch.Tensor, paths_per_pair: int = 8) -> torch.Tensor:
    batch, path_count = base.shape
    grouped_base = base.reshape(batch, -1, paths_per_pair)
    grouped_logits = logits.reshape(batch, -1, paths_per_pair)
    valid = grouped_base > 0
    masked_logits = torch.where(
        valid, grouped_logits, torch.full_like(grouped_logits, -1e9)
    )
    routed = torch.softmax(masked_logits, dim=-1) * grouped_base.sum(dim=-1, keepdim=True)
    return torch.where(valid, routed, torch.zeros_like(routed)).reshape(batch, path_count)


def predicted_fulfillment(model, props, cache, policies: list[torch.Tensor]) -> torch.Tensor:
    tms = list(cache.predicted_tms)
    pte = cache.dataset.pte.coalesce()
    pte_indices = pte.indices()
    pte_info = (pte, pte_indices[0], pte_indices[1], pte.values())
    ratios = model.simulate(
        policies, tms, cache.capacities, pte_info, len(cache), props,
        rate_cap=props.rate_cap,
    )[:3]
    values = []
    for ratio, tm in zip(ratios, tms):
        admitted = (ratio.reshape(len(cache), -1) * tm.squeeze(-1)).sum(dim=1)
        demand = tm.sum(dim=1).squeeze(-1) / 8
        values.append(admitted / demand.clamp_min(1e-9))
    return torch.stack(values, dim=1)


def independent_score(values: torch.Tensor) -> torch.Tensor:
    # Each tensor element owns disjoint policy logits. A plain sum therefore
    # produces the same update as processing every snapshot separately, while
    # retaining an efficient batched implementation.
    return values.sum()


def correct_cache(
    model,
    props,
    cache: runtime.PolicyCache,
    steps: int,
    learning_rate: float,
    low_weight: float,
    anchor_weight: float,
) -> runtime.PolicyCache:
    base = [value.squeeze(-1).detach() for value in cache.policies]
    medium_logits = torch.nn.Parameter(torch.log(base[1].clamp_min(1e-12)))
    low_logits = torch.nn.Parameter(torch.log(base[2].clamp_min(1e-12)))
    optimizer = torch.optim.Adam([medium_logits, low_logits], lr=float(learning_rate))
    base_medium = base[1].reshape(len(cache), -1, 8)
    base_low = base[2].reshape(len(cache), -1, 8)
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        medium = routed_policy(medium_logits, base[1])
        low = routed_policy(low_logits, base[2])
        policies = [cache.policies[0], medium.unsqueeze(-1), low.unsqueeze(-1)]
        fulfill = predicted_fulfillment(model, props, cache, policies)
        current_medium = medium.reshape(len(cache), -1, 8)
        current_low = low.reshape(len(cache), -1, 8)
        # Reverse KL is finite on the original support and keeps the inner loop local.
        anchor = (
            current_medium * torch.log((current_medium + 1e-12) / (base_medium + 1e-12))
        ).sum(dim=-1).mean(dim=1).sum()
        anchor = anchor + (
            current_low * torch.log((current_low + 1e-12) / (base_low + 1e-12))
        ).sum(dim=-1).mean(dim=1).sum()
        objective = (
            independent_score(fulfill[:, 1])
            + float(low_weight) * independent_score(fulfill[:, 2])
            - float(anchor_weight) * anchor
        )
        (-objective).backward()
        optimizer.step()
    with torch.no_grad():
        medium = routed_policy(medium_logits, base[1])
        low = routed_policy(low_logits, base[2])
    return replace(cache, path_features=torch.cat([medium, low], dim=1).detach())


def summary_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def compact(summary: list[dict]) -> dict[str, dict[str, float]]:
    wanted = ("norm_fulfill_mean", "norm_fulfill_p1", "norm_fulfill_p10")
    return {
        key: {metric: float(row[metric]) for metric in wanted}
        for key, row in summary_index(summary).items()
    }


def gaps(candidate: list[dict], baseline: list[dict]) -> dict[str, float]:
    c, b = summary_index(candidate), summary_index(baseline)
    return {
        f"{priority}.{metric}": float(c[priority][metric] - b[priority][metric])
        for priority in ("High", "Medium", "Low")
        for metric in ("norm_fulfill_mean", "norm_fulfill_p1", "norm_fulfill_p10")
    }


def feasible(summary: list[dict], delta: dict[str, float]) -> bool:
    high = summary_index(summary)["High"]
    return (
        float(high["norm_fulfill_mean"]) >= 0.995
        and delta["Low.norm_fulfill_mean"] >= -0.001
        and delta["Low.norm_fulfill_p1"] >= -0.003
        and delta["Low.norm_fulfill_p10"] >= -0.003
    )


def score(delta: dict[str, float]) -> float:
    return (
        delta["Medium.norm_fulfill_mean"]
        + 0.35 * delta["Medium.norm_fulfill_p1"]
        + 0.35 * delta["Medium.norm_fulfill_p10"]
    )


def run_config(model, props, cache, baseline, config, split: str):
    corrected = correct_cache(model, props, cache, *config)
    adapter = CorrectedPolicyAdapter(int(cache.policies[0].shape[1]))
    _, summary = runtime.evaluate_cache(model, props, corrected, adapter, batch_size=32)
    delta = gaps(summary, baseline)
    return {
        "split": split,
        "steps": config[0],
        "learning_rate": config[1],
        "low_weight": config[2],
        "anchor_weight": config[3],
        "feasible": feasible(summary, delta),
        "score": score(delta),
        "metrics": compact(summary),
        "delta": delta,
    }


def main() -> None:
    torch.manual_seed(20260820)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(3, device)
    model, checkpoint = runtime.load_backbone(3, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    small = runtime.build_policy_cache(model, props, 350, 358, batch_size=16)
    _, small_baseline = runtime.evaluate_cache(model, props, small, None, batch_size=16)
    configurations = [
        (steps, learning_rate, low_weight, anchor)
        for steps in (16, 24)
        for learning_rate in (0.03, 0.05, 0.06)
        for low_weight in (0.01, 0.05, 0.1, 0.25, 0.5)
        for anchor in (0.0, 0.01)
    ]
    small_rows = [
        run_config(model, props, small, small_baseline, config, "small_350_358")
        for config in configurations
    ]
    candidates = sorted(
        (row for row in small_rows if row["feasible"]),
        key=lambda row: row["score"], reverse=True,
    )
    selected = [
        (int(row["steps"]), float(row["learning_rate"]),
         float(row["low_weight"]), float(row["anchor_weight"]))
        for row in candidates[:5]
    ]
    holdout_baseline = None
    holdout_rows = []
    if selected:
        holdout = runtime.build_policy_cache(model, props, 358, 400, batch_size=32)
        _, holdout_baseline = runtime.evaluate_cache(model, props, holdout, None, batch_size=32)
        holdout_rows = [
            run_config(model, props, holdout, holdout_baseline, config, "holdout_358_400")
            for config in selected
        ]
    payload = {
        "method": "per-snapshot ESM sequential-admission self-correction",
        "input_contract": "optimization uses ESM-predicted TMs; actual TMs are evaluation only",
        "checkpoint": str(checkpoint),
        "small_range": [350, 358],
        "holdout_range": [358, 400],
        "small_baseline": compact(small_baseline),
        "small_results": small_rows,
        "selected_from_small": selected,
        "holdout_baseline": compact(holdout_baseline) if holdout_baseline else None,
        "holdout_results": holdout_rows,
    }
    output = THIS_DIR / "esm_self_correction_refined.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
