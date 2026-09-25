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
    "shared2x_bottleneck_gate_runtime", RUNTIME_DIR / "run_experiment.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load frozen Hattrick runtime")
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)


class BottleneckGate:
    """Parameter-free Medium gate driven only by ESM-predicted edge pressure."""

    def __init__(self, alpha: float, paths_per_pair: int = 8):
        self.alpha = float(alpha)
        self.paths_per_pair = int(paths_per_pair)

    def adapt_batch(self, policies: list[torch.Tensor], batch: dict) -> list[torch.Tensor]:
        base = policies[1]
        flat = base.squeeze(-1)
        batch_size, path_count = flat.shape
        pressure = batch["path_features"].to(device=flat.device, dtype=flat.dtype)
        if pressure.shape != flat.shape:
            raise ValueError("Path pressure and Medium policy shapes differ")
        grouped = flat.reshape(batch_size, -1, self.paths_per_pair)
        cost = pressure.reshape(batch_size, -1, self.paths_per_pair)
        valid = grouped > 0
        # Only relative path pressure matters within an OD. Keeping exact zeros
        # prevents disabled candidate paths from being re-enabled.
        logits = torch.where(
            valid,
            torch.log(grouped.clamp_min(1e-30)) - self.alpha * cost,
            torch.full_like(grouped, -torch.inf),
        )
        adapted = torch.softmax(logits, dim=-1) * grouped.sum(dim=-1, keepdim=True)
        adapted = torch.where(valid, adapted, torch.zeros_like(adapted))
        policies[1] = adapted.reshape(batch_size, path_count, 1)
        return policies


def attach_top_edge_pressure(cache: runtime.PolicyCache, top_k: int) -> runtime.PolicyCache:
    """Attach the number of active top-k edges traversed by every candidate path."""
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    capacity = cache.capacities.to(dtype=torch.float32).clamp_min(1e-9)
    predicted_high = cache.predicted_tms[0].squeeze(-1).to(dtype=torch.float32)
    predicted_medium = cache.predicted_tms[1].squeeze(-1).to(dtype=torch.float32)
    high_flow = cache.policies[0].squeeze(-1).to(dtype=torch.float32) * predicted_high
    medium_flow = cache.policies[1].squeeze(-1).to(dtype=torch.float32) * predicted_medium
    load = torch.sparse.mm(pte.t(), (high_flow + medium_flow).t()).t()
    utilization = load / capacity
    active_edges = utilization.topk(int(top_k), dim=1).indices
    edge_pressure = torch.zeros_like(utilization)
    # Rank weights keep the method discrete while distinguishing the dominant edge.
    weights = torch.linspace(1.0, 0.5, int(top_k), device=utilization.device)
    edge_pressure.scatter_(1, active_edges, weights.reshape(1, -1).expand(len(cache), -1))
    path_pressure = torch.sparse.mm(pte, edge_pressure.t()).t().detach()
    return replace(cache, path_features=path_pressure)


def index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def compact(summary: list[dict]) -> dict[str, dict[str, float]]:
    wanted = ("norm_fulfill_mean", "norm_fulfill_p1", "norm_fulfill_p10")
    return {
        key: {metric: float(row[metric]) for metric in wanted}
        for key, row in index(summary).items()
    }


def gaps(candidate: list[dict], baseline: list[dict]) -> dict[str, float]:
    c, b = index(candidate), index(baseline)
    result = {}
    for priority in ("High", "Medium", "Low"):
        for metric in ("norm_fulfill_mean", "norm_fulfill_p1", "norm_fulfill_p10"):
            result[f"{priority}.{metric}"] = float(c[priority][metric] - b[priority][metric])
    return result


def feasible(summary: list[dict], delta: dict[str, float]) -> bool:
    by_priority = index(summary)
    return (
        float(by_priority["High"]["norm_fulfill_mean"]) >= 0.995
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


def evaluate_grid(model, props, cache, split: str, configurations: list[tuple[int, float]]):
    _, baseline = runtime.evaluate_cache(model, props, cache, None, batch_size=32)
    rows = []
    pressure_caches = {}
    for top_k, alpha in configurations:
        if top_k not in pressure_caches:
            pressure_caches[top_k] = attach_top_edge_pressure(cache, top_k)
        gate = BottleneckGate(alpha)
        _, summary = runtime.evaluate_cache(
            model, props, pressure_caches[top_k], gate, batch_size=32
        )
        delta = gaps(summary, baseline)
        rows.append(
            {
                "split": split,
                "top_k": top_k,
                "alpha": alpha,
                "feasible": feasible(summary, delta),
                "score": score(delta),
                "metrics": compact(summary),
                "delta": delta,
            }
        )
    return compact(baseline), rows


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(3, device)
    model, checkpoint = runtime.load_backbone(3, 490, props, device)
    small = runtime.build_policy_cache(model, props, 350, 358, batch_size=16)
    grid = [(k, a) for k in (1, 2, 3, 4, 6) for a in (0.25, 0.5, 1.0, 2.0, 4.0)]
    small_baseline, small_rows = evaluate_grid(model, props, small, "small_350_358", grid)
    feasible_rows = sorted(
        (row for row in small_rows if row["feasible"]),
        key=lambda row: row["score"],
        reverse=True,
    )
    # Validate only distinct best configurations; the large holdout has no role in selection.
    selected = [(int(row["top_k"]), float(row["alpha"])) for row in feasible_rows[:5]]
    holdout_baseline = None
    holdout_rows = []
    if selected:
        holdout = runtime.build_policy_cache(model, props, 358, 400, batch_size=32)
        holdout_baseline, holdout_rows = evaluate_grid(
            model, props, holdout, "holdout_358_400", selected
        )
    payload = {
        "method": "ESM bottleneck-conditioned Medium path gate",
        "checkpoint": str(checkpoint),
        "small_range": [350, 358],
        "holdout_range": [358, 400],
        "constraints": {
            "high_mean_minimum": 0.995,
            "low_mean_delta_minimum": -0.001,
            "low_p1_delta_minimum": -0.003,
            "low_p10_delta_minimum": -0.003,
        },
        "small_baseline": small_baseline,
        "small_results": small_rows,
        "selected_from_small": selected,
        "holdout_baseline": holdout_baseline,
        "holdout_results": holdout_rows,
    }
    output = THIS_DIR / "bottleneck_gate_probe.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
