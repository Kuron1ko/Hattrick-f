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
    "shared2x_residual_decoder_runtime", RUNTIME_DIR / "run_experiment.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load frozen Hattrick runtime")
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)


class DecodedPolicyAdapter:
    def __init__(self, path_count: int):
        self.path_count = int(path_count)

    def adapt_batch(self, policies: list[torch.Tensor], batch: dict) -> list[torch.Tensor]:
        decoded = batch["path_features"]
        policies[1] = decoded[:, : self.path_count].unsqueeze(-1)
        policies[2] = decoded[:, self.path_count :].unsqueeze(-1)
        return policies


def edge_load(pte: torch.Tensor, policy: torch.Tensor, demand: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(pte.t(), (policy * demand).t()).t()


def conditional_gradient_decode(
    pte: torch.Tensor,
    base: torch.Tensor,
    demand: torch.Tensor,
    fixed_load: torch.Tensor,
    capacity: torch.Tensor,
    steps: int,
    gamma: float,
    temperature: float,
    paths_per_pair: int = 8,
) -> torch.Tensor:
    """Route toward paths with the lowest marginal smooth-MLU price."""
    batch_size, path_count = base.shape
    grouped = base.reshape(batch_size, -1, paths_per_pair)
    route = grouped.clone()
    valid = grouped > 0
    mass = grouped.sum(dim=-1, keepdim=True)
    for _ in range(int(steps)):
        flat = route.reshape(batch_size, path_count)
        utilization = (fixed_load + edge_load(pte, flat, demand)) / capacity
        # This is the gradient of a smooth maximum utilization. Division by
        # capacity converts edge pressure into marginal path cost.
        edge_price = torch.softmax(utilization / float(temperature), dim=1) / capacity
        path_price = torch.sparse.mm(pte, edge_price.t()).t()
        costs = path_price.reshape(batch_size, -1, paths_per_pair)
        costs = torch.where(valid, costs, torch.full_like(costs, torch.inf))
        best = costs.argmin(dim=-1, keepdim=True)
        vertex = torch.zeros_like(route).scatter_(2, best, mass)
        route = (1.0 - float(gamma)) * route + float(gamma) * vertex
    return route.reshape(batch_size, path_count)


def decode_cache(
    cache: runtime.PolicyCache,
    medium_steps: int,
    medium_gamma: float,
    low_steps: int,
    low_gamma: float,
    temperature: float,
) -> runtime.PolicyCache:
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    capacity = cache.capacities.to(dtype=torch.float32).clamp_min(1e-9)
    base = [value.squeeze(-1).to(dtype=torch.float32) for value in cache.policies]
    demand = [value.squeeze(-1).to(dtype=torch.float32) for value in cache.predicted_tms]
    high_load = edge_load(pte, base[0], demand[0])
    medium = conditional_gradient_decode(
        pte, base[1], demand[1], high_load, capacity,
        medium_steps, medium_gamma, temperature,
    )
    high_medium_load = high_load + edge_load(pte, medium, demand[1])
    low = conditional_gradient_decode(
        pte, base[2], demand[2], high_medium_load, capacity,
        low_steps, low_gamma, temperature,
    )
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
    values = summary_index(summary)
    return (
        float(values["High"]["norm_fulfill_mean"]) >= 0.995
        and delta["Low.norm_fulfill_mean"] >= -0.001
        and delta["Low.norm_fulfill_p1"] >= -0.003
        and delta["Low.norm_fulfill_p10"] >= -0.003
    )


def score(delta: dict[str, float]) -> float:
    return (
        delta["Medium.norm_fulfill_mean"]
        + 0.35 * delta["Medium.norm_fulfill_p1"]
        + 0.35 * delta["Medium.norm_fulfill_p10"]
        + 0.1 * min(0.0, delta["Low.norm_fulfill_mean"])
    )


def run_config(model, props, cache, baseline, config: tuple[int, float, int, float, float], split: str):
    medium_steps, medium_gamma, low_steps, low_gamma, temperature = config
    decoded = decode_cache(cache, *config)
    adapter = DecodedPolicyAdapter(int(cache.policies[0].shape[1]))
    _, summary = runtime.evaluate_cache(model, props, decoded, adapter, batch_size=32)
    delta = gaps(summary, baseline)
    return {
        "split": split,
        "medium_steps": medium_steps,
        "medium_gamma": medium_gamma,
        "low_steps": low_steps,
        "low_gamma": low_gamma,
        "temperature": temperature,
        "feasible": feasible(summary, delta),
        "score": score(delta),
        "metrics": compact(summary),
        "delta": delta,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(3, device)
    model, checkpoint = runtime.load_backbone(3, 490, props, device)
    small = runtime.build_policy_cache(model, props, 350, 358, batch_size=16)
    _, small_baseline_summary = runtime.evaluate_cache(model, props, small, None, batch_size=16)
    configurations = [
        (steps, gamma, 8, low_gamma, temperature)
        for steps in (2, 4, 8, 16)
        for gamma in (0.025, 0.05, 0.1)
        for low_gamma in (0.05, 0.1)
        for temperature in (0.03, 0.07)
    ]
    small_rows = [
        run_config(model, props, small, small_baseline_summary, config, "small_350_358")
        for config in configurations
    ]
    candidates = sorted(
        (row for row in small_rows if row["feasible"]),
        key=lambda row: row["score"], reverse=True,
    )
    selected = [
        (
            int(row["medium_steps"]), float(row["medium_gamma"]),
            int(row["low_steps"]), float(row["low_gamma"]), float(row["temperature"]),
        )
        for row in candidates[:5]
    ]
    holdout_baseline_summary = None
    holdout_rows = []
    if selected:
        holdout = runtime.build_policy_cache(model, props, 358, 400, batch_size=32)
        _, holdout_baseline_summary = runtime.evaluate_cache(
            model, props, holdout, None, batch_size=32
        )
        holdout_rows = [
            run_config(
                model, props, holdout, holdout_baseline_summary,
                config, "holdout_358_400",
            )
            for config in selected
        ]
    payload = {
        "method": "ESM sequential residual-capacity conditional-gradient decoder",
        "checkpoint": str(checkpoint),
        "small_range": [350, 358],
        "holdout_range": [358, 400],
        "small_baseline": compact(small_baseline_summary),
        "small_results": small_rows,
        "selected_from_small": selected,
        "holdout_baseline": compact(holdout_baseline_summary) if holdout_baseline_summary else None,
        "holdout_results": holdout_rows,
    }
    output = THIS_DIR / "residual_decoder_probe.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
