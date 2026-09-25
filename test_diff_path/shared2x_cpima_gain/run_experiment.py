from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
PIMA_DIR = TEST_DIR / "shared2x_medium_adapter"
for item in (str(ROOT), str(TEST_DIR), str(PIMA_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

spec = importlib.util.spec_from_file_location(
    "shared2x_cpima_gain_runtime", PIMA_DIR / "run_experiment.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load C-PIMA runtime")
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)

from adapter import CausalMediumLowAdapter


K = 8
OUTPUT_ROOT = THIS_DIR / "artifacts"
LEVELS = {
    1: {"label": "level1_same_backbone_eight", "range": (350, 358), "backbone": 3, "adapter_level": "level3_validation_only"},
    2: {"label": "level2_same_backbone_holdout", "range": (358, 400), "backbone": 3, "adapter_level": "level3_validation_only"},
    3: {"label": "level3_full_validation", "range": (350, 400), "backbone": 3, "adapter_level": "level3_validation_only"},
    4: {"label": "level4_confirmation", "range": (400, 500), "backbone": 4, "adapter_level": "level4_confirmation"},
}


def adapter_path(adapter_level: str, order_seed: int) -> Path:
    base = (
        PIMA_DIR
        / "artifacts"
        / adapter_level
        / "causal_endpoint"
        / "low_align_0p5"
        / "medium_tail_0p1"
        / "backbone_490"
        / f"order_seed_{order_seed}"
        / "best_adapter.pt"
    )
    if not base.exists():
        raise FileNotFoundError(base)
    return base


def build_adapter(cache: runtime.PolicyCache, checkpoint: Path, device, dtype):
    pairs = list(cache.dataset.pij.keys())
    sources = torch.tensor([int(pair[0]) for pair in pairs], dtype=torch.long)
    destinations = torch.tensor([int(pair[1]) for pair in pairs], dtype=torch.long)
    adapter = CausalMediumLowAdapter(sources, destinations, K).to(device=device, dtype=dtype)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    adapter.load_state_dict(payload["adapter_state_dict"])
    adapter.eval()
    return adapter


def exponential_tilt(base: torch.Tensor, head, temperature: float) -> torch.Tensor:
    flat = base.squeeze(-1)
    batch = flat.shape[0]
    values = flat.reshape(batch, head.num_pairs, K)
    score = torch.tanh(
        head.source_bias[head.pair_sources]
        + head.destination_bias[head.pair_destinations]
        + head.path_rank_bias.unsqueeze(0)
    ).unsqueeze(0)
    tilted = values * torch.exp(float(temperature) * score)
    mass = values.sum(dim=-1, keepdim=True)
    tilted = tilted * mass / tilted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return tilted.reshape_as(flat).unsqueeze(-1)


def extrapolate(
    cache: runtime.PolicyCache, adapter, gain: float, low_mode: str
) -> runtime.PolicyCache:
    with torch.no_grad():
        if low_mode == "original":
            low = adapter.low_head(cache.policies[2])
        elif low_mode == "temperature":
            low = exponential_tilt(cache.policies[2], adapter.low_head, gain)
        else:
            raise ValueError(f"Unknown Low mode: {low_mode}")
        adapted = [
            cache.policies[0],
            exponential_tilt(cache.policies[1], adapter.medium_head, gain),
            low,
        ]
    policies = [cache.policies[0]]
    for class_index in (1, 2):
        value = adapted[class_index]
        if float(value.min().item()) < -1e-8:
            raise RuntimeError("Amplified adapter creates a negative policy")
        policies.append(value.clamp_min(0.0))
    return replace(cache, policies=tuple(policies))


def gaps(candidate: list[dict], baseline: list[dict]) -> dict[str, float]:
    c = runtime.summary_index(candidate)
    b = runtime.summary_index(baseline)
    return {
        f"{name.lower()}_{metric}_gap": float(c[name][f"norm_fulfill_{metric}"])
        - float(b[name][f"norm_fulfill_{metric}"])
        for name in runtime.CLASSES
        for metric in ("mean", "p1", "p10")
    }


def safe_remove(path: Path) -> None:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise RuntimeError(f"Unsafe removal target: {resolved}")
    if path.exists():
        shutil.rmtree(path)


def run_one(
    level: int, gain: float, order_seed: int, low_mode: str, force: bool
) -> Path:
    setting = LEVELS[level]
    gain_label = format(gain, ".4g").replace(".", "p")
    run_dir = (
        OUTPUT_ROOT
        / setting["label"]
        / f"low_{low_mode}"
        / f"gain_{gain_label}"
        / f"order_seed_{order_seed}"
    )
    if force:
        safe_remove(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(int(setting["backbone"]), device)
    model, backbone = runtime.load_backbone(int(setting["backbone"]), 490, props, device)
    cache = runtime.build_policy_cache(model, props, *setting["range"], batch_size=16)
    baseline_rows, baseline_summary = runtime.evaluate_cache(model, props, cache, None)
    checkpoint = adapter_path(str(setting["adapter_level"]), order_seed)
    adapter = build_adapter(cache, checkpoint, device, props.dtype)
    candidate_cache = extrapolate(cache, adapter, gain, low_mode)
    candidate_rows, candidate_summary = runtime.evaluate_cache(
        model, props, candidate_cache, None
    )
    delta = gaps(candidate_summary, baseline_summary)
    high = runtime.summary_index(candidate_summary)["High"]
    feasible = (
        float(high["norm_fulfill_mean"]) >= 0.995
        and abs(delta["high_mean_gap"]) <= 1e-6
        and delta["medium_mean_gap"] > 0.0
        and delta["medium_p1_gap"] > 0.0
        and delta["medium_p10_gap"] > 0.0
        and delta["low_mean_gap"] >= -0.003
        and delta["low_p10_gap"] >= -0.01
        and max(float(row["max_disabled_flow"]) for row in candidate_summary) <= 1e-8
        and max(float(row["max_admitted_capacity_ratio"]) for row in candidate_summary) <= 1.0001
    )
    runtime.write_csv(run_dir / "baseline_metrics.csv", baseline_rows)
    runtime.write_csv(run_dir / "candidate_metrics.csv", candidate_rows)
    result = {
        "status": "complete",
        "feasible": feasible,
        "gain": gain,
        "order_seed": order_seed,
        "low_mode": low_mode,
        "range": list(setting["range"]),
        "gaps": delta,
        "baseline": baseline_summary,
        "candidate": candidate_summary,
        "backbone": str(backbone),
        "adapter": str(checkpoint),
        "method": "Joint gain extrapolation of isolated C-PIMA Medium head and causal Low shield",
    }
    runtime.write_json(run_dir / "summary.json", result)
    runtime.write_json(run_dir / "complete.json", {key: result[key] for key in ("status", "feasible", "gaps")})
    print(json.dumps({key: result[key] for key in ("status", "feasible", "gaps")}), flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=tuple(LEVELS), required=True)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--order-seed", type=int, choices=(490, 491), default=490)
    parser.add_argument("--low-mode", choices=("original", "temperature"), default="original")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_one(args.level, args.gain, args.order_seed, args.low_mode, args.force)


if __name__ == "__main__":
    main()
