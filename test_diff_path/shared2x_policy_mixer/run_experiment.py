from __future__ import annotations

import argparse
import hashlib
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
for item in (str(ROOT), str(TEST_DIR), str(PIMA_DIR), str(THIS_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

runtime_spec = importlib.util.spec_from_file_location(
    "shared2x_policy_mixer_runtime", PIMA_DIR / "run_experiment.py"
)
if runtime_spec is None or runtime_spec.loader is None:
    raise RuntimeError("Unable to load the frozen Hattrick policy runtime")
runtime = importlib.util.module_from_spec(runtime_spec)
sys.modules[runtime_spec.name] = runtime
runtime_spec.loader.exec_module(runtime)

import run_dotemc_priority_mask_experiment as dote


OUTPUT_ROOT = THIS_DIR / "artifacts"
DOTE_CHECKPOINT = (
    TEST_DIR
    / "results_load2x_retrain_shared_strict"
    / "outputs"
    / "shared"
    / "w_1_0p1_0p01"
    / "best_model.pt"
)
LEVELS = {
    1: {"label": "level1_eight_samples", "range": (200, 208), "backbone": 2},
    2: {"label": "level2_proxy", "range": (200, 250), "backbone": 2},
    3: {"label": "level3_validation", "range": (350, 400), "backbone": 4},
    4: {"label": "level4_confirmation", "range": (400, 500), "backbone": 4},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_remove(path: Path) -> None:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise RuntimeError(f"Unsafe removal target: {resolved}")
    if path.exists():
        shutil.rmtree(path)


@torch.no_grad()
def dote_policies(
    cache: runtime.PolicyCache,
    model: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = dote.make_inputs(
        cache.predicted_tms[0].to(dtype=torch.float32),
        cache.predicted_tms[1].to(dtype=torch.float32),
        cache.predicted_tms[2].to(dtype=torch.float32),
        cache.dataset.num_pairs,
        mean,
        std,
    )
    split = model(features, cache.path_masks).reshape(len(cache), 3, -1, 1)
    return split[:, 0], split[:, 1], split[:, 2]


def mix_cache(
    cache: runtime.PolicyCache,
    dote_split: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    alpha: float,
    mode: str,
) -> runtime.PolicyCache:
    if mode not in ("medium", "high_medium", "medium_low"):
        raise ValueError(f"Unknown mixture mode: {mode}")
    high = cache.policies[0]
    if mode == "high_medium":
        high = (1.0 - float(alpha)) * high + float(alpha) * dote_split[0]
    medium = (1.0 - float(alpha)) * cache.policies[1] + float(alpha) * dote_split[1]
    low = cache.policies[2]
    if mode == "medium_low":
        low = (1.0 - float(alpha)) * low + float(alpha) * dote_split[2]
    return replace(cache, policies=(high, medium, low))


def run_one(
    level: int, backbone_seed: int, alpha: float, mode: str, force: bool
) -> Path:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    spec = LEVELS[level]
    alpha_label = format(float(alpha), ".4g").replace(".", "p")
    run_dir = (
        OUTPUT_ROOT
        / spec["label"]
        / mode
        / f"alpha_{alpha_label}"
        / f"backbone_{backbone_seed}"
    )
    if (run_dir / "complete.json").exists() and not force:
        print(f"[skip] {run_dir}", flush=True)
        return run_dir
    if force:
        safe_remove(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_level = int(spec["backbone"])
    props = runtime.build_props(backbone_level, device)
    hattrick, hattrick_checkpoint = runtime.load_backbone(
        backbone_level, backbone_seed, props, device
    )
    start, end = spec["range"]
    cache = runtime.build_policy_cache(hattrick, props, start, end, batch_size=8)
    baseline_rows, baseline_summary = runtime.evaluate_cache(
        hattrick, props, cache, None
    )
    dote_model, dote_mean, dote_std = dote.load_checkpoint(DOTE_CHECKPOINT, device)
    dote_split = dote_policies(cache, dote_model, dote_mean, dote_std)
    candidate_cache = mix_cache(cache, dote_split, alpha, mode)
    candidate_rows, candidate_summary = runtime.evaluate_cache(
        hattrick, props, candidate_cache, None
    )
    gaps = runtime.metric_gaps(candidate_summary, baseline_summary)
    indexed = runtime.summary_index(candidate_summary)
    feasible = (
        float(indexed["High"]["norm_fulfill_mean"]) >= 0.995
        and (mode == "high_medium" or abs(gaps["high_mean_gap"]) <= 1e-6)
        and gaps["low_mean_gap"] >= -0.003
        and gaps["low_p10_gap"] >= -0.01
        and max(float(row["max_admitted_capacity_ratio"]) for row in candidate_summary)
        <= 1.0001
        and max(float(row["max_disabled_flow"]) for row in candidate_summary) <= 1e-8
    )
    config = {
        "method": "Priority-Isolated Hattrick/DOTE Medium Mixture",
        "level": level,
        "range": [start, end],
        "alpha": float(alpha),
        "mode": mode,
        "inference_inputs": "ESM-predicted TMs, topology, capacities, paths and masks only",
        "high_policy": (
            "Hattrick unchanged"
            if mode in ("medium", "medium_low")
            else "convex mixture of Hattrick and DOTE-MC sensitivity"
        ),
        "medium_policy": "convex mixture of Hattrick and DOTE-MC sensitivity",
        "low_policy": (
            "convex mixture of Hattrick and DOTE-MC sensitivity"
            if mode == "medium_low"
            else "Hattrick unchanged"
        ),
        "hattrick_checkpoint": str(hattrick_checkpoint),
        "hattrick_checkpoint_sha256": sha256(hattrick_checkpoint),
        "dote_checkpoint": str(DOTE_CHECKPOINT),
        "dote_checkpoint_sha256": sha256(DOTE_CHECKPOINT),
        "source_sha256": {"run_experiment.py": sha256(Path(__file__).resolve())},
    }
    runtime.write_json(run_dir / "config.json", config)
    runtime.write_csv(run_dir / "baseline_metrics.csv", baseline_rows)
    runtime.write_csv(run_dir / "candidate_metrics.csv", candidate_rows)
    runtime.write_json(
        run_dir / "summary.json",
        {
            "baseline": baseline_summary,
            "candidate": candidate_summary,
            "gaps": gaps,
            "feasible": feasible,
        },
    )
    complete = {"status": "complete", "feasible": feasible, "gaps": gaps}
    runtime.write_json(run_dir / "complete.json", complete)
    print(json.dumps(complete, ensure_ascii=False), flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=tuple(LEVELS), required=True)
    parser.add_argument("--backbone-seed", type=int, default=490)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument(
        "--mode", choices=("medium", "high_medium", "medium_low"), default="medium"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_one(args.level, args.backbone_seed, args.alpha, args.mode, args.force)


if __name__ == "__main__":
    main()
