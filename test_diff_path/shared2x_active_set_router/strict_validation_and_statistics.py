from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
FINAL_PATH = THIS_DIR / "run_final_esm_self_correction.py"
spec = importlib.util.spec_from_file_location("strict_validation_runtime", FINAL_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load final experiment implementation")
final = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = final
spec.loader.exec_module(final)
method = final.method
runtime = final.runtime


def evaluate_range(model, props, start: int, stop: int) -> dict:
    cache = runtime.build_policy_cache(model, props, start, stop, batch_size=32)
    _, baseline = runtime.evaluate_cache(model, props, cache, None, batch_size=32)
    corrected = final.correct_strictly_per_snapshot(model, props, cache)
    adapter = method.CorrectedPolicyAdapter(int(cache.policies[0].shape[1]))
    _, candidate = runtime.evaluate_cache(model, props, corrected, adapter, batch_size=32)
    delta = method.gaps(candidate, baseline)
    return {
        "range": [start, stop],
        "baseline": method.compact(baseline),
        "candidate": method.compact(candidate),
        "delta": delta,
        "constraints_passed": method.feasible(candidate, delta),
    }


def read_norm_fulfill(path: Path) -> dict[str, np.ndarray]:
    values = {name: [] for name in ("High", "Medium", "Low")}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values[row["class"]].append(float(row["norm_fulfill"]))
    return {name: np.asarray(items, dtype=np.float64) for name, items in values.items()}


def paired_bootstrap(base: np.ndarray, candidate: np.ndarray, seed: int) -> dict:
    difference = candidate - base
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(20000, len(difference)))
    means = difference[indices].mean(axis=1)
    return {
        "mean_delta": float(difference.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "improved_snapshot_fraction": float(np.mean(difference > 0)),
        "unchanged_or_improved_snapshot_fraction": float(np.mean(difference >= -1e-7)),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(3, device)
    model, _ = runtime.load_backbone(3, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    exploration = evaluate_range(model, props, 350, 358)
    validation = evaluate_range(model, props, 358, 400)

    baseline = read_norm_fulfill(THIS_DIR / "final_baseline_rows_400_500.csv")
    candidate = read_norm_fulfill(THIS_DIR / "final_candidate_rows_400_500.csv")
    statistics = {
        name: paired_bootstrap(baseline[name], candidate[name], 20260820 + index)
        for index, name in enumerate(("High", "Medium", "Low"))
    }
    payload = {
        "config": {
            "steps": final.FINAL_CONFIG[0],
            "learning_rate": final.FINAL_CONFIG[1],
            "low_weight": final.FINAL_CONFIG[2],
            "anchor_weight": final.FINAL_CONFIG[3],
        },
        "deployment_mode": "strict one-snapshot-at-a-time correction",
        "exploration": exploration,
        "validation": validation,
        "final_paired_bootstrap_20000": statistics,
    }
    output = THIS_DIR / "strict_validation_and_statistics.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
