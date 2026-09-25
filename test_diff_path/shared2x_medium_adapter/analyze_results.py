from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CLASSES = ("High", "Medium", "Low")
QUANTILES = {"mean": None, "p1": 0.01, "p10": 0.10}


def read_rows(path: Path) -> dict[tuple[int, str], dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            (int(row["snapshot"]), row["class"]): row
            for row in csv.DictReader(handle)
        }


def statistic(values: np.ndarray, label: str, axis: int | None = None) -> np.ndarray | float:
    quantile = QUANTILES[label]
    if quantile is None:
        return np.mean(values, axis=axis)
    return np.quantile(values, quantile, axis=axis)


def paired_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    label: str,
    rng: np.random.Generator,
    repetitions: int,
) -> dict[str, float]:
    observed = float(statistic(candidate, label) - statistic(baseline, label))
    sample_count = baseline.size
    draws = rng.integers(0, sample_count, size=(repetitions, sample_count))
    boot = statistic(candidate[draws], label, axis=1) - statistic(baseline[draws], label, axis=1)
    return {
        "gap": observed,
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "bootstrap_probability_gap_le_zero": float(np.mean(boot <= 0.0)),
        "noninferior_at_minus_0p01": bool(float(np.quantile(boot, 0.025)) > -0.01),
    }


def analyze(root: Path, repetitions: int) -> dict:
    rng = np.random.default_rng(20260820)
    result: dict = {"bootstrap_repetitions": repetitions, "seeds": {}, "aggregate": {}}
    per_seed_arrays: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for seed in (490, 491):
        run_dir = root / f"order_seed_{seed}"
        baseline_rows = read_rows(run_dir / "baseline_evaluation_metrics.csv")
        candidate_rows = read_rows(run_dir / "evaluation_metrics.csv")
        snapshots = sorted({snapshot for snapshot, _ in baseline_rows})
        per_seed_arrays[seed] = {}
        seed_result: dict = {}
        for class_name in CLASSES:
            baseline = np.asarray(
                [float(baseline_rows[(snapshot, class_name)]["norm_fulfill"]) for snapshot in snapshots],
                dtype=np.float64,
            )
            candidate = np.asarray(
                [float(candidate_rows[(snapshot, class_name)]["norm_fulfill"]) for snapshot in snapshots],
                dtype=np.float64,
            )
            per_seed_arrays[seed][class_name] = (baseline, candidate)
            seed_result[class_name] = {
                label: paired_bootstrap(baseline, candidate, label, rng, repetitions)
                for label in QUANTILES
            }
            seed_result[class_name]["baseline_mean"] = float(baseline.mean())
            seed_result[class_name]["candidate_mean"] = float(candidate.mean())
        result["seeds"][str(seed)] = seed_result

    for class_name in CLASSES:
        baseline = np.mean(
            np.stack([per_seed_arrays[seed][class_name][0] for seed in (490, 491)]), axis=0
        )
        candidate = np.mean(
            np.stack([per_seed_arrays[seed][class_name][1] for seed in (490, 491)]), axis=0
        )
        result["aggregate"][class_name] = {
            label: paired_bootstrap(baseline, candidate, label, rng, repetitions)
            for label in QUANTILES
        }
        result["aggregate"][class_name]["baseline_mean"] = float(baseline.mean())
        result["aggregate"][class_name]["candidate_mean"] = float(candidate.mean())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.repetitions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
