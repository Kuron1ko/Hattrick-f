from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import run_experiment as runner


def load_best_rows(level: int, penalty: str, multiplier: float, seed: int) -> list[dict]:
    path = runner.run_directory(level, penalty, multiplier, seed) / "best_evaluation_metrics.csv"
    return runner.read_csv(path)


def values(rows: list[dict], class_name: str, metric: str) -> np.ndarray:
    selected = sorted(
        (row for row in rows if row["class"] == class_name),
        key=lambda row: int(row["snapshot"]),
    )
    return np.asarray([float(row[metric]) for row in selected], dtype=np.float64)


def percentile(data: np.ndarray, q: float) -> float:
    return float(np.percentile(data, q))


def compare_seed(
    level: int, penalty: str, multiplier: float, seed: int
) -> tuple[dict, dict[str, bool]]:
    baseline = load_best_rows(level, penalty, 0.0, seed)
    candidate = load_best_rows(level, penalty, multiplier, seed)
    if not baseline or not candidate:
        raise FileNotFoundError(
            f"Missing paired artifacts for level={level}, penalty={penalty}, multiplier={multiplier}, seed={seed}"
        )
    baseline_keys = {(int(row["snapshot"]), row["class"]) for row in baseline}
    candidate_keys = {(int(row["snapshot"]), row["class"]) for row in candidate}
    if baseline_keys != candidate_keys:
        raise RuntimeError(f"Paired row mismatch for seed {seed}, multiplier {multiplier}")

    base_medium = values(baseline, "Medium", "norm_fulfill")
    cand_medium = values(candidate, "Medium", "norm_fulfill")
    base_high = values(baseline, "High", "norm_fulfill")
    cand_high = values(candidate, "High", "norm_fulfill")
    base_low_norm = values(baseline, "Low", "norm_fulfill")
    cand_low_norm = values(candidate, "Low", "norm_fulfill")
    base_low_fulfill = values(baseline, "Low", "fulfill_ratio")
    cand_low_fulfill = values(candidate, "Low", "fulfill_ratio")
    base_gap = base_low_norm - base_medium
    cand_gap = cand_low_norm - cand_medium
    base_severity = float(np.maximum(base_gap, 0.0).mean())
    cand_severity = float(np.maximum(cand_gap, 0.0).mean())
    severity_reduction = (
        (base_severity - cand_severity) / base_severity if base_severity > 1e-12 else 0.0
    )
    paired_medium = cand_medium - base_medium
    candidate_capacity = np.asarray(
        [float(row["admitted_capacity_ratio"]) for row in candidate], dtype=np.float64
    )
    candidate_disabled = np.asarray(
        [float(row["disabled_flow"]) for row in candidate], dtype=np.float64
    )

    item = {
        "level": level,
        "penalty": penalty,
        "lambda_multiplier": float(multiplier),
        "seed": seed,
        "n_snapshots": int(base_medium.size),
        "baseline_medium_mean": float(base_medium.mean()),
        "candidate_medium_mean": float(cand_medium.mean()),
        "medium_mean_gap": float(cand_medium.mean() - base_medium.mean()),
        "baseline_medium_p1": percentile(base_medium, 1),
        "candidate_medium_p1": percentile(cand_medium, 1),
        "medium_p1_gap": percentile(cand_medium, 1) - percentile(base_medium, 1),
        "baseline_medium_p10": percentile(base_medium, 10),
        "candidate_medium_p10": percentile(cand_medium, 10),
        "medium_p10_gap": percentile(cand_medium, 10) - percentile(base_medium, 10),
        "baseline_high_mean": float(base_high.mean()),
        "candidate_high_mean": float(cand_high.mean()),
        "high_mean_gap": float(cand_high.mean() - base_high.mean()),
        "baseline_low_fulfill_mean": float(base_low_fulfill.mean()),
        "candidate_low_fulfill_mean": float(cand_low_fulfill.mean()),
        "low_fulfill_mean_gap": float(cand_low_fulfill.mean() - base_low_fulfill.mean()),
        "baseline_low_norm_mean": float(base_low_norm.mean()),
        "candidate_low_norm_mean": float(cand_low_norm.mean()),
        "candidate_medium_minus_low_norm_mean": float(cand_medium.mean() - cand_low_norm.mean()),
        "baseline_inversion_positive_gap_mean": base_severity,
        "candidate_inversion_positive_gap_mean": cand_severity,
        "inversion_severity_reduction_fraction": severity_reduction,
        "baseline_inversion_violation_fraction": float((base_gap > 0).mean()),
        "candidate_inversion_violation_fraction": float((cand_gap > 0).mean()),
        "paired_medium_positive_fraction": float((paired_medium > 0).mean()),
        "paired_medium_gap_min": float(paired_medium.min()),
        "paired_medium_gap_max": float(paired_medium.max()),
        "candidate_max_post_admission_mlu": float(candidate_capacity.max()),
        "candidate_max_disabled_flow": float(candidate_disabled.max()),
    }
    maximum_violation_fraction = 0.25 if level == 4 else 0.50
    gates = {
        "medium_mean_nondecrease": item["medium_mean_gap"] >= 0.0,
        "medium_p1_gain": item["medium_p1_gap"] >= 0.01,
        "medium_p10_gain": item["medium_p10_gap"] >= 0.01,
        "high_absolute": item["candidate_high_mean"] >= 0.98,
        "high_relative": item["high_mean_gap"] >= -0.005,
        "low_raw_guard": item["low_fulfill_mean_gap"] >= -0.02,
        "inversion_severity_half": item["inversion_severity_reduction_fraction"] >= 0.50,
        "inversion_frequency": item["candidate_inversion_violation_fraction"] <= maximum_violation_fraction,
        "paired_medium_fraction": item["paired_medium_positive_fraction"] >= 0.60,
        "capacity": item["candidate_max_post_admission_mlu"] <= 1.0001,
        "disabled_flow": item["candidate_max_disabled_flow"] <= 1e-8,
    }
    if level >= 3:
        gates["mean_priority_order"] = item["candidate_medium_minus_low_norm_mean"] >= 0.0
    item["gates"] = gates
    item["passes"] = bool(all(gates.values()))
    basic = {
        "medium_mean_nondecrease": gates["medium_mean_nondecrease"],
        "high_absolute": gates["high_absolute"],
        "high_relative": gates["high_relative"],
        "low_raw_guard": gates["low_raw_guard"],
        "capacity": gates["capacity"],
        "disabled_flow": gates["disabled_flow"],
    }
    return item, basic


def hierarchical_bootstrap_p10_gap(
    level: int,
    penalty: str,
    multiplier: float,
    seeds: list[int],
    draws: int = 10_000,
) -> dict:
    paired = {}
    for seed in seeds:
        baseline = values(load_best_rows(level, penalty, 0.0, seed), "Medium", "norm_fulfill")
        candidate = values(
            load_best_rows(level, penalty, multiplier, seed), "Medium", "norm_fulfill"
        )
        paired[seed] = (baseline, candidate)
    rng = np.random.default_rng(20260816)
    estimates = np.empty(draws, dtype=np.float64)
    seed_array = np.asarray(seeds, dtype=np.int64)
    for draw in range(draws):
        sampled_seeds = rng.choice(seed_array, size=len(seed_array), replace=True)
        base_parts = []
        candidate_parts = []
        for seed in sampled_seeds:
            baseline, candidate = paired[int(seed)]
            indices = rng.integers(0, baseline.size, size=baseline.size)
            base_parts.append(baseline[indices])
            candidate_parts.append(candidate[indices])
        estimates[draw] = percentile(np.concatenate(candidate_parts), 10) - percentile(
            np.concatenate(base_parts), 10
        )
    return {
        "draws": draws,
        "rng_seed": 20260816,
        "estimate_mean": float(estimates.mean()),
        "ci95_lower": float(np.percentile(estimates, 2.5)),
        "ci95_upper": float(np.percentile(estimates, 97.5)),
        "passes": float(np.percentile(estimates, 2.5)) > 0.0,
    }


def analyze(level: int, penalty: str, multipliers: list[float], seeds: list[int]) -> Path:
    results = []
    basic_by_multiplier: dict[float, list[bool]] = {}
    for multiplier in multipliers:
        for seed in seeds:
            try:
                item, basic = compare_seed(level, penalty, multiplier, seed)
            except FileNotFoundError:
                continue
            results.append(item)
            basic_by_multiplier.setdefault(float(multiplier), []).append(all(basic.values()))

    seed490 = [row for row in results if int(row["seed"]) == 490]
    safe_seed490 = [
        row
        for row in seed490
        if all(
            row["gates"][key]
            for key in (
                "medium_mean_nondecrease",
                "high_absolute",
                "high_relative",
                "low_raw_guard",
                "capacity",
                "disabled_flow",
            )
        )
    ]
    safe_seed490.sort(
        key=lambda row: (
            float(row["medium_p10_gap"]),
            float(row["medium_p1_gap"]),
            -float(row["candidate_inversion_positive_gap_mean"]),
        ),
        reverse=True,
    )
    top_two = [float(row["lambda_multiplier"]) for row in safe_seed490[:2]]

    required_seed_count = {2: 2, 3: 2, 4: 3}.get(level, 1)
    winners = []
    for multiplier in multipliers:
        selected = [
            row for row in results if float(row["lambda_multiplier"]) == float(multiplier)
        ]
        if len(selected) == len(seeds) and len(selected) >= required_seed_count and all(
            bool(row["passes"]) for row in selected
        ):
            winners.append(
                {
                    "lambda_multiplier": float(multiplier),
                    "mean_medium_p10_gap": float(
                        np.mean([float(row["medium_p10_gap"]) for row in selected])
                    ),
                    "mean_medium_p1_gap": float(
                        np.mean([float(row["medium_p1_gap"]) for row in selected])
                    ),
                    "mean_inversion_severity": float(
                        np.mean(
                            [float(row["candidate_inversion_positive_gap_mean"]) for row in selected]
                        )
                    ),
                }
            )
    winners.sort(
        key=lambda row: (
            row["mean_medium_p10_gap"],
            row["mean_medium_p1_gap"],
            -row["mean_inversion_severity"],
        ),
        reverse=True,
    )
    selected_multiplier = None if not winners else float(winners[0]["lambda_multiplier"])
    bootstrap = None
    if level == 4 and selected_multiplier is not None:
        bootstrap = hierarchical_bootstrap_p10_gap(
            level, penalty, selected_multiplier, seeds
        )
    promote = selected_multiplier is not None and (bootstrap is None or bootstrap["passes"])
    decision = {
        "level": level,
        "penalty": penalty,
        "seeds": seeds,
        "multipliers_requested": multipliers,
        "results": results,
        "top_two_after_seed490_safety_screen": top_two,
        "winners": winners,
        "selected_multiplier": selected_multiplier,
        "bootstrap_medium_p10_gap": bootstrap,
        "promote": promote,
        "thresholds": {
            "medium_mean_gap_min": 0.0,
            "medium_p1_gap_min": 0.01,
            "medium_p10_gap_min": 0.01,
            "high_mean_absolute_min": 0.98,
            "high_mean_gap_min": -0.005,
            "low_fulfill_mean_gap_min": -0.02,
            "inversion_severity_reduction_min": 0.50,
            "inversion_violation_fraction_max": 0.25 if level == 4 else 0.50,
            "paired_medium_positive_fraction_min": 0.60,
            "max_post_admission_mlu": 1.0001,
            "max_disabled_flow": 1e-8,
            "level3_and_4_mean_priority_order": True,
        },
    }
    output_dir = runner.OUTPUT_ROOT / runner.LEVELS[level]["label"] / "analysis"
    output = output_dir / f"{penalty}_decision.json"
    runner.write_json(output, decision)
    flat_rows = []
    for row in results:
        flat = {key: value for key, value in row.items() if key != "gates"}
        flat.update({f"gate_{key}": value for key, value in row["gates"].items()})
        flat_rows.append(flat)
    runner.write_csv(output.with_suffix(".csv"), flat_rows)
    print(json.dumps(decision, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--penalty", choices=runner.PENALTIES, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    analyze(args.level, args.penalty, args.multipliers, args.seeds)


if __name__ == "__main__":
    main()
