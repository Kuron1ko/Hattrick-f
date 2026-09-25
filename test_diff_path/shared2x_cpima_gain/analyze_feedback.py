from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR / "artifacts"
CLASSES = ("High", "Medium", "Low")
SEEDS = (490, 491)


def read_rows(path: Path) -> dict[tuple[int, str], dict]:
    with path.open(encoding="utf-8") as handle:
        return {
            (int(row["snapshot"]), row["class"]): row
            for row in csv.DictReader(handle)
        }


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p1": float(np.percentile(values, 1)),
        "p10": float(np.percentile(values, 10)),
    }


def bootstrap_mean(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(20_000, values.size))
    samples = values[indices].mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def validation_reserve() -> float:
    base = ROOT / "level3_full_validation" / "low_original"
    baseline = read_rows(base / "gain_0p5" / "order_seed_490" / "baseline_metrics.csv")
    aggressive = read_rows(base / "gain_1p5" / "order_seed_490" / "candidate_metrics.csv")
    worst_loss = max(
        float(baseline[(snapshot, "Low")]["admitted_traffic"])
        - float(aggressive[(snapshot, "Low")]["admitted_traffic"])
        for snapshot in range(350, 400)
    )
    return 2.0 * max(worst_loss, 0.0)


def run_controller(seed: int, reserve: float) -> tuple[dict, list[dict], dict[str, np.ndarray]]:
    base = ROOT / "level4_confirmation" / "low_original"
    baseline = read_rows(base / "gain_0p5" / f"order_seed_{seed}" / "baseline_metrics.csv")
    weak = read_rows(base / "gain_0p5" / f"order_seed_{seed}" / "candidate_metrics.csv")
    aggressive = read_rows(base / "gain_1p5" / f"order_seed_{seed}" / "candidate_metrics.csv")
    debt = 0.0
    chosen: dict[tuple[int, str], dict] = {}
    trace = []
    aggressive_count = 0
    for snapshot in range(400, 500):
        use_aggressive = debt >= reserve
        source = aggressive if use_aggressive else weak
        aggressive_count += int(use_aggressive)
        debt_before = debt
        low_gap = (
            float(source[(snapshot, "Low")]["admitted_traffic"])
            - float(baseline[(snapshot, "Low")]["admitted_traffic"])
        )
        debt += low_gap
        for class_name in CLASSES:
            chosen[(snapshot, class_name)] = source[(snapshot, class_name)]
        trace.append(
            {
                "snapshot": snapshot,
                "temperature": 1.5 if use_aggressive else 0.5,
                "debt_before": debt_before,
                "low_admitted_gap": low_gap,
                "debt_after": debt,
            }
        )

    arrays: dict[str, np.ndarray] = {}
    result = {"seed": seed, "aggressive_slices": aggressive_count, "final_debt": debt, "classes": {}}
    for class_name in CLASSES:
        base_values = np.asarray(
            [float(baseline[(snapshot, class_name)]["norm_fulfill"]) for snapshot in range(400, 500)]
        )
        candidate_values = np.asarray(
            [float(chosen[(snapshot, class_name)]["norm_fulfill"]) for snapshot in range(400, 500)]
        )
        arrays[f"{class_name}_gap"] = candidate_values - base_values
        base_summary = summarize(base_values)
        candidate_summary = summarize(candidate_values)
        result["classes"][class_name] = {
            "baseline": base_summary,
            "candidate": candidate_summary,
            "gaps": {
                key: candidate_summary[key] - base_summary[key]
                for key in ("mean", "p1", "p10")
            },
        }
    return result, trace, arrays


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    output = THIS_DIR / "feedback_result"
    output.mkdir(parents=True, exist_ok=True)
    reserve = validation_reserve()
    results = []
    arrays = []
    for seed in SEEDS:
        result, trace, current_arrays = run_controller(seed, reserve)
        results.append(result)
        arrays.append(current_arrays)
        write_csv(output / f"controller_trace_seed_{seed}.csv", trace)

    average = {"classes": {}}
    bootstrap = {}
    for class_name in CLASSES:
        gap = np.mean([item[f"{class_name}_gap"] for item in arrays], axis=0)
        average["classes"][class_name] = {
            "mean_gap": float(np.mean(gap)),
            "p1_gap_average_of_orders": float(
                np.mean([result["classes"][class_name]["gaps"]["p1"] for result in results])
            ),
            "p10_gap_average_of_orders": float(
                np.mean([result["classes"][class_name]["gaps"]["p10"] for result in results])
            ),
        }
        bootstrap[class_name] = {
            "paired_mean_gap_ci95": bootstrap_mean(gap, 20260820 + CLASSES.index(class_name))
        }

    payload = {
        "method": "Causal asymmetric C-PIMA temperature controller",
        "strong_temperature": 1.5,
        "weak_temperature": 0.5,
        "low_shield": "original bounded C-PIMA Low head at both temperatures",
        "reserve": reserve,
        "reserve_rule": "2x the worst one-snapshot aggressive Low admitted-traffic loss on validation 350-399",
        "inference_contract": "current ESM policy and debt accumulated from completed prior snapshots only; no current actual TM enters current policy",
        "results": results,
        "two_order_average": average,
        "bootstrap": bootstrap,
    }
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
