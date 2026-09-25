from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def paired_class_values(rows: list[dict], class_name: str) -> np.ndarray:
    selected = sorted(
        (row for row in rows if row["class"] == class_name),
        key=lambda row: int(row["snapshot"]),
    )
    return np.asarray([float(row["norm_fulfill"]) for row in selected])


def statistic(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [values.mean(), np.percentile(values, 1), np.percentile(values, 10)]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    baseline_rows = read_rows(args.run_dir / "baseline_metrics.csv")
    candidate_rows = read_rows(args.run_dir / "candidate_metrics.csv")
    baseline_medium = paired_class_values(baseline_rows, "Medium")
    candidate_medium = paired_class_values(candidate_rows, "Medium")
    if baseline_medium.shape != candidate_medium.shape:
        raise RuntimeError("Paired Medium rows do not align")

    rng = np.random.default_rng(args.seed)
    samples = baseline_medium.size
    boot = np.empty((args.bootstrap, 3), dtype=np.float64)
    for offset in range(0, args.bootstrap, 1000):
        stop = min(offset + 1000, args.bootstrap)
        indices = rng.integers(0, samples, size=(stop - offset, samples))
        b = baseline_medium[indices]
        c = candidate_medium[indices]
        boot[offset:stop, 0] = c.mean(axis=1) - b.mean(axis=1)
        boot[offset:stop, 1] = np.percentile(c, 1, axis=1) - np.percentile(b, 1, axis=1)
        boot[offset:stop, 2] = np.percentile(c, 10, axis=1) - np.percentile(b, 10, axis=1)

    observed = statistic(candidate_medium) - statistic(baseline_medium)
    names = ("mean", "p1", "p10")
    intervals = {
        name: {
            "observed_gap": float(observed[index]),
            "bootstrap_ci95": [
                float(np.percentile(boot[:, index], 2.5)),
                float(np.percentile(boot[:, index], 97.5)),
            ],
            "bootstrap_probability_positive": float(
                np.mean(boot[:, index] > 0.0)
            ),
        }
        for index, name in enumerate(names)
    }
    per_snapshot_gap = candidate_medium - baseline_medium

    exactness = {}
    for class_name in ("High", "Low"):
        baseline = paired_class_values(baseline_rows, class_name)
        candidate = paired_class_values(candidate_rows, class_name)
        exactness[class_name.lower()] = {
            "max_abs_per_snapshot_norm_gap": float(np.max(np.abs(candidate - baseline))),
            "mean_per_snapshot_norm_gap": float(np.mean(candidate - baseline)),
        }

    diagnostics = read_rows(args.run_dir / "planning_diagnostics.csv")
    result = {
        "snapshots": samples,
        "bootstrap_resamples": args.bootstrap,
        "seed": args.seed,
        "medium": {
            **intervals,
            "improved_snapshots": int(np.sum(per_snapshot_gap > 1e-10)),
            "nondecreasing_snapshots": int(np.sum(per_snapshot_gap >= -1e-10)),
            "min_per_snapshot_norm_gap": float(per_snapshot_gap.min()),
            "median_per_snapshot_norm_gap": float(np.median(per_snapshot_gap)),
            "max_per_snapshot_norm_gap": float(per_snapshot_gap.max()),
            "mean_absolute_traffic_gain": float(
                np.mean([float(row["absolute_gain"]) for row in diagnostics])
            ),
        },
        "exactness": exactness,
        "planning_seconds_per_snapshot": float(
            diagnostics[0]["planning_seconds_per_snapshot"]
        ),
    }
    output = args.run_dir / "paired_bootstrap.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
