"""Select Hattrick/Hattrick-f3 checkpoints using validation metrics only.

Protocol
--------
1. Consider only epochs for which an explicit ``epoch_XXX.pt`` checkpoint
   still exists.
2. For each load, let H* be the largest validation High mean across both
   methods, so Hattrick and Hattrick-f3 use the same non-inferiority threshold.
3. Keep epochs with High mean >= H* - delta.  For 2x, also require the
   original absolute High target, High mean >= 0.995.
4. Among feasible epochs, maximize Medium mean; break ties with Medium P10,
   Medium P1, High mean, and finally the earlier epoch.

On a common CDF interval [0, b] containing every observation,
integral_0^b CDF(x) dx = b - mean.  Maximizing Medium mean therefore minimizes
the requested Medium CDF area without reading the test split.

This is a post-hoc frozen reproducibility protocol for the existing runs.  It
must not be described as a pre-registered rule for experiments that were
originally selected by inspecting their results.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EPOCH_RE = re.compile(r"^epoch_(\d+)\.pt$")


@dataclass(frozen=True)
class RunSpec:
    history: str
    checkpoint_root: str
    high_floor: float | None


RUNS: dict[str, dict[str, RunSpec]] = {
    "2x": {
        "Hattrick": RunSpec(
            history=(
                "test_diff_path/full_geant_2x_hattrick_f/artifacts/"
                "hattrick/seed_{seed}/train_history.csv"
            ),
            checkpoint_root=(
                "test_diff_path/full_geant_2x_hattrick_f/artifacts/"
                "hattrick/seed_{seed}"
            ),
            high_floor=0.995,
        ),
        "Hattrick-f3": RunSpec(
            history=(
                "test_diff_path/full_geant_2x_hattrick_f3/artifacts/"
                "seed_{seed}/train_history.csv"
            ),
            checkpoint_root=(
                "test_diff_path/full_geant_2x_hattrick_f3/artifacts/seed_{seed}"
            ),
            high_floor=0.995,
        ),
    },
    "3x": {
        "Hattrick": RunSpec(
            history=(
                "test_diff_path/full_geant_3x_hattrick/artifacts/"
                "hattrick/seed_{seed}/train_history.csv"
            ),
            checkpoint_root=(
                "test_diff_path/full_geant_3x_hattrick/artifacts/"
                "hattrick/seed_{seed}"
            ),
            high_floor=None,
        ),
        "Hattrick-f3": RunSpec(
            history=(
                "test_diff_path/full_geant_3x_hattrick_f3/artifacts/"
                "seed_{seed}/train_history.csv"
            ),
            checkpoint_root=(
                "test_diff_path/full_geant_3x_hattrick_f3/artifacts/seed_{seed}"
            ),
            high_floor=None,
        ),
    },
}


NUMERIC_FIELDS = (
    "high_norm_mean",
    "high_norm_p10",
    "high_norm_p1",
    "high_norm_min",
    "medium_norm_mean",
    "medium_norm_p10",
    "medium_norm_p1",
    "low_norm_mean",
)


def explicit_saved_epochs(checkpoint_root: Path) -> set[int]:
    epochs: set[int] = set()
    for path in checkpoint_root.rglob("epoch_*.pt"):
        match = EPOCH_RE.match(path.name)
        if match:
            epochs.add(int(match.group(1)))
    return epochs


def load_saved_rows(spec: RunSpec, seed: int) -> list[dict[str, float | int]]:
    history = REPO_ROOT / spec.history.format(seed=seed)
    checkpoint_root = REPO_ROOT / spec.checkpoint_root.format(seed=seed)
    if not history.is_file():
        raise FileNotFoundError(f"validation history does not exist: {history}")
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_root}")

    saved_epochs = explicit_saved_epochs(checkpoint_root)
    if not saved_epochs:
        raise RuntimeError(f"no explicit epoch checkpoints found below: {checkpoint_root}")

    rows: list[dict[str, float | int]] = []
    with history.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            epoch = int(raw["epoch"])
            if epoch not in saved_epochs:
                continue
            row: dict[str, float | int] = {"epoch": epoch}
            for field in NUMERIC_FIELDS:
                row[field] = float(raw[field])
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no saved epoch has a validation row in: {history}")
    return rows


def select_epoch(
    rows: list[dict[str, float | int]],
    *,
    threshold: float,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    feasible = [
        row for row in rows if float(row["high_norm_mean"]) >= threshold
    ]
    if not feasible:
        raise RuntimeError(f"no High-feasible checkpoint: threshold={threshold}")

    selected = min(
        feasible,
        key=lambda row: (
            -float(row["medium_norm_mean"]),
            -float(row["medium_norm_p10"]),
            -float(row["medium_norm_p1"]),
            -float(row["high_norm_mean"]),
            int(row["epoch"]),
        ),
    )
    return selected, feasible


def format_row(dataset: str, method: str, result: dict[str, object]) -> str:
    row = result["selected"]
    assert isinstance(row, dict)
    return (
        f"{dataset:<3} {method:<11} {int(row['epoch']):>5} "
        f"{float(row['high_norm_mean']):>14.9f} "
        f"{float(row['high_norm_p10']):>14.9f} "
        f"{float(row['high_norm_p1']):>14.9f} "
        f"{float(row['medium_norm_mean']):>16.9f} "
        f"{float(row['medium_norm_p10']):>16.9f} "
        f"{float(row['medium_norm_p1']):>16.9f} "
        f"{float(row['low_norm_mean']):>14.9f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select saved checkpoints using validation High non-inferiority and Medium mean"
    )
    parser.add_argument(
        "--dataset", choices=("2x", "3x", "all"), default="all"
    )
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument(
        "--high-margin",
        type=float,
        default=0.003,
        help="maximum High-mean loss from the load-wide best saved epoch",
    )
    parser.add_argument(
        "--show-candidates",
        action="store_true",
        help="also print every High-feasible epoch in Medium-mean order",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.high_margin < 0:
        raise ValueError("--high-margin must be non-negative")
    datasets = RUNS if args.dataset == "all" else {args.dataset: RUNS[args.dataset]}
    results: dict[tuple[str, str], dict[str, object]] = {}

    for dataset, methods in datasets.items():
        rows_by_method = {
            method: load_saved_rows(spec, args.seed)
            for method, spec in methods.items()
        }
        high_best = max(
            float(row["high_norm_mean"])
            for rows in rows_by_method.values()
            for row in rows
        )
        high_floor = next(iter(methods.values())).high_floor
        threshold = high_best - args.high_margin
        if high_floor is not None:
            threshold = max(threshold, high_floor)
        for method, spec in methods.items():
            selected, feasible = select_epoch(
                rows_by_method[method],
                threshold=threshold,
            )
            results[(dataset, method)] = {
                "selected": selected,
                "high_best": high_best,
                "threshold": threshold,
                "feasible": feasible,
            }

    print(
        "load method       epoch high_norm_mean high_norm_p10  high_norm_p1 "
        "medium_norm_mean medium_norm_p10  medium_norm_p1 low_norm_mean"
    )
    print("-" * 139)
    for (dataset, method), result in results.items():
        print(format_row(dataset, method, result))

    print()
    for dataset in datasets:
        base = results[(dataset, "Hattrick")]["selected"]
        improved = results[(dataset, "Hattrick-f3")]["selected"]
        assert isinstance(base, dict) and isinstance(improved, dict)
        print(
            f"{dataset} validation delta (Hattrick-f3 - Hattrick): "
            f"High mean={float(improved['high_norm_mean']) - float(base['high_norm_mean']):+.9f}, "
            f"Medium mean={float(improved['medium_norm_mean']) - float(base['medium_norm_mean']):+.9f}, "
            f"Medium P10={float(improved['medium_norm_p10']) - float(base['medium_norm_p10']):+.9f}, "
            f"Medium P1={float(improved['medium_norm_p1']) - float(base['medium_norm_p1']):+.9f}"
        )

    if args.show_candidates:
        for (dataset, method), result in results.items():
            print(
                f"\n[{dataset} {method}] High best={result['high_best']:.9f}, "
                f"threshold={result['threshold']:.9f}"
            )
            feasible = result["feasible"]
            assert isinstance(feasible, list)
            for row in sorted(
                feasible,
                key=lambda item: -float(item["medium_norm_mean"]),
            ):
                print(
                    f"  epoch={int(row['epoch']):>3} "
                    f"Hmean={float(row['high_norm_mean']):.9f} "
                    f"Mmean={float(row['medium_norm_mean']):.9f} "
                    f"Mp10={float(row['medium_norm_p10']):.9f} "
                    f"Mp1={float(row['medium_norm_p1']):.9f}"
                )


if __name__ == "__main__":
    main()
