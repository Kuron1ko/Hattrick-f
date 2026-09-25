from __future__ import annotations

"""CPU-only paired statistics for Hattrick-f final-validation CSV files."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
OUTPUT_ROOT = ARTIFACTS / "statistics"
BOOTSTRAP_SEED = 20260825
BLOCK_LENGTHS = (5, 12)
RAW_HIGH_DIAGNOSTIC_THRESHOLD = 0.995
COMPARISONS = (
    ("hattrick_f_minus_phase_a", "hattrick_f", "phase_a"),
    (
        "hattrick_f_minus_six_loss_control",
        "hattrick_f",
        "six_loss_control",
    ),
)
POINT_STATISTICS = ("mean", "median", "p1", "p10", "min")
EXPECTED_SEEDS = (490, 491, 492)
EXPECTED_METHODS = ("phase_a", "hattrick_f", "six_loss_control")
EXPECTED_CLASSES = ("High", "Medium", "Low")
EXPECTED_WINDOWS = {"near": 500, "far": 500}
EXPECTED_CAPACITY_AUDIT_SNAPSHOTS = 20
EXPECTED_PRODUCER_EXPERIMENTS = {
    "Hattrick-f method-new temporal holdout and global ESM bias",
    "Hattrick-f strict-ESM robustness",
}
EXPECTED_SNAPSHOT_SETS = {
    "near": set(range(500, 1000)),
    "far": set(range(9000, 9500)),
}
EXPECTED_CAPACITY_SNAPSHOT_SETS = {
    name: {min(indices) + offset for offset in range(0, 500, 25)}
    for name, indices in EXPECTED_SNAPSHOT_SETS.items()
}
EXPECTED_SCENARIOS = {
    "global_bias": {
        "global_bias_0p8",
        "global_bias_0p9",
        "global_bias_1",
        "global_bias_1p1",
        "global_bias_1p2",
    },
    "class_bias": {
        "high_bias_0p8",
        "high_bias_1p2",
        "medium_bias_0p8",
        "medium_bias_1p2",
    },
    "shape_noise": {
        f"od_lognormal_sigma_{sigma}_seed_{seed}"
        for sigma in ("0p1", "0p2")
        for seed in (20260825, 20260826, 20260827)
    },
    "capacity_derating": {
        f"link_{link:02d}_{mode}_factor_{factor}{suffix}"
        for link in range(36)
        for mode, factor, suffix in (
            ("announced", "0p5", ""),
            ("announced", "0p1", ""),
            ("unannounced", "0p5", ""),
            ("unannounced", "0p1", ""),
            ("unannounced", "0p01", "_near_outage"),
        )
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def compact_number(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def discover_inputs() -> tuple[list[Path], str]:
    formal = [
        ARTIFACTS / "holdout" / "snapshots.csv",
        ARTIFACTS / "robustness" / "snapshots.csv",
    ]
    existing_formal = [path for path in formal if path.exists()]
    if existing_formal:
        return existing_formal, "formal"
    smoke = [
        ARTIFACTS / "holdout_smoke" / "snapshots.csv",
        ARTIFACTS / "robustness_smoke" / "snapshots.csv",
    ]
    existing_smoke = [path for path in smoke if path.exists()]
    if existing_smoke:
        return existing_smoke, "smoke_fallback"
    return [], "none"


def normalized_scenario(row: dict) -> tuple[str, str]:
    family = str(row.get("scenario_family", "")).strip()
    scenario_id = str(row.get("scenario_id", "")).strip()
    if family and scenario_id:
        return family, scenario_id
    bias = float(row.get("esm_global_bias", 1.0))
    return "global_bias", f"global_bias_{compact_number(bias)}"


def audit_input_summaries(paths: list[Path]) -> tuple[list[dict], list[str]]:
    """Read producer metadata; paths alone never establish formality."""
    audits = []
    reasons = []
    expected_pairs = {
        (seed, method) for seed in EXPECTED_SEEDS for method in EXPECTED_METHODS
    }
    for csv_path in paths:
        summary_path = csv_path.parent / "summary.json"
        if not summary_path.is_file():
            reasons.append(f"missing producer summary.json for {csv_path}")
            audits.append(
                {
                    "snapshot_csv": str(csv_path),
                    "summary_json": str(summary_path),
                    "summary_present": False,
                    "formal_result": False,
                }
            )
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        experiment = str(summary.get("experiment", ""))
        formal = summary.get("formal_result") is True
        seeds = tuple(int(value) for value in summary.get("seeds", []))
        windows = summary.get("windows", {})
        window_counts = {
            str(name): int(bounds[1]) - int(bounds[0])
            for name, bounds in windows.items()
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2
        }
        skipped = summary.get("skipped_methods", [])
        actual_pairs = {
            (int(item["seed"]), str(item["method"]))
            for item in summary.get("checkpoints", [])
            if "seed" in item and "method" in item
        }
        local_reasons = []
        if not formal:
            local_reasons.append("producer summary formal_result is not true")
        if experiment not in EXPECTED_PRODUCER_EXPERIMENTS:
            local_reasons.append(f"unexpected producer experiment {experiment!r}")
        declared_csv = summary.get("snapshot_csv")
        if not declared_csv or Path(declared_csv).resolve() != csv_path.resolve():
            local_reasons.append(
                "producer summary snapshot_csv does not identify this exact CSV"
            )
        actual_csv_hash = sha256(csv_path)
        if summary.get("snapshot_csv_sha256") != actual_csv_hash:
            local_reasons.append(
                "producer summary snapshot_csv_sha256 is missing or mismatched"
            )
        if seeds != EXPECTED_SEEDS:
            local_reasons.append(
                f"producer seeds expected {list(EXPECTED_SEEDS)}, got {list(seeds)}"
            )
        if window_counts != EXPECTED_WINDOWS:
            local_reasons.append(
                f"producer windows expected {EXPECTED_WINDOWS}, got {window_counts}"
            )
        if actual_pairs != expected_pairs:
            local_reasons.append(
                "producer method/seed checkpoint set is incomplete: "
                f"expected={sorted(expected_pairs)}, actual={sorted(actual_pairs)}"
            )
        if skipped:
            local_reasons.append("producer summary contains skipped_methods")
        completeness = summary.get("completeness")
        if isinstance(completeness, dict):
            if completeness.get("all_methods_and_seeds_complete") is not True:
                local_reasons.append(
                    "producer completeness says methods/seeds are incomplete"
                )
            if completeness.get("all_windows_complete") is not True:
                local_reasons.append(
                    "producer completeness says windows are incomplete"
                )
        formal_validation = summary.get("formal_validation")
        if isinstance(formal_validation, dict):
            coverage = formal_validation.get("coverage", {})
            if coverage.get("complete") is not True:
                local_reasons.append("producer formal coverage audit is incomplete")
            if int(
                formal_validation.get("required_prediction_snapshots_per_window", -1)
            ) != 500:
                local_reasons.append(
                    "producer does not require 500 prediction snapshots per window"
                )
            if int(
                formal_validation.get("required_capacity_snapshots_per_window", -1)
            ) != EXPECTED_CAPACITY_AUDIT_SNAPSHOTS:
                local_reasons.append(
                    "producer capacity audit sample count differs from the preregistered 20"
                )
        reasons.extend(f"{summary_path}: {reason}" for reason in local_reasons)
        audits.append(
            {
                "snapshot_csv": str(csv_path),
                "snapshot_csv_sha256": sha256(csv_path),
                "summary_json": str(summary_path.resolve()),
                "summary_json_sha256": sha256(summary_path),
                "summary_present": True,
                "experiment": experiment,
                "formal_result": formal,
                "seeds": list(seeds),
                "window_snapshot_counts": window_counts,
                "method_seed_pairs": [
                    {"seed": seed, "method": method}
                    for seed, method in sorted(actual_pairs)
                ],
                "skipped_method_count": len(skipped),
                "passes_formal_metadata_audit": not local_reasons,
                "audit_reasons": local_reasons,
            }
        )
    experiments = [
        audit.get("experiment")
        for audit in audits
        if audit.get("summary_present")
    ]
    if len(experiments) != len(EXPECTED_PRODUCER_EXPERIMENTS) or set(
        experiments
    ) != EXPECTED_PRODUCER_EXPERIMENTS:
        reasons.append(
            "formal statistics require exactly one complete holdout producer and "
            "one complete robustness producer; observed experiments="
            f"{experiments}"
        )
    return audits, reasons


def audit_rows_for_formality(
    rows: list[dict], method_summary: list[dict], skipped: list[dict]
) -> list[str]:
    reasons = []
    seeds = sorted({int(row["seed"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    windows = sorted({str(row["window"]) for row in rows})
    classes = sorted({str(row["class"]) for row in rows})
    if seeds != list(EXPECTED_SEEDS):
        reasons.append(f"row seeds expected {list(EXPECTED_SEEDS)}, got {seeds}")
    if methods != sorted(EXPECTED_METHODS):
        reasons.append(
            f"row methods expected {sorted(EXPECTED_METHODS)}, got {methods}"
        )
    if windows != sorted(EXPECTED_WINDOWS):
        reasons.append(
            f"row windows expected {sorted(EXPECTED_WINDOWS)}, got {windows}"
        )
    if classes != sorted(EXPECTED_CLASSES):
        reasons.append(
            f"row classes expected {sorted(EXPECTED_CLASSES)}, got {classes}"
        )
    observed_scenarios: dict[str, set[str]] = {}
    snapshots_by_group: dict[tuple, set[int]] = {}
    for row in rows:
        family = str(row["scenario_family"])
        scenario_id = str(row["scenario_id"])
        observed_scenarios.setdefault(family, set()).add(scenario_id)
        key = (
            int(row["seed"]),
            str(row["method"]),
            str(row["window"]),
            family,
            scenario_id,
            str(row["class"]),
        )
        snapshots_by_group.setdefault(key, set()).add(int(row["snapshot"]))
    if observed_scenarios != EXPECTED_SCENARIOS:
        scenario_differences = {
            family: {
                "missing": sorted(
                    EXPECTED_SCENARIOS.get(family, set())
                    - observed_scenarios.get(family, set())
                ),
                "unexpected": sorted(
                    observed_scenarios.get(family, set())
                    - EXPECTED_SCENARIOS.get(family, set())
                ),
            }
            for family in sorted(set(EXPECTED_SCENARIOS) | set(observed_scenarios))
            if observed_scenarios.get(family, set())
            != EXPECTED_SCENARIOS.get(family, set())
        }
        reasons.append(f"formal scenario set mismatch: {scenario_differences}")
    expected_group_keys = {
        (seed, method, window, family, scenario_id, class_name)
        for seed in EXPECTED_SEEDS
        for method in EXPECTED_METHODS
        for window in EXPECTED_WINDOWS
        for family, scenario_ids in EXPECTED_SCENARIOS.items()
        for scenario_id in scenario_ids
        for class_name in EXPECTED_CLASSES
    }
    actual_group_keys = set(snapshots_by_group)
    missing_groups = expected_group_keys - actual_group_keys
    unexpected_groups = actual_group_keys - expected_group_keys
    if missing_groups or unexpected_groups:
        reasons.append(
            "formal seed/method/window/scenario/class Cartesian coverage mismatch: "
            f"missing_count={len(missing_groups)}, "
            f"unexpected_count={len(unexpected_groups)}, "
            f"missing_examples={sorted(missing_groups)[:3]}, "
            f"unexpected_examples={sorted(unexpected_groups)[:3]}"
        )
    wrong_snapshot_sets = []
    for key, observed in snapshots_by_group.items():
        seed, method, window, family, scenario_id, class_name = key
        expected = (
            EXPECTED_CAPACITY_SNAPSHOT_SETS.get(window)
            if family == "capacity_derating"
            else EXPECTED_SNAPSHOT_SETS.get(window)
        )
        if expected is None or observed != expected:
            wrong_snapshot_sets.append(
                {
                    "seed": seed,
                    "method": method,
                    "window": window,
                    "scenario_family": family,
                    "scenario_id": scenario_id,
                    "class": class_name,
                    "expected_count": len(expected) if expected is not None else None,
                    "actual_count": len(observed),
                    "missing_first": sorted(expected - observed)[:5]
                    if expected is not None
                    else [],
                    "unexpected_first": sorted(observed - expected)[:5]
                    if expected is not None
                    else sorted(observed)[:5],
                }
            )
    if wrong_snapshot_sets:
        reasons.append(
            f"{len(wrong_snapshot_sets)} method/scenario/class groups have an "
            f"incorrect exact snapshot set; examples={wrong_snapshot_sets[:3]}"
        )
    wrong_counts = [
        {
            "seed": row["seed"],
            "method": row["method"],
            "window": row["window"],
            "scenario_id": row["scenario_id"],
            "class": row["class"],
            "expected": (
                EXPECTED_CAPACITY_AUDIT_SNAPSHOTS
                if str(row["scenario_family"]) == "capacity_derating"
                else EXPECTED_WINDOWS.get(str(row["window"]))
            ),
            "actual": int(row["n"]),
        }
        for row in method_summary
        if str(row["window"]) not in EXPECTED_WINDOWS
        or int(row["n"])
        != (
            EXPECTED_CAPACITY_AUDIT_SNAPSHOTS
            if str(row["scenario_family"]) == "capacity_derating"
            else EXPECTED_WINDOWS[str(row["window"])]
        )
    ]
    if wrong_counts:
        reasons.append(
            f"{len(wrong_counts)} method/scenario/class groups have the wrong "
            "snapshot count (prediction=500, capacity audit=20)"
        )
    seeds_by_scenario: dict[tuple, set[int]] = {}
    methods_by_scenario_seed: dict[tuple, set[str]] = {}
    for row in method_summary:
        scenario = (
            row["window"],
            row["scenario_family"],
            row["scenario_id"],
            row["class"],
            row["method"],
        )
        seeds_by_scenario.setdefault(scenario, set()).add(int(row["seed"]))
        scenario_seed = (
            int(row["seed"]),
            row["window"],
            row["scenario_family"],
            row["scenario_id"],
            row["class"],
        )
        methods_by_scenario_seed.setdefault(scenario_seed, set()).add(
            str(row["method"])
        )
    incomplete_seed_groups = sum(
        seeds != set(EXPECTED_SEEDS) for seeds in seeds_by_scenario.values()
    )
    incomplete_method_groups = sum(
        methods != set(EXPECTED_METHODS)
        for methods in methods_by_scenario_seed.values()
    )
    if incomplete_seed_groups:
        reasons.append(
            f"{incomplete_seed_groups} scenario/class/method groups do not contain all three seeds"
        )
    if incomplete_method_groups:
        reasons.append(
            f"{incomplete_method_groups} seed/scenario/class groups do not contain all three methods"
        )
    if skipped:
        reasons.append(f"{len(skipped)} requested paired comparison groups were skipped")
    return reasons


def read_inputs(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    indexed: dict[tuple, dict] = {}
    duplicate_audit = []
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        if not source_rows:
            raise RuntimeError(f"input CSV is empty: {path}")
        required = {
            "seed",
            "method",
            "window",
            "snapshot",
            "class",
            "raw_fulfill_ratio",
            "admitted_traffic",
        }
        missing = required - set(source_rows[0])
        if missing:
            raise KeyError(f"{path} is missing columns {sorted(missing)}")
        for source in source_rows:
            family, scenario_id = normalized_scenario(source)
            row = {
                "seed": int(source["seed"]),
                "method": str(source["method"]),
                "method_label": str(
                    source.get("method_label", source["method"])
                ),
                "window": str(source["window"]),
                "scenario_family": family,
                "scenario_id": scenario_id,
                "snapshot": int(source["snapshot"]),
                "class": str(source["class"]),
                "raw_fulfill_ratio": float(source["raw_fulfill_ratio"]),
                "admitted_traffic": float(source["admitted_traffic"]),
                "source_file": str(path.resolve()),
            }
            key = (
                row["seed"],
                row["method"],
                row["window"],
                row["scenario_id"],
                row["snapshot"],
                row["class"],
            )
            previous = indexed.get(key)
            if previous is not None:
                same = (
                    abs(
                        previous["raw_fulfill_ratio"]
                        - row["raw_fulfill_ratio"]
                    )
                    <= 1e-12
                    and abs(
                        previous["admitted_traffic"] - row["admitted_traffic"]
                    )
                    <= 1e-9
                )
                duplicate_audit.append(
                    {
                        "key": list(key),
                        "first": previous["source_file"],
                        "second": row["source_file"],
                        "identical": same,
                    }
                )
                if not same:
                    raise RuntimeError(
                        f"conflicting duplicate result for key {key}: "
                        f"{previous['source_file']} vs {row['source_file']}"
                    )
                continue
            indexed[key] = row
    rows = sorted(
        indexed.values(),
        key=lambda row: (
            row["seed"],
            row["window"],
            row["scenario_id"],
            row["class"],
            row["method"],
            row["snapshot"],
        ),
    )
    return rows, duplicate_audit


def point_statistics(values: np.ndarray, prefix: str = "") -> dict:
    return {
        f"{prefix}mean": float(values.mean()),
        f"{prefix}median": float(np.median(values)),
        f"{prefix}p1": float(np.percentile(values, 1)),
        f"{prefix}p10": float(np.percentile(values, 10)),
        f"{prefix}min": float(values.min()),
    }


def method_summaries(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["seed"],
            row["window"],
            row["scenario_family"],
            row["scenario_id"],
            row["class"],
            row["method"],
            row["method_label"],
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for key, selected in sorted(grouped.items()):
        (
            seed,
            window,
            family,
            scenario_id,
            class_name,
            method,
            label,
        ) = key
        selected.sort(key=lambda row: row["snapshot"])
        snapshots = [int(row["snapshot"]) for row in selected]
        values = np.asarray(
            [float(row["raw_fulfill_ratio"]) for row in selected],
            dtype=np.float64,
        )
        high_count = (
            int(np.count_nonzero(values < RAW_HIGH_DIAGNOSTIC_THRESHOLD))
            if class_name == "High"
            else ""
        )
        high_fraction = (
            float(high_count / len(values)) if class_name == "High" else ""
        )
        output.append(
            {
                "seed": seed,
                "window": window,
                "scenario_family": family,
                "scenario_id": scenario_id,
                "class": class_name,
                "method": method,
                "method_label": label,
                "n": int(values.size),
                "snapshot_first": min(snapshots),
                "snapshot_last": max(snapshots),
                "snapshots_consecutive": snapshots
                == list(range(min(snapshots), max(snapshots) + 1)),
                **point_statistics(values, "raw_fulfill_"),
                "high_raw_below_0p995_count": high_count,
                "high_raw_below_0p995_fraction": high_fraction,
                "high_threshold_semantics": (
                    "raw FulfillRatio diagnostic only; not a NormFulFill safety gate"
                    if class_name == "High"
                    else ""
                ),
            }
        )
    return output


def moving_block_bootstrap_means(
    values: np.ndarray,
    requested_block_length: int,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    n = int(values.size)
    if repetitions <= 0:
        return np.asarray([], dtype=np.float64), min(requested_block_length, n)
    block_length = min(int(requested_block_length), n)
    if n == 1:
        return np.full(repetitions, values[0], dtype=np.float64), block_length
    doubled = np.concatenate((values, values))
    prefix = np.concatenate(([0.0], np.cumsum(doubled, dtype=np.float64)))

    def circular_sums(length: int) -> np.ndarray:
        starts = np.arange(n)
        return prefix[starts + length] - prefix[starts]

    full_blocks, remainder = divmod(n, block_length)
    draw_count = full_blocks + int(remainder > 0)
    starts = rng.integers(0, n, size=(repetitions, draw_count))
    total = np.zeros(repetitions, dtype=np.float64)
    if full_blocks:
        block_sums = circular_sums(block_length)
        total += block_sums[starts[:, :full_blocks]].sum(axis=1)
    if remainder:
        remainder_sums = circular_sums(remainder)
        total += remainder_sums[starts[:, -1]]
    return total / float(n), block_length


def iid_bootstrap_means(
    values: np.ndarray, repetitions: int, rng: np.random.Generator
) -> np.ndarray:
    if repetitions <= 0:
        return np.asarray([], dtype=np.float64)
    if values.size == 1:
        return np.full(repetitions, values[0], dtype=np.float64)
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    return values[indices].mean(axis=1)


def ci95(values: np.ndarray) -> tuple[float | str, float | str]:
    if values.size == 0:
        return "", ""
    low, high = np.percentile(values, (2.5, 97.5))
    return float(low), float(high)


def paired_summaries(
    rows: list[dict],
    block_repetitions: int,
    iid_repetitions: int,
    rng: np.random.Generator,
) -> tuple[list[dict], list[dict]]:
    by_group: dict[tuple, dict[str, dict[int, dict]]] = {}
    for row in rows:
        group = (
            row["seed"],
            row["window"],
            row["scenario_family"],
            row["scenario_id"],
            row["class"],
        )
        by_group.setdefault(group, {}).setdefault(row["method"], {})[
            row["snapshot"]
        ] = row
    output = []
    skipped = []
    for group, methods in sorted(by_group.items()):
        seed, window, family, scenario_id, class_name = group
        for comparison, candidate_name, baseline_name in COMPARISONS:
            missing = [
                method
                for method in (candidate_name, baseline_name)
                if method not in methods
            ]
            if missing:
                skipped.append(
                    {
                        "seed": seed,
                        "window": window,
                        "scenario_family": family,
                        "scenario_id": scenario_id,
                        "class": class_name,
                        "comparison": comparison,
                        "reason": "method_missing",
                        "missing_methods": missing,
                    }
                )
                continue
            snapshots = sorted(
                set(methods[candidate_name]) & set(methods[baseline_name])
            )
            candidate_only = sorted(
                set(methods[candidate_name]) - set(methods[baseline_name])
            )
            baseline_only = sorted(
                set(methods[baseline_name]) - set(methods[candidate_name])
            )
            if candidate_only or baseline_only:
                raise RuntimeError(
                    f"unpaired snapshots for {group}/{comparison}: "
                    f"candidate_only={candidate_only[:5]}, "
                    f"baseline_only={baseline_only[:5]}"
                )
            candidate = np.asarray(
                [
                    methods[candidate_name][snapshot]["raw_fulfill_ratio"]
                    for snapshot in snapshots
                ],
                dtype=np.float64,
            )
            baseline = np.asarray(
                [
                    methods[baseline_name][snapshot]["raw_fulfill_ratio"]
                    for snapshot in snapshots
                ],
                dtype=np.float64,
            )
            delta = candidate - baseline
            row = {
                "seed": seed,
                "window": window,
                "scenario_family": family,
                "scenario_id": scenario_id,
                "class": class_name,
                "comparison": comparison,
                "candidate": candidate_name,
                "baseline": baseline_name,
                "n": int(delta.size),
                "snapshot_first": min(snapshots),
                "snapshot_last": max(snapshots),
                "snapshots_consecutive": snapshots
                == list(range(min(snapshots), max(snapshots) + 1)),
                **point_statistics(delta, "delta_"),
                "candidate_better_fraction": float(np.mean(delta > 0)),
                "candidate_equal_fraction": float(np.mean(delta == 0)),
            }
            for block_length in BLOCK_LENGTHS:
                samples, effective = moving_block_bootstrap_means(
                    delta, block_length, block_repetitions, rng
                )
                low, high = ci95(samples)
                row[
                    f"moving_block_l{block_length}_effective_length"
                ] = effective
                row[f"moving_block_l{block_length}_mean_ci95_low"] = low
                row[f"moving_block_l{block_length}_mean_ci95_high"] = high
            iid_samples = iid_bootstrap_means(delta, iid_repetitions, rng)
            iid_low, iid_high = ci95(iid_samples)
            row["iid_paired_mean_ci95_low_secondary"] = iid_low
            row["iid_paired_mean_ci95_high_secondary"] = iid_high
            if class_name == "High":
                candidate_count = int(
                    np.count_nonzero(candidate < RAW_HIGH_DIAGNOSTIC_THRESHOLD)
                )
                baseline_count = int(
                    np.count_nonzero(baseline < RAW_HIGH_DIAGNOSTIC_THRESHOLD)
                )
                row.update(
                    {
                        "candidate_high_raw_below_0p995_count": candidate_count,
                        "candidate_high_raw_below_0p995_fraction": float(
                            candidate_count / candidate.size
                        ),
                        "baseline_high_raw_below_0p995_count": baseline_count,
                        "baseline_high_raw_below_0p995_fraction": float(
                            baseline_count / baseline.size
                        ),
                        "high_threshold_semantics": "raw FulfillRatio diagnostic only; not a NormFulFill safety gate",
                    }
                )
            else:
                row.update(
                    {
                        "candidate_high_raw_below_0p995_count": "",
                        "candidate_high_raw_below_0p995_fraction": "",
                        "baseline_high_raw_below_0p995_count": "",
                        "baseline_high_raw_below_0p995_fraction": "",
                        "high_threshold_semantics": "",
                    }
                )
            output.append(row)
    return output, skipped


def across_seed_summaries(paired: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in paired:
        key = (
            row["window"],
            row["scenario_family"],
            row["scenario_id"],
            row["class"],
            row["comparison"],
            row["candidate"],
            row["baseline"],
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for key, selected in sorted(grouped.items()):
        (
            window,
            family,
            scenario_id,
            class_name,
            comparison,
            candidate,
            baseline,
        ) = key
        item = {
            "window": window,
            "scenario_family": family,
            "scenario_id": scenario_id,
            "class": class_name,
            "comparison": comparison,
            "candidate": candidate,
            "baseline": baseline,
            "seed_count": len(selected),
            "seeds": ";".join(str(row["seed"]) for row in sorted(selected, key=lambda x: x["seed"])),
        }
        for statistic in POINT_STATISTICS:
            values = np.asarray(
                [float(row[f"delta_{statistic}"]) for row in selected],
                dtype=np.float64,
            )
            item[f"delta_{statistic}_across_seed_mean"] = float(values.mean())
            item[f"delta_{statistic}_across_seed_range_min"] = float(values.min())
            item[f"delta_{statistic}_across_seed_range_max"] = float(values.max())
        output.append(item)
    return output


def run(args: argparse.Namespace) -> Path:
    if args.inputs:
        paths = [path.resolve() for path in args.inputs]
        input_mode = "explicit"
    else:
        paths, input_mode = discover_inputs()
        paths = [path.resolve() for path in paths]
    if not paths:
        raise FileNotFoundError(
            "no formal or smoke snapshots.csv exists; pass one or more files with --inputs"
        )
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"input CSV files are missing: {missing}")

    input_audits, metadata_reasons = audit_input_summaries(paths)
    if args.block_bootstrap_repetitions <= 0:
        metadata_reasons.append(
            "formal statistics require block-bootstrap repetitions > 0"
        )
    if args.iid_bootstrap_repetitions <= 0:
        metadata_reasons.append(
            "formal statistics require IID-bootstrap repetitions > 0"
        )
    if metadata_reasons and not args.allow_incomplete:
        raise RuntimeError(
            "refusing to summarize non-formal/incomplete producer artifacts; "
            "pass --allow-incomplete for an explicitly exploratory summary: "
            + " | ".join(metadata_reasons)
        )
    rows, duplicate_audit = read_inputs(paths)
    method = method_summaries(rows)
    rng = np.random.default_rng(args.bootstrap_seed)
    paired, skipped = paired_summaries(
        rows,
        args.block_bootstrap_repetitions,
        args.iid_bootstrap_repetitions,
        rng,
    )
    row_reasons = audit_rows_for_formality(rows, method, skipped)
    incompleteness_reasons = metadata_reasons + row_reasons
    if incompleteness_reasons and not args.allow_incomplete:
        raise RuntimeError(
            "formal summary integrity audit failed; pass --allow-incomplete "
            "only for exploratory output: " + " | ".join(incompleteness_reasons)
        )
    across_seed = across_seed_summaries(paired)
    output_dir = args.output_dir.resolve()
    if args.allow_incomplete and output_dir == OUTPUT_ROOT.resolve():
        raise ValueError(
            "--allow-incomplete cannot write into the default formal statistics "
            "directory; choose an explicit exploratory --output-dir"
        )
    formal_result = not incompleteness_reasons and not args.allow_incomplete
    files = {
        "method_summary": output_dir / "method_summary.csv",
        "paired_summary": output_dir / "paired_summary.csv",
        "across_seed_summary": output_dir / "across_seed_summary.csv",
    }
    atomic_csv(files["method_summary"], method)
    atomic_csv(files["paired_summary"], paired)
    atomic_csv(files["across_seed_summary"], across_seed)
    payload = {
        "schema_version": 1,
        "experiment": "Hattrick-f CPU-only paired statistical summary",
        "formal_result": formal_result,
        "result_class": "formal" if formal_result else "exploratory",
        "allow_incomplete_override": bool(args.allow_incomplete),
        "formal_incompleteness_reasons": incompleteness_reasons,
        "input_mode": input_mode,
        "inputs": [
            {
                "path": str(path),
                "sha256": sha256(path),
            }
            for path in paths
        ],
        "row_count_after_identical_deduplication": len(rows),
        "duplicate_audit": duplicate_audit,
        "producer_summary_audit": input_audits,
        "completeness": {
            "expected_producer_experiments": sorted(EXPECTED_PRODUCER_EXPERIMENTS),
            "actual_producer_experiments": sorted(
                audit.get("experiment", "")
                for audit in input_audits
                if audit.get("summary_present")
            ),
            "expected_seeds": list(EXPECTED_SEEDS),
            "actual_seeds": sorted({int(row["seed"]) for row in rows}),
            "expected_methods": list(EXPECTED_METHODS),
            "actual_methods": sorted({str(row["method"]) for row in rows}),
            "expected_classes": list(EXPECTED_CLASSES),
            "actual_classes": sorted({str(row["class"]) for row in rows}),
            "expected_window_snapshot_counts": EXPECTED_WINDOWS,
            "expected_capacity_audit_snapshots_per_window": EXPECTED_CAPACITY_AUDIT_SNAPSHOTS,
            "expected_scenario_counts": {
                family: len(scenarios)
                for family, scenarios in EXPECTED_SCENARIOS.items()
            },
            "actual_scenario_counts": {
                family: len(
                    {
                        str(row["scenario_id"])
                        for row in rows
                        if str(row["scenario_family"]) == family
                    }
                )
                for family in sorted(
                    {str(row["scenario_family"]) for row in rows}
                )
            },
            "actual_windows": sorted({str(row["window"]) for row in rows}),
            "all_requested_comparisons_present": not skipped,
            "all_method_groups_have_expected_snapshot_count": not any(
                str(row["window"]) not in EXPECTED_WINDOWS
                or int(row["n"])
                != (
                    EXPECTED_CAPACITY_AUDIT_SNAPSHOTS
                    if str(row["scenario_family"]) == "capacity_derating"
                    else EXPECTED_WINDOWS[str(row["window"])]
                )
                for row in method
            ),
        },
        "methods_present": sorted({row["method"] for row in rows}),
        "comparisons_requested": [item[0] for item in COMPARISONS],
        "comparison_groups_completed": len(paired),
        "comparison_groups_skipped": skipped,
        "statistics": {
            "point_estimates": list(POINT_STATISTICS),
            "paired_unit": "snapshot; candidate raw FulfillRatio minus baseline raw FulfillRatio",
            "time_order": "ascending snapshot within seed/window/scenario/class",
            "moving_block_bootstrap": {
                "primary": True,
                "circular_blocks": True,
                "requested_block_lengths": list(BLOCK_LENGTHS),
                "repetitions": args.block_bootstrap_repetitions,
                "ci": "percentile 95% CI for paired mean delta",
                "short_series_rule": "effective block length=min(requested length,n)",
            },
            "ordinary_paired_bootstrap": {
                "primary": False,
                "role": "secondary check",
                "repetitions": args.iid_bootstrap_repetitions,
                "ci": "percentile 95% CI for paired mean delta",
            },
            "bootstrap_seed": args.bootstrap_seed,
            "across_seed": "mean and observed min/max range of each per-seed paired point estimate",
        },
        "high_raw_threshold": {
            "value": RAW_HIGH_DIAGNOSTIC_THRESHOLD,
            "metric": "raw FulfillRatio",
            "interpretation": "diagnostic count/fraction only",
            "explicit_non_interpretation": "not a NormFulFill threshold and not the Hattrick validation safety gate",
        },
        "output_files": {name: str(path.resolve()) for name, path in files.items()},
        "source": {
            "summarize_results.py": sha256(Path(__file__).resolve()),
        },
    }
    atomic_json(output_dir / "summary.json", payload)
    return output_dir


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only paired statistics for holdout/robustness snapshots.csv; "
            "uses formal artifacts when present and GPU smoke artifacts otherwise"
        )
    )
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--block-bootstrap-repetitions", type=int, default=2000
    )
    parser.add_argument("--iid-bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="produce an explicitly exploratory summary from incomplete/smoke inputs",
    )
    args = parser.parse_args()
    if args.block_bootstrap_repetitions < 0:
        parser.error("block bootstrap repetitions cannot be negative")
    if args.iid_bootstrap_repetitions < 0:
        parser.error("iid bootstrap repetitions cannot be negative")
    return args


def main() -> None:
    print(run(parse_cli()), flush=True)


if __name__ == "__main__":
    main()
