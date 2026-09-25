from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_hattrick_residual_lp as residual
import run_hattrick_strict2x_research as research


def keyed(rows: list[dict]) -> dict[tuple[int, str], dict]:
    result = {(int(row["snapshot"]), row["class"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("Duplicate snapshot/class rows")
    return result


def values(rows: dict[tuple[int, str], dict], class_name: str, field: str) -> np.ndarray:
    return np.asarray(
        [float(rows[key][field]) for key in sorted(rows) if key[1] == class_name],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen Level-3 promotion gate for residual LP.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[490, 491])
    args = parser.parse_args()
    if sorted(args.seeds) != [490, 491] or len(args.seeds) != 2:
        raise RuntimeError("This precommitted Level-3 gate requires exactly seeds 490 and 491")

    level = 3
    spec = research.LEVELS[level]
    level_dir = residual.OUTPUT_ROOT / spec["label"]
    expected_runner = residual.sha256(Path(residual.__file__).resolve())
    expected_baseline_runner = residual.sha256(Path(research.__file__).resolve())
    expected_reader = residual.sha256(residual.ROOT / "utils" / "build_dataset_within_cluster.py")
    expected_core = residual.sha256(residual.ROOT / "frameworks" / "hattrick_system.py")
    expected_keys = {
        (snapshot, class_name)
        for snapshot in range(spec["evaluation"][0], spec["evaluation"][1])
        for class_name in research.CLASSES
    }
    results = []
    for seed in args.seeds:
        baseline_dir = level_dir / "unchanged" / f"seed_{seed}"
        candidate_dir = level_dir / residual.APPROACH / f"seed_{seed}"
        baseline = keyed(research.read_csv(baseline_dir / "best_evaluation_metrics.csv"))
        candidate = keyed(research.read_csv(candidate_dir / "best_evaluation_metrics.csv"))
        if set(baseline) != expected_keys or set(candidate) != expected_keys:
            raise RuntimeError(f"Seed {seed} does not contain exactly the frozen evaluation keys")

        baseline_config = json.loads((baseline_dir / "config.json").read_text(encoding="utf-8"))
        candidate_config = json.loads((candidate_dir / "config.json").read_text(encoding="utf-8"))
        baseline_checkpoint_hash = residual.sha256(baseline_dir / "best_model.pt")
        provenance_passes = bool(
            baseline_config["source_sha256"]["test_diff_path/run_hattrick_strict2x_research.py"]
            == expected_baseline_runner
            and baseline_config["source_sha256"]["utils/build_dataset_within_cluster.py"]
            == expected_reader
            and baseline_config["source_sha256"]["frameworks/hattrick_system.py"] == expected_core
            and candidate_config["source_sha256"]["test_diff_path/run_hattrick_residual_lp.py"]
            == expected_runner
            and candidate_config["source_sha256"]["utils/build_dataset_within_cluster.py"]
            == expected_reader
            and candidate_config["source_sha256"]["frameworks/hattrick_system.py"] == expected_core
            and candidate_config["source_sha256"]["matched_baseline_checkpoint"]
            == baseline_checkpoint_hash
            and baseline_config["data_access"]["max_source_index_read"] == 399
            and candidate_config["dataset_access"]["max_source_index_read"] == 399
            and not baseline_config["evaluation_is_final_test"]
            and not candidate_config["evaluation_is_final_test"]
        )

        b_medium = values(baseline, "Medium", "norm_fulfill")
        c_medium = values(candidate, "Medium", "norm_fulfill")
        b_high = values(baseline, "High", "norm_fulfill")
        c_high = values(candidate, "High", "norm_fulfill")
        b_low = values(baseline, "Low", "norm_fulfill")
        c_low = values(candidate, "Low", "norm_fulfill")
        b_mlu = values(baseline, "Low", "admitted_capacity_ratio")
        c_mlu = values(candidate, "Low", "admitted_capacity_ratio")

        medium_gap = c_medium - b_medium
        item = {
            "seed": seed,
            "baseline_medium_p10": float(np.percentile(b_medium, 10)),
            "candidate_medium_p10": float(np.percentile(c_medium, 10)),
            "medium_p10_gap": float(np.percentile(c_medium, 10) - np.percentile(b_medium, 10)),
            "baseline_medium_p1": float(np.percentile(b_medium, 1)),
            "candidate_medium_p1": float(np.percentile(c_medium, 1)),
            "medium_p1_gap": float(np.percentile(c_medium, 1) - np.percentile(b_medium, 1)),
            "high_mean_gap": float(c_high.mean() - b_high.mean()),
            "common_post_admission_mlu_mean_gap": float(c_mlu.mean() - b_mlu.mean()),
            "candidate_max_admitted_capacity_ratio": float(
                max(float(row["admitted_capacity_ratio"]) for row in candidate.values())
            ),
            "candidate_max_disabled_flow": float(
                max(float(row["disabled_flow"]) for row in candidate.values())
            ),
            "paired_medium_mean_gap": float(medium_gap.mean()),
            "paired_medium_min_gap": float(medium_gap.min()),
            "paired_medium_positive_slices": int((medium_gap > 0.0).sum()),
            "paired_medium_total_slices": int(medium_gap.size),
            "low_mean_gap": float(c_low.mean() - b_low.mean()),
            "low_p10_gap": float(np.percentile(c_low, 10) - np.percentile(b_low, 10)),
            "provenance_passes": provenance_passes,
        }
        item["passes"] = bool(
            item["medium_p10_gap"] >= 0.02
            and item["high_mean_gap"] >= -0.01
            and item["common_post_admission_mlu_mean_gap"] <= 0.01
            and item["candidate_max_admitted_capacity_ratio"] <= 1.0001
            and item["candidate_max_disabled_flow"] <= 1e-8
            and item["paired_medium_positive_slices"] >= 0.5 * item["paired_medium_total_slices"]
            and provenance_passes
        )
        results.append(item)

    output = {
        "level": level,
        "approach": residual.APPROACH,
        "evaluation_window": list(spec["evaluation"]),
        "final_test_opened": False,
        "results": results,
        "promote": all(item["passes"] for item in results),
        "rule_frozen_before_level3_results": {
            "medium_p10_gap_min": 0.02,
            "high_mean_gap_min": -0.01,
            "common_post_admission_mlu_mean_gap_max": 0.01,
            "candidate_max_admitted_capacity_ratio": 1.0001,
            "candidate_max_disabled_flow": 1e-8,
            "paired_positive_fraction_min": 0.5,
            "provenance_required": True,
            "low_is_reported_but_not_a_priority_gate": True,
        },
        "source_sha256": {
            "test_diff_path/analyze_residual_lp_level3.py": residual.sha256(Path(__file__).resolve()),
            "test_diff_path/run_hattrick_residual_lp.py": expected_runner,
            "test_diff_path/run_hattrick_strict2x_research.py": expected_baseline_runner,
            "utils/build_dataset_within_cluster.py": expected_reader,
            "frameworks/hattrick_system.py": expected_core,
        },
    }
    path = level_dir / "hattrick_residual_lp_level3_gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
