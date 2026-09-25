from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_hattrick_residual_lp as residual
import run_hattrick_strict2x_research as research


SEEDS = (490, 491, 492)
BOOTSTRAP_SEED = 20260815
BOOTSTRAP_DRAWS = 10_000
MANIFEST = residual.OUTPUT_ROOT / "final_frozen_manifest.json"


def keyed(rows: list[dict]) -> dict[tuple[int, str], dict]:
    result = {(int(row["snapshot"]), row["class"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("Duplicate snapshot/class rows")
    return result


def class_values(rows: dict[tuple[int, str], dict], class_name: str, field: str) -> np.ndarray:
    return np.asarray(
        [float(rows[(snapshot, class_name)][field]) for snapshot in range(400, 500)],
        dtype=np.float64,
    )


def ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def hierarchical_bootstrap(
    baseline: np.ndarray, candidate: np.ndarray, rng: np.random.Generator
) -> dict:
    # Shape is seeds x snapshots.  Resample seeds and paired snapshots within
    # each sampled seed; candidate/control always share the sampled indices.
    mean_gap = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    p10_gap = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    p1_gap = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        seed_index = rng.integers(0, baseline.shape[0], baseline.shape[0])
        sampled_baseline = []
        sampled_candidate = []
        for selected_seed in seed_index:
            snapshot_index = rng.integers(0, baseline.shape[1], baseline.shape[1])
            sampled_baseline.append(baseline[selected_seed, snapshot_index])
            sampled_candidate.append(candidate[selected_seed, snapshot_index])
        b = np.concatenate(sampled_baseline)
        c = np.concatenate(sampled_candidate)
        mean_gap[draw] = np.mean(c - b)
        p10_gap[draw] = np.percentile(c, 10) - np.percentile(b, 10)
        p1_gap[draw] = np.percentile(c, 1) - np.percentile(b, 1)
    return {
        "scheme": "resample seeds, then paired snapshots within each sampled seed",
        "draws": BOOTSTRAP_DRAWS,
        "rng_seed": BOOTSTRAP_SEED,
        "mean_gap_95ci": ci(mean_gap),
        "p10_gap_95ci": ci(p10_gap),
        "p1_gap_95ci": ci(p1_gap),
    }


def summarize(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "p1": float(np.percentile(values, 1)),
        "p10": float(np.percentile(values, 10)),
    }


def main() -> None:
    if not MANIFEST.exists():
        raise RuntimeError("Final-test manifest is absent; test analysis remains sealed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not manifest.get("authorized", False) or manifest.get("seeds") != list(SEEDS):
        raise RuntimeError("Final-test manifest is not authorized for exactly the frozen seeds")
    own_hash = residual.sha256(Path(__file__).resolve())
    if manifest["source_sha256"]["test_diff_path/analyze_residual_lp_level4.py"] != own_hash:
        raise RuntimeError("Final analyzer differs from the frozen manifest")

    level_dir = residual.OUTPUT_ROOT / research.LEVELS[4]["label"]
    expected_keys = {
        (snapshot, class_name)
        for snapshot in range(400, 500)
        for class_name in research.CLASSES
    }
    seed_rows = []
    arrays: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"baseline": [], "candidate": []} for name in research.CLASSES
    }
    provenance = []
    for seed in SEEDS:
        baseline_dir = level_dir / "unchanged" / f"seed_{seed}"
        candidate_dir = level_dir / residual.APPROACH / f"seed_{seed}"
        baseline = keyed(research.read_csv(baseline_dir / "best_evaluation_metrics.csv"))
        candidate = keyed(research.read_csv(candidate_dir / "best_evaluation_metrics.csv"))
        if set(baseline) != expected_keys or set(candidate) != expected_keys:
            raise RuntimeError(f"Seed {seed} does not contain exactly snapshots 400-499 x classes")

        b_config = json.loads((baseline_dir / "config.json").read_text(encoding="utf-8"))
        c_config = json.loads((candidate_dir / "config.json").read_text(encoding="utf-8"))
        checkpoint_hash = residual.sha256(baseline_dir / "best_model.pt")
        policy_artifact = np.load(candidate_dir / "emitted_policy_and_admitted_flow.npz")
        if not np.array_equal(policy_artifact["snapshots"], np.arange(400, 500, dtype=np.int64)):
            raise RuntimeError(f"Seed {seed} policy artifact has the wrong snapshot vector")
        policies = policy_artifact["policies"]
        if policies.shape[0] != 100 or policies.shape[1] != 3 or policies.shape[2] % research.K:
            raise RuntimeError(f"Seed {seed} policy artifact has the wrong shape: {policies.shape}")
        source = manifest["source_sha256"]
        provenance_passes = bool(
            b_config["source_sha256"]["frameworks/hattrick_system.py"]
            == source["frameworks/hattrick_system.py"]
            and b_config["source_sha256"]["utils/build_dataset_within_cluster.py"]
            == source["utils/build_dataset_within_cluster.py"]
            and b_config["source_sha256"]["test_diff_path/run_hattrick_strict2x_research.py"]
            == source["test_diff_path/run_hattrick_strict2x_research.py"]
            and b_config["source_sha256"]["utils/training_utils.py"]
            == source["utils/training_utils.py"]
            and b_config["source_sha256"]["utils/robust_proj_utils.py"]
            == source["utils/robust_proj_utils.py"]
            and c_config["source_sha256"]["frameworks/hattrick_system.py"]
            == source["frameworks/hattrick_system.py"]
            and c_config["source_sha256"]["utils/build_dataset_within_cluster.py"]
            == source["utils/build_dataset_within_cluster.py"]
            and c_config["source_sha256"]["test_diff_path/run_hattrick_residual_lp.py"]
            == source["test_diff_path/run_hattrick_residual_lp.py"]
            and c_config["source_sha256"]["matched_baseline_checkpoint"] == checkpoint_hash
            and b_config["data_access"]["max_source_index_read"] == 499
            and c_config["dataset_access"]["max_source_index_read"] == 499
            and b_config["evaluation_is_final_test"]
            and c_config["evaluation_is_final_test"]
        )
        provenance.append({"seed": seed, "passes": provenance_passes, "checkpoint_sha256": checkpoint_hash})

        item = {
            "seed": seed,
            "provenance_passes": provenance_passes,
            "classes": {},
            "policy_pair_mass": {},
        }
        for class_index, class_name in enumerate(research.CLASSES):
            b = class_values(baseline, class_name, "norm_fulfill")
            c = class_values(candidate, class_name, "norm_fulfill")
            arrays[class_name]["baseline"].append(b)
            arrays[class_name]["candidate"].append(c)
            b_admitted = class_values(baseline, class_name, "admitted_traffic")
            c_admitted = class_values(candidate, class_name, "admitted_traffic")
            item["classes"][class_name] = {
                "baseline_norm_fulfill": summarize(b),
                "candidate_norm_fulfill": summarize(c),
                "norm_fulfill_mean_gap": float(np.mean(c - b)),
                "norm_fulfill_p10_gap": float(np.percentile(c, 10) - np.percentile(b, 10)),
                "norm_fulfill_p1_gap": float(np.percentile(c, 1) - np.percentile(b, 1)),
                "baseline_admitted_traffic_mean": float(b_admitted.mean()),
                "candidate_admitted_traffic_mean": float(c_admitted.mean()),
            }
            pair_mass = policies[:, class_index].reshape(100, -1, research.K).sum(axis=2)
            zero = pair_mass <= 1e-8
            full = pair_mass >= 1.0 - 1e-6
            item["policy_pair_mass"][class_name] = {
                "zero_fraction": float(zero.mean()),
                "partial_fraction": float((~zero & ~full).mean()),
                "full_fraction": float(full.mean()),
                "min": float(pair_mass.min()),
                "max": float(pair_mass.max()),
            }
        b_mlu = class_values(baseline, "Low", "admitted_capacity_ratio")
        c_mlu = class_values(candidate, "Low", "admitted_capacity_ratio")
        item["common_post_admission_mlu"] = {
            "baseline_mean": float(b_mlu.mean()),
            "candidate_mean": float(c_mlu.mean()),
            "mean_gap": float(c_mlu.mean() - b_mlu.mean()),
            "candidate_max": float(
                max(float(row["admitted_capacity_ratio"]) for row in candidate.values())
            ),
        }
        item["candidate_max_disabled_flow"] = float(
            max(float(row["disabled_flow"]) for row in candidate.values())
        )
        item["meets_absolute_targets"] = bool(
            item["classes"]["Medium"]["candidate_norm_fulfill"]["p10"] >= 0.8235
            and item["classes"]["Medium"]["candidate_norm_fulfill"]["p1"] > 0.73
            and item["classes"]["Medium"]["norm_fulfill_p1_gap"] > 0.0
            and item["classes"]["High"]["candidate_norm_fulfill"]["mean"] >= 0.98
            and item["common_post_admission_mlu"]["candidate_max"] <= 1.0001
            and item["candidate_max_disabled_flow"] <= 1e-8
            and provenance_passes
        )
        seed_rows.append(item)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    pooled = {}
    bootstrap = {}
    for class_name in research.CLASSES:
        b = np.stack(arrays[class_name]["baseline"])
        c = np.stack(arrays[class_name]["candidate"])
        pooled[class_name] = {
            "baseline_norm_fulfill": summarize(b.reshape(-1)),
            "candidate_norm_fulfill": summarize(c.reshape(-1)),
        }
        bootstrap[class_name] = hierarchical_bootstrap(b, c, rng)

    medium_ci_positive = bootstrap["Medium"]["p10_gap_95ci"][0] > 0.0
    output = {
        "level": 4,
        "approach": residual.APPROACH,
        "evaluation_window": [400, 500],
        "seeds": list(SEEDS),
        "seed_results": seed_rows,
        "pooled": pooled,
        "paired_hierarchical_bootstrap": bootstrap,
        "all_seed_absolute_targets_pass": all(item["meets_absolute_targets"] for item in seed_rows),
        "medium_p10_gap_ci_excludes_zero": medium_ci_positive,
        "effective_improvement": bool(
            all(item["meets_absolute_targets"] for item in seed_rows) and medium_ci_positive
        ),
        "interpretation_constraints": {
            "pre_admission_normalized_mlu_comparable": False,
            "common_post_admission_mlu_is_capacity_feasibility_metric": True,
            "low_and_pair_starvation_must_be_reported": True,
        },
        "provenance": provenance,
        "source_sha256": {"test_diff_path/analyze_residual_lp_level4.py": own_hash},
    }
    path = level_dir / "hattrick_residual_lp_final_analysis.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
