from __future__ import annotations

import importlib.util
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
MODULE_PATH = THIS_DIR / "probe_esm_dominance_gate.py"
spec = importlib.util.spec_from_file_location("safety_memory_gate", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {MODULE_PATH}")
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)
runtime = gate.runtime


DATA_RANGE = (0, 400)
TRAIN_RANGE = (0, 318)
SAFETY_RANGE = (318, 350)
VALIDATION_RANGE = (350, 400)
LEVEL = int(os.environ.get("HATTRICK_MEMORY_LEVEL", "3"))


def gain_labels(baseline_rows, candidate_rows):
    baseline = gate.row_map(baseline_rows)
    candidate = gate.row_map(candidate_rows)
    snapshots = sorted({key[0] for key in baseline})
    return {
        snapshot: {
            class_name: candidate[(snapshot, class_name)] - baseline[(snapshot, class_name)]
            for class_name in ("High", "Medium", "Low")
        }
        for snapshot in snapshots
    }


def hybrid_diagnostics(baseline_rows, candidate_rows, accepted_by_snapshot):
    baseline = gate.row_map(baseline_rows)
    candidate = gate.row_map(candidate_rows)
    snapshots = sorted(accepted_by_snapshot)
    hybrid_rows = []
    for snapshot in snapshots:
        for class_name in ("High", "Medium", "Low"):
            value = (
                candidate[(snapshot, class_name)]
                if accepted_by_snapshot[snapshot] and class_name != "High"
                else baseline[(snapshot, class_name)]
            )
            hybrid_rows.append(
                {
                    "snapshot": snapshot,
                    "class": class_name,
                    "norm_fulfill": value,
                }
            )
    subset_baseline = [
        {
            "snapshot": snapshot,
            "class": class_name,
            "norm_fulfill": baseline[(snapshot, class_name)],
        }
        for snapshot in snapshots
        for class_name in ("High", "Medium", "Low")
    ]
    accepted = [accepted_by_snapshot[snapshot] for snapshot in snapshots]
    return gate.diagnostics(subset_baseline, hybrid_rows, accepted)


def feature_sets(keys):
    compact = [
        key
        for key in keys
        if key.startswith("baseline_predicted_")
        or key.startswith("candidate_predicted_")
        or key.startswith("predicted_")
        or key.endswith("util_max")
        or key.endswith("util_p90")
        or key.endswith("_l1")
        or key.endswith("_kl")
    ]
    proxy = [
        "baseline_predicted_high",
        "baseline_predicted_medium",
        "baseline_predicted_low",
        "predicted_medium_gain",
        "predicted_low_gain",
    ]
    return {"proxy": proxy, "compact": compact, "all": keys}


def standardize(train_x, all_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    return (train_x - mean) / std, (all_x - mean) / std


def leave_one_out_radius(train_x, k, quantile):
    distance = np.sqrt(((train_x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(distance, np.inf)
    kth = np.partition(distance, k - 1, axis=1)[:, k - 1]
    return float(np.quantile(kth, quantile))


def predict_acceptance(train_x, train_y, query_x, k, lower_quantile, radius, margin):
    decisions = []
    bounds = []
    for query in query_x:
        distance = np.sqrt(((train_x - query) ** 2).sum(axis=1))
        indices = np.argpartition(distance, k - 1)[:k]
        neighbor_gain = train_y[indices]
        lower = np.quantile(neighbor_gain, lower_quantile, axis=0)
        within_memory = float(np.partition(distance, k - 1)[k - 1]) <= radius
        decisions.append(
            bool(within_memory and lower[0] >= margin and lower[1] >= margin)
        )
        bounds.append(lower.tolist())
    return decisions, bounds


def public_config(config):
    return {
        key: value
        for key, value in config.items()
        if key not in ("train_x", "train_y", "all_x", "feature_keys")
    }


def main() -> None:
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(LEVEL, device)
    model, checkpoint = runtime.load_backbone(LEVEL, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    level_suffix = "" if LEVEL == 3 else f"_level{LEVEL}"
    feature_path = THIS_DIR / f"memory_features{level_suffix}_0_400.csv"
    baseline_path = THIS_DIR / f"memory_baseline_rows{level_suffix}_0_400.csv"
    candidate_path = THIS_DIR / f"memory_candidate_rows{level_suffix}_0_400.csv"
    if feature_path.exists() and baseline_path.exists() and candidate_path.exists():
        def read_rows(path):
            with path.open("r", newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))

        proxy_rows = read_rows(feature_path)
        baseline_rows = read_rows(baseline_path)
        candidate_rows = read_rows(candidate_path)
        elapsed = 0.0
    else:
        cache = runtime.build_policy_cache(model, props, *DATA_RANGE, batch_size=32)
        baseline_rows, _ = runtime.evaluate_cache(model, props, cache, None, batch_size=32)
        started = time.perf_counter()
        proposed, proxy_rows = gate.build_proposals_and_proxy(model, props, cache)
        adapter = gate.sar.CorrectedPolicyAdapter(int(cache.policies[0].shape[1]))
        candidate_rows, _ = runtime.evaluate_cache(
            model, props, proposed, adapter, batch_size=32
        )
        elapsed = time.perf_counter() - started
    labels = gain_labels(baseline_rows, candidate_rows)

    feature_keys = [key for key in proxy_rows[0] if key != "snapshot"]
    snapshots = np.asarray([int(row["snapshot"]) for row in proxy_rows])
    feature_matrix = np.asarray(
        [[float(row[key]) for key in feature_keys] for row in proxy_rows],
        dtype=np.float64,
    )
    gain_matrix = np.asarray(
        [[labels[int(snapshot)]["Medium"], labels[int(snapshot)]["Low"]] for snapshot in snapshots],
        dtype=np.float64,
    )
    train_mask = (snapshots >= TRAIN_RANGE[0]) & (snapshots < TRAIN_RANGE[1])
    safety_mask = (snapshots >= SAFETY_RANGE[0]) & (snapshots < SAFETY_RANGE[1])
    validation_mask = (snapshots >= VALIDATION_RANGE[0]) & (snapshots < VALIDATION_RANGE[1])

    candidates = []
    for feature_name, selected_keys in feature_sets(feature_keys).items():
        indices = [feature_keys.index(key) for key in selected_keys]
        train_x_raw = feature_matrix[train_mask][:, indices]
        all_x_raw = feature_matrix[:, indices]
        train_x, all_x = standardize(train_x_raw, all_x_raw)
        train_y = gain_matrix[train_mask]
        for k in (3, 5, 8, 12, 20):
            for radius_quantile in (0.8, 0.9, 1.0):
                radius = leave_one_out_radius(train_x, k, radius_quantile)
                for lower_quantile in (0.0, 0.1, 0.2):
                    for margin in (0.0, 0.001, 0.003, 0.005):
                        safety_decisions, _ = predict_acceptance(
                            train_x,
                            train_y,
                            all_x[safety_mask],
                            k,
                            lower_quantile,
                            radius,
                            margin,
                        )
                        safety_snapshots = snapshots[safety_mask].tolist()
                        safety_diag = hybrid_diagnostics(
                            baseline_rows,
                            candidate_rows,
                            dict(zip(safety_snapshots, safety_decisions)),
                        )
                        candidates.append(
                            {
                                "feature_set": feature_name,
                                "feature_keys": selected_keys,
                                "k": k,
                                "radius_quantile": radius_quantile,
                                "radius": radius,
                                "lower_quantile": lower_quantile,
                                "margin": margin,
                                "safety": safety_diag,
                                "train_x": train_x,
                                "train_y": train_y,
                                "all_x": all_x,
                            }
                        )

    safety_feasible = [
        row
        for row in candidates
        if row["safety"]["all_three_empirical_cdfs_noninferior"]
        and row["safety"]["accepted"] > 0
    ]
    safety_feasible.sort(
        key=lambda row: (
            row["safety"]["classes"]["Medium"]["mean_gain"],
            row["safety"]["classes"]["Low"]["mean_gain"],
        ),
        reverse=True,
    )
    finalists = []
    for row in safety_feasible:
        decisions, bounds = predict_acceptance(
            row["train_x"],
            row["train_y"],
            row["all_x"][validation_mask],
            row["k"],
            row["lower_quantile"],
            row["radius"],
            row["margin"],
        )
        validation_snapshots = snapshots[validation_mask].tolist()
        validation_diag = hybrid_diagnostics(
            baseline_rows,
            candidate_rows,
            dict(zip(validation_snapshots, decisions)),
        )
        finalists.append({**public_config(row), "validation": validation_diag})
    validation_feasible = [
        row
        for row in finalists
        if row["validation"]["all_three_empirical_cdfs_noninferior"]
        and row["validation"]["accepted"] > 0
    ]
    validation_feasible.sort(
        key=lambda row: (
            row["validation"]["classes"]["Medium"]["mean_gain"],
            row["validation"]["classes"]["Low"]["mean_gain"],
        ),
        reverse=True,
    )
    winner = validation_feasible[0] if validation_feasible else None
    payload = {
        "method": "counterfactual safety-memory gate",
        "level": LEVEL,
        "data_protocol": {
            "memory_train": list(TRAIN_RANGE),
            "safety_selection": list(SAFETY_RANGE),
            "independent_validation": list(VALIDATION_RANGE),
            "actual_tm_online": False,
        },
        "checkpoint": str(checkpoint),
        "proposal_seconds_400_snapshots": elapsed,
        "candidate_count": len(candidates),
        "safety_feasible_count": len(safety_feasible),
        "validation_feasible_count": len(validation_feasible),
        "top_safety_candidates": [public_config(row) for row in safety_feasible[:10]],
        "top_validation_feasible": validation_feasible[:20],
        "winner": winner,
    }
    runtime.write_json(THIS_DIR / f"safety_memory_result_level{LEVEL}.json", payload)
    runtime.write_csv(feature_path, proxy_rows)
    runtime.write_csv(baseline_path, baseline_rows)
    runtime.write_csv(candidate_path, candidate_rows)
    print(json.dumps({
        "safety_feasible_count": len(safety_feasible),
        "validation_feasible_count": len(validation_feasible),
        "winner": winner,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
