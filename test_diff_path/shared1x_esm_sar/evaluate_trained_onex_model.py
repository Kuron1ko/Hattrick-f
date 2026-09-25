from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TRAIN_PATH = THIS_DIR / "train_onex_esm_sar_model.py"
RUN_DIR = THIS_DIR / "artifacts" / "level4_full" / "seed_490"
CHECKPOINT_PATH = RUN_DIR / "best_model.pt"
TEST_RANGE = (400, 500)

spec = importlib.util.spec_from_file_location("trained_onex_runtime", TRAIN_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load the from-scratch 1x training runtime")
train = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train
spec.loader.exec_module(train)
runtime = train.runtime
method = train.method
probe = train.probe
Hattrick = train.Hattrick


def class_values(rows: list[dict], class_name: str) -> np.ndarray:
    ordered = sorted(
        (
            (int(row["snapshot"]), float(row["norm_fulfill"]))
            for row in rows
            if row["class"] == class_name
        ),
        key=lambda item: item[0],
    )
    return np.asarray([value for _snapshot, value in ordered], dtype=np.float64)


def paired_bootstrap(base: np.ndarray, candidate: np.ndarray, seed: int) -> dict:
    if base.shape != candidate.shape:
        raise RuntimeError("Final-test rows are not paired")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(base), size=(20000, len(base)))
    base_samples = base[indices]
    candidate_samples = candidate[indices]
    metric_samples = {
        "mean": candidate_samples.mean(axis=1) - base_samples.mean(axis=1),
        "p1": np.quantile(candidate_samples, 0.01, axis=1)
        - np.quantile(base_samples, 0.01, axis=1),
        "p10": np.quantile(candidate_samples, 0.10, axis=1)
        - np.quantile(base_samples, 0.10, axis=1),
    }
    point = {
        "mean": float(candidate.mean() - base.mean()),
        "p1": float(np.quantile(candidate, 0.01) - np.quantile(base, 0.01)),
        "p10": float(np.quantile(candidate, 0.10) - np.quantile(base, 0.10)),
    }
    result = {}
    for name, samples in metric_samples.items():
        result[name] = {
            "delta": point[name],
            "ci95": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
        }
    difference = candidate - base
    result["per_snapshot"] = {
        "improved_fraction": float(np.mean(difference > 1e-7)),
        "degraded_fraction": float(np.mean(difference < -1e-7)),
        "equal_fraction": float(np.mean(np.abs(difference) <= 1e-7)),
    }
    return result


def state_distance(candidate, original) -> dict:
    candidate_state = candidate.state_dict()
    original_state = original.state_dict()
    common = sorted(set(candidate_state) & set(original_state))
    squared = 0.0
    equal_tensors = 0
    for name in common:
        left = candidate_state[name].detach().to(device="cpu", dtype=torch.float64)
        right = original_state[name].detach().to(device="cpu", dtype=torch.float64)
        if left.shape != right.shape:
            continue
        squared += float(torch.square(left - right).sum().item())
        equal_tensors += int(torch.equal(left, right))
    return {
        "common_state_tensors": len(common),
        "exactly_equal_tensors": equal_tensors,
        "parameter_l2_distance": squared ** 0.5,
    }


def main() -> None:
    train.set_seed(490)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime.shared.TOPOLOGY = probe.TOPOLOGY
    props = runtime.build_props(4, device)
    props.checkpoint = 0
    props.research_return_policy = True

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    candidate = Hattrick(props).to(device=device, dtype=props.dtype)
    candidate.load_state_dict(checkpoint["model_state_dict"])
    original, _original_props = probe.load_onex(device)
    independence_audit = state_distance(candidate, original)

    original_eval = train.evaluate_model(original, props, *TEST_RANGE, strict=False)
    original_rows = original_eval["base_rows"]
    original_summary = original_eval["base_summary"]

    started = time.perf_counter()
    candidate_eval = train.evaluate_model(candidate, props, *TEST_RANGE, strict=True)
    evaluation_seconds = time.perf_counter() - started
    base_rows = candidate_eval["base_rows"]
    base_summary = candidate_eval["base_summary"]
    sar_rows = candidate_eval["rows"]
    sar_summary = candidate_eval["summary"]

    comparisons = {}
    for label, rows in (("trained_base", base_rows), ("trained_esm_sar", sar_rows)):
        comparisons[label] = {
            class_name: paired_bootstrap(
                class_values(original_rows, class_name),
                class_values(rows, class_name),
                20260821 + 10 * method_index + class_index,
            )
            for class_index, class_name in enumerate(("High", "Medium", "Low"))
            for method_index in [0 if label == "trained_base" else 1]
        }

    payload = {
        "method": "from-scratch 1x Hattrick trained through first-order unrolled ESM-SAR",
        "topology": probe.TOPOLOGY,
        "traffic_scale": "1x original data",
        "protocol": {
            "random_initialization_seed": 490,
            "train": [0, 350],
            "validation_model_selection": [350, 400],
            "untouched_final_test": list(TEST_RANGE),
            "epochs_completed": 60,
            "selected_epoch": int(checkpoint["epoch"]),
            "original_hattrick_role": "comparison only; never loaded into candidate",
            "sar_traffic": "ESM-predicted High/Medium/Low traffic in the unrolled sequential-admission objective",
        },
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_sha256": runtime.sha256(CHECKPOINT_PATH),
        "training_script_sha256": runtime.sha256(TRAIN_PATH),
        "original_hattrick": str(probe.MODEL_PATH),
        "original_hattrick_sha256": runtime.sha256(probe.MODEL_PATH),
        "independence_audit": independence_audit,
        "evaluation_seconds_candidate_100_snapshots": evaluation_seconds,
        "original_hattrick_test": method.compact(original_summary),
        "trained_model_base_test": method.compact(base_summary),
        "trained_model_esm_sar_test": method.compact(sar_summary),
        "trained_base_vs_original_delta": method.gaps(base_summary, original_summary),
        "trained_esm_sar_vs_original_delta": method.gaps(sar_summary, original_summary),
        "paired_bootstrap_20000": comparisons,
        "target_checks": {
            "high_mean_at_least_0_995": (
                runtime.summary_index(sar_summary)["High"]["norm_fulfill_mean"] >= 0.995
            ),
            "low_mean_decline_significant_95pct": (
                comparisons["trained_esm_sar"]["Low"]["mean"]["ci95"][1] < 0.0
            ),
        },
    }
    (RUN_DIR / "final_test_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    runtime.write_csv(RUN_DIR / "final_test_original_hattrick_rows.csv", original_rows)
    runtime.write_csv(RUN_DIR / "final_test_trained_base_rows.csv", base_rows)
    runtime.write_csv(RUN_DIR / "final_test_trained_esm_sar_rows.csv", sar_rows)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
