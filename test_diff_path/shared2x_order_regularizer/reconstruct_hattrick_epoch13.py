from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_experiment as runner


SEED = 490
OUTPUT_ROOT = runner.OUTPUT_ROOT / "reconstructed_hattrick_epoch13"
LABEL = "matched_level3_epoch13"
REFERENCE_METRICS = (
    runner.OUTPUT_ROOT
    / "level3_validation_only"
    / "baseline"
    / f"seed_{SEED}"
    / "validation_epoch_013_metrics.csv"
)
REFERENCE_CONFIG = REFERENCE_METRICS.parent / "config.json"


def main() -> None:
    for path in (REFERENCE_METRICS, REFERENCE_CONFIG):
        if not path.exists():
            raise FileNotFoundError(path)
    runner.OUTPUT_ROOT = OUTPUT_ROOT
    runner.LEVELS[3] = {
        "label": LABEL,
        "train": (0, 350),
        "validation": (350, 400),
        "evaluation": (350, 400),
        "epochs": 13,
    }
    run_dir = runner.run_one(
        level=3,
        penalty="tail_directional_squared",
        multiplier=0.0,
        seed=SEED,
        force=False,
    )
    checkpoint_path = run_dir / "final_model.pt"
    evaluation_path = run_dir / "final_evaluation_metrics.csv"
    if not checkpoint_path.exists() or not evaluation_path.exists():
        raise RuntimeError("Reconstructed epoch-13 run is incomplete")

    expected = runner.read_csv(REFERENCE_METRICS)
    actual = runner.read_csv(evaluation_path)
    expected_by_key = {
        (int(row["snapshot"]), row["class"]): row
        for row in expected
    }
    actual_by_key = {
        (int(row["snapshot"]), row["class"]): row
        for row in actual
    }
    if expected_by_key.keys() != actual_by_key.keys():
        raise RuntimeError("Reconstructed epoch-13 validation keys differ from the original run")
    numeric_fields = (
        "admitted_traffic",
        "demand",
        "fulfill_ratio",
        "oracle_admitted_traffic",
        "norm_fulfill",
        "admitted_capacity_ratio",
        "disabled_flow",
    )
    deltas_by_field: dict[str, list[float]] = {field: [] for field in numeric_fields}
    for key in expected_by_key:
        for field in numeric_fields:
            deltas_by_field[field].append(
                abs(float(expected_by_key[key][field]) - float(actual_by_key[key][field]))
            )
    delta_summary = {
        field: {
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
        }
        for field, values in deltas_by_field.items()
    }
    original_config = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    reconstructed_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    identity_fields = (
        "penalty",
        "actual_lambda",
        "seed",
        "topology",
        "paths_per_pair",
        "shared_paths",
        "train",
        "validation",
        "batch_size",
        "learning_rate",
        "objectives",
        "source_sha256",
    )
    configuration_matches = all(
        original_config.get(field) == reconstructed_config.get(field)
        for field in identity_fields
    )
    # CUDA scatter/reduction kernels used by the routing model are not bitwise
    # deterministic.  The original run retained the epoch-13 metrics but not its
    # checkpoint, so accept only a tightly bounded, configuration-identical
    # reconstruction and preserve the observed deltas in the audit.
    acceptance_limits = {
        "norm_fulfill_max": 0.05,
        "norm_fulfill_mean": 0.01,
        "fulfill_ratio_max": 0.03,
        "disabled_flow_max": 1e-8,
    }
    passes = bool(
        configuration_matches
        and delta_summary["norm_fulfill"]["max"] <= acceptance_limits["norm_fulfill_max"]
        and delta_summary["norm_fulfill"]["mean"] <= acceptance_limits["norm_fulfill_mean"]
        and delta_summary["fulfill_ratio"]["max"] <= acceptance_limits["fulfill_ratio_max"]
        and delta_summary["disabled_flow"]["max"] <= acceptance_limits["disabled_flow_max"]
    )
    audit = {
        "status": (
            "configuration-matched epoch-13 reconstruction accepted"
            if passes
            else "epoch-13 reconstruction rejected"
        ),
        "checkpoint_origin": (
            "retrained for exactly 13 epochs because the original run retained "
            "epoch-13 metrics but overwrote the epoch-13 checkpoint"
        ),
        "bitwise_identity_claimed": False,
        "nondeterminism_note": (
            "CUDA routing scatter/reduction operations are not bitwise deterministic; "
            "all observed validation deltas are reported below"
        ),
        "seed": SEED,
        "epochs": 13,
        "train": [0, 350],
        "validation": [350, 400],
        "reference_metrics": str(REFERENCE_METRICS),
        "reference_metrics_sha256": runner.sha256(REFERENCE_METRICS),
        "reconstructed_metrics": str(evaluation_path),
        "reconstructed_metrics_sha256": runner.sha256(evaluation_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": runner.sha256(checkpoint_path),
        "reference_config": str(REFERENCE_CONFIG),
        "reference_config_sha256": runner.sha256(REFERENCE_CONFIG),
        "configuration_matches": configuration_matches,
        "validation_delta_by_field": delta_summary,
        "acceptance_limits": acceptance_limits,
        "passes": passes,
        "source_sha256": runner.sha256(Path(__file__).resolve()),
    }
    runner.write_json(run_dir / "reconstruction_audit.json", audit)
    if not audit["passes"]:
        raise RuntimeError(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
