from __future__ import annotations

import csv
import json
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
ARTIFACTS = THIS_DIR / "artifacts"
OLD_LEVEL1 = (
    THIS_DIR.parent
    / "shared2x_full_objectives"
    / "artifacts"
    / "level1_correctness"
    / "seed_490"
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric_replay_delta(left: Path, right: Path) -> float:
    fields = (
        "admitted_traffic",
        "fulfill_ratio",
        "norm_fulfill",
        "raw_mlu",
        "normalized_mlu",
        "disabled_flow",
        "admitted_capacity_ratio",
    )
    expected = read_csv(left)
    actual = read_csv(right)
    if [(row["snapshot"], row["class"]) for row in expected] != [
        (row["snapshot"], row["class"]) for row in actual
    ]:
        raise AssertionError("metric keys differ")
    return max(
        abs(float(a[field]) - float(b[field]))
        for a, b in zip(expected, actual)
        for field in fields
    )


def run_checks() -> dict:
    phase_a = ARTIFACTS / "phase_a" / "level1_correctness" / "seed_490"
    branches = (
        ARTIFACTS / "phase_b" / "level1_correctness" / "control" / "seed_490",
        ARTIFACTS / "phase_b" / "level1_correctness" / "swap" / "seed_490",
        ARTIFACTS
        / "phase_b"
        / "level1_correctness"
        / "epsilon"
        / "epsilon_0p005"
        / "seed_490",
    )
    complete = [
        json.loads((branch / "complete.json").read_text(encoding="utf-8"))
        for branch in branches
    ]
    phase_a_hashes = {item["phase_a_parameter_sha256"] for item in complete}
    optimizer_hashes = {item["phase_a_optimizer_repr_sha256"] for item in complete}
    if len(phase_a_hashes) != 1 or len(optimizer_hashes) != 1:
        raise AssertionError("Phase-B branches did not start identically")
    if not all(item["all_parameters_trainable"] for item in complete):
        raise AssertionError("a Phase-B branch froze model parameters")
    max_recovery = max(item["checkpoint_recovery_max_delta"] for item in complete)
    if max_recovery > 1e-5:
        raise AssertionError("checkpoint recovery exceeded tolerance")
    regression_delta = numeric_replay_delta(
        OLD_LEVEL1 / "validation_epoch_001_metrics.csv",
        phase_a / "validation_epoch_001_metrics.csv",
    )
    if regression_delta > 1e-5:
        raise AssertionError("Phase-A control no longer matches restored objectives")
    required = (
        "route_snapshot_metrics.csv",
        "high_path_policy_changes.csv",
        "high_link_load_changes.csv",
    )
    for branch in branches:
        config = json.loads((branch / "config.json").read_text(encoding="utf-8"))
        if config["train"] != [0, 32] or config["validation"] != [32, 40]:
            raise AssertionError("Level-1 split drift")
        for checkpoint in ("final", "best"):
            for name in required:
                if not (branch / checkpoint / name).exists():
                    raise AssertionError(f"missing route diagnostic {branch / checkpoint / name}")
    return {
        "status": "PASS",
        "phase_a_parameter_sha256": next(iter(phase_a_hashes)),
        "phase_a_optimizer_repr_sha256": next(iter(optimizer_hashes)),
        "phase_a_restored_numeric_regression_max_delta": regression_delta,
        "checkpoint_recovery_max_delta": max_recovery,
        "all_phase_b_parameters_trainable": True,
        "split_safe": True,
        "route_diagnostics_complete": True,
    }


if __name__ == "__main__":
    result = run_checks()
    output = ARTIFACTS / "level1_integration_checks.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
