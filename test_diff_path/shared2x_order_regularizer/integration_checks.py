from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

import run_experiment as runner
from frameworks.hattrick_system import Hattrick
from utils.AdamOptimizer import ADAMOptimizer
from utils.training_utils import train as unchanged_train


def lambda_zero_update_regression() -> dict:
    # Run the numerical equivalence probe on CPU.  The production lambda=0
    # branch is literally the unchanged training routine; comparing two
    # separate CUDA model instances adds cuBLAS replay noise unrelated to the
    # branch under test.
    device = torch.device("cpu")
    runner.set_seed(490)
    props_reference = runner.build_props(1, device)
    props_candidate = runner.build_props(1, device)
    reference_dataset = runner.DM_Dataset_within_Cluster(props_reference, 0, 0, 8)
    candidate_dataset = runner.DM_Dataset_within_Cluster(props_candidate, 0, 0, 8)
    reference_model = Hattrick(props_reference).to(device=device, dtype=props_reference.dtype)
    candidate_model = Hattrick(props_candidate).to(device=device, dtype=props_candidate.dtype)
    candidate_model.load_state_dict(copy.deepcopy(reference_model.state_dict()))
    reference_optimizer = ADAMOptimizer(reference_model.parameters(), lr=props_reference.lr)
    candidate_optimizer = ADAMOptimizer(candidate_model.parameters(), lr=props_candidate.lr)

    runner.set_seed(20260816)
    unchanged_train(
        0,
        1,
        reference_model,
        props_reference,
        [reference_dataset],
        [runner.data_loader(reference_dataset, 8, False, 490)],
        [reference_optimizer],
    )
    runner.set_seed(20260816)
    runner.train_epoch_dispatch(
        1,
        1,
        candidate_model,
        props_candidate,
        candidate_dataset,
        runner.data_loader(candidate_dataset, 8, False, 490),
        candidate_optimizer,
        "hinge",
        0.0,
    )
    deltas = [
        float((reference_model.state_dict()[key] - candidate_model.state_dict()[key]).abs().max())
        for key in reference_model.state_dict()
    ]
    return {
        "max_parameter_delta": max(deltas),
        "tensor_count": len(deltas),
        "tolerance": 0.0,
        "note": "Deterministic CPU replay; production multiplier=0 delegates directly to unchanged_train.",
        "passes": max(deltas) == 0.0,
    }


def level1_artifact_audit() -> dict:
    root = runner.OUTPUT_ROOT / runner.LEVELS[1]["label"]
    complete_files = sorted(root.rglob("complete.json"))
    runs = []
    for path in complete_files:
        complete = json.loads(path.read_text(encoding="utf-8"))
        best = next(item for item in complete["evaluations"] if item["checkpoint"] == "best")
        maximum_capacity = max(
            float(row["max_admitted_capacity_ratio"]) for row in best["classes"]
        )
        maximum_disabled = max(float(row["max_disabled_flow"]) for row in best["classes"])
        runs.append(
            {
                "run": str(path.parent.relative_to(root)),
                "checkpoint_recovery": bool(complete["checkpoint_recovery"]["passes"]),
                "max_post_admission_mlu": maximum_capacity,
                "max_disabled_flow": maximum_disabled,
                "passes": bool(complete["checkpoint_recovery"]["passes"])
                and maximum_capacity <= 1.0001
                and maximum_disabled <= 1e-8,
            }
        )
    return {
        "run_count": len(runs),
        "runs": runs,
        "passes": len(runs) == 6 and all(run["passes"] for run in runs),
    }


def main() -> None:
    result = {
        "lambda_zero_update_regression": lambda_zero_update_regression(),
        "level1_artifact_audit": level1_artifact_audit(),
    }
    result["passes"] = all(value["passes"] for value in result.values())
    runner.write_json(runner.OUTPUT_ROOT / "level1_correctness" / "integration_audit.json", result)
    print(json.dumps(result, indent=2))
    if not result["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
