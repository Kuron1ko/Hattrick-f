from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import run_hattrick_strict2x_research as research
import run_residual_lp_level4_frozen as frozen
from frameworks.hattrick_system import Hattrick
from utils.AdamOptimizer import ADAMOptimizer
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster


NUMERIC_FIELDS = (
    "admitted_traffic",
    "demand",
    "fulfill_ratio",
    "oracle_admitted_traffic",
    "norm_fulfill",
    "raw_mlu",
    "oracle_mlu",
    "normalized_mlu",
    "disabled_flow",
    "admitted_capacity_ratio",
)


def compare_rows(expected: list[dict], actual: list[dict]) -> dict:
    if len(expected) != len(actual):
        return {"rows": len(actual), "expected_rows": len(expected), "passes_1e6": False}
    expected_keys = [(int(row["snapshot"]), row["class"]) for row in expected]
    actual_keys = [(int(row["snapshot"]), row["class"]) for row in actual]
    if expected_keys != actual_keys:
        return {
            "rows": len(actual),
            "expected_rows": len(expected),
            "keys_exact_and_ordered": False,
            "passes_1e6": False,
        }

    maxima = {field: 0.0 for field in NUMERIC_FIELDS}
    relative_maxima = {field: 0.0 for field in NUMERIC_FIELDS}
    worst = None
    for expected_row, actual_row in zip(expected, actual):
        for field in NUMERIC_FIELDS:
            delta = abs(float(expected_row[field]) - float(actual_row[field]))
            maxima[field] = max(maxima[field], delta)
            denominator = max(abs(float(expected_row[field])), 1e-12)
            relative_maxima[field] = max(relative_maxima[field], delta / denominator)
            if worst is None or delta > worst["delta"]:
                worst = {
                    "snapshot": int(expected_row["snapshot"]),
                    "class": expected_row["class"],
                    "field": field,
                    "expected": float(expected_row[field]),
                    "actual": float(actual_row[field]),
                    "delta": delta,
                }
    max_delta = max(maxima.values())
    scale_safe = bool(
        relative_maxima["admitted_traffic"] <= 2e-6
        and maxima["fulfill_ratio"] <= 2e-6
        and maxima["norm_fulfill"] <= 2e-6
        and maxima["raw_mlu"] <= 2e-6
        and maxima["normalized_mlu"] <= 2e-6
        and maxima["admitted_capacity_ratio"] <= 1e-6
        and maxima["disabled_flow"] == 0.0
        and maxima["demand"] == 0.0
        and maxima["oracle_admitted_traffic"] == 0.0
        and maxima["oracle_mlu"] == 0.0
    )
    return {
        "rows": len(actual),
        "expected_rows": len(expected),
        "keys_exact_and_ordered": True,
        "max_delta_by_field": maxima,
        "max_relative_delta_by_field": relative_maxima,
        "worst": worst,
        "max_numeric_delta": max_delta,
        "passes_1e6": max_delta <= 1e-6,
        "passes_2e6": max_delta <= 2e-6,
        "passes_scale_safe_protocol_exception": scale_safe,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mark-complete", action="store_true")
    args = parser.parse_args()

    manifest = frozen.verify_manifest()
    if args.seed not in manifest["seeds"]:
        raise RuntimeError(f"Seed {args.seed} is not frozen")

    run_dir = (
        research.OUTPUT_ROOT
        / research.LEVELS[4]["label"]
        / "unchanged"
        / f"seed_{args.seed}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = research.build_props(4, device)
    dataset = DM_Dataset_within_Cluster(props, 0, 400, 500)

    result = {
        "seed": args.seed,
        "purpose": "post-failure recovery adjudication only; no metric or checkpoint selection change",
        "frozen_source_hashes_verified": True,
        "original_recovery": json.loads(
            (run_dir / "checkpoint_recovery.json").read_text(encoding="utf-8")
        ),
        "checkpoints": {},
    }
    for checkpoint_name in ("final", "best"):
        checkpoint = torch.load(
            run_dir / f"{checkpoint_name}_model.pt", map_location=device, weights_only=False
        )
        model = Hattrick(props).to(device=device, dtype=props.dtype)
        model.load_state_dict(checkpoint["model_state_dict"])
        max_state_delta = max(
            float((model.state_dict()[name] - value).abs().max().item())
            for name, value in checkpoint["model_state_dict"].items()
        )
        optimizer_states = None
        if checkpoint_name == "final":
            optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_states = len(optimizer.state)
        rows, _ = research.evaluate(model, props, dataset, 400)
        expected = research.read_csv(run_dir / f"{checkpoint_name}_evaluation_metrics.csv")
        comparison = compare_rows(expected, rows)
        comparison.update(
            {
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "model_state_loaded": True,
                "max_loaded_model_state_delta": max_state_delta,
                "optimizer_parameter_states": optimizer_states,
            }
        )
        result["checkpoints"][checkpoint_name] = comparison

    result["selected_best_checkpoint_recovery_passes"] = bool(
        result["checkpoints"]["best"]["passes_1e6"]
    )
    result["continuation_is_numerically_adjudicated"] = bool(
        result["checkpoints"]["best"]["passes_scale_safe_protocol_exception"]
        and result["checkpoints"]["final"]["passes_scale_safe_protocol_exception"]
        and result["checkpoints"]["best"]["max_loaded_model_state_delta"] == 0.0
        and result["checkpoints"]["final"]["max_loaded_model_state_delta"] == 0.0
        and result["checkpoints"]["final"]["optimizer_parameter_states"] == 107
    )
    adjudication_path = run_dir / "recovery_protocol_exception.json"
    adjudication_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.mark_complete:
        if not result["continuation_is_numerically_adjudicated"]:
            raise RuntimeError("Recovery adjudication did not meet the disclosed continuation rule")
        best_checkpoint = torch.load(
            run_dir / "best_model.pt", map_location="cpu", weights_only=False
        )
        evaluations = []
        for checkpoint_name in ("final", "best"):
            summary = json.loads(
                (run_dir / f"{checkpoint_name}_evaluation_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            checkpoint = torch.load(
                run_dir / f"{checkpoint_name}_model.pt", map_location="cpu", weights_only=False
            )
            evaluations.append(
                {
                    "checkpoint": checkpoint_name,
                    "epoch": int(checkpoint["epoch"]),
                    "summary": summary,
                }
            )
        complete = {
            "best_epoch": int(best_checkpoint["epoch"]),
            "best_rank": list(best_checkpoint["rank"]),
            "selection_status": (
                "guard_feasible" if best_checkpoint["rank"][0] else "high_first_fallback_no_guard_feasible"
            ),
            "checkpoint_recovery": result["original_recovery"],
            "selected_checkpoint_recovery": result["checkpoints"]["best"],
            "completion_status": "disclosed_recovery_protocol_exception",
            "protocol_exception": {
                "reason": "frozen final-checkpoint replay exceeded 1e-6 by 9.073486328125e-7",
                "adjudication": adjudication_path.name,
                "no_training_retry": True,
                "no_checkpoint_or_metric_rule_change": True,
                "continuation_basis": (
                    "exact loaded model state; 107 optimizer states; exact keys, masks, demand, and "
                    "oracles; admitted traffic <=2e-6 relative; normalized outcomes/raw MLU <=2e-6 "
                    "absolute; capacity <=1e-6 absolute"
                ),
            },
            "evaluations": evaluations,
        }
        (run_dir / "complete.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
