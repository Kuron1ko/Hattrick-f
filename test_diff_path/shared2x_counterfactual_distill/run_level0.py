from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
LEVEL3_ROOT = ROOT.parent / "shared2x_counterfactual_level3"
SHARED_RUNTIME = TEST_DIR / "shared2x_order_regularizer"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(SHARED_RUNTIME))

import run_experiment as shared  # noqa: E402
from counterfactual_search import search_counterfactual  # noqa: E402
from frameworks.hattrick_system import Hattrick  # noqa: E402
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster  # noqa: E402


SPLITS = {
    "train32": (0, 32),
    "train160": (0, 160),
    "train350": (0, 350),
    "proxy50": (200, 250),
}


def phase_a_path(split: str, seed: int) -> Path:
    if split == "train350":
        label = "level3_validation_only"
        independent = LEVEL3_ROOT / "phase_a" / label / f"seed_{seed}" / "best_model.pt"
        if independent.exists():
            return independent
    else:
        # Preserve the original Level-0/1/2 calibration contract.
        label = "level2_proxy"
        seed = 490
    return (
        TEST_DIR
        / "shared2x_order_epsilon"
        / "artifacts"
        / "phase_a"
        / label
        / f"seed_{seed}"
        / "best_model.pt"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=tuple(SPLITS), default="train32")
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--profile", choices=("full", "fast"), default="full")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    start, end = SPLITS[args.split]
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        end = min(end, start + args.limit)
    output = THIS_DIR / "artifacts" / "level0_search" / args.split
    if args.split == "train350":
        output = (
            LEVEL3_ROOT
            / "level0_search"
            / args.split
            / f"seed_{args.seed}"
        )
    complete_path = output / "complete.json"
    if complete_path.exists() and not args.force and args.limit is None:
        print(complete_path.read_text(encoding="utf-8"), flush=True)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared.set_seed(args.seed)
    level = 3 if args.split == "train350" else 2
    props = shared.build_props(level, device)
    props.research_return_policy = False
    props.research_return_admitted = False
    dataset = DM_Dataset_within_Cluster(props, 0, start, end)
    path_masks = shared.base.move_dataset_static(dataset, device)
    phase_a = phase_a_path(args.split, args.seed)
    if not phase_a.exists():
        parser.error(f"missing Phase-A checkpoint: {phase_a}")
    checkpoint = torch.load(phase_a, map_location=device, weights_only=False)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    props.mode = "test"
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    props.sim_mf_mlu = 0
    loader = shared.data_loader(dataset, 1, False, 0)
    rows = []
    targets = {}
    partial_metrics = output / "search_metrics.partial.csv"
    partial_targets = output / "targets.partial.pt"
    if not args.force and partial_metrics.exists() and partial_targets.exists():
        rows = read_csv(partial_metrics)
        targets = torch.load(partial_targets, map_location="cpu", weights_only=False)
    elif args.split == "train160":
        # The first 32 snapshots were already searched in Level 0.  They use
        # the same frozen Phase-A checkpoint and search implementation, so
        # reuse them instead of paying the search cost twice.
        base = THIS_DIR / "artifacts" / "level0_search" / "train32"
        if (base / "search_metrics.csv").exists() and (base / "targets.pt").exists():
            rows = read_csv(base / "search_metrics.csv")
            targets = torch.load(base / "targets.pt", map_location="cpu", weights_only=False)
    processed = {int(row["snapshot"]) for row in rows}
    started = time.perf_counter()
    for local_index, inputs in enumerate(loader):
        snapshot = start + local_index
        if snapshot in processed:
            continue
        values = shared.unpack_to_device(inputs, props)
        props.research_return_policy = True
        with torch.no_grad():
            policies, capacities = shared.model_forward(
                model, props, dataset, values, path_masks
            )
        props.research_return_policy = False
        props.sim_mf_mlu = 1
        with torch.no_grad():
            direct_admitted, _ = shared.model_forward(
                model, props, dataset, values, path_masks
            )
        props.sim_mf_mlu = 0
        direct_totals = [float(value.reshape(1, -1).sum().item()) for value in direct_admitted]
        tms = (values[2], values[4], values[6])
        if path_masks is None:
            masks = tuple(
                torch.ones(dataset.pte.shape[0], dtype=torch.bool, device=device)
                for _ in range(3)
            )
        else:
            masks = tuple(path_masks[index] for index in range(3))
        try:
            search_kwargs = {"rounds": 8}
            if args.profile == "fast":
                search_kwargs = {
                    "rounds": 4,
                    "step_grid": (4.0, 1.0),
                    "coordinate_rounds": 2,
                    "coordinate_candidate_limit": 32,
                    "coordinate_move_fractions": (0.4, 0.1),
                }
            result = search_counterfactual(
                model,
                props,
                policies,
                masks,
                tms,
                capacities,
                dataset.pte,
                values[11],
                **search_kwargs,
            )
            status = "OK"
            error = ""
        except ValueError as exc:
            result = None
            status = "INFEASIBLE_START"
            error = str(exc)
        if result is None:
            row = {
                "snapshot": snapshot,
                "status": status,
                "error": error,
                "initial_high": "",
                "direct_high": direct_totals[0],
                "target_high": "",
                "high_norm_initial": "",
                "high_norm_target": "",
                "initial_medium": "",
                "target_medium": "",
                "medium_gain": "",
                "medium_norm_gain": "",
                "initial_low": "",
                "target_low": "",
                "low_delta": "",
                "accepted_rounds": 0,
                "attempted_candidates": 0,
                "feasible_candidates": 0,
                "policy_l1": 0,
            }
        else:
            initial = result.initial.totals[0].detach().cpu().numpy()
            target = result.target.totals[0].detach().cpu().numpy()
            oracle_high = float(values[11].reshape(-1)[0].item())
            oracle_medium = float((values[12] - values[11]).reshape(-1)[0].item())
            row = {
                "snapshot": snapshot,
                "status": status,
                "error": error,
                "initial_high": float(initial[0]),
                "direct_high": direct_totals[0],
                "target_high": float(target[0]),
                "high_norm_initial": float(initial[0] / oracle_high),
                "high_norm_target": float(target[0] / oracle_high),
                "initial_medium": float(initial[1]),
                "target_medium": float(target[1]),
                "medium_gain": float(target[1] - initial[1]),
                "medium_norm_gain": float((target[1] - initial[1]) / oracle_medium),
                "initial_low": float(initial[2]),
                "target_low": float(target[2]),
                "low_delta": float(target[2] - initial[2]),
                "accepted_rounds": result.accepted_rounds,
                "attempted_candidates": result.attempted_candidates,
                "feasible_candidates": result.feasible_candidates,
                "policy_l1": result.policy_l1,
            }
            targets[snapshot] = {
                "policies": tuple(
                    value.detach().cpu() for value in result.target_policies
                ),
                "initial_policies": tuple(
                    value.detach().cpu() for value in result.initial_policies
                ),
                "initial_totals": result.initial.totals.detach().cpu(),
                "target_totals": result.target.totals.detach().cpu(),
                "high_floor": result.high_floor,
            }
        rows.append(row)
        # Counterfactual search is deliberately more expensive than ordinary
        # inference.  Persist after every snapshot so an interrupted Level-0
        # calibration still leaves reusable evidence and teacher policies.
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / "search_metrics.partial.csv", rows)
        torch.save(targets, output / "targets.partial.pt")
        print(
            f"[{args.split}] {snapshot}: status={status} "
            f"medium_gain={row['medium_gain']} high_norm={row['high_norm_target']}",
            flush=True,
        )
    output.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.limit is None else f"_limit_{args.limit}"
    metrics_path = output / f"search_metrics{suffix}.csv"
    write_csv(metrics_path, rows)
    targets_path = output / f"targets{suffix}.pt"
    torch.save(targets, targets_path)
    valid = [row for row in rows if row["status"] == "OK"]
    gains = np.asarray([float(row["medium_norm_gain"]) for row in valid])
    summary = {
        "status": "COMPLETE",
        "split": args.split,
        "range": [start, end],
        "samples": len(rows),
        "feasible_start_fraction": len(valid) / max(len(rows), 1),
        "improved_fraction_all": float(
            np.mean([float(row["medium_norm_gain"] or 0.0) > 1e-6 for row in rows])
        ),
        "medium_norm_gain_mean_valid": float(gains.mean()) if len(gains) else None,
        "medium_norm_gain_p10_valid": float(np.percentile(gains, 10)) if len(gains) else None,
        "medium_norm_gain_max_valid": float(gains.max()) if len(gains) else None,
        "accepted_rounds_mean_valid": float(
            np.mean([float(row["accepted_rounds"]) for row in valid])
        ) if valid else None,
        "runtime_seconds": time.perf_counter() - started,
        "seed": args.seed,
        "search_profile": args.profile,
        "search_parameters": (
            {
                "rounds": 4,
                "step_grid": [4.0, 1.0],
                "coordinate_rounds": 2,
                "coordinate_candidate_limit": 32,
                "coordinate_move_fractions": [0.4, 0.1],
            }
            if args.profile == "fast"
            else {
                "rounds": 8,
                "step_grid": [8.0, 4.0, 2.0, 1.0, 0.5, 0.25],
                "coordinate_rounds": 4,
                "coordinate_candidate_limit": 128,
                "coordinate_move_fractions": [0.4, 0.2, 0.1, 0.05],
            }
        ),
        "phase_a_checkpoint": str(phase_a.resolve()),
        "phase_a_checkpoint_sha256": sha256(phase_a),
        "search_source_sha256": sha256(THIS_DIR / "counterfactual_search.py"),
        "metrics": str(metrics_path.resolve()),
        "targets": str(targets_path.resolve()),
    }
    summary_path = output / f"complete{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
