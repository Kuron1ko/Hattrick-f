from __future__ import annotations

import argparse
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
from counterfactual_search import search_counterfactual_batch  # noqa: E402
from frameworks.hattrick_system import Hattrick  # noqa: E402
from run_level0 import phase_a_path, read_csv, sha256, write_csv  # noqa: E402
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batched exact-admission counterfactual teacher search for Level 3"
    )
    parser.add_argument("--seed", type=int, choices=(490, 491), required=True)
    parser.add_argument("--limit", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--candidate-limit", type=int, choices=(8, 16, 32), default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0 or args.limit > 350:
        parser.error("--limit must lie in [1, 350]")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    output = LEVEL3_ROOT / "level0_search" / "train350_batched" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    partial_metrics = output / "search_metrics.partial.csv"
    partial_targets = output / "targets.partial.pt"
    final_metrics = output / "search_metrics.csv"
    final_targets = output / "targets.pt"
    complete_path = output / "complete.json"
    if complete_path.exists() and not args.force:
        payload = json.loads(complete_path.read_text(encoding="utf-8"))
        if int(payload.get("samples", 0)) >= args.limit:
            print(complete_path.read_text(encoding="utf-8"), flush=True)
            return

    rows: list[dict] = []
    targets: dict[int, dict] = {}
    if not args.force and partial_metrics.exists() and partial_targets.exists():
        rows = read_csv(partial_metrics)
        targets = torch.load(partial_targets, map_location="cpu", weights_only=False)
    processed = {int(row["snapshot"]) for row in rows}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared.set_seed(args.seed)
    props = shared.build_props(3, device)
    props.research_return_policy = True
    props.research_return_admitted = False
    props.sim_mf_mlu = 0
    dataset = DM_Dataset_within_Cluster(props, 0, 0, args.limit)
    path_masks = shared.base.move_dataset_static(dataset, device)
    if path_masks is None:
        masks = tuple(
            torch.ones(dataset.pte.shape[0], dtype=torch.bool, device=device)
            for _ in range(3)
        )
    else:
        masks = tuple(path_masks[index] for index in range(3))

    checkpoint_path = phase_a_path("train350", args.seed)
    if not checkpoint_path.exists():
        parser.error(f"missing Phase-A checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    props.mode = "test"
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")

    loader = shared.data_loader(dataset, args.batch_size, False, 0)
    cursor = 0
    started = time.perf_counter()
    for inputs in loader:
        values = shared.unpack_to_device(inputs, props)
        batch = values[2].shape[0]
        ids = list(range(cursor, cursor + batch))
        cursor += batch
        if all(snapshot in processed for snapshot in ids):
            continue
        if any(snapshot in processed for snapshot in ids):
            raise RuntimeError("partial resume must end on a complete batch boundary")
        # Hattrick caches static-topology transformer output, but its cached
        # tensor retains the previous data batch dimension while capacities
        # are deliberately collapsed to one static row.  Recompute per data
        # batch to keep those dimensions consistent.
        if hasattr(model, "transformer_output"):
            delattr(model, "transformer_output")
        with torch.no_grad():
            policies, capacities = shared.model_forward(
                model, props, dataset, values, path_masks
            )
        results = search_counterfactual_batch(
            model,
            props,
            policies,
            masks,
            (values[2], values[4], values[6]),
            capacities,
            dataset.pte,
            values[11],
            coordinate_rounds=2,
            coordinate_candidate_limit=args.candidate_limit,
            coordinate_move_fractions=(0.4, 0.1),
        )
        for local, (snapshot, result) in enumerate(zip(ids, results)):
            initial = result.initial.totals[0].detach().cpu().numpy()
            target = result.target.totals[0].detach().cpu().numpy()
            oracle_high = float(values[11][local].reshape(-1)[0].item())
            oracle_medium = float(
                (values[12][local] - values[11][local]).reshape(-1)[0].item()
            )
            row = {
                "snapshot": snapshot,
                "status": "OK",
                "initial_high": float(initial[0]),
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
            rows.append(row)
            targets[snapshot] = {
                "policies": tuple(value.detach().cpu() for value in result.target_policies),
                "initial_policies": tuple(
                    value.detach().cpu() for value in result.initial_policies
                ),
                "initial_totals": result.initial.totals.detach().cpu(),
                "target_totals": result.target.totals.detach().cpu(),
                "high_floor": result.high_floor,
            }
        write_csv(partial_metrics, rows)
        torch.save(targets, partial_targets)
        gains = [float(row["medium_norm_gain"]) for row in rows]
        print(
            f"[batched seed={args.seed}] completed={len(rows)}/{args.limit} "
            f"gain_mean={np.mean(gains):.5f} gain_p10={np.percentile(gains, 10):.5f}",
            flush=True,
        )

    write_csv(final_metrics, rows)
    torch.save(targets, final_targets)
    gains = np.asarray([float(row["medium_norm_gain"]) for row in rows])
    summary = {
        "status": "COMPLETE",
        "seed": args.seed,
        "range": [0, args.limit],
        "samples": len(rows),
        "improved_fraction": float((gains > 1e-6).mean()),
        "medium_norm_gain_mean": float(gains.mean()),
        "medium_norm_gain_p10": float(np.percentile(gains, 10)),
        "medium_norm_gain_max": float(gains.max()),
        "accepted_rounds_mean": float(
            np.mean([float(row["accepted_rounds"]) for row in rows])
        ),
        "runtime_seconds_this_invocation": time.perf_counter() - started,
        "batch_size": args.batch_size,
        "search": {
            "gradient_logit_rounds": 0,
            "coordinate_rounds": 2,
            "candidate_limit": args.candidate_limit,
            "move_fractions": [0.4, 0.1],
            "acceptance": "exact per-snapshot High floor and positive Medium gain",
        },
        "phase_a_checkpoint": str(checkpoint_path.resolve()),
        "phase_a_checkpoint_sha256": sha256(checkpoint_path),
        "search_source_sha256": sha256(THIS_DIR / "counterfactual_search.py"),
        "metrics": str(final_metrics.resolve()),
        "targets": str(final_targets.resolve()),
    }
    complete_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
