from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
SHARED_DIR = TEST_DIR / "shared2x_order_regularizer"
FULL_DIR = TEST_DIR / "shared2x_full_objectives"
OUTPUT_ROOT = THIS_DIR / "artifacts"
DEFAULT_CHECKPOINT = (
    FULL_DIR / "artifacts" / "level4_confirmation" / "seed_490" / "best_model.pt"
)
K = 8
CLASSES = ("High", "Medium", "Low")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(THIS_DIR))

_shared_spec = importlib.util.spec_from_file_location(
    "shared2x_runtime", SHARED_DIR / "run_experiment.py"
)
if _shared_spec is None or _shared_spec.loader is None:
    raise RuntimeError("unable to load shared-2x runtime")
shared = importlib.util.module_from_spec(_shared_spec)
_shared_spec.loader.exec_module(shared)

from frameworks.hattrick_system import Hattrick
from repair import (
    solve_medium_low_pareto,
    solve_medium_low_per_od_pareto,
    solve_medium_with_edge_reservation,
)
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def scipy_pte(dataset) -> csr_matrix:
    matrix = dataset.pte.coalesce().cpu()
    indices = matrix.indices().numpy()
    values = matrix.values().numpy().astype(np.float64, copy=False)
    return coo_matrix(
        (values, (indices[0], indices[1])), shape=matrix.shape
    ).tocsr()


def torch_pte_info(dataset):
    matrix = dataset.pte.coalesce()
    indices = matrix.indices()
    return matrix, indices[0], indices[1], matrix.values()


def policy_forward(model, props, dataset, values, path_masks):
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_policy = True
    try:
        policies, capacities = shared.model_forward(
            model, props, dataset, values, path_masks
        )
    finally:
        props.research_return_policy = False
    return tuple(policy.reshape(1, -1, 1) for policy in policies), capacities


def simulate(model, props, policies, tms, capacities, pte_info):
    fractions = model.simulate(
        list(policies),
        list(tms),
        capacities,
        pte_info,
        1,
        props,
        rate_cap=props.rate_cap,
    )[:3]
    flows = tuple(
        fraction.reshape(1, -1) * tm.squeeze(-1)
        for fraction, tm in zip(fractions, tms)
    )
    return fractions, flows


def link_load(dataset, path_flow: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(
        dataset.pte.to(dtype=torch.float32).t(),
        path_flow.to(dtype=torch.float32).t(),
    ).t()


def rows_for_policy(
    dataset,
    values,
    policies,
    admitted,
    capacities,
    start_index: int,
    local_index: int,
    flat_masks,
    method: str,
    solver_runtime: float,
) -> list[dict]:
    capacities_full = values[1]
    tms = (values[2], values[4], values[6])
    oracle_flow = (values[11], values[12] - values[11], values[13] - values[12])
    oracle_mlu = (values[8], values[9], values[10])
    cumulative_admitted = torch.zeros_like(admitted[0])
    cumulative_emitted = torch.zeros_like(admitted[0])
    rows = []
    for class_index, class_name in enumerate(CLASSES):
        class_admitted = admitted[class_index]
        emitted = policies[class_index].reshape(1, -1) * tms[class_index].squeeze(-1)
        cumulative_admitted = cumulative_admitted + class_admitted
        cumulative_emitted = cumulative_emitted + emitted
        admitted_ratio = float(
            (link_load(dataset, cumulative_admitted) / capacities_full[:1]).max().item()
        )
        raw_mlu = float(
            (link_load(dataset, cumulative_emitted) / capacities[:1]).max().item()
        )
        admitted_total = float(class_admitted.sum().item())
        demand_total = float((tms[class_index].sum() / K).item())
        oracle = max(float(oracle_flow[class_index].item()), 1e-9)
        disabled_flow = 0.0
        if flat_masks is not None:
            disabled = ~flat_masks[class_index]
            if disabled.any():
                disabled_flow = float(class_admitted[:, disabled].abs().max().item())
        rows.append(
            {
                "snapshot": start_index + local_index,
                "class": class_name,
                "method": method,
                "admitted_traffic": admitted_total,
                "demand": demand_total,
                "fulfill_ratio": admitted_total / max(demand_total, 1e-9),
                "oracle_admitted_traffic": oracle,
                "norm_fulfill": admitted_total / oracle,
                "raw_mlu": raw_mlu,
                "oracle_mlu": float(oracle_mlu[class_index].item()),
                "normalized_mlu": raw_mlu / max(float(oracle_mlu[class_index].item()), 1e-9),
                "disabled_flow": disabled_flow,
                "admitted_capacity_ratio": admitted_ratio,
                "solver_runtime_seconds": solver_runtime,
            }
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    for method in ("baseline", "protected_pareto"):
        for class_name in CLASSES:
            selected = [
                row for row in rows
                if row["method"] == method and row["class"] == class_name
            ]
            item = {"method": method, "class": class_name, "n": len(selected)}
            for metric in (
                "admitted_traffic", "fulfill_ratio", "norm_fulfill",
                "admitted_capacity_ratio", "solver_runtime_seconds",
            ):
                values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
                item[f"{metric}_mean"] = float(values.mean())
                item[f"{metric}_p1"] = float(np.percentile(values, 1))
                item[f"{metric}_p10"] = float(np.percentile(values, 10))
            item["max_disabled_flow"] = max(float(row["disabled_flow"]) for row in selected)
            item["max_admitted_capacity_ratio"] = max(
                float(row["admitted_capacity_ratio"]) for row in selected
            )
            result.append(item)
    return result


def evaluate(
    model,
    props,
    dataset,
    start_index: int,
    reserve_factor: float,
    mode: str,
    predicted_dominance_guard: bool,
    policy_step: float,
    low_floor_scope: str,
):
    model.eval()
    props.mode = "test"
    props.sim_mf_mlu = 0
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    path_masks = shared.base.move_dataset_static(dataset, props.device)
    if path_masks is None:
        flat_masks = [
            torch.ones(dataset.pte.shape[0], dtype=torch.bool, device=props.device)
            for _ in CLASSES
        ]
    else:
        flat_masks = [path_masks[index].reshape(-1).to(dtype=torch.bool) for index in range(3)]
    pte_scipy = scipy_pte(dataset)
    pte_info = torch_pte_info(dataset)
    rows: list[dict] = []
    diagnostics: list[dict] = []
    policies_saved = []
    loader = shared.data_loader(dataset, 1, False, 0)
    with torch.no_grad():
        for local_index, inputs in enumerate(loader):
            values = shared.unpack_to_device(inputs, props)
            raw_policies, capacities = policy_forward(
                model, props, dataset, values, path_masks
            )
            predicted_tms = (values[3], values[5], values[7])
            actual_tms = (values[2], values[4], values[6])
            _, baseline_predicted = simulate(
                model, props, raw_policies, predicted_tms, capacities, pte_info
            )
            _, baseline_actual = simulate(
                model, props, raw_policies, actual_tms, capacities, pte_info
            )
            high_predicted_load = link_load(dataset, baseline_predicted[0])
            low_predicted_load = link_load(dataset, baseline_predicted[2])
            capacity_after_high = (capacities - high_predicted_load).clamp_min(0.0)
            if mode == "edge":
                medium_policy_np, solver = solve_medium_with_edge_reservation(
                    pte_scipy,
                    predicted_tms[1].squeeze().detach().cpu().numpy(),
                    flat_masks[1].detach().cpu().numpy(),
                    capacity_after_high.squeeze().detach().cpu().numpy(),
                    low_predicted_load.squeeze().detach().cpu().numpy(),
                    K,
                    reserve_factor,
                )
                low_policy_np = raw_policies[2].reshape(-1).detach().cpu().numpy()
            elif mode == "joint" and low_floor_scope == "aggregate":
                medium_policy_np, low_policy_np, solver = solve_medium_low_pareto(
                    pte_scipy,
                    predicted_tms[1].squeeze().detach().cpu().numpy(),
                    predicted_tms[2].squeeze().detach().cpu().numpy(),
                    flat_masks[1].detach().cpu().numpy(),
                    flat_masks[2].detach().cpu().numpy(),
                    capacity_after_high.squeeze().detach().cpu().numpy(),
                    float(baseline_predicted[2].sum().item()),
                    K,
                    reserve_factor,
                )
            elif mode == "joint" and low_floor_scope == "per_od":
                medium_policy_np, low_policy_np, solver = solve_medium_low_per_od_pareto(
                    pte_scipy,
                    predicted_tms[1].squeeze().detach().cpu().numpy(),
                    predicted_tms[2].squeeze().detach().cpu().numpy(),
                    baseline_predicted[2].reshape(-1).detach().cpu().numpy(),
                    flat_masks[1].detach().cpu().numpy(),
                    flat_masks[2].detach().cpu().numpy(),
                    capacity_after_high.squeeze().detach().cpu().numpy(),
                    K,
                    reserve_factor,
                )
            else:
                raise ValueError(f"unknown mode: {mode}")
            medium_policy = torch.from_numpy(medium_policy_np).to(
                device=props.device, dtype=props.dtype
            ).reshape(1, -1, 1)
            low_policy = torch.from_numpy(low_policy_np).to(
                device=props.device, dtype=props.dtype
            ).reshape(1, -1, 1)
            repaired_policies = (raw_policies[0], medium_policy, low_policy)
            _, repaired_predicted = simulate(
                model, props, repaired_policies, predicted_tms, capacities, pte_info
            )
            _, repaired_actual = simulate(
                model, props, repaired_policies, actual_tms, capacities, pte_info
            )
            predicted_medium_before = float(baseline_predicted[1].sum().item())
            predicted_low_before = float(baseline_predicted[2].sum().item())
            predicted_medium_after = float(repaired_predicted[1].sum().item())
            predicted_low_after = float(repaired_predicted[2].sum().item())
            guard_fallback = bool(
                predicted_dominance_guard
                and (
                    predicted_medium_after + 1e-6 < predicted_medium_before
                    or predicted_low_after + 1e-6 < predicted_low_before
                )
            )
            if guard_fallback:
                repaired_policies = raw_policies
                repaired_predicted = baseline_predicted
                repaired_actual = baseline_actual
                medium_policy_np = raw_policies[1].reshape(-1).detach().cpu().numpy()
                low_policy_np = raw_policies[2].reshape(-1).detach().cpu().numpy()
            elif policy_step < 1.0:
                repaired_policies = (
                    raw_policies[0],
                    raw_policies[1] + float(policy_step) * (repaired_policies[1] - raw_policies[1]),
                    raw_policies[2] + float(policy_step) * (repaired_policies[2] - raw_policies[2]),
                )
                _, repaired_predicted = simulate(
                    model, props, repaired_policies, predicted_tms, capacities, pte_info
                )
                _, repaired_actual = simulate(
                    model, props, repaired_policies, actual_tms, capacities, pte_info
                )
                medium_policy_np = repaired_policies[1].reshape(-1).detach().cpu().numpy()
                low_policy_np = repaired_policies[2].reshape(-1).detach().cpu().numpy()
            baseline_rows = rows_for_policy(
                dataset, values, raw_policies, baseline_actual, capacities,
                start_index, local_index, flat_masks, "baseline", 0.0,
            )
            repaired_rows = rows_for_policy(
                dataset, values, repaired_policies, repaired_actual, capacities,
                start_index, local_index, flat_masks, "protected_pareto",
                solver["runtime_seconds"],
            )
            rows.extend(baseline_rows)
            rows.extend(repaired_rows)
            predicted_totals_before = [float(flow.sum().item()) for flow in baseline_predicted]
            predicted_totals_after = [float(flow.sum().item()) for flow in repaired_predicted]
            diagnostics.append(
                {
                    "snapshot": start_index + local_index,
                    **solver,
                    "predicted_dominance_guard": bool(predicted_dominance_guard),
                    "guard_fallback": guard_fallback,
                    "policy_step": float(policy_step),
                    "baseline_predicted_high": predicted_totals_before[0],
                    "baseline_predicted_medium": predicted_totals_before[1],
                    "baseline_predicted_low": predicted_totals_before[2],
                    "repaired_predicted_high": predicted_totals_after[0],
                    "repaired_predicted_medium": predicted_totals_after[1],
                    "repaired_predicted_low": predicted_totals_after[2],
                }
            )
            policies_saved.append(
                np.stack(
                    [
                        raw_policies[0].reshape(-1).detach().cpu().numpy(),
                        medium_policy_np,
                        low_policy_np,
                    ]
                )
            )
    if any(
        not math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key not in ("class", "method")
    ):
        raise RuntimeError("non-finite evaluation output")
    return rows, summarize(rows), diagnostics, np.stack(policies_saved)


def paired_audit(rows: list[dict]) -> dict:
    keyed = {
        (int(row["snapshot"]), row["class"], row["method"]): row
        for row in rows
    }
    snapshots = sorted({int(row["snapshot"]) for row in rows})
    result = {}
    for class_name in CLASSES:
        baseline = np.asarray(
            [float(keyed[(snapshot, class_name, "baseline")]["norm_fulfill"]) for snapshot in snapshots]
        )
        repaired = np.asarray(
            [float(keyed[(snapshot, class_name, "protected_pareto")]["norm_fulfill"]) for snapshot in snapshots]
        )
        gap = repaired - baseline
        result[class_name] = {
            "baseline_mean": float(baseline.mean()),
            "repaired_mean": float(repaired.mean()),
            "mean_gap": float(gap.mean()),
            "p1_gap": float(np.percentile(repaired, 1) - np.percentile(baseline, 1)),
            "p10_gap": float(np.percentile(repaired, 10) - np.percentile(baseline, 10)),
            "paired_positive_fraction": float(np.mean(gap > 0.0)),
            "paired_min_gap": float(gap.min()),
            "paired_max_gap": float(gap.max()),
        }
    result["max_high_admitted_abs_delta"] = max(
        abs(
            float(keyed[(snapshot, "High", "protected_pareto")]["admitted_traffic"])
            - float(keyed[(snapshot, "High", "baseline")]["admitted_traffic"])
        )
        for snapshot in snapshots
    )
    result["max_capacity_ratio"] = max(
        float(row["admitted_capacity_ratio"]) for row in rows
    )
    result["max_disabled_flow"] = max(float(row["disabled_flow"]) for row in rows)
    return result


def run(args) -> Path:
    if args.end <= args.start:
        raise ValueError("end must be greater than start")
    if not (0.0 <= float(args.policy_step) <= 1.0):
        raise ValueError("policy_step must be in [0, 1]")
    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    shared.set_seed(490)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    props = shared.build_props(4, device)
    props.research_return_admitted = False
    props.research_return_policy = False
    dataset = DM_Dataset_within_Cluster(props, 0, args.start, args.end)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(checkpoint["model_state_dict"])
    label = args.label or f"{args.mode}_{args.start}_{args.end}_reserve_{args.reserve_factor:g}"
    run_dir = OUTPUT_ROOT / label
    complete_path = run_dir / "complete.json"
    if complete_path.exists() and not args.force:
        print(complete_path.read_text(encoding="utf-8"), flush=True)
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "approach": "protected_pareto_joint_lp" if args.mode == "joint" else "protected_pareto_edge_reservation",
        "evaluation": [args.start, args.end],
        "reserve_factor": float(args.reserve_factor),
        "predicted_dominance_guard": bool(args.predicted_dominance_guard),
        "policy_step": float(args.policy_step),
        "low_floor_scope": args.low_floor_scope,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": sha256(checkpoint_path),
        "inference_inputs": "ESM-predicted TMs, topology, capacities, paths, and masks only",
        "mechanism": (
            "freeze High; jointly reroute Medium/Low; constrain predicted Low at the configured aggregate or per-OD scope; lexicographically maximize Medium then Low"
            if args.mode == "joint"
            else "freeze High and Low policies; reserve baseline predicted admitted Low load edge-wise; maximize predicted Medium admission"
        ),
        "source_sha256": {
            "run_experiment.py": sha256(Path(__file__).resolve()),
            "repair.py": sha256(THIS_DIR / "repair.py"),
            "frameworks/hattrick_system.py": sha256(ROOT / "frameworks" / "hattrick_system.py"),
        },
        "data_access": {
            "loaded_index_range": list(dataset.loaded_index_range),
            "max_source_index_read": int(dataset.max_source_index_read),
        },
        "device": str(device),
    }
    write_json(run_dir / "config.json", config)
    started = time.perf_counter()
    rows, summary, diagnostics, policies = evaluate(
        model,
        props,
        dataset,
        args.start,
        float(args.reserve_factor),
        args.mode,
        bool(args.predicted_dominance_guard),
        float(args.policy_step),
        args.low_floor_scope,
    )
    elapsed = time.perf_counter() - started
    audit = paired_audit(rows)
    write_csv(run_dir / "metrics.csv", rows)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "solver_diagnostics.json", diagnostics)
    np.savez_compressed(
        run_dir / "policies.npz",
        policies=policies,
        snapshots=np.arange(args.start, args.end, dtype=np.int64),
    )
    complete = {
        "evaluation_rows": len(rows),
        "runtime_seconds": elapsed,
        "paired_audit": audit,
        "prediction_contract": {
            "minimum_medium_gap": min(
                row["repaired_predicted_medium"] - row["baseline_predicted_medium"]
                for row in diagnostics
            ),
            "minimum_low_gap": min(
                row["repaired_predicted_low"] - row["baseline_predicted_low"]
                for row in diagnostics
            ),
        },
        "passes_correctness": bool(
            len(rows) == 6 * (args.end - args.start)
            and audit["max_high_admitted_abs_delta"] <= 1e-5
            and audit["max_capacity_ratio"] <= 1.0001
            and audit["max_disabled_flow"] <= 1e-8
        ),
    }
    write_json(complete_path, complete)
    if not complete["passes_correctness"]:
        raise RuntimeError(f"correctness audit failed: {complete}")
    print(json.dumps(complete, indent=2), flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--reserve-factor", type=float, required=True)
    parser.add_argument("--mode", choices=("edge", "joint"), default="joint")
    parser.add_argument("--predicted-dominance-guard", action="store_true")
    parser.add_argument("--policy-step", type=float, default=1.0)
    parser.add_argument("--low-floor-scope", choices=("aggregate", "per_od"), default="aggregate")
    parser.add_argument("--label")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
