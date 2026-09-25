from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
OUTPUT_ROOT = THIS_DIR / "results_hattrick_improvement_strict2x"
APPROACH = "hattrick_residual_lp"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(THIS_DIR))

from frameworks.hattrick_system import Hattrick
import run_hattrick_strict2x_research as research


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scipy_pte(dataset) -> csr_matrix:
    pte = dataset.pte.coalesce().cpu()
    indices = pte.indices().numpy()
    values = pte.values().numpy().astype(np.float64, copy=False)
    return coo_matrix((values, (indices[0], indices[1])), shape=pte.shape).tocsr()


def torch_pte_info(dataset):
    pte = dataset.pte.coalesce()
    indices = pte.indices()
    return pte, indices[0], indices[1], pte.values()


def solve_residual_class(
    pte: csr_matrix,
    predicted_tm_per_path: np.ndarray,
    enabled_mask: np.ndarray,
    residual_capacity: np.ndarray,
    k: int,
) -> tuple[np.ndarray, dict]:
    """Maximize predicted admitted flow on enabled paths under exact residual capacity."""
    started = time.perf_counter()
    predicted_tm_per_path = np.asarray(predicted_tm_per_path, dtype=np.float64).reshape(-1)
    enabled_mask = np.asarray(enabled_mask, dtype=bool).reshape(-1)
    residual_capacity = np.maximum(np.asarray(residual_capacity, dtype=np.float64).reshape(-1), 0.0)
    active = np.flatnonzero(enabled_mask)
    num_paths = predicted_tm_per_path.size
    num_pairs = num_paths // k
    if active.size == 0:
        raise RuntimeError("Residual LP received no enabled variables")

    demand = predicted_tm_per_path[active]
    edge_constraints = pte[active, :].T.multiply(demand)
    pair_rows = active // k
    pair_constraints = coo_matrix(
        (np.ones(active.size, dtype=np.float64), (pair_rows, np.arange(active.size))),
        shape=(num_pairs, active.size),
    ).tocsr()
    a_ub = vstack((edge_constraints, pair_constraints), format="csr")
    b_ub = np.concatenate((residual_capacity, np.ones(num_pairs, dtype=np.float64)))
    result = linprog(
        -demand,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=(0.0, 1.0),
        method="highs",
        options={"presolve": True},
    )
    elapsed = time.perf_counter() - started
    if not result.success or result.x is None or not np.isfinite(result.x).all():
        raise RuntimeError(f"Residual LP failed: status={result.status} message={result.message}")

    policy = np.zeros(num_paths, dtype=np.float32)
    policy[active] = np.clip(result.x, 0.0, 1.0).astype(np.float32)
    edge_load = pte.T @ (policy.astype(np.float64) * predicted_tm_per_path)
    pair_mass = policy.reshape(num_pairs, k).sum(axis=1)
    diagnostics = {
        "status": int(result.status),
        "iterations": int(result.nit),
        "runtime_seconds": float(elapsed),
        "predicted_admitted": float(-(result.fun)),
        "max_edge_excess": float(np.max(edge_load - residual_capacity)),
        "max_pair_mass": float(pair_mass.max()),
    }
    if diagnostics["max_edge_excess"] > 1e-6 or diagnostics["max_pair_mass"] > 1.0 + 1e-7:
        raise RuntimeError(f"Residual LP invariant failure: {diagnostics}")
    return policy, diagnostics


def policy_forward(model, props, dataset, values, path_masks):
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_policy = True
    try:
        policies, capacities = research.model_forward(model, props, dataset, values, path_masks)
    finally:
        props.research_return_policy = False
    return tuple(policy.reshape(1, -1, 1) for policy in policies), capacities


def evaluate_hybrid(model, props, dataset, start_index: int):
    model.eval()
    props.mode = "test"
    props.research_return_policy = False
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    path_masks = research.move_dataset_static(dataset, props.device)
    if path_masks is None:
        raise RuntimeError("Strict residual-LP evaluation requires path masks")
    flat_masks = [path_masks[index].reshape(-1) for index in range(3)]
    flat_masks_np = [mask.detach().cpu().numpy().astype(bool) for mask in flat_masks]
    pte_scipy = scipy_pte(dataset)
    pte_info = torch_pte_info(dataset)

    rows: list[dict] = []
    policies_saved = []
    admitted_saved = []
    solver_saved = []
    loader = research.data_loader(dataset, 1, False, 0)
    with torch.no_grad():
        for local_index, inputs in enumerate(loader):
            values = research.unpack_to_device(inputs, props)
            (
                _node_features,
                capacities_full,
                tm1,
                tm1_pred,
                tm2,
                tm2_pred,
                tm3,
                tm3_pred,
                opt1,
                opt2,
                opt3,
                opt1_mf,
                opt2_mf,
                opt3_mf,
                _snapshots,
            ) = values
            raw_policy, capacities = policy_forward(model, props, dataset, values, path_masks)
            high_policy = raw_policy[0]
            zeros = torch.zeros_like(high_policy)

            predicted_high_fraction, _, _, _ = model.simulate(
                [high_policy, zeros, zeros],
                [tm1_pred, torch.zeros_like(tm2_pred), torch.zeros_like(tm3_pred)],
                capacities,
                pte_info,
                1,
                props,
                rate_cap=props.rate_cap,
            )
            predicted_high_rate = predicted_high_fraction * tm1_pred.squeeze(-1)
            predicted_high_load = torch.sparse.mm(
                dataset.pte.to(dtype=torch.float32).t(), predicted_high_rate.to(dtype=torch.float32).t()
            ).t()
            residual_medium = (capacities - predicted_high_load).clamp_min(0.0)
            medium_policy_np, medium_diag = solve_residual_class(
                pte_scipy,
                tm2_pred.squeeze().detach().cpu().numpy(),
                flat_masks_np[1],
                residual_medium.squeeze().detach().cpu().numpy(),
                research.K,
            )
            medium_policy = torch.from_numpy(medium_policy_np).to(
                device=props.device, dtype=props.dtype
            ).reshape(1, -1, 1)

            predicted_medium_rate = medium_policy.reshape(1, -1) * tm2_pred.squeeze(-1)
            predicted_medium_load = torch.sparse.mm(
                dataset.pte.to(dtype=torch.float32).t(), predicted_medium_rate.to(dtype=torch.float32).t()
            ).t()
            residual_low = (residual_medium - predicted_medium_load).clamp_min(0.0)
            low_policy_np, low_diag = solve_residual_class(
                pte_scipy,
                tm3_pred.squeeze().detach().cpu().numpy(),
                flat_masks_np[2],
                residual_low.squeeze().detach().cpu().numpy(),
                research.K,
            )
            low_policy = torch.from_numpy(low_policy_np).to(
                device=props.device, dtype=props.dtype
            ).reshape(1, -1, 1)

            admitted_fraction = model.simulate(
                [high_policy, medium_policy, low_policy],
                [tm1, tm2, tm3],
                capacities,
                pte_info,
                1,
                props,
                rate_cap=props.rate_cap,
            )[:3]
            admitted = tuple(
                fraction.reshape(1, -1) * tm.squeeze(-1)
                for fraction, tm in zip(admitted_fraction, (tm1, tm2, tm3))
            )
            emitted = (
                high_policy.reshape(1, -1) * tm1.squeeze(-1),
                medium_policy.reshape(1, -1) * tm2.squeeze(-1),
                low_policy.reshape(1, -1) * tm3.squeeze(-1),
            )
            policies_saved.append(
                np.stack(
                    [
                        high_policy.reshape(-1).detach().cpu().numpy(),
                        medium_policy_np,
                        low_policy_np,
                    ]
                )
            )
            admitted_saved.append(np.stack([item.reshape(-1).detach().cpu().numpy() for item in admitted]))
            solver_saved.append(
                {
                    "snapshot": start_index + local_index,
                    "medium": medium_diag,
                    "low": low_diag,
                }
            )

            oracle_flow = (opt1_mf, opt2_mf - opt1_mf, opt3_mf - opt2_mf)
            oracle_mlu = (opt1, opt2, opt3)
            tms = (tm1, tm2, tm3)
            cumulative_admitted = torch.zeros_like(admitted[0])
            cumulative_emitted = torch.zeros_like(emitted[0])
            for class_index, class_name in enumerate(research.CLASSES):
                class_admitted = admitted[class_index]
                cumulative_admitted = cumulative_admitted + class_admitted
                cumulative_emitted = cumulative_emitted + emitted[class_index]
                admitted_on_links = torch.sparse.mm(
                    dataset.pte.to(dtype=torch.float32).t(), cumulative_admitted.to(dtype=torch.float32).t()
                ).t()
                emitted_on_links = torch.sparse.mm(
                    dataset.pte.to(dtype=torch.float32).t(), cumulative_emitted.to(dtype=torch.float32).t()
                ).t()
                admitted_capacity_ratio = float(
                    (admitted_on_links / capacities_full[:1].to(dtype=torch.float32)).max().item()
                )
                raw_mlu = float(
                    (emitted_on_links / capacities_full[:1].to(dtype=torch.float32)).max().item()
                )
                admitted_total = float(class_admitted.sum().item())
                demand_total = float((tms[class_index].sum() / research.K).item())
                oracle = max(float(oracle_flow[class_index].item()), 1e-9)
                disabled = ~flat_masks[class_index]
                disabled_flow = (
                    float(class_admitted[:, disabled].abs().max().item()) if disabled.any() else 0.0
                )
                rows.append(
                    {
                        "snapshot": start_index + local_index,
                        "class": class_name,
                        "admitted_traffic": admitted_total,
                        "demand": demand_total,
                        "fulfill_ratio": admitted_total / max(demand_total, 1e-9),
                        "oracle_admitted_traffic": oracle,
                        "norm_fulfill": admitted_total / oracle,
                        "raw_mlu": raw_mlu,
                        "oracle_mlu": float(oracle_mlu[class_index].item()),
                        "normalized_mlu": raw_mlu / max(float(oracle_mlu[class_index].item()), 1e-9),
                        "disabled_flow": disabled_flow,
                        "admitted_capacity_ratio": admitted_capacity_ratio,
                        "medium_lp_runtime_seconds": medium_diag["runtime_seconds"],
                        "low_lp_runtime_seconds": low_diag["runtime_seconds"],
                    }
                )

    if any(
        not math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key != "class"
    ):
        raise RuntimeError("Hybrid evaluation produced NaN or Inf")
    return (
        rows,
        research.summarize_rows(rows),
        np.stack(policies_saved),
        np.stack(admitted_saved),
        solver_saved,
    )


def selected_baseline_dir(level: int, seed: int) -> Path:
    return OUTPUT_ROOT / research.LEVELS[level]["label"] / "unchanged" / f"seed_{seed}"


def run_one(level: int, seed: int, force: bool = False) -> Path:
    spec = research.LEVELS[level]
    baseline_dir = selected_baseline_dir(level, seed)
    checkpoint_path = baseline_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Matched unchanged checkpoint is missing: {checkpoint_path}")
    run_dir = OUTPUT_ROOT / spec["label"] / APPROACH / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    complete_path = run_dir / "complete.json"
    if complete_path.exists() and not force:
        print(f"[skip] complete {run_dir}", flush=True)
        return run_dir

    research.set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = research.build_props(level, device)
    props.research_return_policy = False
    eval_start, eval_end = spec["evaluation"]
    dataset = research.DM_Dataset_within_Cluster(props, 0, eval_start, eval_end)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(checkpoint["model_state_dict"])

    config = {
        "level": level,
        "approach": APPROACH,
        "seed": seed,
        "evaluation": [eval_start, eval_end],
        "evaluation_is_final_test": level == 4,
        "matched_baseline_checkpoint": str(checkpoint_path),
        "matched_baseline_checkpoint_epoch": int(checkpoint["epoch"]),
        "mechanism": "freeze Hattrick High path policy; maximize predicted Medium then Low admitted flow over exact residual capacity with disabled paths eliminated",
        "inference_inputs": "ESM-predicted TMs, topology, capacities, candidate paths, strict masks only",
        "capacity_margin": 0.0,
        "solver": "scipy.optimize.linprog(method='highs')",
        "metric_contract": {
            "promotion_mlu": "common post-admission cumulative link load divided by capacity",
            "pre_admission_normalized_mlu": "diagnostic only; not comparable because this policy jointly emits admission and routing fractions",
        },
        "source_sha256": {
            "frameworks/hattrick_system.py": sha256(ROOT / "frameworks" / "hattrick_system.py"),
            "utils/build_dataset_within_cluster.py": sha256(ROOT / "utils" / "build_dataset_within_cluster.py"),
            "test_diff_path/run_hattrick_residual_lp.py": sha256(Path(__file__).resolve()),
            "matched_baseline_checkpoint": sha256(checkpoint_path),
        },
        "dataset_access": {
            "loaded_index_range": list(dataset.loaded_index_range),
            "max_source_index_read": dataset.max_source_index_read,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows, summary, policies, admitted, solver_rows = evaluate_hybrid(model, props, dataset, eval_start)
    research.write_csv(run_dir / "best_evaluation_metrics.csv", rows)
    (run_dir / "best_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        run_dir / "emitted_policy_and_admitted_flow.npz",
        policies=policies,
        admitted=admitted,
        snapshots=np.arange(eval_start, eval_end, dtype=np.int64),
    )
    (run_dir / "solver_diagnostics.json").write_text(json.dumps(solver_rows, indent=2), encoding="utf-8")

    baseline_rows = research.read_csv(baseline_dir / "best_evaluation_metrics.csv")
    baseline_high = {
        int(row["snapshot"]): float(row["admitted_traffic"])
        for row in baseline_rows
        if row["class"] == "High"
    }
    candidate_high = {
        int(row["snapshot"]): float(row["admitted_traffic"])
        for row in rows
        if row["class"] == "High"
    }
    high_max_delta = max(abs(candidate_high[key] - baseline_high[key]) for key in baseline_high)
    max_capacity = max(float(row["admitted_capacity_ratio"]) for row in rows)
    max_disabled = max(float(row["disabled_flow"]) for row in rows)
    complete = {
        "source_checkpoint_epoch": int(checkpoint["epoch"]),
        "evaluation_rows": len(rows),
        "solver_calls": 2 * (eval_end - eval_start),
        "all_solver_calls_successful": True,
        "high_admitted_max_abs_delta_from_matched_baseline": high_max_delta,
        "max_admitted_capacity_ratio": max_capacity,
        "max_disabled_flow": max_disabled,
        "finite": all(
            math.isfinite(float(value))
            for row in rows
            for key, value in row.items()
            if key != "class"
        ),
        "passes_correctness": bool(
            len(rows) == 3 * (eval_end - eval_start)
            and high_max_delta <= 1e-5
            and max_capacity <= 1.0001
            and max_disabled <= 1e-8
        ),
    }
    complete_path.write_text(json.dumps(complete, indent=2), encoding="utf-8")
    if not complete["passes_correctness"]:
        raise RuntimeError(f"Hybrid correctness failure: {complete}")
    return run_dir


def analyze_level2(seeds: list[int]) -> Path:
    level_dir = OUTPUT_ROOT / research.LEVELS[2]["label"]
    results = []
    for seed in seeds:
        baseline = research.read_csv(selected_baseline_dir(2, seed) / "best_evaluation_metrics.csv")
        candidate = research.read_csv(level_dir / APPROACH / f"seed_{seed}" / "best_evaluation_metrics.csv")

        def values(rows, class_name, metric):
            return np.asarray(
                [float(row[metric]) for row in rows if row["class"] == class_name], dtype=np.float64
            )

        base_medium = values(baseline, "Medium", "norm_fulfill")
        cand_medium = values(candidate, "Medium", "norm_fulfill")
        base_high = values(baseline, "High", "norm_fulfill")
        cand_high = values(candidate, "High", "norm_fulfill")
        cand_capacity = values(candidate, "Low", "admitted_capacity_ratio")
        base_capacity = values(baseline, "Low", "admitted_capacity_ratio")
        cand_mlu = values(candidate, "Low", "normalized_mlu")
        base_mlu = values(baseline, "Low", "normalized_mlu")
        paired = cand_medium - base_medium
        item = {
            "seed": seed,
            "baseline_medium_p10": research.percentile(base_medium.tolist(), 10),
            "candidate_medium_p10": research.percentile(cand_medium.tolist(), 10),
            "medium_p10_gap": research.percentile(cand_medium.tolist(), 10) - research.percentile(base_medium.tolist(), 10),
            "baseline_medium_p1": research.percentile(base_medium.tolist(), 1),
            "candidate_medium_p1": research.percentile(cand_medium.tolist(), 1),
            "medium_p1_gap": research.percentile(cand_medium.tolist(), 1) - research.percentile(base_medium.tolist(), 1),
            "baseline_high_mean": float(base_high.mean()),
            "candidate_high_mean": float(cand_high.mean()),
            "high_mean_gap": float(cand_high.mean() - base_high.mean()),
            "admitted_capacity_ratio_mean_gap": float(cand_capacity.mean() - base_capacity.mean()),
            "common_post_admission_mlu_mean_gap": float(cand_capacity.mean() - base_capacity.mean()),
            "baseline_normalized_mlu_mean": float(base_mlu.mean()),
            "candidate_normalized_mlu_mean": float(cand_mlu.mean()),
            "normalized_mlu_mean_gap": float(cand_mlu.mean() - base_mlu.mean()),
            "normalized_mlu_comparable": False,
            "candidate_max_admitted_capacity_ratio": float(cand_capacity.max()),
            "candidate_max_disabled_flow": max(float(row["disabled_flow"]) for row in candidate),
            "paired_medium_mean_gap": float(paired.mean()),
            "paired_medium_median_gap": float(np.median(paired)),
            "paired_medium_positive_slices": int((paired > 0).sum()),
            "paired_medium_total_slices": int(paired.size),
            "paired_medium_min_gap": float(paired.min()),
            "paired_medium_max_gap": float(paired.max()),
            "median_online_solver_seconds": float(
                np.median(
                    values(candidate, "High", "medium_lp_runtime_seconds")
                    + values(candidate, "High", "low_lp_runtime_seconds")
                )
            ),
        }
        item["passes"] = bool(
            item["medium_p10_gap"] >= 0.02
            and item["high_mean_gap"] >= -0.01
            and item["common_post_admission_mlu_mean_gap"] <= 0.01
            and item["candidate_max_admitted_capacity_ratio"] <= 1.0001
            and item["candidate_max_disabled_flow"] <= 1e-8
            and item["paired_medium_positive_slices"] >= math.ceil(0.5 * item["paired_medium_total_slices"])
        )
        results.append(item)

    decision = {
        "level": 2,
        "approach": APPROACH,
        "evaluation_window": list(research.LEVELS[2]["evaluation"]),
        "final_test_opened": False,
        "required_seeds": seeds,
        "results": results,
        "promote": len(results) == len(seeds) and len(results) >= 2 and all(item["passes"] for item in results),
        "rule": {
            "medium_p10_gap_min": 0.02,
            "high_mean_gap_min": -0.01,
            "common_post_admission_mlu_mean_gap_max": 0.01,
            "candidate_max_admitted_capacity_ratio": 1.0001,
            "candidate_max_disabled_flow": 1e-8,
            "paired_positive_fraction_min": 0.5,
        },
    }
    output = level_dir / "hattrick_residual_lp_level2_gate.json"
    output.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=sorted(research.LEVELS), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[490])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.level == 2 and len(args.seeds) < 2:
        raise SystemExit("Level 2 requires at least two matched baseline seeds")
    if args.level == 4:
        raise SystemExit("Level 4 is locked until the hybrid policy is frozen after Level 3")
    for seed in args.seeds:
        run_dir = run_one(args.level, seed, force=args.force)
        print(f"[ok] {run_dir}", flush=True)
    if args.level == 2:
        gate = analyze_level2(args.seeds)
        print(f"[ok] {gate}", flush=True)


if __name__ == "__main__":
    main()
