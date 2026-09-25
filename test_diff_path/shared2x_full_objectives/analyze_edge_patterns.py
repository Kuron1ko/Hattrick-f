from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.sparse import coo_matrix
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
SHARED_RUNNER_DIR = TEST_DIR / "shared2x_order_regularizer"
TWO_PHASE_DIR = THIS_DIR / "two_phase"
DEFAULT_RUN_DIR = THIS_DIR / "artifacts" / "level4_confirmation" / "seed_490"
DEFAULT_OUTPUT_DIR = THIS_DIR / "artifacts" / "restored_edge_pattern_analysis"
CLASSES = ("High", "Medium", "Low")
NODE10_CORRIDOR = {(4, 10), (10, 4), (10, 21), (21, 10)}
NODE6_INCIDENT = {
    (1, 6), (6, 1), (2, 6), (6, 2), (4, 6), (6, 4),
    (5, 6), (6, 5), (13, 6), (6, 13), (21, 6), (6, 21),
}
RANGE_EDGES = NODE10_CORRIDOR | NODE6_INCIDENT

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(SHARED_RUNNER_DIR))

import run_experiment as runner  # noqa: E402
from frameworks.gurobi_utils import GurobiModel  # noqa: E402
from frameworks.hattrick_system import Hattrick  # noqa: E402
from utils.args_parser import parse_args as parse_project_args  # noqa: E402
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_gurobi_props():
    return parse_project_args(
        [
            "--topo", runner.TOPOLOGY,
            "--framework", "gurobi",
            "--num_paths_per_pair", str(runner.K),
            "--pred", "0",
            "--priority", "3",
            "--objs", "mf", "mf", "mf",
            "--gur_mode", "flexile",
            "--cluster", "0",
            "--path_mask", "0",
            "--tol", "0.00001",
        ]
    )


def scipy_pte(tensor: torch.Tensor):
    tensor = tensor.coalesce().cpu()
    indices = tensor.indices().numpy()
    values = tensor.values().to(dtype=torch.float64).numpy()
    return coo_matrix((values, (indices[0], indices[1])), shape=tensor.shape).tocsr()


def solve_oracle(
    snapshot,
    pte,
    opt1: float,
    opt2: float,
    opt3: float,
    props,
    range_edge_ids: dict[tuple[int, int], int],
    compute_edge_ranges: bool = True,
) -> tuple[
    np.ndarray,
    float,
    dict[tuple[int, int], tuple[float, float]],
    dict[str, tuple[float, float]],
]:
    solver = GurobiModel(props, snapshot, "flexile")
    solver.model.setParam("OutputFlag", 0)
    solver.model.setParam("NumericFocus", 3)
    solver.model.setParam("FeasibilityTol", 1e-6)
    solver.add_variables(props.objs, 0)
    solver.add_mlu_variables(props.objs)
    solver.mul_vars_by_tms()
    solver.add_demand_constraints(props.objs)
    solver.add_optimality_constraints(opt1, opt2, props.objs)
    solver.add_capacity_constraints(props.objs, pte)
    solver.define_objective(props.objs)
    solver.model.optimize()
    if solver.model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status {solver.model.Status}")
    class_flows = []
    for class_index in range(3):
        ratios = np.asarray(solver.model_variables[class_index].X, dtype=np.float64)
        tm = np.asarray(solver.tms[class_index], dtype=np.float64).reshape(-1, runner.K)
        class_flows.append((ratios * tm).reshape(-1))
    total_path_flow = sum(class_flows)
    total_link_load = np.asarray(pte.T @ total_path_flow.reshape(-1, 1)).reshape(-1)
    objective = float(solver.model.ObjVal)
    if abs(objective - opt3) > 2e-3:
        raise RuntimeError(f"Oracle objective mismatch: {objective} vs {opt3}")

    # A single max-flow optimum can have many link allocations.  Hold all three
    # lexicographic flow levels at their optima, then measure the feasible load
    # range on the suspicious links.  A positive (minimum-optimal - Hattrick)
    # gap is stronger evidence than comparison with one arbitrary optimum.
    total_flow = sum(value.sum() for value in solver.model_variables_by_tms.values())
    solver.model.addConstr(total_flow >= opt3 * (1 - props.tol), name="hold_total_optimum")
    optimal_ranges: dict[tuple[int, int], tuple[float, float]] = {}
    if compute_edge_ranges:
        for edge, edge_id in sorted(range_edge_ids.items()):
            link_load = solver.commodities_on_links[2][edge_id][0]
            solver.model.setObjective(link_load, GRB.MINIMIZE)
            solver.model.optimize()
            if solver.model.Status != GRB.OPTIMAL:
                raise RuntimeError(f"Gurobi min-range status {solver.model.Status} on {edge}")
            minimum = float(link_load.getValue())
            solver.model.setObjective(link_load, GRB.MAXIMIZE)
            solver.model.optimize()
            if solver.model.Status != GRB.OPTIMAL:
                raise RuntimeError(f"Gurobi max-range status {solver.model.Status} on {edge}")
            maximum = float(link_load.getValue())
            optimal_ranges[edge] = (minimum, maximum)
    group_ranges: dict[str, tuple[float, float]] = {}
    for name, members in (
        ("node10_corridor", NODE10_CORRIDOR),
        ("node6_incident", NODE6_INCIDENT),
    ):
        utilization_sum = sum(
            solver.commodities_on_links[2][range_edge_ids[edge]][0]
            / float(solver.capacities[range_edge_ids[edge]])
            for edge in members
        )
        solver.model.setObjective(utilization_sum, GRB.MINIMIZE)
        solver.model.optimize()
        if solver.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi min group-range status {solver.model.Status} on {name}")
        minimum = float(utilization_sum.getValue())
        solver.model.setObjective(utilization_sum, GRB.MAXIMIZE)
        solver.model.optimize()
        if solver.model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi max group-range status {solver.model.Status} on {name}")
        maximum = float(utilization_sum.getValue())
        group_ranges[name] = (minimum, maximum)
    solver.model.dispose()
    return total_link_load, objective, optimal_ranges, group_ranges


def edge_group(source: int, target: int) -> str:
    if (source, target) in NODE10_CORRIDOR:
        return "node10_corridor"
    if source == 6 or target == 6:
        return "node6_incident"
    return "other_edges"


def safe_corr(left: list[float], right: list[float]) -> float | None:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.size < 2 or float(a.std()) <= 1e-12 or float(b.std()) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def replay_split(
    split_name: str,
    start: int,
    end: int,
    model: Hattrick,
    props,
    seed: int,
    gurobi_props,
    compute_edge_ranges: bool,
) -> tuple[list[dict], list[dict]]:
    dataset = DM_Dataset_within_Cluster(props, 0, start, end)
    if int(dataset.max_source_index_read) != end - 1:
        raise RuntimeError("Split-safe data audit failed")
    pte = scipy_pte(dataset.pte)
    path_masks = runner.base.move_dataset_static(dataset, props.device)
    loader = runner.data_loader(dataset, 1, False, seed)
    edge_rows: list[dict] = []
    snapshot_rows: list[dict] = []

    for local_index, inputs in enumerate(loader):
        snapshot_id = start + local_index
        values = runner.unpack_to_device(inputs, props)
        snapshot = values[14][0]
        capacities = values[1][:1].reshape(-1).to(dtype=torch.float32).cpu().numpy()
        actual_tms = (values[2], values[4], values[6])
        oracle_totals = (
            float(values[11].reshape(-1)[0].item()),
            float((values[12] - values[11]).reshape(-1)[0].item()),
            float((values[13] - values[12]).reshape(-1)[0].item()),
        )

        props.mode = "test"
        props.sim_mf_mlu = 0
        props.research_return_policy = True
        with torch.no_grad():
            policies, _ = runner.model_forward(model, props, dataset, values, path_masks)
        props.research_return_policy = False
        props.sim_mf_mlu = 1
        with torch.no_grad():
            admitted, _ = runner.model_forward(model, props, dataset, values, path_masks)
        props.sim_mf_mlu = 0

        planned_path = np.zeros(pte.shape[0], dtype=np.float64)
        admitted_path = np.zeros_like(planned_path)
        class_admitted = []
        for class_index in range(3):
            policy = policies[class_index].detach().to(dtype=torch.float32).reshape(-1).cpu().numpy()
            demand = actual_tms[class_index].detach().to(dtype=torch.float32).reshape(-1).cpu().numpy()
            planned_path += policy * demand
            admitted_values = admitted[class_index].detach().to(dtype=torch.float32).reshape(-1).cpu().numpy()
            admitted_path += admitted_values
            class_admitted.append(float(admitted_values.sum()))

        planned_link = np.asarray(pte.T @ planned_path.reshape(-1, 1)).reshape(-1)
        admitted_link = np.asarray(pte.T @ admitted_path.reshape(-1, 1)).reshape(-1)
        opt1 = oracle_totals[0]
        opt2 = oracle_totals[0] + oracle_totals[1]
        opt3 = sum(oracle_totals)
        edge_names = list(snapshot.graph.edges())
        if len(edge_names) != len(capacities):
            raise RuntimeError("Edge ordering mismatch")
        range_edge_ids = {
            (int(source), int(target)): edge_id
            for edge_id, (source, target) in enumerate(edge_names)
            if (int(source), int(target)) in RANGE_EDGES
        }
        if set(range_edge_ids) != RANGE_EDGES:
            raise RuntimeError(f"Missing range-analysis edges: {RANGE_EDGES - set(range_edge_ids)}")
        oracle_link, resolved_objective, optimal_ranges, optimal_group_ranges = solve_oracle(
            snapshot,
            pte,
            opt1,
            opt2,
            opt3,
            gurobi_props,
            range_edge_ids,
            compute_edge_ranges,
        )

        for edge_id, (source, target) in enumerate(edge_names):
            capacity = float(capacities[edge_id])
            oracle_util = float(oracle_link[edge_id] / capacity)
            planned_util = float(planned_link[edge_id] / capacity)
            admitted_util = float(admitted_link[edge_id] / capacity)
            optimal_range = optimal_ranges.get((int(source), int(target)))
            optimal_min_util = (
                float(optimal_range[0] / capacity) if optimal_range is not None else None
            )
            optimal_max_util = (
                float(optimal_range[1] / capacity) if optimal_range is not None else None
            )
            edge_rows.append(
                {
                    "split": split_name,
                    "snapshot": snapshot_id,
                    "edge_id": edge_id,
                    "source": int(source),
                    "target": int(target),
                    "group": edge_group(int(source), int(target)),
                    "capacity": capacity,
                    "oracle_total_load": float(oracle_link[edge_id]),
                    "planned_total_load": float(planned_link[edge_id]),
                    "admitted_total_load": float(admitted_link[edge_id]),
                    "oracle_utilization": oracle_util,
                    "planned_utilization": planned_util,
                    "admitted_utilization": admitted_util,
                    "oracle_minus_admitted_utilization": oracle_util - admitted_util,
                    "oracle_minus_admitted_load": float(oracle_link[edge_id] - admitted_link[edge_id]),
                    "oracle_hot_hattrick_cold": int(oracle_util >= 0.8 and admitted_util <= 0.2),
                    "oracle_hot_policy_starved": int(oracle_util >= 0.8 and planned_util <= 0.2),
                    "optimal_min_utilization": optimal_min_util,
                    "optimal_max_utilization": optimal_max_util,
                    "required_underuse_gap": (
                        max(optimal_min_util - admitted_util, 0.0)
                        if optimal_min_util is not None else None
                    ),
                }
            )

        snapshot_row = {
                "split": split_name,
                "snapshot": snapshot_id,
                "high_norm_fulfill": class_admitted[0] / max(oracle_totals[0], 1e-12),
                "medium_norm_fulfill": class_admitted[1] / max(oracle_totals[1], 1e-12),
                "low_norm_fulfill": class_admitted[2] / max(oracle_totals[2], 1e-12),
                "high_shortfall": oracle_totals[0] - class_admitted[0],
                "medium_shortfall": oracle_totals[1] - class_admitted[1],
                "low_excess": class_admitted[2] - oracle_totals[2],
                "resolved_oracle_objective": resolved_objective,
                "stored_oracle_objective": opt3,
            }
        for group_name, members in (
            ("node10_corridor", NODE10_CORRIDOR),
            ("node6_incident", NODE6_INCIDENT),
        ):
            member_ids = [range_edge_ids[edge] for edge in members]
            planned_sum = float(sum(planned_link[i] / capacities[i] for i in member_ids))
            admitted_sum = float(sum(admitted_link[i] / capacities[i] for i in member_ids))
            minimum, maximum = optimal_group_ranges[group_name]
            snapshot_row[f"{group_name}_planned_utilization_sum"] = planned_sum
            snapshot_row[f"{group_name}_admitted_utilization_sum"] = admitted_sum
            snapshot_row[f"{group_name}_optimal_min_utilization_sum"] = minimum
            snapshot_row[f"{group_name}_optimal_max_utilization_sum"] = maximum
            snapshot_row[f"{group_name}_required_underuse_sum"] = max(minimum - admitted_sum, 0.0)
        snapshot_rows.append(snapshot_row)
        if (local_index + 1) % 10 == 0 or snapshot_id == end - 1:
            print(
                f"[{split_name}] {local_index + 1}/{end - start} snapshots complete",
                flush=True,
            )
    return edge_rows, snapshot_rows


def summarize_edges(edge_rows: list[dict]) -> list[dict]:
    for split_name in sorted({row["split"] for row in edge_rows}):
        snapshots = sorted({int(row["snapshot"]) for row in edge_rows if row["split"] == split_name})
        for snapshot in snapshots:
            selected = [
                row for row in edge_rows
                if row["split"] == split_name and int(row["snapshot"]) == snapshot
            ]
            selected.sort(
                key=lambda row: float(row["oracle_minus_admitted_utilization"]),
                reverse=True,
            )
            for rank, row in enumerate(selected, start=1):
                row["underuse_rank"] = rank

    summaries: list[dict] = []
    keys = sorted(
        {
            (row["split"], int(row["edge_id"]), int(row["source"]), int(row["target"]), row["group"])
            for row in edge_rows
        }
    )
    for split_name, edge_id, source, target, group in keys:
        rows = [
            row for row in edge_rows
            if row["split"] == split_name and int(row["edge_id"]) == edge_id
        ]
        gaps = np.asarray([float(row["oracle_minus_admitted_utilization"]) for row in rows])
        summaries.append(
            {
                "split": split_name,
                "edge_id": edge_id,
                "source": source,
                "target": target,
                "group": group,
                "capacity": float(rows[0]["capacity"]),
                "n": len(rows),
                "oracle_utilization_mean": float(np.mean([float(row["oracle_utilization"]) for row in rows])),
                "planned_utilization_mean": float(np.mean([float(row["planned_utilization"]) for row in rows])),
                "admitted_utilization_mean": float(np.mean([float(row["admitted_utilization"]) for row in rows])),
                "gap_mean": float(gaps.mean()),
                "gap_median": float(np.median(gaps)),
                "gap_p90": float(np.quantile(gaps, 0.9)),
                "gap_gt_0p2_fraction": float(np.mean(gaps >= 0.2)),
                "gap_gt_0p5_fraction": float(np.mean(gaps >= 0.5)),
                "oracle_hot_hattrick_cold_fraction": float(np.mean([int(row["oracle_hot_hattrick_cold"]) for row in rows])),
                "oracle_hot_policy_starved_fraction": float(np.mean([int(row["oracle_hot_policy_starved"]) for row in rows])),
                "top5_underuse_fraction": float(np.mean([int(row["underuse_rank"]) <= 5 for row in rows])),
                "top10_underuse_fraction": float(np.mean([int(row["underuse_rank"]) <= 10 for row in rows])),
            }
        )
    return summaries


def summarize_groups(edge_rows: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    split_names = sorted({row["split"] for row in edge_rows})
    for split_name in split_names + ["combined"]:
        split_rows = edge_rows if split_name == "combined" else [row for row in edge_rows if row["split"] == split_name]
        for group in ("node10_corridor", "node6_incident", "other_edges"):
            rows = [row for row in split_rows if row["group"] == group]
            gaps = np.asarray([float(row["oracle_minus_admitted_utilization"]) for row in rows])
            summaries.append(
                {
                    "split": split_name,
                    "group": group,
                    "distinct_edges": len({int(row["edge_id"]) for row in rows}),
                    "edge_snapshot_count": len(rows),
                    "oracle_utilization_mean": float(np.mean([float(row["oracle_utilization"]) for row in rows])),
                    "planned_utilization_mean": float(np.mean([float(row["planned_utilization"]) for row in rows])),
                    "admitted_utilization_mean": float(np.mean([float(row["admitted_utilization"]) for row in rows])),
                    "gap_mean": float(gaps.mean()),
                    "gap_median": float(np.median(gaps)),
                    "gap_gt_0p2_fraction": float(np.mean(gaps >= 0.2)),
                    "gap_gt_0p5_fraction": float(np.mean(gaps >= 0.5)),
                    "oracle_hot_hattrick_cold_fraction": float(np.mean([int(row["oracle_hot_hattrick_cold"]) for row in rows])),
                    "oracle_hot_policy_starved_fraction": float(np.mean([int(row["oracle_hot_policy_starved"]) for row in rows])),
                    "top5_underuse_fraction": float(np.mean([int(row["underuse_rank"]) <= 5 for row in rows])),
                    "top10_underuse_fraction": float(np.mean([int(row["underuse_rank"]) <= 10 for row in rows])),
                }
            )
    return summaries


def attach_snapshot_group_metrics(edge_rows: list[dict], snapshot_rows: list[dict]) -> dict:
    lookup = {(row["split"], int(row["snapshot"])): row for row in snapshot_rows}
    for key, snapshot in lookup.items():
        split_name, snapshot_id = key
        selected = [
            row for row in edge_rows
            if row["split"] == split_name and int(row["snapshot"]) == snapshot_id
        ]
        for group in ("node10_corridor", "node6_incident", "other_edges"):
            rows = [row for row in selected if row["group"] == group]
            snapshot[f"{group}_mean_gap"] = float(
                np.mean([float(row["oracle_minus_admitted_utilization"]) for row in rows])
            )
            snapshot[f"{group}_positive_wasted_load"] = float(
                sum(max(float(row["oracle_minus_admitted_load"]), 0.0) for row in rows)
            )

    correlations: dict[str, dict] = {}
    for split_name in sorted({row["split"] for row in snapshot_rows}) + ["combined"]:
        rows = snapshot_rows if split_name == "combined" else [row for row in snapshot_rows if row["split"] == split_name]
        correlations[split_name] = {}
        for group in ("node10_corridor", "node6_incident", "other_edges"):
            correlations[split_name][group] = {
                "corr_medium_shortfall_vs_mean_gap": safe_corr(
                    [float(row["medium_shortfall"]) for row in rows],
                    [float(row[f"{group}_mean_gap"]) for row in rows],
                ),
                "corr_medium_shortfall_vs_positive_wasted_load": safe_corr(
                    [float(row["medium_shortfall"]) for row in rows],
                    [float(row[f"{group}_positive_wasted_load"]) for row in rows],
                ),
            }
    return correlations


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-snapshot edge-pattern analysis")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--skip-edge-ranges",
        action="store_true",
        help="Only solve joint group ranges; useful after the per-edge range audit is complete.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("validation", "confirmation"),
        default=("validation", "confirmation"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    runner.set_seed(args.seed)
    props = runner.build_props(4, device)
    props.batch_size = 1
    checkpoint = torch.load(
        args.run_dir.resolve() / "best_model.pt", map_location=device, weights_only=False
    )
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    gurobi_props = build_gurobi_props()

    split_specs = {
        "validation": runner.LEVELS[4]["validation"],
        "confirmation": runner.LEVELS[4]["evaluation"],
    }
    edge_rows: list[dict] = []
    snapshot_rows: list[dict] = []
    started = time.perf_counter()
    for split_name in args.splits:
        start, end = split_specs[split_name]
        split_edges, split_snapshots = replay_split(
            split_name,
            start,
            end,
            model,
            props,
            args.seed,
            gurobi_props,
            not args.skip_edge_ranges,
        )
        edge_rows.extend(split_edges)
        snapshot_rows.extend(split_snapshots)

    edge_summary = summarize_edges(edge_rows)
    group_summary = summarize_groups(edge_rows)
    correlations = attach_snapshot_group_metrics(edge_rows, snapshot_rows)
    write_csv(output_dir / "edge_snapshot_metrics.csv", edge_rows)
    write_csv(output_dir / "edge_summary.csv", edge_summary)
    write_csv(output_dir / "group_summary.csv", group_summary)
    write_csv(output_dir / "snapshot_summary.csv", snapshot_rows)
    manifest = {
        "model": "original Hattrick with restored Fh and cumulative Fhm",
        "seed": args.seed,
        "checkpoint": str(args.run_dir.resolve() / "best_model.pt"),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "splits": {name: list(split_specs[name]) for name in args.splits},
        "edge_flow_definition": "cumulative High + Medium + Low total admitted flow",
        "oracle": "deterministic re-solve of the same three-stage lexicographic max-flow LP; link allocation can vary under degeneracy",
        "oracle_range_test": "for all node-10-corridor and node-6-incident edges, minimize and maximize link load while holding all three lexicographic flow optima",
        "oracle_group_range_test": "jointly minimize and maximize summed utilization for the node-10 corridor and all node-6 incident edges while holding all three lexicographic flow optima",
        "correlations": correlations,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "manifest.json", manifest)
    gp.disposeDefaultEnv()
    print(json.dumps({"output_dir": str(output_dir), "runtime_seconds": manifest["runtime_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("GRB_LICENSE_FILE", r"D:\kuroresearch\gurobi_license\gurobi.lic")
    main()
