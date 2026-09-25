from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("GUROBI_HOME", r"D:\kuroresearch\gurobi1302\win64")
os.environ.setdefault(
    "GRB_LICENSE_FILE", r"D:\kuroresearch\gurobi_license\gurobi.lic"
)

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from scipy.sparse import csr_matrix, eye, kron
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EDGE_RUNNER_DIR = HERE.parent / "shared2x_learned_edge_cost"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EDGE_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_RUNNER_DIR))

import run_experiment as edge_runner  # noqa: E402
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def scipy_pte(tensor: torch.Tensor) -> csr_matrix:
    tensor = tensor.coalesce().cpu()
    indices = tensor.indices().numpy()
    values = tensor.values().to(dtype=torch.float64).numpy()
    return csr_matrix(
        (values, (indices[0], indices[1])), shape=tuple(tensor.shape)
    )


def solve_high_medium_shadow_prices(
    pte: csr_matrix,
    capacities: np.ndarray,
    tm_high: np.ndarray,
    tm_medium: np.ndarray,
    num_pairs: int,
    num_paths: int,
    high_tolerance: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return Medium shadow prices with High held at its max-flow optimum.

    Both class path allocations remain variables in phase two.  Consequently High
    may move among any max-flow-equivalent paths while Medium is maximized.
    """
    path_count, edge_count = pte.shape
    if path_count != num_pairs * num_paths:
        raise ValueError("Unexpected path matrix shape")
    high_demand = np.asarray(tm_high, dtype=np.float64).reshape(-1)
    medium_demand = np.asarray(tm_medium, dtype=np.float64).reshape(-1)
    capacities = np.asarray(capacities, dtype=np.float64).reshape(-1)
    if high_demand.shape != (path_count,) or medium_demand.shape != (path_count,):
        raise ValueError("Traffic matrix/path count mismatch")
    if capacities.shape != (edge_count,):
        raise ValueError("Capacity/edge count mismatch")

    # The project represents one split-ratio variable for each OD/path.  These
    # sparse matrices reproduce its demand and capacity constraints exactly.
    od_to_path = kron(
        eye(num_pairs, format="csr"),
        np.ones((1, num_paths), dtype=np.float64),
        format="csr",
    )
    edge_by_path = pte.transpose().tocsr()
    high_edge_load = edge_by_path.multiply(high_demand.reshape(1, -1))
    medium_edge_load = edge_by_path.multiply(medium_demand.reshape(1, -1))

    model = gp.Model("high_medium_shadow_price")
    model.Params.OutputFlag = 0
    model.Params.NumericFocus = 2
    model.Params.FeasibilityTol = 1e-7
    model.Params.OptimalityTol = 1e-7
    high = model.addMVar(path_count, lb=0.0, ub=1.0, name="high")
    medium = model.addMVar(path_count, lb=0.0, ub=1.0, name="medium")
    model.addConstr(od_to_path @ high <= np.ones(num_pairs), name="high_demand")
    model.addConstr(
        od_to_path @ medium <= np.ones(num_pairs), name="medium_demand"
    )
    capacity_constraints = model.addConstr(
        high_edge_load @ high + medium_edge_load @ medium <= capacities,
        name="capacity",
    )

    high_flow = high_demand @ high
    medium_flow = medium_demand @ medium
    model.setObjective(high_flow, GRB.MAXIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"High max-flow Gurobi status {model.Status}")
    high_optimum = float(model.ObjVal)

    model.addConstr(
        high_flow >= high_optimum * (1.0 - high_tolerance),
        name="hold_high_optimum",
    )
    model.setObjective(medium_flow, GRB.MAXIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Medium max-flow Gurobi status {model.Status}")

    prices = np.maximum(
        np.asarray(capacity_constraints.Pi, dtype=np.float64).reshape(-1), 0.0
    )
    high_achieved = float(high_flow.getValue())
    medium_optimum = float(medium_flow.getValue())
    cumulative_load = np.asarray(
        high_edge_load @ high.X + medium_edge_load @ medium.X,
        dtype=np.float64,
    ).reshape(-1)
    max_capacity_violation = float(np.max(cumulative_load - capacities))
    model.dispose()
    return prices, {
        "high_optimum": high_optimum,
        "high_achieved": high_achieved,
        "medium_optimum": medium_optimum,
        "high_shortfall": high_optimum - high_achieved,
        "positive_price_edges": int(np.count_nonzero(prices > 1e-10)),
        "tight_edges": int(np.count_nonzero(capacities - cumulative_load < 1e-6)),
        "max_capacity_violation": max_capacity_violation,
    }


def fit_shadow_price_weights(
    start: int,
    end: int,
    quantile: float,
    delta: float,
    high_tolerance: float,
    output_path: Path,
) -> dict:
    device = torch.device("cpu")
    props = edge_runner.make_policy_props(start, end, device)
    dataset = DM_Dataset_within_Cluster(props, 0, start, end)
    if int(dataset.max_source_index_read) != end - 1:
        raise RuntimeError("Split-safe reader audit failed")
    pte = scipy_pte(dataset.pte)
    edge_count = pte.shape[1]

    raw_prices = []
    normalized_prices = []
    solve_rows = []
    started = time.perf_counter()
    for local_index, snapshot in enumerate(dataset.list_snapshots):
        prices, solve_stats = solve_high_medium_shadow_prices(
            pte=pte,
            capacities=np.asarray(snapshot.capacities, dtype=np.float64),
            tm_high=np.asarray(snapshot.tm1, dtype=np.float64),
            tm_medium=np.asarray(snapshot.tm2, dtype=np.float64),
            num_pairs=int(snapshot.num_demands),
            num_paths=edge_runner.K,
            high_tolerance=high_tolerance,
        )
        scale = float(prices.max())
        normalized = prices / scale if scale > 1e-12 else np.zeros_like(prices)
        raw_prices.append(prices)
        normalized_prices.append(normalized)

        expected_high = float(dataset.list_optimal_values_1_mf[local_index])
        expected_cumulative = float(dataset.list_optimal_values_2_mf[local_index])
        solve_stats.update(
            {
                "snapshot": start + local_index,
                "expected_high": expected_high,
                "expected_cumulative_high_medium": expected_cumulative,
                "high_reference_error": solve_stats["high_optimum"] - expected_high,
                "cumulative_reference_error": (
                    solve_stats["high_achieved"]
                    + solve_stats["medium_optimum"]
                    - expected_cumulative
                ),
                "raw_price_max": scale,
            }
        )
        solve_rows.append(solve_stats)
        if (local_index + 1) % 16 == 0 or local_index + 1 == len(dataset):
            print(
                f"[shadow] solved {local_index + 1}/{len(dataset)} snapshots",
                flush=True,
            )

    raw = np.stack(raw_prices)
    normalized = np.stack(normalized_prices)
    score = np.quantile(normalized, quantile, axis=0)
    if float(score.max()) > 0.0:
        score = score / float(score.max())
    weights = np.ones((3, edge_count), dtype=np.float32)
    weights[0] += np.float32(delta) * score.astype(np.float32)

    edge_names = [
        [int(source), int(target)]
        for source, target in dataset.list_snapshots[0].graph.edges()
    ]
    ranked_edges = []
    for edge_id in np.argsort(-score):
        ranked_edges.append(
            {
                "edge_id": int(edge_id),
                "edge": edge_names[int(edge_id)],
                "score": float(score[edge_id]),
                "weight": float(weights[0, edge_id]),
                "positive_frequency": float(
                    np.mean(normalized[:, edge_id] > 1e-10)
                ),
                "mean_normalized_price": float(normalized[:, edge_id].mean()),
            }
        )

    max_high_reference_error = float(
        max(abs(row["high_reference_error"]) for row in solve_rows)
    )
    max_cumulative_reference_error = float(
        max(abs(row["cumulative_reference_error"]) for row in solve_rows)
    )
    max_capacity_violation = float(
        max(row["max_capacity_violation"] for row in solve_rows)
    )
    payload = {
        "method": "lexicographic Medium capacity shadow-price High tie-break",
        "strict_esm_inference": True,
        "supervision": "actual train traffic only",
        "train_range": [start, end],
        "quantile": quantile,
        "delta": delta,
        "high_tolerance": high_tolerance,
        "weights": torch.from_numpy(weights),
        "high_shadow_score": torch.from_numpy(score.astype(np.float32)),
        "raw_medium_capacity_prices": torch.from_numpy(raw.astype(np.float32)),
        "normalized_medium_capacity_prices": torch.from_numpy(
            normalized.astype(np.float32)
        ),
        "edge_index": dataset.edge_index.detach().cpu(),
        "edge_names": edge_names,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    stats = {
        "method": payload["method"],
        "strict_esm_inference": True,
        "train_range": [start, end],
        "n_samples": len(dataset),
        "n_edges": edge_count,
        "quantile": quantile,
        "delta": delta,
        "high_tolerance": high_tolerance,
        "elapsed_seconds": time.perf_counter() - started,
        "weight_min": weights.min(axis=1).tolist(),
        "weight_mean": weights.mean(axis=1).tolist(),
        "weight_max": weights.max(axis=1).tolist(),
        "non_unit_high_edges": int(np.count_nonzero(weights[0] > 1.0)),
        "snapshots_with_positive_prices": int(
            np.count_nonzero(normalized.max(axis=1) > 0.0)
        ),
        "mean_positive_price_edges_per_snapshot": float(
            np.mean(np.count_nonzero(normalized > 1e-10, axis=1))
        ),
        "max_high_reference_error": max_high_reference_error,
        "max_cumulative_reference_error": max_cumulative_reference_error,
        "max_capacity_violation": max_capacity_violation,
        "top_edges": ranked_edges[:20],
        "solve_rows": solve_rows,
        "path": str(output_path),
    }
    write_json(output_path.parent / "shadow_price_stats.json", stats)
    return stats


def paired_bootstrap(
    control: np.ndarray, candidate: np.ndarray, samples: int = 20000
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(490)
    differences = candidate - control
    sample_count = differences.shape[0]
    draws = rng.integers(0, sample_count, size=(samples, sample_count))
    bootstrap_means = differences[draws].mean(axis=1)
    result = {}
    for class_index, class_name in enumerate(edge_runner.CLASSES):
        result[class_name] = {
            "mean_delta": float(differences[:, class_index].mean()),
            "ci95_low": float(np.percentile(bootstrap_means[:, class_index], 2.5)),
            "ci95_high": float(np.percentile(bootstrap_means[:, class_index], 97.5)),
            "positive_probability": float(
                np.mean(bootstrap_means[:, class_index] > 0.0)
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=edge_runner.PHASES, default="small")
    parser.add_argument("--quantile", type=float, default=0.75)
    parser.add_argument("--delta", type=float, default=0.003)
    parser.add_argument("--high-tolerance", type=float, default=1e-5)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    if not 0.5 <= args.quantile < 1.0:
        raise SystemExit("--quantile must be in [0.5, 1.0)")
    if not 0.0 <= args.delta <= 0.05:
        raise SystemExit("--delta must be in [0, 0.05]")
    if not 0.0 < args.high_tolerance <= 1e-3:
        raise SystemExit("--high-tolerance must be in (0, 1e-3]")

    cfg = edge_runner.PHASES[args.phase]
    suffix = (
        f"{args.phase}_q{args.quantile:g}_delta{args.delta:g}"
        .replace(".", "p")
    )
    output_dir = HERE / "artifacts" / suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_path = output_dir / "shadow_price_weights.pt"
    stats = fit_shadow_price_weights(
        *cfg["train"],
        quantile=args.quantile,
        delta=args.delta,
        high_tolerance=args.high_tolerance,
        output_path=weight_path,
    )
    print(json.dumps({key: value for key, value in stats.items() if key != "solve_rows"}, indent=2))
    if args.extract_only:
        return

    control_label = f"shadow_control_{suffix}"
    candidate_label = f"shadow_candidate_{suffix}"
    all_values = {}
    models = {}
    training_seconds = {}
    for label, weights in (
        (control_label, None),
        (candidate_label, weight_path),
    ):
        model_path, elapsed = edge_runner.train_model(
            args.phase,
            label,
            weights,
            "weighted_mlu_actual",
            4.0,
            0.0,
            False,
            0.0,
            0.0,
        )
        values = edge_runner.evaluate(args.phase, label, model_path, False)
        all_values[label] = values
        models[label] = str(model_path)
        training_seconds[label] = elapsed
        np.savetxt(
            output_dir / f"{label}_norm_fulfill.csv",
            values,
            delimiter=",",
            header=",".join(edge_runner.CLASSES),
            comments="",
        )

    control_summary = edge_runner.summarize(all_values[control_label])
    candidate_summary = edge_runner.summarize(all_values[candidate_label])
    delta = {
        class_name: {
            metric: candidate_summary[class_name][metric]
            - control_summary[class_name][metric]
            for metric in ("mean", "p1", "p10", "median")
        }
        for class_name in edge_runner.CLASSES
    }
    bootstrap = paired_bootstrap(
        all_values[control_label], all_values[candidate_label]
    )
    gates = {
        "high_mean_at_least_0.995": candidate_summary["High"]["mean"] >= 0.995,
        "low_mean_drop_at_most_0.003": delta["Low"]["mean"] >= -0.003,
        "low_p10_drop_at_most_0.005": delta["Low"]["p10"] >= -0.005,
        "medium_mean_improves": delta["Medium"]["mean"] > 0.0,
        "medium_bootstrap_ci_excludes_zero": bootstrap["Medium"]["ci95_low"] > 0.0,
    }
    report = {
        "method": "lexicographic Medium shadow-price tie-break inside native actual MLU",
        "formula": (
            "phase 1 max High; phase 2 max Medium subject to High >= (1-tol) High*; "
            "score_e=Q_q(normalized phase-2 capacity dual_e); "
            "High training MLU=max_e((1+delta*score_e)*real_util_e)"
        ),
        "strict_esm_inference": True,
        "inference_change": None,
        "phase": args.phase,
        "protocol": cfg,
        "same_initial_model": str(edge_runner.BASE_MODEL),
        "weight_stats": {key: value for key, value in stats.items() if key != "solve_rows"},
        "models": models,
        "training_seconds": training_seconds,
        "summaries": {
            control_label: control_summary,
            candidate_label: candidate_summary,
        },
        "candidate_minus_control": delta,
        "paired_bootstrap": bootstrap,
        "expansion_gates": gates,
        "expand": all(gates.values()),
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
