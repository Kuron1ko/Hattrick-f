from __future__ import annotations

import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import block_diag, coo_matrix, csr_matrix, hstack, vstack


def solve_medium_with_edge_reservation(
    path_to_edge: csr_matrix,
    predicted_demand_per_path: np.ndarray,
    enabled_mask: np.ndarray,
    capacity_after_high: np.ndarray,
    baseline_low_link_load: np.ndarray,
    paths_per_pair: int,
    reserve_factor: float,
) -> tuple[np.ndarray, dict]:
    """Maximize Medium flow after reserving baseline Low load on every edge.

    The decision variable is an admitted fraction for every enabled path.  A
    pair may admit at most one unit of its duplicated per-path demand.  This is
    the same admission-plus-routing interpretation used by the existing
    residual-LP evaluator.
    """
    started = time.perf_counter()
    demand = np.asarray(predicted_demand_per_path, dtype=np.float64).reshape(-1)
    enabled = np.asarray(enabled_mask, dtype=bool).reshape(-1)
    after_high = np.asarray(capacity_after_high, dtype=np.float64).reshape(-1)
    low_load = np.asarray(baseline_low_link_load, dtype=np.float64).reshape(-1)
    if demand.size % int(paths_per_pair):
        raise ValueError("path count must be divisible by paths_per_pair")
    if path_to_edge.shape[0] != demand.size:
        raise ValueError("path_to_edge and demand size disagree")
    if path_to_edge.shape[1] != after_high.size or low_load.size != after_high.size:
        raise ValueError("edge vector size mismatch")
    if reserve_factor < 0.0:
        raise ValueError("reserve_factor must be non-negative")

    active = np.flatnonzero(enabled & (demand > 0.0))
    if active.size == 0:
        raise RuntimeError("no enabled positive-demand Medium paths")
    residual = np.maximum(after_high - float(reserve_factor) * low_load, 0.0)
    active_demand = demand[active]
    edge_constraints = path_to_edge[active, :].T.multiply(active_demand)
    pair_ids = active // int(paths_per_pair)
    num_pairs = demand.size // int(paths_per_pair)
    pair_constraints = coo_matrix(
        (np.ones(active.size, dtype=np.float64), (pair_ids, np.arange(active.size))),
        shape=(num_pairs, active.size),
    ).tocsr()
    constraints = vstack((edge_constraints, pair_constraints), format="csr")
    bounds = np.concatenate((residual, np.ones(num_pairs, dtype=np.float64)))
    result = linprog(
        -active_demand,
        A_ub=constraints,
        b_ub=bounds,
        bounds=(0.0, 1.0),
        method="highs",
        options={"presolve": True},
    )
    elapsed = time.perf_counter() - started
    if not result.success or result.x is None or not np.isfinite(result.x).all():
        raise RuntimeError(
            f"protected Medium LP failed: status={result.status} message={result.message}"
        )

    policy64 = np.zeros(demand.size, dtype=np.float64)
    policy64[active] = np.clip(result.x, 0.0, 1.0)
    link_load = np.asarray(path_to_edge.T @ (policy64 * demand)).reshape(-1)
    pair_mass = policy64.reshape(num_pairs, int(paths_per_pair)).sum(axis=1)
    max_edge_excess = float(np.max(link_load - residual))
    max_pair_mass = float(pair_mass.max())
    if max_edge_excess > 1e-6 or max_pair_mass > 1.0 + 1e-9:
        raise RuntimeError(
            f"protected Medium LP invariant failure: edge={max_edge_excess}, pair={max_pair_mass}"
        )
    diagnostics = {
        "status": int(result.status),
        "iterations": int(result.nit),
        "runtime_seconds": float(elapsed),
        "predicted_medium_admitted": float(-result.fun),
        "reserved_low_link_load_sum": float(low_load.sum()),
        "reserve_factor": float(reserve_factor),
        "min_residual_capacity": float(residual.min()),
        "max_edge_excess": max_edge_excess,
        "max_pair_mass": max_pair_mass,
    }
    return policy64.astype(np.float32), diagnostics


def solve_medium_low_pareto(
    path_to_edge: csr_matrix,
    predicted_medium_per_path: np.ndarray,
    predicted_low_per_path: np.ndarray,
    medium_enabled_mask: np.ndarray,
    low_enabled_mask: np.ndarray,
    capacity_after_high: np.ndarray,
    baseline_low_admitted: float,
    paths_per_pair: int,
    low_floor_factor: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Lexicographically maximize Medium while protecting aggregate Low.

    First determine the maximum Low flow available after frozen High.  Then
    maximize Medium subject to a Low floor, and finally maximize Low on that
    Medium-optimal face.  The only tunable safety value is a multiplicative
    margin on the baseline predicted Low admission.
    """
    started = time.perf_counter()
    medium_demand = np.asarray(predicted_medium_per_path, dtype=np.float64).reshape(-1)
    low_demand = np.asarray(predicted_low_per_path, dtype=np.float64).reshape(-1)
    medium_enabled = np.asarray(medium_enabled_mask, dtype=bool).reshape(-1)
    low_enabled = np.asarray(low_enabled_mask, dtype=bool).reshape(-1)
    residual = np.maximum(
        np.asarray(capacity_after_high, dtype=np.float64).reshape(-1), 0.0
    )
    k = int(paths_per_pair)
    if medium_demand.size != low_demand.size or medium_demand.size % k:
        raise ValueError("Medium/Low path vectors must have equal K-divisible size")
    if path_to_edge.shape != (medium_demand.size, residual.size):
        raise ValueError("path_to_edge shape mismatch")
    if low_floor_factor < 0.0:
        raise ValueError("low_floor_factor must be non-negative")

    medium_active = np.flatnonzero(medium_enabled & (medium_demand > 0.0))
    low_active = np.flatnonzero(low_enabled & (low_demand > 0.0))
    if medium_active.size == 0 or low_active.size == 0:
        raise RuntimeError("joint repair requires enabled Medium and Low paths")
    dm = medium_demand[medium_active]
    dl = low_demand[low_active]
    num_medium = medium_active.size
    num_low = low_active.size
    num_pairs = medium_demand.size // k

    edge_medium = path_to_edge[medium_active, :].T.multiply(dm)
    edge_low = path_to_edge[low_active, :].T.multiply(dl)
    edge_constraints = hstack((edge_medium, edge_low), format="csr")
    pair_medium = coo_matrix(
        (
            np.ones(num_medium, dtype=np.float64),
            (medium_active // k, np.arange(num_medium)),
        ),
        shape=(num_pairs, num_medium),
    ).tocsr()
    pair_low = coo_matrix(
        (
            np.ones(num_low, dtype=np.float64),
            (low_active // k, np.arange(num_low)),
        ),
        shape=(num_pairs, num_low),
    ).tocsr()
    pair_constraints = block_diag((pair_medium, pair_low), format="csr")
    base_constraints = vstack((edge_constraints, pair_constraints), format="csr")
    base_bounds = np.concatenate((residual, np.ones(2 * num_pairs, dtype=np.float64)))
    medium_objective = np.concatenate((-dm, np.zeros(num_low, dtype=np.float64)))
    low_objective = np.concatenate((np.zeros(num_medium, dtype=np.float64), -dl))
    zero_medium_low_row = np.concatenate((np.zeros(num_medium, dtype=np.float64), -dl))

    low_max_result = linprog(
        low_objective,
        A_ub=base_constraints,
        b_ub=base_bounds,
        bounds=(0.0, 1.0),
        method="highs",
        options={"presolve": True},
    )
    if not low_max_result.success or low_max_result.x is None:
        raise RuntimeError(f"Low ceiling LP failed: {low_max_result.message}")
    low_ceiling = float(-low_max_result.fun)
    requested_floor = float(low_floor_factor) * float(baseline_low_admitted)
    # Keep the target inside the exactly feasible polytope despite float32
    # simulator totals being used to define the baseline.
    low_floor = max(min(requested_floor, low_ceiling) - 1e-6, 0.0)
    first_constraints = vstack((base_constraints, csr_matrix(zero_medium_low_row)), format="csr")
    first_bounds = np.concatenate((base_bounds, np.asarray([-low_floor])))
    medium_result = linprog(
        medium_objective,
        A_ub=first_constraints,
        b_ub=first_bounds,
        bounds=(0.0, 1.0),
        method="highs",
        options={"presolve": True},
    )
    if not medium_result.success or medium_result.x is None:
        raise RuntimeError(f"Medium-with-Low-floor LP failed: {medium_result.message}")
    medium_optimum = float(-medium_result.fun)
    medium_floor_row = np.concatenate((-dm, np.zeros(num_low, dtype=np.float64)))
    final_constraints = vstack(
        (first_constraints, csr_matrix(medium_floor_row)), format="csr"
    )
    final_bounds = np.concatenate(
        (first_bounds, np.asarray([-max(medium_optimum - 1e-6, 0.0)]))
    )
    final_result = linprog(
        low_objective,
        A_ub=final_constraints,
        b_ub=final_bounds,
        bounds=(0.0, 1.0),
        method="highs",
        options={"presolve": True},
    )
    if not final_result.success or final_result.x is None or not np.isfinite(final_result.x).all():
        raise RuntimeError(f"Low tie-break LP failed: {final_result.message}")

    medium_policy64 = np.zeros(medium_demand.size, dtype=np.float64)
    low_policy64 = np.zeros(low_demand.size, dtype=np.float64)
    medium_policy64[medium_active] = np.clip(final_result.x[:num_medium], 0.0, 1.0)
    low_policy64[low_active] = np.clip(final_result.x[num_medium:], 0.0, 1.0)
    medium_total = float(np.dot(medium_policy64, medium_demand))
    low_total = float(np.dot(low_policy64, low_demand))
    total_link_load = np.asarray(
        path_to_edge.T
        @ (medium_policy64 * medium_demand + low_policy64 * low_demand)
    ).reshape(-1)
    medium_pair_mass = medium_policy64.reshape(num_pairs, k).sum(axis=1)
    low_pair_mass = low_policy64.reshape(num_pairs, k).sum(axis=1)
    max_edge_excess = float(np.max(total_link_load - residual))
    max_pair_mass = float(max(medium_pair_mass.max(), low_pair_mass.max()))
    if max_edge_excess > 1e-6 or max_pair_mass > 1.0 + 1e-9:
        raise RuntimeError(
            f"joint Pareto LP invariant failure: edge={max_edge_excess}, pair={max_pair_mass}"
        )
    diagnostics = {
        "status": int(final_result.status),
        "iterations": int(
            low_max_result.nit + medium_result.nit + final_result.nit
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "baseline_predicted_low_floor_source": float(baseline_low_admitted),
        "requested_low_floor": requested_floor,
        "effective_low_floor": low_floor,
        "predicted_low_ceiling": low_ceiling,
        "predicted_medium_admitted": medium_total,
        "predicted_low_admitted": low_total,
        "low_floor_factor": float(low_floor_factor),
        "max_edge_excess": max_edge_excess,
        "max_pair_mass": max_pair_mass,
    }
    return (
        medium_policy64.astype(np.float32),
        low_policy64.astype(np.float32),
        diagnostics,
    )


def solve_medium_low_per_od_pareto(
    path_to_edge: csr_matrix,
    predicted_medium_per_path: np.ndarray,
    predicted_low_per_path: np.ndarray,
    baseline_low_admitted_per_path: np.ndarray,
    medium_enabled_mask: np.ndarray,
    low_enabled_mask: np.ndarray,
    capacity_after_high: np.ndarray,
    paths_per_pair: int,
    low_floor_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Maximize Medium while preserving every Low OD's predicted admission."""
    started = time.perf_counter()
    medium_demand = np.asarray(predicted_medium_per_path, dtype=np.float64).reshape(-1)
    low_demand = np.asarray(predicted_low_per_path, dtype=np.float64).reshape(-1)
    baseline_low = np.asarray(baseline_low_admitted_per_path, dtype=np.float64).reshape(-1)
    medium_enabled = np.asarray(medium_enabled_mask, dtype=bool).reshape(-1)
    low_enabled = np.asarray(low_enabled_mask, dtype=bool).reshape(-1)
    residual = np.maximum(
        np.asarray(capacity_after_high, dtype=np.float64).reshape(-1), 0.0
    )
    k = int(paths_per_pair)
    if not (
        medium_demand.size == low_demand.size == baseline_low.size
        and medium_demand.size % k == 0
    ):
        raise ValueError("path vectors must have equal K-divisible size")
    if path_to_edge.shape != (medium_demand.size, residual.size):
        raise ValueError("path_to_edge shape mismatch")
    if low_floor_factor < 0.0:
        raise ValueError("low_floor_factor must be non-negative")

    medium_active = np.flatnonzero(medium_enabled & (medium_demand > 0.0))
    low_active = np.flatnonzero(low_enabled & (low_demand > 0.0))
    if medium_active.size == 0 or low_active.size == 0:
        raise RuntimeError("per-OD repair requires enabled Medium and Low paths")
    dm = medium_demand[medium_active]
    dl = low_demand[low_active]
    num_medium = medium_active.size
    num_low = low_active.size
    num_pairs = medium_demand.size // k
    pair_ids_low = low_active // k

    edge_constraints = hstack(
        (
            path_to_edge[medium_active, :].T.multiply(dm),
            path_to_edge[low_active, :].T.multiply(dl),
        ),
        format="csr",
    )
    pair_medium = coo_matrix(
        (np.ones(num_medium), (medium_active // k, np.arange(num_medium))),
        shape=(num_pairs, num_medium),
    ).tocsr()
    pair_low = coo_matrix(
        (np.ones(num_low), (pair_ids_low, np.arange(num_low))),
        shape=(num_pairs, num_low),
    ).tocsr()
    pair_constraints = block_diag((pair_medium, pair_low), format="csr")
    low_floor_block = coo_matrix(
        (-dl, (pair_ids_low, np.arange(num_low))),
        shape=(num_pairs, num_low),
    ).tocsr()
    low_floor_constraints = hstack(
        (csr_matrix((num_pairs, num_medium)), low_floor_block), format="csr"
    )
    baseline_pair = baseline_low.reshape(num_pairs, k).sum(axis=1)
    pair_demand = low_demand.reshape(num_pairs, k)[:, 0]
    requested_pair_floor = np.minimum(
        float(low_floor_factor) * baseline_pair, pair_demand
    )
    effective_pair_floor = np.maximum(requested_pair_floor - 1e-6, 0.0)
    constraints = vstack(
        (edge_constraints, pair_constraints, low_floor_constraints), format="csr"
    )
    bounds = np.concatenate(
        (residual, np.ones(2 * num_pairs, dtype=np.float64), -effective_pair_floor)
    )
    medium_objective = np.concatenate((-dm, np.zeros(num_low, dtype=np.float64)))
    low_objective = np.concatenate((np.zeros(num_medium, dtype=np.float64), -dl))
    medium_result = linprog(
        medium_objective,
        A_ub=constraints,
        b_ub=bounds,
        bounds=(0.0, 1.0),
        method="highs",
        options={"presolve": True},
    )
    if not medium_result.success or medium_result.x is None:
        raise RuntimeError(f"per-OD Medium LP failed: {medium_result.message}")
    medium_optimum = float(-medium_result.fun)
    medium_floor_row = np.concatenate((-dm, np.zeros(num_low, dtype=np.float64)))
    final_constraints = vstack(
        (constraints, csr_matrix(medium_floor_row)), format="csr"
    )
    final_bounds = np.concatenate(
        (bounds, np.asarray([-max(medium_optimum - 1e-6, 0.0)]))
    )
    final_result = linprog(
        low_objective,
        A_ub=final_constraints,
        b_ub=final_bounds,
        bounds=(0.0, 1.0),
        method="highs",
        options={"presolve": True},
    )
    if not final_result.success or final_result.x is None or not np.isfinite(final_result.x).all():
        raise RuntimeError(f"per-OD Low tie-break LP failed: {final_result.message}")

    medium_policy64 = np.zeros(medium_demand.size, dtype=np.float64)
    low_policy64 = np.zeros(low_demand.size, dtype=np.float64)
    medium_policy64[medium_active] = np.clip(final_result.x[:num_medium], 0.0, 1.0)
    low_policy64[low_active] = np.clip(final_result.x[num_medium:], 0.0, 1.0)
    low_pair_total = (low_policy64 * low_demand).reshape(num_pairs, k).sum(axis=1)
    total_load = np.asarray(
        path_to_edge.T
        @ (medium_policy64 * medium_demand + low_policy64 * low_demand)
    ).reshape(-1)
    medium_pair_mass = medium_policy64.reshape(num_pairs, k).sum(axis=1)
    low_pair_mass = low_policy64.reshape(num_pairs, k).sum(axis=1)
    max_edge_excess = float(np.max(total_load - residual))
    max_pair_mass = float(max(medium_pair_mass.max(), low_pair_mass.max()))
    min_low_pair_slack = float(np.min(low_pair_total - effective_pair_floor))
    if max_edge_excess > 1e-6 or max_pair_mass > 1.0 + 1e-9 or min_low_pair_slack < -1e-6:
        raise RuntimeError(
            "per-OD Pareto invariant failure: "
            f"edge={max_edge_excess}, pair_mass={max_pair_mass}, low_slack={min_low_pair_slack}"
        )
    diagnostics = {
        "status": int(final_result.status),
        "iterations": int(medium_result.nit + final_result.nit),
        "runtime_seconds": float(time.perf_counter() - started),
        "low_floor_factor": float(low_floor_factor),
        "predicted_medium_admitted": float(np.dot(medium_policy64, medium_demand)),
        "predicted_low_admitted": float(np.dot(low_policy64, low_demand)),
        "low_pairs_with_positive_floor": int(np.sum(effective_pair_floor > 0.0)),
        "minimum_low_pair_floor_slack": min_low_pair_slack,
        "max_edge_excess": max_edge_excess,
        "max_pair_mass": max_pair_mass,
    }
    return (
        medium_policy64.astype(np.float32),
        low_policy64.astype(np.float32),
        diagnostics,
    )
