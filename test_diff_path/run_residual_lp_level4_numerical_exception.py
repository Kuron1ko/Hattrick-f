from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import run_hattrick_residual_lp as residual
import run_residual_lp_level4_frozen as frozen


ADDENDUM = residual.THIS_DIR / "level4_numerical_exception_addendum.json"
PAIR_MASS_LIMIT = float(1.0 + np.finfo(np.float32).eps)


def solve_residual_class_one_ulp(
    pte,
    predicted_tm_per_path: np.ndarray,
    enabled_mask: np.ndarray,
    residual_capacity: np.ndarray,
    k: int,
):
    """Frozen solver logic with only the pair-mass assertion set to one float32 ULP."""
    started = time.perf_counter()
    predicted_tm_per_path = np.asarray(predicted_tm_per_path, dtype=np.float64).reshape(-1)
    enabled_mask = np.asarray(enabled_mask, dtype=bool).reshape(-1)
    residual_capacity = np.maximum(np.asarray(residual_capacity, dtype=np.float64).reshape(-1), 0.0)
    num_paths = predicted_tm_per_path.size
    num_pairs = num_paths // k
    active = np.flatnonzero(enabled_mask & (predicted_tm_per_path > 0.0))
    if active.size == 0:
        raise RuntimeError("Residual LP received no enabled variables")

    demand = predicted_tm_per_path[active]
    edge_constraints = pte[active, :].T.multiply(demand)
    pair_rows = active // k
    pair_constraints = residual.coo_matrix(
        (np.ones(active.size, dtype=np.float64), (pair_rows, np.arange(active.size))),
        shape=(num_pairs, active.size),
    ).tocsr()
    a_ub = residual.vstack((edge_constraints, pair_constraints), format="csr")
    b_ub = np.concatenate((residual_capacity, np.ones(num_pairs, dtype=np.float64)))
    result = residual.linprog(
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
    if diagnostics["max_edge_excess"] > 1e-6 or diagnostics["max_pair_mass"] > PAIR_MASS_LIMIT:
        raise RuntimeError(f"Residual LP invariant failure after one-ULP addendum: {diagnostics}")
    return policy, diagnostics


def main() -> None:
    manifest = frozen.verify_manifest()
    if not ADDENDUM.exists():
        raise RuntimeError("Numerical-exception addendum is absent")
    addendum = json.loads(ADDENDUM.read_text(encoding="utf-8"))
    own_hash = residual.sha256(Path(__file__).resolve())
    if not addendum.get("authorized", False) or addendum["wrapper_sha256"] != own_hash:
        raise RuntimeError("Numerical-exception wrapper is not frozen by its addendum")
    if addendum["original_manifest_sha256"] != residual.sha256(frozen.MANIFEST):
        raise RuntimeError("Original Level-4 manifest changed after the addendum")
    if addendum["pair_mass_limit"] != PAIR_MASS_LIMIT:
        raise RuntimeError("Numerical-exception threshold differs from one float32 ULP")

    residual.solve_residual_class = solve_residual_class_one_ulp
    for seed in frozen.SEEDS:
        run_dir = residual.run_one(4, seed, force=False)
        provenance = {
            "seed": seed,
            "original_frozen_manifest_sha256": addendum["original_manifest_sha256"],
            "numerical_exception_addendum_sha256": residual.sha256(ADDENDUM),
            "wrapper_sha256": own_hash,
            "policy_values_adjusted": False,
            "solver_or_objective_changed": False,
            "edge_excess_limit": 1e-6,
            "pair_mass_limit": PAIR_MASS_LIMIT,
            "success_gates_changed": False,
        }
        (run_dir / "numerical_exception_provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
