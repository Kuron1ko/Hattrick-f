from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


@dataclass(frozen=True)
class MediumPlan:
    path_flow: np.ndarray
    total: float
    status: int


@dataclass(frozen=True)
class JointPlan:
    medium_flow: np.ndarray
    low_flow: np.ndarray
    medium_total: float
    low_total: float
    status: int


class SandwichMediumPlanner:
    """Maximize Medium while reserving the exact High and Low link footprints."""

    def __init__(self, links_by_paths: sparse.spmatrix, od_by_paths: sparse.spmatrix):
        self.links_by_paths = sparse.csr_matrix(links_by_paths, dtype=np.float64)
        self.od_by_paths = sparse.csr_matrix(od_by_paths, dtype=np.float64)
        self.num_links, self.num_paths = self.links_by_paths.shape
        self.num_pairs = self.od_by_paths.shape[0]
        if self.od_by_paths.shape[1] != self.num_paths:
            raise ValueError("OD and link matrices must share the path dimension")
        self.constraints = sparse.vstack(
            [self.links_by_paths, self.od_by_paths], format="csr"
        )
        self.objective = -np.ones(self.num_paths, dtype=np.float64)
        zero_links = sparse.csr_matrix((self.num_links, self.num_paths))
        zero_pairs = sparse.csr_matrix((self.num_pairs, self.num_paths))
        self.joint_constraints = sparse.vstack(
            [
                sparse.hstack([self.links_by_paths, self.links_by_paths]),
                sparse.hstack([self.od_by_paths, zero_pairs]),
                sparse.hstack([zero_pairs, self.od_by_paths]),
                sparse.hstack([zero_pairs, -self.od_by_paths]),
            ],
            format="csr",
        )
        # Lexicographic approximation: maximize Medium, then avoid admitting
        # unnecessary Low above its per-OD entitlement.
        self.joint_objective = np.concatenate(
            [-np.ones(self.num_paths), 1e-7 * np.ones(self.num_paths)]
        )

    def solve(
        self,
        capacity: np.ndarray,
        high_flow: np.ndarray,
        low_flow: np.ndarray,
        medium_demand: np.ndarray,
        enabled_paths: np.ndarray | None = None,
    ) -> MediumPlan:
        capacity = np.asarray(capacity, dtype=np.float64)
        high_flow = np.asarray(high_flow, dtype=np.float64)
        low_flow = np.asarray(low_flow, dtype=np.float64)
        medium_demand = np.asarray(medium_demand, dtype=np.float64)
        reserved_load = self.links_by_paths @ (high_flow + low_flow)
        residual = np.maximum(capacity - reserved_load, 0.0)
        rhs = np.concatenate([residual, np.maximum(medium_demand, 0.0)])
        if enabled_paths is None:
            bounds = [(0.0, None)] * self.num_paths
        else:
            enabled = np.asarray(enabled_paths, dtype=bool).reshape(-1)
            if enabled.shape != (self.num_paths,):
                raise ValueError("enabled path mask shape mismatch")
            bounds = [(0.0, None) if flag else (0.0, 0.0) for flag in enabled]
        result = linprog(
            self.objective,
            A_ub=self.constraints,
            b_ub=rhs,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )
        if not result.success:
            raise RuntimeError(
                f"Sandwich Medium LP failed ({result.status}): {result.message}"
            )
        flow = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
        return MediumPlan(flow, float(flow.sum()), int(result.status))

    def solve_with_low_entitlements(
        self,
        capacity: np.ndarray,
        high_flow: np.ndarray,
        medium_demand: np.ndarray,
        low_demand: np.ndarray,
        baseline_low_by_od: np.ndarray,
        medium_enabled: np.ndarray | None = None,
        low_enabled: np.ndarray | None = None,
    ) -> JointPlan:
        capacity = np.asarray(capacity, dtype=np.float64)
        high_flow = np.asarray(high_flow, dtype=np.float64)
        medium_demand = np.maximum(np.asarray(medium_demand, dtype=np.float64), 0.0)
        low_demand = np.maximum(np.asarray(low_demand, dtype=np.float64), 0.0)
        low_floor = np.clip(
            np.asarray(baseline_low_by_od, dtype=np.float64), 0.0, low_demand
        )
        residual = np.maximum(capacity - self.links_by_paths @ high_flow, 0.0)
        rhs = np.concatenate([residual, medium_demand, low_demand, -low_floor])
        bounds: list[tuple[float, float | None]] = []
        for enabled_paths in (medium_enabled, low_enabled):
            if enabled_paths is None:
                bounds.extend([(0.0, None)] * self.num_paths)
            else:
                enabled = np.asarray(enabled_paths, dtype=bool).reshape(-1)
                if enabled.shape != (self.num_paths,):
                    raise ValueError("enabled path mask shape mismatch")
                bounds.extend(
                    [(0.0, None) if flag else (0.0, 0.0) for flag in enabled]
                )
        result = linprog(
            self.joint_objective,
            A_ub=self.joint_constraints,
            b_ub=rhs,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )
        if not result.success:
            raise RuntimeError(
                f"Entitlement sandwich LP failed ({result.status}): {result.message}"
            )
        medium = np.maximum(np.asarray(result.x[: self.num_paths]), 0.0)
        low = np.maximum(np.asarray(result.x[self.num_paths :]), 0.0)
        return JointPlan(
            medium, low, float(medium.sum()), float(low.sum()), int(result.status)
        )


def od_incidence(num_pairs: int, paths_per_pair: int) -> sparse.csr_matrix:
    rows = np.repeat(np.arange(num_pairs), paths_per_pair)
    cols = np.arange(num_pairs * paths_per_pair)
    return sparse.csr_matrix(
        (np.ones_like(cols, dtype=np.float64), (rows, cols)),
        shape=(num_pairs, num_pairs * paths_per_pair),
    )
