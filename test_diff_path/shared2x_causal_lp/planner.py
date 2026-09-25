from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


@dataclass(frozen=True)
class PlanResult:
    medium_flow: np.ndarray
    low_flow: np.ndarray
    medium_total: float
    low_total: float
    status: int
    message: str


@dataclass(frozen=True)
class GateResult:
    admitted_by_od: np.ndarray
    admitted_total: float
    status: int
    message: str


class CausalResidualAdmissionPlanner:
    """Joint Medium/Low path-flow LP after High has been admitted."""

    def __init__(self, links_by_paths: sparse.spmatrix, od_by_paths: sparse.spmatrix):
        self.links_by_paths = sparse.csr_matrix(links_by_paths, dtype=np.float64)
        self.od_by_paths = sparse.csr_matrix(od_by_paths, dtype=np.float64)
        self.num_links, self.num_paths = self.links_by_paths.shape
        self.num_pairs = self.od_by_paths.shape[0]
        if self.od_by_paths.shape[1] != self.num_paths:
            raise ValueError("OD and link matrices must use the same path dimension")
        zero_links = sparse.csr_matrix((self.num_links, self.num_paths), dtype=np.float64)
        zero_pairs = sparse.csr_matrix((self.num_pairs, self.num_paths), dtype=np.float64)
        self.constraint_matrix = sparse.vstack(
            [
                sparse.hstack([self.links_by_paths, self.links_by_paths]),
                sparse.hstack([self.od_by_paths, zero_pairs]),
                sparse.hstack([zero_pairs, self.od_by_paths]),
                sparse.hstack([zero_pairs, -self.od_by_paths]),
            ],
            format="csr",
        )
        self.objective = np.concatenate(
            [-np.ones(self.num_paths), -1e-6 * np.ones(self.num_paths)]
        )

    def solve(
        self,
        residual_capacity: np.ndarray,
        medium_demand: np.ndarray,
        low_demand: np.ndarray,
        low_entitlement: np.ndarray,
        medium_enabled: np.ndarray | None = None,
        low_enabled: np.ndarray | None = None,
    ) -> PlanResult:
        residual_capacity = np.maximum(np.asarray(residual_capacity, dtype=np.float64), 0.0)
        medium_demand = np.maximum(np.asarray(medium_demand, dtype=np.float64), 0.0)
        low_demand = np.maximum(np.asarray(low_demand, dtype=np.float64), 0.0)
        low_entitlement = np.clip(
            np.asarray(low_entitlement, dtype=np.float64), 0.0, low_demand
        )
        if residual_capacity.shape != (self.num_links,):
            raise ValueError("residual capacity shape mismatch")
        for name, value in (
            ("medium_demand", medium_demand),
            ("low_demand", low_demand),
            ("low_entitlement", low_entitlement),
        ):
            if value.shape != (self.num_pairs,):
                raise ValueError(f"{name} shape mismatch")

        bounds: list[tuple[float, float | None]] = []
        for enabled in (medium_enabled, low_enabled):
            if enabled is None:
                bounds.extend([(0.0, None)] * self.num_paths)
            else:
                enabled = np.asarray(enabled, dtype=bool).reshape(-1)
                if enabled.shape != (self.num_paths,):
                    raise ValueError("path mask shape mismatch")
                bounds.extend([(0.0, None) if flag else (0.0, 0.0) for flag in enabled])

        rhs = np.concatenate(
            [residual_capacity, medium_demand, low_demand, -low_entitlement]
        )
        result = linprog(
            self.objective,
            A_ub=self.constraint_matrix,
            b_ub=rhs,
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )
        if not result.success:
            raise RuntimeError(f"Residual admission LP failed ({result.status}): {result.message}")
        medium = np.asarray(result.x[: self.num_paths], dtype=np.float64)
        low = np.asarray(result.x[self.num_paths :], dtype=np.float64)
        return PlanResult(
            medium_flow=medium,
            low_flow=low,
            medium_total=float(medium.sum()),
            low_total=float(low.sum()),
            status=int(result.status),
            message=str(result.message),
        )

    def solve_medium_gates(
        self,
        residual_capacity: np.ndarray,
        medium_demand: np.ndarray,
        base_medium_policy: np.ndarray,
    ) -> GateResult:
        residual_capacity = np.maximum(np.asarray(residual_capacity, dtype=np.float64), 0.0)
        medium_demand = np.maximum(np.asarray(medium_demand, dtype=np.float64), 0.0)
        base_medium_policy = np.maximum(
            np.asarray(base_medium_policy, dtype=np.float64).reshape(-1), 0.0
        )
        if residual_capacity.shape != (self.num_links,):
            raise ValueError("residual capacity shape mismatch")
        if medium_demand.shape != (self.num_pairs,):
            raise ValueError("medium demand shape mismatch")
        if base_medium_policy.shape != (self.num_paths,):
            raise ValueError("base policy shape mismatch")
        path_flow_per_od = sparse.diags(base_medium_policy) @ self.od_by_paths.T
        link_flow_per_od = self.links_by_paths @ path_flow_per_od
        result = linprog(
            -np.ones(self.num_pairs, dtype=np.float64),
            A_ub=link_flow_per_od,
            b_ub=residual_capacity,
            bounds=[(0.0, float(demand)) for demand in medium_demand],
            method="highs",
            options={"presolve": True},
        )
        if not result.success:
            raise RuntimeError(f"Medium admission-gate LP failed ({result.status}): {result.message}")
        admitted = np.asarray(result.x, dtype=np.float64)
        return GateResult(
            admitted_by_od=admitted,
            admitted_total=float(admitted.sum()),
            status=int(result.status),
            message=str(result.message),
        )


def od_incidence(num_pairs: int, paths_per_pair: int) -> sparse.csr_matrix:
    rows = np.repeat(np.arange(num_pairs), paths_per_pair)
    cols = np.arange(num_pairs * paths_per_pair)
    data = np.ones_like(cols, dtype=np.float64)
    return sparse.csr_matrix(
        (data, (rows, cols)), shape=(num_pairs, num_pairs * paths_per_pair)
    )
