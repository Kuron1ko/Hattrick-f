from __future__ import annotations

import numpy as np
from scipy import sparse

from planner import CausalResidualAdmissionPlanner, od_incidence


def test_medium_is_maximized_while_low_entitlement_is_preserved() -> None:
    links_by_paths = sparse.csr_matrix([[1.0, 1.0]])
    planner = CausalResidualAdmissionPlanner(links_by_paths, od_incidence(2, 1))
    result = planner.solve(
        residual_capacity=np.asarray([10.0]),
        medium_demand=np.asarray([8.0, 8.0]),
        low_demand=np.asarray([4.0, 4.0]),
        low_entitlement=np.asarray([2.0, 3.0]),
    )
    assert abs(result.medium_total - 5.0) < 1e-7
    assert result.low_flow[0] >= 2.0 - 1e-7
    assert result.low_flow[1] >= 3.0 - 1e-7
    assert result.medium_total + result.low_total <= 10.0 + 1e-7


def test_disabled_paths_are_zero() -> None:
    planner = CausalResidualAdmissionPlanner(
        sparse.eye(2, format="csr"), od_incidence(1, 2)
    )
    result = planner.solve(
        residual_capacity=np.asarray([5.0, 5.0]),
        medium_demand=np.asarray([4.0]),
        low_demand=np.asarray([2.0]),
        low_entitlement=np.asarray([1.0]),
        medium_enabled=np.asarray([True, False]),
        low_enabled=np.asarray([False, True]),
    )
    assert result.medium_flow[1] == 0.0
    assert result.low_flow[0] == 0.0


def test_gate_lp_preserves_path_proportions_and_maximizes_admission() -> None:
    links = sparse.csr_matrix([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    planner = CausalResidualAdmissionPlanner(links, od_incidence(2, 2))
    result = planner.solve_medium_gates(
        residual_capacity=np.asarray([3.0, 3.0]),
        medium_demand=np.asarray([5.0, 5.0]),
        base_medium_policy=np.asarray([0.5, 0.5, 0.5, 0.5]),
    )
    assert abs(result.admitted_total - 6.0) < 1e-7
