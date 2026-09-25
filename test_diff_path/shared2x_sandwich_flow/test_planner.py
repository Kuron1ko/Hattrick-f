import numpy as np
from scipy import sparse

from planner import SandwichMediumPlanner, od_incidence


def test_reserved_classes_are_respected_and_medium_improves() -> None:
    # Two ODs, two paths each, on two links.  The baseline Medium uses only the
    # crowded first link; the planner can move its second OD to the spare link.
    links = sparse.csr_matrix(
        np.asarray([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float64)
    )
    planner = SandwichMediumPlanner(links, od_incidence(2, 2))
    high = np.asarray([2, 0, 0, 0], dtype=np.float64)
    low = np.asarray([1, 0, 0, 0], dtype=np.float64)
    result = planner.solve(
        np.asarray([5, 5], dtype=np.float64),
        high,
        low,
        np.asarray([2, 4], dtype=np.float64),
    )
    combined = links @ (high + low + result.path_flow)
    assert result.total == 6.0
    assert np.all(combined <= np.asarray([5, 5]) + 1e-9)


def test_low_od_entitlements_allow_lossless_rerouting() -> None:
    links = sparse.csr_matrix(
        np.asarray([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float64)
    )
    planner = SandwichMediumPlanner(links, od_incidence(2, 2))
    result = planner.solve_with_low_entitlements(
        np.asarray([5, 5], dtype=np.float64),
        np.asarray([2, 0, 0, 0], dtype=np.float64),
        np.asarray([2, 4], dtype=np.float64),
        np.asarray([1, 1], dtype=np.float64),
        np.asarray([1, 1], dtype=np.float64),
    )
    low_by_od = result.low_flow.reshape(2, 2).sum(axis=1)
    combined = links @ (
        np.asarray([2, 0, 0, 0], dtype=np.float64)
        + result.medium_flow
        + result.low_flow
    )
    assert np.all(low_by_od >= np.asarray([1, 1]) - 1e-8)
    assert np.all(combined <= np.asarray([5, 5]) + 1e-8)


if __name__ == "__main__":
    test_reserved_classes_are_respected_and_medium_improves()
    test_low_od_entitlements_allow_lossless_rerouting()
