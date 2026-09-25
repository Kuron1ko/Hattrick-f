import numpy as np

from run_experiment import margin_vectors


def test_guards_are_nonnegative_and_monotone() -> None:
    ratios = np.asarray([[1.0, 2.0], [1.5, 3.0], [2.0, 4.0]])
    additive = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    r1, a1 = margin_vectors(ratios, additive, 0.5, 1.0)
    r2, a2 = margin_vectors(ratios, additive, 0.9, 1.0)
    assert np.all(r1 >= 1.0)
    assert np.all(a1 >= 0.0)
    assert np.all(r2 >= r1)
    assert np.all(a2 >= a1)


if __name__ == "__main__":
    test_guards_are_nonnegative_and_monotone()
