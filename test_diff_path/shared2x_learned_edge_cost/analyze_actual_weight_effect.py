from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
CLASSES = ("High", "Medium", "Low")
EXPERIMENTS = {
    "blend_0.25": ARTIFACTS / "small_weighted_mlu_actual_ratio_q0p9_eps0p01_blend0p25",
    "blend_0.5": ARTIFACTS / "small_weighted_mlu_actual_ratio_q0p9_eps0p01_blend0p5",
}
OUTPUT = ARTIFACTS / "actual_weight_effect_validation.json"


def load_pair(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    control_paths = sorted(directory.glob("control_*_norm_fulfill.csv"))
    candidate_paths = sorted(directory.glob("learned_*_norm_fulfill.csv"))
    if len(control_paths) != 1 or len(candidate_paths) != 1:
        raise RuntimeError(
            f"Expected one control and one candidate in {directory}, got "
            f"{len(control_paths)} and {len(candidate_paths)}"
        )
    control = np.loadtxt(control_paths[0], delimiter=",", skiprows=1)
    candidate = np.loadtxt(candidate_paths[0], delimiter=",", skiprows=1)
    if control.shape != candidate.shape or control.shape[1] != 3:
        raise RuntimeError(f"Invalid paired shapes {control.shape} and {candidate.shape}")
    return control, candidate


def paired_summary(delta: np.ndarray, rng: np.random.Generator) -> dict[str, float | bool]:
    n = len(delta)
    indices = rng.integers(0, n, size=(20000, n))
    bootstrap_means = delta[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "n": n,
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "paired_positive_fraction": float((delta > 0).mean()),
        "bootstrap_95_lower": float(lower),
        "bootstrap_95_upper": float(upper),
        "mean_effect_excludes_zero": bool(lower > 0 or upper < 0),
    }


def main() -> None:
    result: dict[str, object] = {
        "method": "paired nonparametric bootstrap over strict-ESM evaluation snapshots",
        "bootstrap_resamples": 20000,
        "identity_check": json.loads(
            (ARTIFACTS / "weighted_mlu_identity.json").read_text(encoding="utf-8")
        ),
        "experiments": {},
    }
    for experiment_index, (label, directory) in enumerate(EXPERIMENTS.items()):
        control, candidate = load_pair(directory)
        rng = np.random.default_rng(490 + experiment_index)
        result["experiments"][label] = {
            class_name: paired_summary(candidate[:, index] - control[:, index], rng)
            for index, class_name in enumerate(CLASSES)
        }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
