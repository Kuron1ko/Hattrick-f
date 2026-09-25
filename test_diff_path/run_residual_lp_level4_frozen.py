from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_residual_lp_level4 as final_analysis
import evaluate_strict2x_common_methods as common_evaluator
import run_hattrick_residual_lp as residual
import run_hattrick_strict2x_research as research


SEEDS = (490, 491, 492)
MANIFEST = residual.OUTPUT_ROOT / "final_frozen_manifest.json"


def verify_manifest() -> dict:
    if not MANIFEST.exists():
        raise RuntimeError("Final-test manifest is absent; indices 400-499 remain sealed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not manifest.get("authorized", False) or manifest.get("seeds") != list(SEEDS):
        raise RuntimeError("Manifest does not authorize exactly the frozen three seeds")
    source_paths = {
        "frameworks/hattrick_system.py": residual.ROOT / "frameworks" / "hattrick_system.py",
        "utils/build_dataset_within_cluster.py": residual.ROOT / "utils" / "build_dataset_within_cluster.py",
        "utils/training_utils.py": residual.ROOT / "utils" / "training_utils.py",
        "utils/robust_proj_utils.py": residual.ROOT / "utils" / "robust_proj_utils.py",
        "test_diff_path/run_hattrick_strict2x_research.py": Path(research.__file__).resolve(),
        "test_diff_path/run_hattrick_residual_lp.py": Path(residual.__file__).resolve(),
        "test_diff_path/run_residual_lp_level4_frozen.py": Path(__file__).resolve(),
        "test_diff_path/analyze_residual_lp_level4.py": Path(final_analysis.__file__).resolve(),
        "test_diff_path/evaluate_strict2x_common_methods.py": Path(common_evaluator.__file__).resolve(),
        "test_diff_path/run_dotemc_priority_mask_experiment.py": Path(common_evaluator.dote.__file__).resolve(),
    }
    for name, path in source_paths.items():
        actual = residual.sha256(path)
        if manifest["source_sha256"].get(name) != actual:
            raise RuntimeError(f"Frozen source mismatch for {name}: {actual}")
    level3_gate = residual.OUTPUT_ROOT / "level3_validation_only" / "hattrick_residual_lp_level3_gate.json"
    if residual.sha256(level3_gate) != manifest["level3_gate_sha256"]:
        raise RuntimeError("Level-3 promotion gate differs from the frozen manifest")
    return manifest


def run_baselines() -> None:
    for seed in SEEDS:
        research.run_one(4, "unchanged", seed, force=False)
    research.build_level_summary(4)


def run_hybrid() -> None:
    for seed in SEEDS:
        baseline = residual.selected_baseline_dir(4, seed) / "complete.json"
        if not baseline.exists():
            raise RuntimeError(f"Frozen matched baseline is incomplete for seed {seed}")
        residual.run_one(4, seed, force=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot frozen Level-4 execution.")
    parser.add_argument(
        "--phase", choices=("baseline", "hybrid", "analyze", "common", "all"), default="all"
    )
    args = parser.parse_args()
    verify_manifest()
    if args.phase in ("baseline", "all"):
        run_baselines()
    if args.phase in ("hybrid", "all"):
        run_hybrid()
    if args.phase in ("analyze", "all"):
        final_analysis.main()
    if args.phase in ("common", "all"):
        common_evaluator.main()


if __name__ == "__main__":
    main()
