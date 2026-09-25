from __future__ import annotations

"""Hard-locked strict test replay for the sparse-e48-matched edge-toll head.

The numerical evaluator and all counterfactual checks are shared with the
previous fixed-head experiment.  Only immutable paths, hashes, labels, and the
experiment-design caveat are replaced here.
"""

import importlib.util
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
BASE_EVALUATOR = TEST_DIR / "sparse_e48_hattricke_head" / "evaluate_fixed_hybrid.py"


def load_base():
    spec = importlib.util.spec_from_file_location(
        "sparse_e48_matched_head_fixed_evaluator", BASE_EVALUATOR
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {BASE_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    fixed = load_base()
    head_checkpoint = THIS_DIR / "artifacts" / "best_edge_toll_head.pt"
    training_report = THIS_DIR / "artifacts" / "training_validation.json"

    fixed.EDGE_CHECKPOINT = head_checkpoint
    fixed.OUTPUT_DIR = (
        ROOT
        / "output"
        / "comparisons"
        / "sparse_e48_retrained_edge_toll_head_strict2x"
    )
    fixed.EXPECTED_HASHES = {
        fixed.SPARSE_CHECKPOINT: (
            "55dfbe1506148099c838f3e5e08ca871f280877e56748ae3ea5af574c0a3acad"
        ),
        fixed.SELECTION_MANIFEST: (
            "a481a9ff750a00a5ab8abc5a76a66e8db05c82c0cecca7d1f807437282e09d6d"
        ),
        head_checkpoint: (
            "b94bf13248555c7d4844bd250d547681b215522cd59a380ec5fcb9e9afd4c8d9"
        ),
        training_report: (
            "50ff73fd69ba055529f087d212a9a20d51776a74dd7fabef18701871d3977101"
        ),
        fixed.SPARSE_RUNNER: (
            "3b40dff9801bc71e6606888d8a690adbe2e695b52fd1a4710f6bb97b71b19f56"
        ),
        fixed.EDGE_RUNNER: (
            "839c6a8b2a091553fe817abedc97016b35e498980a4eb2d8f936cb5cbae4c866"
        ),
    }
    fixed.METHOD_NAME = (
        "sparse cross-attention e48 High + distribution-matched Medium/Low "
        "edge-toll head"
    )
    fixed.METHOD_DESCRIPTION = (
        "Fixed validation-selected sparse-e48 High backbone plus its "
        "training/safety-selected distribution-matched edge-toll head; strict "
        "ESM sequential admission"
    )
    fixed.HEAD_LABEL = "sparse-e48-matched edge-toll head"
    fixed.EXPERIMENT_DIRECTION_TEST_INFORMED = True
    fixed.EXPERIMENT_DESIGN_CAVEAT = (
        "The head checkpoint itself was trained and selected without snapshots "
        "400-499, but the decision to train a distribution-matched head was made "
        "after inspecting the bare sparse-e48 result on that window. Therefore "
        "this replay is exploratory/post-hoc, not untouched confirmatory evidence."
    )
    fixed.main()


if __name__ == "__main__":
    main()
