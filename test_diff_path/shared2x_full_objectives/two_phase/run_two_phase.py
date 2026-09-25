from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
FULL_OBJECTIVES_DIR = THIS_DIR.parent
TEST_DIR = FULL_OBJECTIVES_DIR.parent
ROOT = TEST_DIR.parent
LEGACY_TWO_PHASE_DIR = TEST_DIR / "shared2x_order_regularizer" / "round2_high_frozen"

# Load the already-tested hard-freeze implementation under its own module
# name, then bind it to the restored-Fh/Fhm Phase-A checkpoint and a fresh
# artifact root.  No earlier artifact directory is modified.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(LEGACY_TWO_PHASE_DIR))

import run_round2 as core


core.OUTPUT_ROOT = THIS_DIR / "artifacts"
core.WARMSTART_ROOT = FULL_OBJECTIVES_DIR / "artifacts" / "level4_confirmation"
core.PHASE_A_SOURCE_DESCRIPTION = (
    "shared2x_full_objectives six-objective Phase A (-Fh, Uh, -Fhm, Uhm, -Fhml, Uhml)"
)
core.HIGH_GATE = 0.9975
core.HIGH_GATE_CONSECUTIVE_EPOCHS = 2


def source_hashes() -> dict[str, str]:
    return {
        "two_phase/run_two_phase.py": core.sha256(Path(__file__).resolve()),
        "two_phase/core_run_round2.py": core.sha256(LEGACY_TWO_PHASE_DIR / "run_round2.py"),
        "two_phase/hybrid_model.py": core.sha256(LEGACY_TWO_PHASE_DIR / "hybrid_model.py"),
        "two_phase/penalty.py": core.sha256(LEGACY_TWO_PHASE_DIR / "penalty.py"),
        "phase_a/run_experiment.py": core.sha256(FULL_OBJECTIVES_DIR / "run_experiment.py"),
        "frameworks/hattrick_system.py": core.sha256(ROOT / "frameworks" / "hattrick_system.py"),
        "utils/robust_proj_utils.py": core.sha256(ROOT / "utils" / "robust_proj_utils.py"),
    }


core.source_hashes = source_hashes


def main() -> None:
    core.main()


if __name__ == "__main__":
    main()
