from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
SOURCE_DIR = TEST_DIR / "shared2x_full_objectives"
SOURCE_PATH = SOURCE_DIR / "run_experiment.py"

sys.path.insert(0, str(SOURCE_DIR))
spec = importlib.util.spec_from_file_location("full_hattrick_onex", SOURCE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load complete-objective runner: {SOURCE_PATH}")
full = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = full
spec.loader.exec_module(full)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Complete six-objective Hattrick, strict-ESM 1x Level-4"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    full.OUTPUT_ROOT = THIS_DIR / "artifacts"
    full.shared.TOPOLOGY = "geant_priomask500_shared"
    full.LOAD_FACTOR = 1.0
    full.run_one(level=4, seed=490, force=args.force)


if __name__ == "__main__":
    main()
