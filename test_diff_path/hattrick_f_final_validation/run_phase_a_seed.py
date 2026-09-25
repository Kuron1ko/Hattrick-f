from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from frozen_sources import verify as verify_frozen_sources


HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
SOURCE = TEST_DIR / "shared2x_full_objectives" / "run_experiment.py"
OUTPUT_ROOT = HERE / "artifacts" / "phase_a_current"
LEVEL = 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_phase_a() -> ModuleType:
    # The original six-loss runner imports ordered_projection by module name;
    # loading it through a wrapper therefore needs its own directory on sys.path.
    sys.path.insert(0, str(SOURCE.parent))
    spec = importlib.util.spec_from_file_location(
        "hattrick_f_final_validation_phase_a", SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(seed: int) -> Path:
    verify_frozen_sources()
    run_dir = OUTPUT_ROOT / f"seed_{seed}"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        # Exclusive creation closes the race between the existence audit and
        # the shared runner opening its first artifact.
        run_dir.mkdir(exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"refusing to overwrite an existing Phase-A artifact: {run_dir}"
        ) from None

    phase_a = load_phase_a()
    phase_a.OUTPUT_ROOT = OUTPUT_ROOT
    phase_a.run_directory = lambda level, run_seed: (
        OUTPUT_ROOT / f"seed_{run_seed}"
    )

    original_source_hashes = phase_a.source_hashes

    def source_hashes() -> dict[str, str]:
        hashes = original_source_hashes()
        hashes["run_phase_a_seed.py"] = sha256(Path(__file__).resolve())
        hashes["frozen_sources.py"] = sha256(HERE / "frozen_sources.py")
        return hashes

    phase_a.source_hashes = source_hashes
    return phase_a.run_one(LEVEL, int(seed), force=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen-source six-loss Hattrick Phase-A Level-4 training for one seed"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(run(args.seed), flush=True)


if __name__ == "__main__":
    main()
