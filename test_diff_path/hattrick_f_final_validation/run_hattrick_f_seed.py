from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

from frozen_sources import EXPECTED as FROZEN_EXPECTED
from frozen_sources import ROOT as FROZEN_ROOT
from frozen_sources import verify as verify_frozen_sources


HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
SOURCE = TEST_DIR / "shared2x_hattrick_f" / "run_level4.py"
PHASE_A_ROOT = (
    TEST_DIR
    / "shared2x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
)
CURRENT_PHASE_A_ROOT = HERE / "artifacts" / "phase_a_current"
OUTPUT_BASE = HERE / "artifacts" / "hattrick_f"
PHASE_A_OBJECTIVES = ("Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml")
PHASE_A_SOURCE_HASHES = {
    "run_experiment.py": "a77c112d25f549e2cbe73b9a41f4ef4cebe10b60061e8683f9faebd86b112a42",
    "ordered_projection.py": "a8ed960f641e9cb0786ab8edee3b4a3c571255c3e76c939f0d6803ce01543e3a",
    "frameworks/hattrick_system.py": "4f7b15dba81b23f0e23d432e0b65f8ae895b459070dd8249a22c1345b15eb862",
    "utils/training_utils.py": "b4b8c08ef353194ae107fc6756bde6f7668f12e5d6e87bf5f26280dbf4057185",
    "shared2x_order_regularizer/run_experiment.py": "317d0b796773963352a5b92652e6273074707bad42b3078a734e569366789310",
}
FROZEN_CORE_SOURCE_HASHES = {
    f"frozen::{path.resolve().relative_to(FROZEN_ROOT.resolve()).as_posix()}": digest
    for path, digest in FROZEN_EXPECTED.items()
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_phase_a_artifact(directory: Path, seed: int) -> tuple[Path, dict]:
    """Require a completed, internally consistent Phase-A artifact."""
    directory = directory.resolve()
    required = {
        "checkpoint": directory / "best_model.pt",
        "config": directory / "config.json",
        "complete": directory / "complete.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"incomplete Phase-A artifact for seed {seed} at {directory}; "
            f"missing {missing}; refusing legacy fallback"
        )
    config = json.loads(required["config"].read_text(encoding="utf-8"))
    complete = json.loads(required["complete"].read_text(encoding="utf-8"))
    checkpoint = torch.load(
        required["checkpoint"], map_location="cpu", weights_only=False
    )
    if int(config.get("seed", -1)) != int(seed):
        raise RuntimeError(
            f"Phase-A config seed mismatch: expected {seed}, got {config.get('seed')}"
        )
    checkpoint_config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    if not isinstance(checkpoint_config, dict):
        raise RuntimeError("Phase-A best checkpoint has no config dictionary")
    if int(checkpoint_config.get("seed", -1)) != int(seed):
        raise RuntimeError(
            "Phase-A checkpoint config seed mismatch: "
            f"expected {seed}, got {checkpoint_config.get('seed')}"
        )
    if checkpoint_config != config:
        raise RuntimeError("Phase-A config.json and checkpoint config differ")
    expected_training = {
        "level": 4,
        "label": "level4_confirmation",
        "epochs": 60,
        "topology": "geant_priomask500_shared_load2x_train",
        "paths_per_pair": 8,
        "load_factor": 2.0,
        "shared_paths": True,
        "train": [0, 350],
        "validation": [350, 400],
        "evaluation": [400, 500],
    }
    mismatched_training = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expected_training.items()
        if config.get(key) != expected
    }
    if tuple(config.get("objectives", ())) != PHASE_A_OBJECTIVES:
        mismatched_training["objectives"] = {
            "expected": list(PHASE_A_OBJECTIVES),
            "actual": config.get("objectives"),
        }
    if mismatched_training:
        raise RuntimeError(
            f"Phase-A is not the frozen six-loss 60-epoch Level-4 run: {mismatched_training}"
        )
    source_hashes = config.get("source_sha256", {})
    source_mismatches = {
        key: {"expected": expected, "actual": source_hashes.get(key)}
        for key, expected in PHASE_A_SOURCE_HASHES.items()
        if source_hashes.get(key) != expected
    }
    if source_mismatches:
        raise RuntimeError(
            f"Phase-A was not produced by the frozen current sources: {source_mismatches}"
        )
    if complete.get("status") not in (None, "COMPLETE"):
        raise RuntimeError(f"Phase-A completion status is {complete.get('status')!r}")
    actual_hash = sha256(required["checkpoint"])
    recorded_hash = complete.get("artifact_sha256", {}).get("best_model.pt")
    if recorded_hash is None:
        recorded_hash = complete.get("best_checkpoint_sha256")
    if not recorded_hash or recorded_hash != actual_hash:
        raise RuntimeError(
            "Phase-A complete/checkpoint SHA-256 mismatch: "
            f"recorded={recorded_hash!r}, actual={actual_hash}"
        )
    if int(complete.get("best_epoch", -1)) != int(checkpoint.get("epoch", -2)):
        raise RuntimeError("Phase-A complete/checkpoint best_epoch mismatch")
    if list(complete.get("best_rank", [])) != list(checkpoint.get("rank", [])):
        raise RuntimeError("Phase-A complete/checkpoint best_rank mismatch")
    return required["checkpoint"], {
        "directory": str(directory),
        "checkpoint_sha256": actual_hash,
        "seed": int(seed),
        "complete_verified": True,
        "frozen_source_provenance_verified": True,
        "six_loss_60_epoch_provenance_verified": True,
    }


def select_phase_a(seed: int) -> tuple[Path, dict]:
    current_dir = CURRENT_PHASE_A_ROOT / f"seed_{seed}"
    legacy_dir = PHASE_A_ROOT / f"seed_{seed}"
    # Directory existence is intentional: a half-written current run must fail
    # loudly instead of being silently replaced by an older legacy artifact.
    if current_dir.exists():
        return validate_phase_a_artifact(current_dir, seed)
    if legacy_dir.exists():
        return validate_phase_a_artifact(legacy_dir, seed)
    raise FileNotFoundError(
        f"Phase-A artifact is missing for seed {seed}; checked "
        f"{current_dir.resolve()} and {legacy_dir.resolve()}"
    )


def load_level4():
    spec = importlib.util.spec_from_file_location(
        "hattrick_f_final_validation_level4", SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(seed: int, epochs: int, low_budget: float, force: bool) -> Path:
    verify_frozen_sources()
    level4 = load_level4()
    phase_a, phase_a_integrity = select_phase_a(seed)
    level4.SEED = int(seed)
    level4.PHASE_A_PATH = phase_a
    level4.PHASE_A_DIR = phase_a.parent
    level4.OUTPUT_BASE = OUTPUT_BASE
    level4.OUTPUT_DIR = OUTPUT_BASE / "unused_until_run_sets_it"

    original_config = level4.config_for_run

    def config_for_run(run_epochs: int, run_low_budget: float) -> dict:
        config = original_config(run_epochs, run_low_budget)
        config["orchestrator"] = {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
            "protocol": str((HERE / "PROTOCOL.md").resolve()),
            "protocol_sha256": sha256(HERE / "PROTOCOL.md"),
        }
        config["source_sha256"].update(FROZEN_CORE_SOURCE_HASHES)
        config["phase_a_integrity"] = phase_a_integrity
        return config

    level4.config_for_run = config_for_run
    return level4.run(epochs=epochs, force=force, low_budget=low_budget)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen-source Hattrick-f Level-4 continuation for one seed"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--low-budget", type=float, default=0.03)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("epochs must be positive")
    if args.low_budget < 0:
        parser.error("low budget must be non-negative")
    print(run(args.seed, args.epochs, args.low_budget, args.force), flush=True)


if __name__ == "__main__":
    main()
