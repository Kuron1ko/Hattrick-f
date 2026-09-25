from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable

import torch

from frozen_sources import EXPECTED as FROZEN_EXPECTED
from frozen_sources import ROOT as FROZEN_ROOT
from frozen_sources import verify as verify_frozen_sources


HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
HATTRICK_F_LEVEL4_SOURCE = TEST_DIR / "shared2x_hattrick_f" / "run_level4.py"
SIX_LOSS_SOURCE = TEST_DIR / "shared2x_full_objectives" / "run_experiment.py"
CURRENT_PHASE_A_ROOT = HERE / "artifacts" / "phase_a_current"
LEGACY_PHASE_A_ROOT = (
    TEST_DIR
    / "shared2x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
)
OUTPUT_BASE = HERE / "artifacts" / "six_loss_continuation_control"
SIX_LOSS_OBJECTIVES = ("Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml")
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
    if tuple(config.get("objectives", ())) != SIX_LOSS_OBJECTIVES:
        mismatched_training["objectives"] = {
            "expected": list(SIX_LOSS_OBJECTIVES),
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
    legacy_dir = LEGACY_PHASE_A_ROOT / f"seed_{seed}"
    if current_dir.exists():
        return validate_phase_a_artifact(current_dir, seed)
    if legacy_dir.exists():
        return validate_phase_a_artifact(legacy_dir, seed)
    raise FileNotFoundError(
        f"Phase-A artifact is missing for seed {seed}; checked "
        f"{current_dir.resolve()} and {legacy_dir.resolve()}"
    )


def load_source(name: str, source: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_six_loss_adapter(
    six_loss: ModuleType,
) -> Callable[..., dict[str, float]]:
    """Adapt the audited six-loss epoch to Hattrick-f's Level-4 call shape."""

    def train_epoch(
        model,
        _teacher,
        props,
        dataset,
        loader,
        optimizer,
        _high_floor,
        _mlu_slack,
        _keep_fh,
    ) -> dict[str, float]:
        return six_loss.train_epoch(model, props, dataset, loader, optimizer)

    return train_epoch


def configure_control(seed: int) -> ModuleType:
    """Load the common runner and change only its continuation objectives."""
    verify_frozen_sources()
    level4 = load_source(
        "hattrick_f_six_loss_control_level4", HATTRICK_F_LEVEL4_SOURCE
    )
    six_loss = load_source(
        "hattrick_f_six_loss_control_objectives", SIX_LOSS_SOURCE
    )

    phase_a, phase_a_integrity = select_phase_a(seed)

    # Keep the same initial checkpoint, Adam reset, epoch/batch seeds, validation
    # gate, checkpoint ranking and post-selection evaluation as Hattrick-f.
    level4.SEED = int(seed)
    level4.PHASE_A_PATH = phase_a
    level4.PHASE_A_DIR = phase_a.parent
    level4.OUTPUT_BASE = OUTPUT_BASE
    level4.OUTPUT_DIR = OUTPUT_BASE / "unused_until_run_sets_it"
    level4.core.train_epoch = build_six_loss_adapter(six_loss)

    original_config = level4.config_for_run
    original_class_comparison = level4.class_comparison

    def config_for_run(run_epochs: int, run_low_budget: float) -> dict:
        config = original_config(run_epochs, run_low_budget)
        config["method"] = "Hattrick six-loss continuation control"
        config["experiment_role"] = (
            "same-budget control for Hattrick-f Phase-F; only the ordered "
            "continuation objectives differ"
        )
        config["phase_f"] = {
            "objectives": list(SIX_LOSS_OBJECTIVES),
            "optimizer_reset": True,
            "all_parameters_trainable": True,
            "persistent_mlu_minimization": True,
        }
        config["controlled_against"] = {
            "method": "Hattrick-f",
            "objectives": ["Fh", "Fhm", "Fhml"],
            "identical_components": [
                "Phase-A checkpoint",
                "fresh Adam optimizer",
                "epoch count",
                "epoch and data-loader seeds",
                "training split",
                "validation safety gate",
                "checkpoint rank",
                "strict-ESM evaluation",
            ],
        }
        for name, digest in six_loss.source_hashes().items():
            config["source_sha256"][f"six_loss_dependency::{name}"] = digest
        config["source_sha256"]["run_six_loss_control.py"] = sha256(
            Path(__file__).resolve()
        )
        config["source_sha256"]["frozen_sources.py"] = sha256(
            HERE / "frozen_sources.py"
        )
        config["source_sha256"].update(FROZEN_CORE_SOURCE_HASHES)
        config["phase_a_integrity"] = phase_a_integrity
        return config

    def class_comparison(
        baseline_summary: list[dict], candidate_summary: list[dict]
    ) -> dict:
        comparison = original_class_comparison(
            baseline_summary, candidate_summary
        )
        for metrics in comparison.values():
            metrics["control_norm_mean"] = metrics.pop("hattrick_f_norm_mean")
            metrics["control_norm_p10"] = metrics.pop("hattrick_f_norm_p10")
            metrics["control_norm_p1"] = metrics.pop("hattrick_f_norm_p1")
        return comparison

    level4.config_for_run = config_for_run
    level4.class_comparison = class_comparison
    return level4


def run(seed: int, epochs: int, low_budget: float) -> Path:
    level4 = configure_control(seed)
    # force=False intentionally prevents this control from deleting or
    # overwriting an existing run. An incomplete matching run may resume.
    return level4.run(epochs=epochs, force=False, low_budget=low_budget)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Level-4 same-budget six-loss continuation control for Hattrick-f"
        )
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--low-budget", type=float, default=0.03)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("epochs must be positive")
    if args.low_budget < 0:
        parser.error("low budget must be non-negative")
    print(run(args.seed, args.epochs, args.low_budget), flush=True)


if __name__ == "__main__":
    main()
