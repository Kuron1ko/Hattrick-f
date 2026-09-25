from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

try:
    import numpy._core  # type: ignore[import-not-found]  # noqa: F401
except ModuleNotFoundError:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
BASE_1X_PATH = TEST_DIR / "full_geant_1x_hattrick" / "run_full_experiment.py"
ALGORITHM_PATH = TEST_DIR / "full_geant_2x_hattrick_f3" / "run_full_experiment.py"
DEFAULT_PHASE_A = (
    TEST_DIR / "full_geant_1x_hattrick" / "artifacts" / "hattrick"
    / "seed_490" / "best_model.pt"
)
ARTIFACT_ROOT = THIS_DIR / "artifacts"
DATASET_NAME = "1x"
TRAIN_RANGE = (0, 6000)
VALIDATION_RANGE = (6000, 7500)
TEST_RANGE = (7500, 10200)
DEFAULT_EPOCHS = 40
DEFAULT_ANNEAL_EPOCHS = 12
DEFAULT_TOP_K = 5
DEFAULT_SAVE_EVERY = 5


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base1 = load_module("full_geant_1x_hattrick_runtime", BASE_1X_PATH)
algorithm = load_module("full_geant_1x_f3_algorithm", ALGORITHM_PATH)
base = base1.base


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_hashes() -> dict[str, str]:
    paths = {
        "full_geant_1x_hattrick_f3.py": Path(__file__).resolve(),
        "full_geant_1x_hattrick.py": BASE_1X_PATH,
        "hattrick_f3_algorithm.py": ALGORITHM_PATH,
        "six_objective_runtime.py": base.FULL_SOURCE,
        "ordered_projection.py": base.FULL_SOURCE.parent / "ordered_projection.py",
        "hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
        "training_utils.py": ROOT / "utils" / "training_utils.py",
        "dataset.py": ROOT / "utils" / "build_dataset_within_cluster.py",
    }
    return {name: base.sha256(path) for name, path in paths.items()}


def experiment_config(
    args: argparse.Namespace, phase_a: Path, phase_a_selection: dict
) -> dict:
    return {
        "method": "Hattrick-f3",
        "dataset": DATASET_NAME,
        "seed": args.seed,
        "topology": base1.TARGET_TOPOLOGY,
        "load_factor": base1.LOAD_FACTOR,
        "strict_esm": True,
        "train": list(TRAIN_RANGE),
        "validation": list(VALIDATION_RANGE),
        "test": list(TEST_RANGE),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "low_budget": args.low_budget,
        "top_k": args.top_k,
        "save_every": args.save_every,
        "phase_a_checkpoint": str(phase_a.resolve()),
        "phase_a_sha256": base.sha256(phase_a),
        "phase_a_selection": phase_a_selection,
        "optimizer_reset": True,
        "all_parameters_trainable": True,
        "annealing": {
            "quantity": "ordered projected update",
            "formula": "alpha*g_six + (1-alpha)*g_flow",
            "six_objectives": list(algorithm.OBJECTIVE_NAMES),
            "flow_objectives": list(algorithm.FLOW_OBJECTIVE_NAMES),
            "schedule": "front-loaded cosine",
            "anneal_epochs": args.anneal_epochs,
            "alpha_epoch_1": 1.0,
            "alpha_epoch_anneal_end": 0.0,
            "alpha_after_anneal": 0.0,
        },
        "selection": (
            "validation only: strict High per-snapshot floor, Low mean budget, "
            "simulator safety; then Medium mean/P10/P1, High mean, Low mean"
        ),
        "source_sha256": source_hashes(),
    }


def archive_upgrade_compatible(saved: dict, current: dict) -> bool:
    """Allow old runs to resume after checkpoint-only runner changes."""
    saved_copy = copy.deepcopy(saved)
    current_copy = copy.deepcopy(current)
    saved_sources = saved_copy.pop("source_sha256", None)
    current_sources = current_copy.pop("source_sha256", None)
    if saved_copy != current_copy:
        return False
    if not isinstance(saved_sources, dict) or not isinstance(current_sources, dict):
        return False
    if saved_sources.keys() != current_sources.keys():
        return False
    allowed_legacy_hashes = {
        "full_geant_1x_hattrick_f3.py":
            "43175d6b97d86b0232bef70171e4d0763317e211466cfd133624fc5b84a36814",
        "full_geant_1x_hattrick.py":
            "7f75c1d015ccd35bcf42c1bca668341f97c25e2a085e995fb38ddf6347f81135",
    }
    return all(
        saved_sources[key] == current_sources[key]
        or saved_sources[key] == allowed_legacy_hashes.get(key)
        for key in saved_sources
    )


def run_directory(seed: int) -> Path:
    return ARTIFACT_ROOT / f"seed_{seed}"


def check_inputs(
    args: argparse.Namespace,
    phase_a_override: Path | None = None,
    *,
    require_selection_index: bool = True,
) -> dict:
    base1.require_training_inputs()
    phase_a = (phase_a_override or args.phase_a_checkpoint).resolve()
    if not phase_a.exists():
        raise FileNotFoundError(f"Missing 1x Phase-A Hattrick checkpoint: {phase_a}")
    checkpoint = torch.load(phase_a, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "Hattrick":
        raise RuntimeError(f"Phase-A checkpoint is not Hattrick: {phase_a}")
    if checkpoint.get("dataset") != DATASET_NAME:
        raise RuntimeError(f"Phase-A checkpoint is not a 1x model: {phase_a}")
    if int(checkpoint.get("seed", -1)) != args.seed:
        raise RuntimeError("Phase-A checkpoint seed mismatch")
    config = checkpoint.get("config", {})
    if (
        config.get("topology") != base1.TARGET_TOPOLOGY
        or float(config.get("load_factor", -1)) != base1.LOAD_FACTOR
    ):
        raise RuntimeError("Phase-A checkpoint topology/load mismatch")
    selection = {
        "status": "PINNED",
        "selected_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": base.sha256(phase_a),
    }
    if require_selection_index:
        entries = base1.load_top_k(phase_a.parent)
        if not entries:
            raise FileNotFoundError(
                f"Phase-A has no validation selection index: {phase_a.parent / 'top5.json'}"
            )
        selected = entries[0]
        if (
            int(selected["epoch"]) != int(checkpoint["epoch"])
            or selected["sha256"] != selection["checkpoint_sha256"]
        ):
            raise RuntimeError("Phase-A best_model does not match top5.json")
        complete_path = phase_a.parent / "complete.json"
        selection["status"] = "COMPLETE" if complete_path.exists() else "IN_PROGRESS"
        selection["selection_index"] = str((phase_a.parent / "top5.json").resolve())
        selection["selection_index_sha256"] = base.sha256(phase_a.parent / "top5.json")
        resume_path = phase_a.parent / "resume_state.pt"
        if resume_path.exists():
            resume = torch.load(resume_path, map_location="cpu", weights_only=False)
            selection["phase_a_trained_through_epoch"] = int(resume["epoch"])
    print(json.dumps({
        "status": "READY",
        "method": "Hattrick-f3",
        "dataset": DATASET_NAME,
        "phase_a_epoch": int(checkpoint["epoch"]),
        "phase_a_sha256": base.sha256(phase_a),
        "phase_a_selection_status": selection["status"],
        "phase_a_trained_through_epoch": selection.get(
            "phase_a_trained_through_epoch", int(checkpoint["epoch"])
        ),
        "train": list(TRAIN_RANGE),
        "validation": list(VALIDATION_RANGE),
        "test_reserved_for_unified_inference": list(TEST_RANGE),
        "epochs": args.epochs,
        "anneal_epochs": args.anneal_epochs,
        "save_every": args.save_every,
        "checkpoint_archive_policy": "every_epoch",
        "top_k": args.top_k,
        "cuda_available": torch.cuda.is_available(),
    }, indent=2, ensure_ascii=False), flush=True)
    return selection


def train(args: argparse.Namespace) -> None:
    run_dir = run_directory(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    resume_path = run_dir / "resume_state.pt"
    if args.resume and config_path.exists():
        saved_config = base.read_json(config_path)
        if not isinstance(saved_config, dict):
            raise RuntimeError(f"Malformed existing configuration: {config_path}")
        phase_a_path = Path(saved_config["phase_a_checkpoint"]).resolve()
        phase_a_selection = check_inputs(
            args, phase_a_path, require_selection_index=False
        )
        phase_a_selection = saved_config["phase_a_selection"]
    else:
        source_phase_a = args.phase_a_checkpoint.resolve()
        phase_a_selection = check_inputs(args, source_phase_a)
        phase_a_path = run_dir / (
            f"phase_a_hattrick_epoch_{phase_a_selection['selected_epoch']:03d}.pt"
        )
        if phase_a_path.exists():
            if base.sha256(phase_a_path) != phase_a_selection["checkpoint_sha256"]:
                raise RuntimeError(f"Pinned Phase-A checkpoint differs: {phase_a_path}")
        else:
            base.atomic_copy(source_phase_a, phase_a_path)
    current_config = experiment_config(args, phase_a_path, phase_a_selection)
    if config_path.exists():
        saved_config = base.read_json(config_path)
        if saved_config == current_config:
            config = current_config
        elif args.resume and archive_upgrade_compatible(saved_config, current_config):
            config = saved_config
            print(
                "[compat] preserving the existing experiment identity; "
                "new checkpoints will be archived every epoch",
                flush=True,
            )
        else:
            raise RuntimeError(f"Existing 1x Hattrick-f3 configuration differs: {config_path}")
    else:
        config = current_config
        base.write_json(config_path, config)
    config_hash = canonical_json_sha256(config)
    if resume_path.exists() and not args.resume:
        raise RuntimeError(f"Existing state found: {resume_path}; use --resume")
    if args.resume and not resume_path.exists():
        raise FileNotFoundError(f"--resume requested but no state exists: {resume_path}")

    shared, full, _hattrick_f = base.load_runtimes()
    from frameworks.hattrick_system import Hattrick
    from utils.AdamOptimizer import ADAMOptimizer

    device = base1.resolve_device(args.device)
    base.set_seed(args.seed)
    props = base.build_props(
        device, batch_size=args.batch_size, epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    train_dataset, validation_dataset, test_dataset = base.make_datasets(props)
    if test_dataset is not None:
        raise RuntimeError("Training unexpectedly instantiated the test split")
    phase_a = torch.load(phase_a_path, map_location=device, weights_only=False)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(phase_a["model_state_dict"])
    optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)
    baseline_rows, baseline_summary, baseline_diagnostics = shared.evaluate(
        model, props, validation_dataset, VALIDATION_RANGE[0]
    )
    base.write_csv(run_dir / "phase_a_validation_metrics.csv", baseline_rows)
    base.write_json(
        run_dir / "phase_a_validation_summary.json",
        {"classes": baseline_summary, "diagnostics": baseline_diagnostics},
    )
    history_path = run_dir / "train_history.csv"
    history = base.read_csv(history_path)
    top_entries = base1.load_top_k(run_dir)
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("method") != "Hattrick-f3" or checkpoint.get("dataset") != DATASET_NAME:
            raise RuntimeError("Resume-state method/dataset mismatch")
        if int(checkpoint.get("seed", -1)) != args.seed:
            raise RuntimeError("Resume-state seed mismatch")
        if checkpoint.get("config") != config or checkpoint.get("config_sha256") != config_hash:
            raise RuntimeError("Resume-state configuration mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[resume] 1x Hattrick-f3 at epoch {start_epoch}", flush=True)

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        base.set_seed(args.seed + epoch * 104729)
        alpha = algorithm.anneal_alpha(epoch, args.anneal_epochs)
        loader = shared.data_loader(
            train_dataset, props.batch_size, True, args.seed + epoch * 1009
        )
        train_metrics = algorithm.train_epoch(
            shared=shared, full=full, model=model, props=props,
            dataset=train_dataset, loader=loader, optimizer=optimizer, alpha=alpha,
        )
        rows, summary, diagnostics = shared.evaluate(
            model, props, validation_dataset, VALIDATION_RANGE[0]
        )
        rank = base.hattrick_f_rank(rows, summary, baseline_summary, args.low_budget)
        base.save_validation(run_dir, epoch, rows, summary, diagnostics)
        row = base.history_row(epoch, rank, rows, summary, diagnostics, train_metrics)
        history = [item for item in history if int(item["epoch"]) < epoch]
        history.append(row)
        base.write_csv(history_path, history)
        payload = {
            "method": "Hattrick-f3",
            "dataset": DATASET_NAME,
            "seed": args.seed,
            "epoch": epoch,
            "anneal_alpha": alpha,
            "rank": tuple(float(x) for x in rank),
            "config": config,
            "config_sha256": config_hash,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_summary": summary,
            "validation_diagnostics": diagnostics,
        }
        torch.save(payload, resume_path)
        archive = base1.save_epoch_checkpoint(run_dir, payload)
        top_entries = base1.update_top_k(run_dir, top_entries, payload, args.top_k)
        print(
            f"[1x Hattrick-f3] epoch={epoch}/{args.epochs} alpha={alpha:.4f} "
            f"eligible={rank[0]} H={row['high_norm_mean']:.6f} "
            f"Hmin={row['high_norm_min']:.6f} M={row['medium_norm_mean']:.6f} "
            f"L={row['low_norm_mean']:.6f} "
            f"archive={archive.name if archive else '-'} "
            f"best{args.top_k}={[int(x['epoch']) for x in top_entries]}",
            flush=True,
        )
    if len(top_entries) != args.top_k:
        raise RuntimeError(f"Expected {args.top_k} best checkpoints, found {len(top_entries)}")
    complete = {
        "status": "COMPLETE",
        "method": "Hattrick-f3",
        "dataset": DATASET_NAME,
        "selection_used_test": False,
        "seed": args.seed,
        "epochs": args.epochs,
        "anneal_epochs": args.anneal_epochs,
        "save_every": args.save_every,
        "checkpoint_archive_policy": "every_epoch",
        "top_k": args.top_k,
        "best_checkpoints": top_entries,
        "selected_epoch": int(top_entries[0]["epoch"]),
        "saved_epoch_checkpoints": base1.saved_epoch_numbers(run_dir),
        "runtime_seconds_this_invocation": time.perf_counter() - started,
        "resume_state": str(resume_path.resolve()),
        "resume_state_sha256": base.sha256(resume_path),
        "config_sha256": config_hash,
    }
    base.write_json(run_dir / "complete.json", complete)
    print(json.dumps(complete, indent=2, ensure_ascii=False), flush=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train full-GEANT 1x Hattrick-f3")
    parser.add_argument("--stage", choices=("check", "train", "all"), default="check")
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--epochs", type=positive_int, default=DEFAULT_EPOCHS)
    parser.add_argument("--anneal-epochs", type=positive_int, default=DEFAULT_ANNEAL_EPOCHS)
    parser.add_argument("--top-k", type=positive_int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--save-every", type=positive_int, default=DEFAULT_SAVE_EVERY,
        help="Legacy configuration field; independent checkpoints are archived every epoch.",
    )
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--low-budget", type=float, default=0.03)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--phase-a-checkpoint", type=Path, default=DEFAULT_PHASE_A)
    args = parser.parse_args()
    if args.anneal_epochs < 2 or args.anneal_epochs > args.epochs:
        parser.error("--anneal-epochs must be between 2 and --epochs")
    if args.top_k > args.epochs:
        parser.error("--top-k cannot exceed --epochs")
    if args.learning_rate <= 0 or args.low_budget < 0:
        parser.error("learning rate must be positive and low budget non-negative")
    return args


def main() -> None:
    args = parse_args()
    if args.stage in ("check", "all"):
        check_inputs(args)
    if args.stage in ("train", "all"):
        train(args)


if __name__ == "__main__":
    main()
