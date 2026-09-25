from __future__ import annotations

"""Prepare full GEANT 1x data/Oracle and train the original Hattrick.

This is the 1x counterpart of ``full_geant_2x_hattrick_f``.  It deliberately
reuses that runner's data, Oracle, strict-ESM evaluation, and six-loss training
runtime while redirecting every mutable path to an independent 1x workspace.

Fixed ranges: train [0,6000), validation [6000,7500), test [7500,10200).
The trainer never instantiates the test split.  It saves a resumable state each
epoch, archives every epoch, and retains a validation-selected top-K.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import pickle
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
BASE_2X_PATH = TEST_DIR / "full_geant_2x_hattrick_f" / "run_full_experiment.py"

SOURCE_TOPOLOGY = "geant"
TARGET_TOPOLOGY = "geant_full_shared_load1x_train"
DATASET_NAME = "1x"
LOAD_FACTOR = 1.0
NUM_PATHS = 8
TRAIN_RANGE = (0, 6000)
VALIDATION_RANGE = (6000, 7500)
TEST_RANGE = (7500, 10200)
TOTAL_SNAPSHOTS = TEST_RANGE[1]
DEFAULT_EPOCHS = 40
DEFAULT_TOP_K = 5
DEFAULT_SAVE_EVERY = 5
ORACLE_CLUSTER_BASE = 840000
ARTIFACT_ROOT = THIS_DIR / "artifacts"
PREPARE_MARKER = ARTIFACT_ROOT / "prepared_1x.json"
FINAL_RESULT_DIR = ROOT / "results" / TARGET_TOPOLOGY / f"{NUM_PATHS}sp" / "0"
SOURCE_RESULT_DIR = ROOT / "results" / SOURCE_TOPOLOGY / f"{NUM_PATHS}sp" / "0"
ORACLE_REUSE_MARKER = ARTIFACT_ROOT / "reused_source_oracle.json"
LEGACY_HATTRICK_49 = (
    TEST_DIR / "shared1x_full_objectives" / "artifacts" / "level4_confirmation"
    / "seed_490" / "best_model.pt"
)
LEGACY_HATTRICK_TOPOLOGY = "geant_priomask500_shared"


def load_base_2x():
    spec = importlib.util.spec_from_file_location("full_geant_1x_base_infra", BASE_2X_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import complete 2x infrastructure: {BASE_2X_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_2x()


def configure_base() -> None:
    """Redirect the audited complete-data infrastructure to the 1x workspace."""
    base.SOURCE_TOPOLOGY = SOURCE_TOPOLOGY
    base.TARGET_TOPOLOGY = TARGET_TOPOLOGY
    base.NUM_PATHS = NUM_PATHS
    base.LOAD_FACTOR = LOAD_FACTOR
    base.TRAIN_RANGE = TRAIN_RANGE
    base.VALIDATION_RANGE = VALIDATION_RANGE
    base.TEST_RANGE = TEST_RANGE
    base.TOTAL_SNAPSHOTS = TOTAL_SNAPSHOTS
    base.ORACLE_CLUSTER_BASE = ORACLE_CLUSTER_BASE
    base.ARTIFACT_ROOT = ARTIFACT_ROOT
    base.LOG_ROOT = ARTIFACT_ROOT / "logs"
    base.ORACLE_STATE_ROOT = ARTIFACT_ROOT / "oracle_chunks"
    base.PREPARE_MARKER = PREPARE_MARKER
    base.SELECTION_PATH = ARTIFACT_ROOT / "selected_models.json"
    base.FINAL_RESULT_DIR = FINAL_RESULT_DIR
    base._RUNTIMES = None


configure_base()


def load_runtimes():
    return base.load_runtimes()


def build_props(device: torch.device, *, batch_size: int, epochs: int, learning_rate: float):
    return base.build_props(
        device, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate
    )


def test_diagnostics(rows: list[dict]) -> dict:
    return base.test_diagnostics(rows)


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_prepared_data() -> None:
    target_rows = base.manifest_rows(TARGET_TOPOLOGY)
    source_rows = base.manifest_rows(SOURCE_TOPOLOGY)[:TOTAL_SNAPSHOTS]
    if len(target_rows) != TOTAL_SNAPSHOTS or target_rows != source_rows:
        raise RuntimeError("The 1x manifest is not the first 10200 GEANT rows")
    for index in (0, 1, 5999, 6000, 7499, 7500, 10199):
        filename = target_rows[index][2]
        for priority in (1, 2, 3):
            for predicted in (False, True):
                with base.source_tm_path(priority, predicted, filename).open("rb") as handle:
                    source = np.asarray(pickle.load(handle), dtype=np.float64)
                with base.target_tm_path(priority, predicted, filename).open("rb") as handle:
                    target = np.asarray(pickle.load(handle), dtype=np.float64)
                if not np.allclose(target, source * LOAD_FACTOR, rtol=1e-7, atol=1e-10):
                    kind = "esm" if predicted else "actual"
                    raise RuntimeError(
                        f"1x audit failed at snapshot={index}, class={priority}, kind={kind}"
                    )
    source_topology = ROOT / "topologies" / SOURCE_TOPOLOGY / "t1.json"
    target_topology = ROOT / "topologies" / TARGET_TOPOLOGY / "t1.json"
    source_pairs = ROOT / "pairs" / SOURCE_TOPOLOGY / "t1.pkl"
    target_pairs = ROOT / "pairs" / TARGET_TOPOLOGY / "t1.pkl"
    if base.sha256(source_topology) != base.sha256(target_topology):
        raise RuntimeError("Topology changed while preparing the 1x dataset")
    if base.sha256(source_pairs) != base.sha256(target_pairs):
        raise RuntimeError("OD pairs changed while preparing the 1x dataset")
    print("[ok] exact 1x actual+ESM data on all split boundaries", flush=True)


# The reused prepare/oracle functions resolve this symbol dynamically.
base.validate_prepared_data = validate_prepared_data


def source_hashes() -> dict[str, str]:
    paths = {
        "full_geant_1x_runner.py": Path(__file__).resolve(),
        "complete_2x_infrastructure.py": BASE_2X_PATH,
        "six_objective_runtime.py": base.FULL_SOURCE,
        "ordered_projection.py": base.FULL_SOURCE.parent / "ordered_projection.py",
        "hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
        "training_utils.py": ROOT / "utils" / "training_utils.py",
        "dataset.py": ROOT / "utils" / "build_dataset_within_cluster.py",
    }
    return {name: base.sha256(path) for name, path in paths.items()}


def experiment_config(args: argparse.Namespace) -> dict:
    return {
        "method": "Hattrick",
        "dataset": DATASET_NAME,
        "seed": args.seed,
        "topology": TARGET_TOPOLOGY,
        "load_factor": LOAD_FACTOR,
        "strict_esm": True,
        "train": list(TRAIN_RANGE),
        "validation": list(VALIDATION_RANGE),
        "test": list(TEST_RANGE),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "top_k": args.top_k,
        "save_every": args.save_every,
        "ordered_objectives": ["Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml"],
        "selection": (
            "validation only: High per-snapshot NormFulFill >= 0.995-1e-4 and "
            "simulator safety; then Medium mean/P10/P1, High mean, Low mean"
        ),
        "source_sha256": source_hashes(),
    }


def archive_upgrade_compatible(saved: dict, current: dict) -> bool:
    """Allow old runs to resume after the checkpoint-only runner change."""
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
        "full_geant_1x_runner.py":
            "7f75c1d015ccd35bcf42c1bca668341f97c25e2a085e995fb38ddf6347f81135",
    }
    return all(
        saved_sources[key] == current_sources[key]
        or saved_sources[key] == allowed_legacy_hashes.get(key)
        for key in saved_sources
    )


def run_directory(seed: int) -> Path:
    return ARTIFACT_ROOT / "hattrick" / f"seed_{seed}"


def save_epoch_checkpoint(run_dir: Path, payload: dict) -> Path:
    directory = run_dir / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"epoch_{int(payload['epoch']):03d}.pt"
    torch.save(payload, path)
    return path


def saved_epoch_numbers(run_dir: Path) -> list[int]:
    result: list[int] = []
    for path in sorted((run_dir / "checkpoints").glob("epoch_*.pt")):
        try:
            result.append(int(path.stem.removeprefix("epoch_")))
        except ValueError:
            continue
    return result


def load_top_k(run_dir: Path) -> list[dict]:
    path = run_dir / "top5.json"
    if not path.exists():
        return []
    value = base.read_json(path)
    entries = value.get("checkpoints") if isinstance(value, dict) else None
    if not isinstance(entries, list):
        raise RuntimeError(f"Malformed validation checkpoint index: {path}")
    for entry in entries:
        checkpoint = Path(entry["path"])
        if not checkpoint.exists() or base.sha256(checkpoint) != entry["sha256"]:
            raise RuntimeError(f"Validation checkpoint audit failed: {checkpoint}")
    return entries


def update_top_k(run_dir: Path, entries: list[dict], payload: dict, top_k: int) -> list[dict]:
    epoch = int(payload["epoch"])
    combined = [entry for entry in entries if int(entry["epoch"]) != epoch]
    combined.append({"epoch": epoch, "rank": [float(x) for x in payload["rank"]]})
    combined.sort(
        key=lambda item: (tuple(float(x) for x in item["rank"]), -int(item["epoch"])),
        reverse=True,
    )
    kept = combined[:top_k]
    kept_epochs = {int(item["epoch"]) for item in kept}
    top_dir = run_dir / "top5"
    top_dir.mkdir(parents=True, exist_ok=True)
    if epoch in kept_epochs:
        torch.save(payload, top_dir / f"epoch_{epoch:03d}.pt")
    for old in entries:
        old_epoch = int(old["epoch"])
        if old_epoch in kept_epochs:
            continue
        old_path = Path(old["path"]).resolve()
        if old_path.parent != top_dir.resolve() or old_path.suffix != ".pt":
            raise RuntimeError(f"Refusing unsafe checkpoint removal: {old_path}")
        if old_path.exists():
            old_path.unlink()
    audited: list[dict] = []
    for item in kept:
        kept_epoch = int(item["epoch"])
        path = top_dir / f"epoch_{kept_epoch:03d}.pt"
        if not path.exists():
            raise RuntimeError(f"Missing retained checkpoint: {path}")
        audited.append({
            "epoch": kept_epoch,
            "rank": [float(x) for x in item["rank"]],
            "eligible": bool(float(item["rank"][0])),
            "path": str(path.resolve()),
            "sha256": base.sha256(path),
        })
    base.write_json(run_dir / "top5.json", {
        "selection_used_test": False,
        "dataset": DATASET_NAME,
        "validation": list(VALIDATION_RANGE),
        "checkpoints": audited,
    })
    base.atomic_copy(Path(audited[0]["path"]), run_dir / "best_model.pt")
    return audited


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def require_training_inputs() -> None:
    if not PREPARE_MARKER.exists() or not base.prepared_file_counts_ok():
        raise RuntimeError("Run --stage prepare before training the 1x model")
    validate_prepared_data()
    if not base.final_oracle_complete():
        raise RuntimeError("Run --stage oracle before training the 1x model")


def _source_oracle_lines(filename: str) -> list[str]:
    path = SOURCE_RESULT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing reusable 1x Oracle file: {path}")
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = [line.rstrip("\r\n") for line in handle if line.strip()]
    if len(lines) < TOTAL_SNAPSHOTS:
        raise RuntimeError(
            f"Reusable 1x Oracle is incomplete: {path} has {len(lines)} rows"
        )
    return lines[:TOTAL_SNAPSHOTS]


def reuse_source_oracle() -> None:
    """Reuse the exact source GEANT Oracle because 1x applies no scaling."""
    if not PREPARE_MARKER.exists():
        raise RuntimeError("Run --stage prepare before --stage oracle")
    validate_prepared_data()
    filenames = _source_oracle_lines("filenames.txt")
    expected = [",".join(row) for row in base.manifest_rows(TARGET_TOPOLOGY)]
    if filenames != expected:
        raise RuntimeError(
            "The reusable source Oracle does not match the 1x target manifest"
        )
    if base.final_oracle_complete() and ORACLE_REUSE_MARKER.exists():
        print("[skip] audited 1x source Oracle is already installed", flush=True)
        return

    FINAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    source_hashes_by_file: dict[str, str] = {}
    target_hashes_by_file: dict[str, str] = {}
    for filename in (*base.ORACLE_FILES, "filenames.txt"):
        source = SOURCE_RESULT_DIR / filename
        lines = filenames if filename == "filenames.txt" else _source_oracle_lines(filename)
        temporary = FINAL_RESULT_DIR / f".{filename}.{os.getpid()}.tmp"
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines) + "\n")
        os.replace(temporary, FINAL_RESULT_DIR / filename)
        source_hashes_by_file[filename] = base.sha256(source)
        target_hashes_by_file[filename] = base.sha256(FINAL_RESULT_DIR / filename)

    if not base.final_oracle_complete():
        raise RuntimeError("The copied 1x Oracle failed the 10200-row audit")
    base.write_json(ORACLE_REUSE_MARKER, {
        "dataset": DATASET_NAME,
        "load_factor": LOAD_FACTOR,
        "source_result_dir": str(SOURCE_RESULT_DIR.resolve()),
        "target_result_dir": str(FINAL_RESULT_DIR.resolve()),
        "rows": TOTAL_SNAPSHOTS,
        "filenames_match_manifest": True,
        "source_sha256": source_hashes_by_file,
        "target_sha256": target_hashes_by_file,
    })
    print(f"[done] reused exact 1x Oracle in {FINAL_RESULT_DIR}", flush=True)


def check(args: argparse.Namespace) -> None:
    source_manifest = ROOT / "manifest" / f"{SOURCE_TOPOLOGY}_manifest.txt"
    source_rows = base.manifest_rows(SOURCE_TOPOLOGY)
    report = {
        "status": "READY" if len(source_rows) >= TOTAL_SNAPSHOTS else "BLOCKED",
        "dataset": DATASET_NAME,
        "target_topology": TARGET_TOPOLOGY,
        "load_factor": LOAD_FACTOR,
        "source_manifest": str(source_manifest),
        "source_snapshots": len(source_rows),
        "train": list(TRAIN_RANGE),
        "validation": list(VALIDATION_RANGE),
        "test": list(TEST_RANGE),
        "prepared": PREPARE_MARKER.exists() and base.prepared_file_counts_ok(),
        "oracle_complete": base.final_oracle_complete(),
        "epochs": args.epochs,
        "save_every": args.save_every,
        "checkpoint_archive_policy": "every_epoch",
        "top_k": args.top_k,
        "cuda_available": torch.cuda.is_available(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if report["status"] != "READY":
        raise RuntimeError("The source GEANT manifest has fewer than 10200 snapshots")


def train(args: argparse.Namespace) -> None:
    require_training_inputs()
    shared, full, _hattrick_f = base.load_runtimes()
    from frameworks.hattrick_system import Hattrick
    from utils.AdamOptimizer import ADAMOptimizer

    run_dir = run_directory(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    current_config = experiment_config(args)
    config_path = run_dir / "config.json"
    resume_path = run_dir / "resume_state.pt"
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
            raise RuntimeError(f"Existing 1x Hattrick configuration differs: {config_path}")
    else:
        config = current_config
        base.write_json(config_path, config)
    config_hash = canonical_json_sha256(config)
    if resume_path.exists() and not args.resume:
        raise RuntimeError(f"Existing state found: {resume_path}; use --resume")
    if args.resume and not resume_path.exists():
        raise FileNotFoundError(f"--resume requested but no state exists: {resume_path}")

    device = resolve_device(args.device)
    base.set_seed(args.seed)
    props = base.build_props(
        device, batch_size=args.batch_size, epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    train_dataset, validation_dataset, test_dataset = base.make_datasets(props)
    if test_dataset is not None:
        raise RuntimeError("Training unexpectedly instantiated the test split")
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)
    history_path = run_dir / "train_history.csv"
    history = base.read_csv(history_path)
    top_entries = load_top_k(run_dir)
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        if checkpoint.get("method") != "Hattrick" or checkpoint.get("dataset") != DATASET_NAME:
            raise RuntimeError("Resume-state method/dataset mismatch")
        if int(checkpoint.get("seed", -1)) != args.seed:
            raise RuntimeError("Resume-state seed mismatch")
        if checkpoint.get("config") != config or checkpoint.get("config_sha256") != config_hash:
            raise RuntimeError("Resume-state configuration mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"[resume] 1x Hattrick at epoch {start_epoch}", flush=True)

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        base.set_seed(args.seed + epoch * 104729)
        loader = shared.data_loader(
            train_dataset, props.batch_size, True, args.seed + epoch * 1009
        )
        train_metrics = full.train_epoch(model, props, train_dataset, loader, optimizer)
        rows, summary, diagnostics = shared.evaluate(
            model, props, validation_dataset, VALIDATION_RANGE[0]
        )
        rank = base.hattrick_rank(rows, summary)
        base.save_validation(run_dir, epoch, rows, summary, diagnostics)
        row = base.history_row(epoch, rank, rows, summary, diagnostics, train_metrics)
        history = [item for item in history if int(item["epoch"]) < epoch]
        history.append(row)
        base.write_csv(history_path, history)
        payload = {
            "method": "Hattrick",
            "dataset": DATASET_NAME,
            "seed": args.seed,
            "epoch": epoch,
            "rank": tuple(float(x) for x in rank),
            "config": config,
            "config_sha256": config_hash,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_summary": summary,
            "validation_diagnostics": diagnostics,
        }
        torch.save(payload, resume_path)
        archive = save_epoch_checkpoint(run_dir, payload)
        top_entries = update_top_k(run_dir, top_entries, payload, args.top_k)
        print(
            f"[1x Hattrick] epoch={epoch}/{args.epochs} eligible={rank[0]} "
            f"H={row['high_norm_mean']:.6f} Hmin={row['high_norm_min']:.6f} "
            f"M={row['medium_norm_mean']:.6f} L={row['low_norm_mean']:.6f} "
            f"archive={archive.name if archive else '-'} "
            f"best{args.top_k}={[int(x['epoch']) for x in top_entries]}",
            flush=True,
        )
    if len(top_entries) != args.top_k:
        raise RuntimeError(f"Expected {args.top_k} best checkpoints, found {len(top_entries)}")
    complete = {
        "status": "COMPLETE",
        "method": "Hattrick",
        "dataset": DATASET_NAME,
        "selection_used_test": False,
        "seed": args.seed,
        "epochs": args.epochs,
        "save_every": args.save_every,
        "checkpoint_archive_policy": "every_epoch",
        "top_k": args.top_k,
        "best_checkpoints": top_entries,
        "selected_epoch": int(top_entries[0]["epoch"]),
        "saved_epoch_checkpoints": saved_epoch_numbers(run_dir),
        "runtime_seconds_this_invocation": time.perf_counter() - started,
        "resume_state": str(resume_path.resolve()),
        "resume_state_sha256": base.sha256(resume_path),
        "config_sha256": config_hash,
    }
    base.write_json(run_dir / "complete.json", complete)
    print(json.dumps(complete, indent=2, ensure_ascii=False), flush=True)


def adopt_hattrick_49(args: argparse.Namespace) -> None:
    """Register the existing six-loss epoch-49 weights without retraining them."""
    if args.seed != 490:
        raise RuntimeError("The existing Hattrick-49 checkpoint belongs to seed 490")
    require_training_inputs()
    source = LEGACY_HATTRICK_49.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Missing existing Hattrick-49 checkpoint: {source}")
    original = torch.load(source, map_location="cpu", weights_only=False)
    if int(original.get("epoch", -1)) != 49:
        raise RuntimeError("The requested legacy checkpoint is not epoch 49")
    original_config = original.get("config")
    if not isinstance(original_config, dict):
        raise RuntimeError("Hattrick-49 has no auditable source configuration")
    expected_objectives = ["Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml"]
    if (
        original_config.get("topology") != LEGACY_HATTRICK_TOPOLOGY
        or float(original_config.get("load_factor", -1)) != LOAD_FACTOR
        or int(original_config.get("paths_per_pair", -1)) != NUM_PATHS
        or original_config.get("objectives") != expected_objectives
    ):
        raise RuntimeError("Hattrick-49 source protocol is not the expected six-loss 1x run")

    old_topology = ROOT / "topologies" / LEGACY_HATTRICK_TOPOLOGY / "t1.json"
    new_topology = ROOT / "topologies" / TARGET_TOPOLOGY / "t1.json"
    old_pairs = ROOT / "pairs" / LEGACY_HATTRICK_TOPOLOGY / "t1.pkl"
    new_pairs = ROOT / "pairs" / TARGET_TOPOLOGY / "t1.pkl"
    if base.sha256(old_topology) != base.sha256(new_topology):
        raise RuntimeError("Hattrick-49 topology differs from the full 1x workspace")
    if base.sha256(old_pairs) != base.sha256(new_pairs):
        raise RuntimeError("Hattrick-49 OD pairs differ from the full 1x workspace")

    base.load_runtimes()
    from frameworks.hattrick_system import Hattrick

    props = build_props(
        torch.device("cpu"), batch_size=args.batch_size,
        epochs=args.epochs, learning_rate=args.learning_rate,
    )
    model = Hattrick(props).to(device="cpu", dtype=props.dtype)
    model.load_state_dict(original["model_state_dict"], strict=True)

    source_sha256 = base.sha256(source)
    adopted_config = {
        "method": "Hattrick",
        "dataset": DATASET_NAME,
        "seed": args.seed,
        "topology": TARGET_TOPOLOGY,
        "load_factor": LOAD_FACTOR,
        "strict_esm": True,
        "train": list(TRAIN_RANGE),
        "validation": list(VALIDATION_RANGE),
        "test": list(TEST_RANGE),
        "epoch": 49,
        "adopted_weights_not_retrained": True,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_sha256,
        "source_training_protocol": original_config,
        "compatibility_audit": {
            "state_dict_strict": True,
            "parameters": len(original["model_state_dict"]),
            "topology_sha256": base.sha256(new_topology),
            "pairs_sha256": base.sha256(new_pairs),
        },
    }
    config_hash = canonical_json_sha256(adopted_config)
    payload = {
        "method": "Hattrick",
        "dataset": DATASET_NAME,
        "seed": args.seed,
        "epoch": 49,
        "rank": tuple(float(value) for value in original.get("rank", ())),
        "config": adopted_config,
        "config_sha256": config_hash,
        "model_state_dict": copy.deepcopy(original["model_state_dict"]),
        "optimizer_state_dict": original.get("optimizer_state_dict"),
        "adoption": {
            "weights_changed": False,
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": source_sha256,
            "source_training_snapshots": 500,
            "registered_inference_snapshots": TOTAL_SNAPSHOTS,
        },
    }
    run_dir = run_directory(args.seed)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "epoch_049.pt"
    if checkpoint.exists():
        existing = torch.load(checkpoint, map_location="cpu", weights_only=False)
        adoption = existing.get("adoption", {})
        if (
            existing.get("method") != "Hattrick"
            or existing.get("dataset") != DATASET_NAME
            or int(existing.get("epoch", -1)) != 49
            or adoption.get("source_checkpoint_sha256") != source_sha256
        ):
            raise RuntimeError(f"Existing adopted checkpoint differs: {checkpoint}")
    else:
        torch.save(payload, checkpoint)
    checkpoint_sha256 = base.sha256(checkpoint)
    base.atomic_copy(checkpoint, run_dir / "best_model.pt")
    base.write_json(run_dir / "config.json", adopted_config)
    history_path = run_dir / "train_history.csv"
    validation_summary_path = run_dir / "validation_epoch_049_summary.json"
    if not history_path.exists() or not validation_summary_path.exists():
        shared, _full, _hattrick_f = base.load_runtimes()
        from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster

        device = resolve_device(args.device)
        eval_props = build_props(
            device, batch_size=args.batch_size,
            epochs=args.epochs, learning_rate=args.learning_rate,
        )
        validation_dataset = DM_Dataset_within_Cluster(
            eval_props, 0, *VALIDATION_RANGE
        )
        if int(validation_dataset.max_source_index_read) != VALIDATION_RANGE[1] - 1:
            raise RuntimeError("Adopted Hattrick-49 validation split audit failed")
        eval_model = Hattrick(eval_props).to(device=device, dtype=eval_props.dtype)
        eval_model.load_state_dict(original["model_state_dict"], strict=True)
        rows, validation_summary, validation_diagnostics = shared.evaluate(
            eval_model, eval_props, validation_dataset, VALIDATION_RANGE[0]
        )
        validation_rank = base.hattrick_rank(rows, validation_summary)
        base.save_validation(
            run_dir, 49, rows, validation_summary, validation_diagnostics
        )
        base.write_csv(history_path, [base.history_row(
            49, validation_rank, rows, validation_summary,
            validation_diagnostics, {"training_status": "adopted_not_retrained"},
        )])
        print(
            "[validate] evaluated adopted Hattrick-49 on full 1x validation "
            "snapshots [6000,7500)",
            flush=True,
        )
    entry = {
        "epoch": 49,
        "rank": [float(value) for value in payload["rank"]],
        "eligible": bool(payload["rank"] and payload["rank"][0] >= 1.0),
        "path": str(checkpoint.resolve()),
        "sha256": checkpoint_sha256,
        "adopted": True,
    }
    base.write_json(run_dir / "top5.json", {
        "selection_used_test": False,
        "selection_protocol": "legacy validation [350,400); user-pinned epoch 49",
        "dataset": DATASET_NAME,
        "validation": list(original_config["validation"]),
        "checkpoints": [entry],
    })
    base.write_json(run_dir / "complete.json", {
        "status": "ADOPTED",
        "method": "Hattrick",
        "dataset": DATASET_NAME,
        "seed": args.seed,
        "selected_epoch": 49,
        "selection_used_test": False,
        "weights_changed": False,
        "source_training_snapshots": 500,
        "registered_inference_snapshots": TOTAL_SNAPSHOTS,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_sha256,
        "config_sha256": config_hash,
    })
    print(json.dumps({
        "status": "ADOPTED",
        "model": "Hattrick-49",
        "weights_changed": False,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
    }, indent=2, ensure_ascii=False), flush=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full GEANT 1x data, Oracle, and Hattrick")
    parser.add_argument(
        "--stage", choices=("check", "prepare", "oracle", "adopt-49", "train", "all"),
        default="check",
    )
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--epochs", type=positive_int, default=DEFAULT_EPOCHS)
    parser.add_argument("--top-k", type=positive_int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--save-every", type=positive_int, default=DEFAULT_SAVE_EVERY,
        help="Legacy configuration field; independent checkpoints are archived every epoch.",
    )
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--oracle-chunk-size", type=positive_int, default=256)
    parser.add_argument("--oracle-workers", type=positive_int, default=1)
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.top_k > args.epochs:
        parser.error("--top-k cannot exceed --epochs")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    return args


def main() -> None:
    args = parse_args()
    check(args)
    if args.stage in ("prepare", "all"):
        base.prepare_data(force=args.force_prepare)
    if args.stage in ("oracle", "all"):
        reuse_source_oracle()
    if args.stage == "adopt-49":
        adopt_hattrick_49(args)
    if args.stage in ("train", "all"):
        train(args)


if __name__ == "__main__":
    main()
