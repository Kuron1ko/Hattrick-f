from __future__ import annotations

"""Full GEANT 2x experiment for Hattrick-f3.

Hattrick-f3 starts from the validation-selected six-loss Hattrick checkpoint.
It smoothly releases the MLU objectives by blending two ordered-projection
updates computed from the same batch:

    g_f3 = alpha * g_six + (1 - alpha) * g_flow

``g_six`` uses Fh -> Uh -> Fhm -> Uhm -> Fhml -> Uhml, while ``g_flow`` uses
Fh -> Fhm -> Fhml. Alpha follows a cosine schedule from 1 to 0 during the
first 12 epochs and stays at 0 afterwards. Scaling an MLU loss directly is not
used because even a small nonzero gradient would still alter the projection
basis and would therefore not provide a smooth priority release.

Fixed half-open snapshot ranges:
    train      [0, 6000)
    validation [6000, 7500)
    test       [7500, 10200)

Every epoch is retained as a complete, resumable checkpoint. Validation results
still determine the separate top-five index, and the test split is first
instantiated after training and top-five selection are complete.
"""

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
BASE_RUNNER_PATH = (
    TEST_DIR / "full_geant_2x_hattrick_f" / "run_full_experiment.py"
)
DEFAULT_PHASE_A = (
    TEST_DIR
    / "full_geant_2x_hattrick_f"
    / "artifacts"
    / "hattrick"
    / "seed_490"
    / "best_model.pt"
)
ARTIFACT_ROOT = THIS_DIR / "artifacts"

TRAIN_RANGE = (0, 6000)
VALIDATION_RANGE = (6000, 7500)
TEST_RANGE = (7500, 10200)
OBJECTIVE_NAMES = ("Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml")
FLOW_OBJECTIVE_NAMES = ("Fh", "Fhm", "Fhml")
FLOW_OBJECTIVE_INDICES = (0, 2, 4)
ZERO_TOLERANCE = 1e-12
DEFAULT_EPOCHS = 40
DEFAULT_ANNEAL_EPOCHS = 12
DEFAULT_TOP_K = 5


# 加载模型，在BASE_RUNNER_PATH的基础上训练Hattrick
def load_base_runner():
    spec = importlib.util.spec_from_file_location(
        "full_geant_2x_hattrick_f3_base", BASE_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import base runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_runner()

# 计算退火的参数 a =  ( 1/2 ) * ( 1 + cos( (e-1) / (T-1) * pi ) )
def anneal_alpha(epoch: int, anneal_epochs: int) -> float:
    """Front-loaded cosine MLU release; zero after the release window."""
    if anneal_epochs < 2:
        raise ValueError("anneal_epochs must be at least 2")
    if epoch >= anneal_epochs:
        return 0.0
    progress = float(epoch - 1) / float(anneal_epochs - 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

# 投影

def ordered_projection_from_raw(
    raw_gradients: list[torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Apply the same ordered null-space projection to cached raw gradients."""
    basis: list[torch.Tensor] = []
    projected: list[torch.Tensor] = []
    for raw in raw_gradients:
        work = raw.to(dtype=torch.float64)
        raw_norm = torch.linalg.vector_norm(work)
        if not torch.isfinite(raw_norm).item():
            raise RuntimeError("Non-finite raw Hattrick-f3 gradient")
        if float(raw_norm.item()) <= ZERO_TOLERANCE:
            projected.append(torch.zeros_like(raw))
            continue
        for _ in range(2):
            for basis_vector in basis:
                work = work - torch.dot(work, basis_vector) * basis_vector
        projected_norm = torch.linalg.vector_norm(work)
        if not torch.isfinite(projected_norm).item():
            raise RuntimeError("Non-finite projected Hattrick-f3 gradient")
        if float(projected_norm.item()) <= ZERO_TOLERANCE:
            projected.append(torch.zeros_like(raw))
            continue
        basis.append(work / projected_norm)
        projected.append(work.to(dtype=raw.dtype))
    final = torch.stack(projected, dim=0).sum(dim=0)
    if not torch.isfinite(final).all().item():
        raise RuntimeError("Non-finite flow-only Hattrick-f3 update")
    return projected, final

# 计算两个向量的余弦相似度， 用来观察 g1 和 g2 的相似程度，不参与更新

def gradient_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(left.double()) * torch.linalg.vector_norm(
        right.double()
    )
    if float(denominator.item()) <= 1e-20:
        return 0.0
    return float(torch.dot(left.double(), right.double()).item() / denominator.item())

# 完成Hattrick-f3的一个epoch训练

def train_epoch(
    *,
    shared,
    full,
    model,
    props,
    dataset,
    loader,
    optimizer,
    alpha: float,
) -> dict[str, float]:
    model.train()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    props.research_return_policy = False
    path_masks = shared.base.move_dataset_static(dataset, props.device)
    totals: dict[str, float] = {}
    first_probe: dict[str, float] = {}
    count = 0

    for batch_index, inputs in enumerate(loader):
        values = shared.unpack_to_device(inputs, props)
        losses, reported = full.build_objectives(
            model, props, dataset, values, path_masks
        )
        if any(not torch.isfinite(loss).item() for loss in losses):
            raise RuntimeError("Non-finite Hattrick-f3 objective")

        six = full.ordered_project_gradients(model, losses)
        flow_raw = [six.raw_gradients[index] for index in FLOW_OBJECTIVE_INDICES]
        flow_projected, flow_update = ordered_projection_from_raw(flow_raw)
        final_update = float(alpha) * six.final_gradient + (
            1.0 - float(alpha)
        ) * flow_update
        if not torch.isfinite(final_update).all().item():
            raise RuntimeError("Non-finite annealed Hattrick-f3 update")

        if batch_index == 0:
            first_probe = full.projection_diagnostics(six, OBJECTIVE_NAMES)
            for name, gradient in zip(FLOW_OBJECTIVE_NAMES, flow_projected):
                first_probe[f"flow_projected_gradient_norm_{name}"] = float(
                    torch.linalg.vector_norm(gradient).item()
                )

        full.assign_gradients_and_step(
            model,
            final_update,
            optimizer,
            six.parameter_shapes,
        )
        diagnostics = {
            f"reported_{name}": float(value)
            for name, value in zip(OBJECTIVE_NAMES, reported)
        }
        diagnostics.update(
            {
                "anneal_alpha": float(alpha),
                "six_update_norm": float(
                    torch.linalg.vector_norm(six.final_gradient).item()
                ),
                "flow_update_norm": float(
                    torch.linalg.vector_norm(flow_update).item()
                ),
                "annealed_update_norm": float(
                    torch.linalg.vector_norm(final_update).item()
                ),
                "six_vs_flow_update_cosine": gradient_cosine(
                    six.final_gradient, flow_update
                ),
            }
        )
        for name, value in diagnostics.items():
            totals[name] = totals.get(name, 0.0) + value
        count += 1

    props.research_return_admitted = False
    result = {name: value / max(count, 1) for name, value in totals.items()}
    result.update(first_probe)
    result["train_batches"] = count
    return result


def source_hashes() -> dict[str, str]:
    paths = {
        "hattrick_f3_runner.py": Path(__file__).resolve(),
        "full_geant_base_runner.py": BASE_RUNNER_PATH,
        "six_objective_runtime.py": base.FULL_SOURCE,
        "ordered_projection.py": base.FULL_SOURCE.parent / "ordered_projection.py",
        "hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
        "training_utils.py": ROOT / "utils" / "training_utils.py",
        "dataset.py": ROOT / "utils" / "build_dataset_within_cluster.py",
    }
    return {name: base.sha256(path) for name, path in paths.items()}

# 当前实验的配置参数

def experiment_config(args: argparse.Namespace, phase_a: Path) -> dict:
    return {
        "method": "Hattrick-f3",
        "seed": args.seed,
        "topology": base.TARGET_TOPOLOGY,
        "load_factor": 2.0,
        "strict_esm": True,
        "train": list(TRAIN_RANGE),
        "validation": list(VALIDATION_RANGE),
        "test": list(TEST_RANGE),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "low_budget": args.low_budget,
        "top_k": args.top_k,
        "phase_a_checkpoint": str(phase_a.resolve()),
        "phase_a_sha256": base.sha256(phase_a),
        "optimizer_reset": True,
        "all_parameters_trainable": True,
        "annealing": {
            "quantity": "ordered projected update",
            "formula": "alpha*g_six + (1-alpha)*g_flow",
            "six_objectives": list(OBJECTIVE_NAMES),
            "flow_objectives": list(FLOW_OBJECTIVE_NAMES),
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


# The active run was started before registry-backed inference was added to the
# base runner. These two audited hashes differ only in experiment orchestration;
# the model, objectives, projection, data, and optimizer sources are unchanged.
AUDITED_RESUME_SOURCE_HASHES = {
    "hattrick_f3_runner.py": {
        "4bfa0409f6cff3b75c597cc10d5fa3ed16adee28321fbb24d36a748c5fb9c86f",
        "0e0df4b0c0a437dc16a8b484eb982c9571adac370b083bb297114c620b321c54",
    },
    "full_geant_base_runner.py": {
        "a25983d966fb39bb7c54f90c5aee63ea556e14538ca71a2eca5a240174450fd5",
        "5d227738fdd7addb7652845054d1dc8468baae4715f7621674b895061b60d5d3",
    },
}

# 判断能否在新参数下继续训练，可以用完全相同的配置，或者修改epoch

def resume_config_compatible(saved: dict, requested: dict) -> bool:
    if saved == requested:
        return True
    saved_copy = copy.deepcopy(saved)
    requested_copy = copy.deepcopy(requested)
    saved_epochs = saved_copy.pop("epochs", None)
    requested_epochs = requested_copy.pop("epochs", None)
    if not isinstance(saved_epochs, int) or not isinstance(requested_epochs, int):
        return False
    if requested_epochs < saved_epochs:
        return False
    # Increasing only the total epoch target is a normal continuation and does
    # not change the optimizer, data, objective, annealing, or model contract.
    if saved_copy == requested_copy:
        return True
    saved_sources = saved_copy.get("source_sha256")
    requested_sources = requested_copy.get("source_sha256")
    if not isinstance(saved_sources, dict) or not isinstance(requested_sources, dict):
        return False
    for name, allowed_saved_hashes in AUDITED_RESUME_SOURCE_HASHES.items():
        saved_hash = saved_sources.pop(name, None)
        requested_sources.pop(name, None)
        if saved_hash not in allowed_saved_hashes:
            return False
    return saved_copy == requested_copy


def run_directory(seed: int) -> Path:
    return ARTIFACT_ROOT / f"seed_{seed}"


def save_epoch_checkpoint(run_dir: Path, payload: dict) -> Path:
    """Persist one complete training state without pruning older epochs."""
    epoch = int(payload["epoch"])
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    torch.save(payload, path)
    return path


def saved_epoch_numbers(run_dir: Path) -> list[int]:
    epochs: list[int] = []
    for path in sorted((run_dir / "checkpoints").glob("epoch_*.pt")):
        try:
            epochs.append(int(path.stem.removeprefix("epoch_")))
        except ValueError:
            continue
    return epochs


def load_top_k(run_dir: Path) -> list[dict]:
    index_path = run_dir / "top5.json"
    if not index_path.exists():
        return []
    value = base.read_json(index_path)
    if not isinstance(value, dict) or not isinstance(value.get("checkpoints"), list):
        raise RuntimeError(f"Malformed top-five index: {index_path}")
    entries = value["checkpoints"]
    for entry in entries:
        path = Path(entry["path"])
        if not path.exists() or base.sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Top-five checkpoint audit failed: {path}")
    return entries


def update_top_k(
    *,
    run_dir: Path,
    entries: list[dict],
    payload: dict,
    top_k: int,
) -> list[dict]:
    epoch = int(payload["epoch"])
    rank = tuple(float(value) for value in payload["rank"])
    candidate = {"epoch": epoch, "rank": list(rank)}
    combined = [entry for entry in entries if int(entry["epoch"]) != epoch]
    combined.append(candidate)
    combined.sort(
        key=lambda entry: (tuple(float(value) for value in entry["rank"]), -int(entry["epoch"])),
        reverse=True,
    )
    kept = combined[:top_k]
    kept_epochs = {int(entry["epoch"]) for entry in kept}
    top_dir = run_dir / "top5"
    top_dir.mkdir(parents=True, exist_ok=True)

    if epoch in kept_epochs:
        checkpoint_path = top_dir / f"epoch_{epoch:03d}.pt"
        torch.save(payload, checkpoint_path)

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
    for entry in kept:
        kept_epoch = int(entry["epoch"])
        path = top_dir / f"epoch_{kept_epoch:03d}.pt"
        if not path.exists():
            raise RuntimeError(f"Missing retained checkpoint: {path}")
        audited.append(
            {
                "epoch": kept_epoch,
                "rank": [float(value) for value in entry["rank"]],
                "eligible": bool(float(entry["rank"][0])),
                "path": str(path.resolve()),
                "sha256": base.sha256(path),
            }
        )
    base.write_json(
        run_dir / "top5.json",
        {
            "selection_used_test": False,
            "validation": list(VALIDATION_RANGE),
            "checkpoints": audited,
        },
    )
    base.atomic_copy(Path(audited[0]["path"]), run_dir / "best_model.pt")
    return audited


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def check_inputs(args: argparse.Namespace) -> None:
    phase_a = args.phase_a_checkpoint.resolve()
    if not phase_a.exists():
        raise FileNotFoundError(f"Missing Phase-A Hattrick checkpoint: {phase_a}")
    if not base.final_oracle_complete():
        raise RuntimeError(
            "The complete 2x oracle is missing; finish the existing prepare/oracle stage"
        )
    checkpoint = torch.load(phase_a, map_location="cpu", weights_only=False)
    if checkpoint.get("method") != "Hattrick":
        raise RuntimeError(f"Phase-A checkpoint is not Hattrick: {phase_a}")
    report = {
        "status": "READY",
        "method": "Hattrick-f3",
        "phase_a_epoch": int(checkpoint["epoch"]),
        "phase_a_sha256": base.sha256(phase_a),
        "train": list(TRAIN_RANGE),
        "validation": list(VALIDATION_RANGE),
        "test": list(TEST_RANGE),
        "epochs": args.epochs,
        "anneal_epochs": args.anneal_epochs,
        "top_k": args.top_k,
        "save_all_epoch_checkpoints": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


def train(args: argparse.Namespace) -> None:
    check_inputs(args)
    phase_a_path = args.phase_a_checkpoint.resolve()
    run_dir = run_directory(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    config = experiment_config(args, phase_a_path)
    config_path = run_dir / "config.json"
    if config_path.exists():
        previous = base.read_json(config_path)
        if not resume_config_compatible(previous, config):
            raise RuntimeError(
                f"Existing Hattrick-f3 configuration differs: {config_path}"
            )
        if previous != config:
            base.write_json(config_path, config)
    else:
        base.write_json(config_path, config)

    shared, full, _hattrick_f = base.load_runtimes()
    from frameworks.hattrick_system import Hattrick
    from utils.AdamOptimizer import ADAMOptimizer

    device = resolve_device(args.device)
    base.set_seed(args.seed)
    props = base.build_props(
        device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    train_dataset, validation_dataset, _test = base.make_datasets(props)

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

    final_path = run_dir / "resume_state.pt"
    history_path = run_dir / "train_history.csv"
    history = base.read_csv(history_path)
    top_entries = load_top_k(run_dir)
    start_epoch = 1
    if final_path.exists():
        checkpoint = torch.load(final_path, map_location=device, weights_only=False)
        if not resume_config_compatible(checkpoint.get("config", {}), config):
            raise RuntimeError("Resume-state Hattrick-f3 configuration mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        resume_epoch_path = save_epoch_checkpoint(run_dir, checkpoint)
        if int(checkpoint["epoch"]) not in {
            int(entry["epoch"]) for entry in top_entries
        }:
            top_entries = update_top_k(
                run_dir=run_dir,
                entries=top_entries,
                payload=checkpoint,
                top_k=args.top_k,
            )
        print(
            f"[resume] Hattrick-f3 at epoch {start_epoch}; "
            f"retained {resume_epoch_path.name}",
            flush=True,
        )

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        base.set_seed(args.seed + epoch * 104729)
        alpha = anneal_alpha(epoch, args.anneal_epochs)
        loader = shared.data_loader(
            train_dataset,
            props.batch_size,
            True,
            args.seed + epoch * 1009,
        )
        train_metrics = train_epoch(
            shared=shared,
            full=full,
            model=model,
            props=props,
            dataset=train_dataset,
            loader=loader,
            optimizer=optimizer,
            alpha=alpha,
        )
        rows, summary, diagnostics = shared.evaluate(
            model, props, validation_dataset, VALIDATION_RANGE[0]
        )
        rank = base.hattrick_f_rank(
            rows, summary, baseline_summary, args.low_budget
        )
        base.save_validation(run_dir, epoch, rows, summary, diagnostics)
        row = base.history_row(
            epoch, rank, rows, summary, diagnostics, train_metrics
        )
        history = [item for item in history if int(item["epoch"]) < epoch]
        history.append(row)
        base.write_csv(history_path, history)

        payload = {
            "method": "Hattrick-f3",
            "seed": args.seed,
            "epoch": epoch,
            "anneal_alpha": alpha,
            "rank": tuple(float(value) for value in rank),
            "config": config,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_summary": summary,
            "validation_diagnostics": diagnostics,
        }
        epoch_path = save_epoch_checkpoint(run_dir, payload)
        base.atomic_copy(epoch_path, final_path)
        top_entries = update_top_k(
            run_dir=run_dir,
            entries=top_entries,
            payload=payload,
            top_k=args.top_k,
        )
        retained = [int(entry["epoch"]) for entry in top_entries]
        print(
            f"[Hattrick-f3] epoch={epoch}/{args.epochs} alpha={alpha:.4f} "
            f"eligible={rank[0]} H={row['high_norm_mean']:.6f} "
            f"Hmin={row['high_norm_min']:.6f} M={row['medium_norm_mean']:.6f} "
            f"L={row['low_norm_mean']:.6f} saved={epoch_path.name} "
            f"top5={retained}",
            flush=True,
        )

    if not top_entries:
        raise RuntimeError("Hattrick-f3 produced no retained checkpoints")
    complete = {
        "status": "COMPLETE",
        "method": "Hattrick-f3",
        "selection_used_test": False,
        "seed": args.seed,
        "epochs": args.epochs,
        "anneal_epochs": args.anneal_epochs,
        "top5": top_entries,
        "selected_epoch": int(top_entries[0]["epoch"]),
        "saved_epoch_checkpoints": saved_epoch_numbers(run_dir),
        "checkpoint_directory": str((run_dir / "checkpoints").resolve()),
        "runtime_seconds_this_invocation": time.perf_counter() - started,
        "resume_state": str(final_path.resolve()),
        "resume_state_sha256": base.sha256(final_path),
    }
    base.write_json(run_dir / "complete.json", complete)
    print(json.dumps(complete, indent=2, ensure_ascii=False), flush=True)


def checkpoint_metadata(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = {
        "path": str(path.resolve()),
        "sha256": base.sha256(path),
        "method": checkpoint["method"],
        "seed": int(checkpoint["seed"]),
        "epoch": int(checkpoint["epoch"]),
        "rank": [float(value) for value in checkpoint["rank"]],
    }
    if "anneal_alpha" in checkpoint:
        metadata["anneal_alpha"] = float(checkpoint["anneal_alpha"])
    return metadata


def test(args: argparse.Namespace) -> None:
    check_inputs(args)
    run_dir = run_directory(args.seed)
    complete_path = run_dir / "complete.json"
    if not complete_path.exists():
        raise FileNotFoundError("Train Hattrick-f3 before running the test stage")
    complete = base.read_json(complete_path)
    entries = load_top_k(run_dir)
    if len(entries) != args.top_k:
        raise RuntimeError(
            f"Expected {args.top_k} retained checkpoints, found {len(entries)}"
        )
    if int(complete["selected_epoch"]) != int(entries[0]["epoch"]):
        raise RuntimeError("Selected Hattrick-f3 checkpoint audit failed")

    shared, _full, _hattrick_f = base.load_runtimes()
    from frameworks.hattrick_system import Hattrick
    from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster

    device = resolve_device(args.device)
    props = base.build_props(
        device,
        batch_size=args.batch_size,
        epochs=1,
        learning_rate=args.learning_rate,
    )
    test_dataset = DM_Dataset_within_Cluster(props, 0, *TEST_RANGE)
    if int(test_dataset.max_source_index_read) != TEST_RANGE[1] - 1:
        raise RuntimeError("Test split audit failed")

    def evaluate(path: Path) -> tuple[list[dict], dict]:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = Hattrick(props).to(device=device, dtype=props.dtype)
        model.load_state_dict(checkpoint["model_state_dict"])
        rows, summary, diagnostics = shared.evaluate(
            model, props, test_dataset, TEST_RANGE[0]
        )
        return rows, {
            "classes": summary,
            "diagnostics": diagnostics,
            "tail_diagnostics": base.test_diagnostics(rows),
            "checkpoint": checkpoint_metadata(path),
        }

    phase_a_path = args.phase_a_checkpoint.resolve()
    baseline_rows, baseline_summary = evaluate(phase_a_path)
    base.write_csv(run_dir / "test_hattrick_metrics.csv", baseline_rows)
    base.write_json(run_dir / "test_hattrick_summary.json", baseline_summary)

    candidate_results: dict[str, dict] = {}
    for entry in entries:
        epoch = int(entry["epoch"])
        path = Path(entry["path"])
        rows, summary = evaluate(path)
        label = f"epoch_{epoch:03d}"
        base.write_csv(run_dir / f"test_hattrick_f3_{label}_metrics.csv", rows)
        base.write_json(run_dir / f"test_hattrick_f3_{label}_summary.json", summary)
        candidate_results[label] = {
            "validation_rank": entry["rank"],
            "selected_by_validation": epoch == int(entries[0]["epoch"]),
            "summary": summary,
            "paired_delta_vs_hattrick": base.paired_test_delta(
                baseline_rows, rows
            ),
        }

    comparison = {
        "status": "COMPLETE",
        "method": "Hattrick-f3",
        "selection_used_test": False,
        "strict_esm": (
            "routing policy consumes ESM predictions; actual 2x traffic is used "
            "only by sequential admission and metric computation"
        ),
        "train": list(TRAIN_RANGE),
        "validation": list(VALIDATION_RANGE),
        "test": list(TEST_RANGE),
        "selected_epoch": int(entries[0]["epoch"]),
        "top5_order_is_validation_only": [int(entry["epoch"]) for entry in entries],
        "hattrick": baseline_summary,
        "hattrick_f3": candidate_results,
    }
    base.write_json(run_dir / "test_top5_comparison.json", comparison)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "selected_epoch": int(entries[0]["epoch"]),
                "tested_top5": [int(entry["epoch"]) for entry in entries],
                "comparison": str((run_dir / "test_top5_comparison.json").resolve()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and test Hattrick-f3 on full GEANT 2x with front-loaded "
            "cosine MLU release"
        )
    )
    parser.add_argument(
        "--stage", choices=("check", "train", "test", "all"), default="check"
    )
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--epochs", type=positive_int, default=DEFAULT_EPOCHS)
    parser.add_argument(
        "--anneal-epochs", type=positive_int, default=DEFAULT_ANNEAL_EPOCHS
    )
    parser.add_argument("--top-k", type=positive_int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--low-budget", type=float, default=0.03)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--phase-a-checkpoint", type=Path, default=DEFAULT_PHASE_A
    )
    args = parser.parse_args()
    if args.epochs < 2:
        parser.error("--epochs must be at least 2")
    if args.anneal_epochs < 2 or args.anneal_epochs > args.epochs:
        parser.error("--anneal-epochs must be between 2 and --epochs")
    if args.top_k > args.epochs:
        parser.error("--top-k cannot exceed --epochs")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.low_budget < 0:
        parser.error("--low-budget must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    if args.stage in ("check", "all"):
        check_inputs(args)
    if args.stage in ("train", "all"):
        train(args)
    if args.stage in ("test", "all"):
        test(args)


if __name__ == "__main__":
    main()
