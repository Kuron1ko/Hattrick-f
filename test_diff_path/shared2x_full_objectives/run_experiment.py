from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
SHARED_RUNNER_DIR = TEST_DIR / "shared2x_order_regularizer"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(SHARED_RUNNER_DIR))

import run_experiment as shared
from frameworks.hattrick_system import Hattrick
from ordered_projection import ordered_project_gradients, projection_diagnostics
from utils.AdamOptimizer import ADAMOptimizer
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster
from utils.robust_proj_utils import assign_gradients_and_step
from utils.training_utils import loss_mf, loss_mlu


OUTPUT_ROOT = THIS_DIR / "artifacts"
LOAD_FACTOR = 2.0
OBJECTIVE_NAMES = ("Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def run_directory(level: int, seed: int) -> Path:
    return OUTPUT_ROOT / shared.LEVELS[level]["label"] / f"seed_{seed}"


def source_hashes() -> dict[str, str]:
    paths = {
        "run_experiment.py": Path(__file__).resolve(),
        "ordered_projection.py": THIS_DIR / "ordered_projection.py",
        "frameworks/hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
        "utils/training_utils.py": ROOT / "utils" / "training_utils.py",
        "shared2x_order_regularizer/run_experiment.py": SHARED_RUNNER_DIR / "run_experiment.py",
    }
    return {name: sha256(path) for name, path in paths.items()}


def build_objectives(model, props, dataset, values, path_masks):
    (
        _node_features,
        _capacities,
        _tm1,
        _tm1_pred,
        _tm2,
        _tm2_pred,
        _tm3,
        _tm3_pred,
        opt1,
        opt2,
        opt3,
        opt1_mf,
        opt2_mf,
        opt3_mf,
        _snapshots,
    ) = values
    output, _ = shared.model_forward(model, props, dataset, values, path_masks)
    (
        edges_high,
        edges_high_medium,
        edges_all,
        _edges_high_final,
        _edges_high_medium_final,
        all_traffic,
        admitted_high,
        admitted_medium,
        _admitted_low,
    ) = output

    # The paper's complete lexicographic objective sequence.  Fhm is cumulative
    # High+Medium admission, not the Medium increment alone.
    loss_fh, value_fh = loss_mf(admitted_high, opt1_mf.detach())
    loss_uh, value_uh = loss_mlu(edges_high, opt1.detach())
    loss_fhm, value_fhm = loss_mf(admitted_high + admitted_medium, opt2_mf.detach())
    loss_uhm, value_uhm = loss_mlu(edges_high_medium, opt2.detach())
    loss_fhml, value_fhml = loss_mf(all_traffic, opt3_mf.detach())
    loss_uhml, value_uhml = loss_mlu(edges_all, opt3.detach())
    return (
        (loss_fh, loss_uh, loss_fhm, loss_uhm, loss_fhml, loss_uhml),
        (value_fh, value_uh, value_fhm, value_uhm, value_fhml, value_uhml),
    )


def train_epoch(model, props, dataset, loader, optimizer) -> dict[str, float]:
    model.train()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_admitted = True
    path_masks = shared.base.move_dataset_static(dataset, props.device)
    totals = {f"reported_{name}": 0.0 for name in OBJECTIVE_NAMES}
    first_probe: dict[str, float] = {}
    count = 0
    for batch_index, inputs in enumerate(loader):
        values = shared.unpack_to_device(inputs, props)
        losses, reported = build_objectives(model, props, dataset, values, path_masks)
        if any(not torch.isfinite(loss).item() for loss in losses):
            raise RuntimeError("Non-finite six-objective loss")
        result = ordered_project_gradients(model, losses)
        if batch_index == 0:
            first_probe = projection_diagnostics(result, OBJECTIVE_NAMES)
        assign_gradients_and_step(
            model,
            result.final_gradient,
            optimizer,
            result.parameter_shapes,
        )
        for name, value in zip(OBJECTIVE_NAMES, reported):
            totals[f"reported_{name}"] += float(value)
        count += 1
    props.research_return_admitted = False
    means = {name: value / max(count, 1) for name, value in totals.items()}
    means.update(first_probe)
    means["train_batches"] = count
    return means


def summary_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def checkpoint_rank(summary: list[dict]) -> tuple:
    indexed = summary_index(summary)
    high = indexed["High"]
    medium = indexed["Medium"]
    safe = (
        max(float(row["max_admitted_capacity_ratio"]) for row in summary) <= 1.0001
        and max(float(row["max_disabled_flow"]) for row in summary) <= 1e-8
    )
    # Validation-only diagnostic checkpoint: follow the restored first objective.
    return (
        int(safe),
        float(high["norm_fulfill_mean"]),
        float(high["norm_fulfill_p10"]),
        float(high["norm_fulfill_p1"]),
        float(medium["norm_fulfill_mean"]),
    )


def safe_remove_run(path: Path) -> None:
    resolved = path.resolve()
    root = OUTPUT_ROOT.resolve()
    if root not in resolved.parents or resolved == root:
        raise RuntimeError(f"Refusing to remove unsafe path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def critical_config(config: dict) -> tuple:
    return (
        config["level"],
        config["seed"],
        tuple(config["train"]),
        tuple(config["validation"]),
        tuple(config["evaluation"]),
        config["epochs"],
        tuple(config["objectives"]),
        config["source_sha256"],
    )


def evaluate_checkpoint(model, props, dataset, start: int):
    return shared.evaluate(model, props, dataset, start)


def run_one(level: int, seed: int, force: bool = False) -> Path:
    spec = shared.LEVELS[level]
    run_dir = run_directory(level, seed)
    complete_path = run_dir / "complete.json"
    if complete_path.exists() and not force:
        print(f"[skip] complete {run_dir}", flush=True)
        return run_dir
    if force:
        safe_remove_run(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    props = shared.build_props(level, device)
    train_start, train_end = spec["train"]
    val_start, val_end = spec["validation"]
    eval_start, eval_end = spec["evaluation"]
    config = {
        "level": level,
        "label": spec["label"],
        "seed": seed,
        "topology": shared.TOPOLOGY,
        "paths_per_pair": shared.K,
        "load_factor": LOAD_FACTOR,
        "shared_paths": True,
        "train": [train_start, train_end],
        "validation": [val_start, val_end],
        "evaluation": [eval_start, eval_end],
        "epochs": spec["epochs"],
        "batch_size": props.batch_size,
        "learning_rate": props.lr,
        "only_change_from_source": "restore Fh and cumulative Fhm objectives",
        "objectives": list(OBJECTIVE_NAMES),
        "final_checkpoint_is_primary_comparison": True,
        "best_checkpoint_rule": "validation lexicographic High NormFulFill mean/P10/P1, then Medium mean",
        "source_sha256": source_hashes(),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
        },
    }
    config_path = run_dir / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if critical_config(previous) != critical_config(config):
            raise RuntimeError("Existing run configuration does not match current source")
    else:
        write_json(config_path, config)

    train_dataset = DM_Dataset_within_Cluster(props, 0, train_start, train_end)
    val_dataset = DM_Dataset_within_Cluster(props, 0, val_start, val_end)
    eval_dataset = DM_Dataset_within_Cluster(props, 0, eval_start, eval_end)
    for dataset, expected in (
        (train_dataset, train_end - 1),
        (val_dataset, val_end - 1),
        (eval_dataset, eval_end - 1),
    ):
        if int(dataset.max_source_index_read) != expected:
            raise RuntimeError("Split-safe data audit failed")

    model = Hattrick(props).to(device=device, dtype=props.dtype)
    optimizer = ADAMOptimizer(model.parameters(), lr=props.lr)
    history_path = run_dir / "train_history.csv"
    history = read_csv(history_path)
    final_path = run_dir / "final_model.pt"
    best_path = run_dir / "best_model.pt"
    start_epoch = 1
    best_rank = None
    best_epoch = None
    if final_path.exists() and not force:
        checkpoint = torch.load(final_path, map_location=device, weights_only=False)
        if critical_config(checkpoint["config"]) != critical_config(config):
            raise RuntimeError("Resume checkpoint configuration mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if best_path.exists():
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            best_rank = tuple(best["rank"])
            best_epoch = int(best["epoch"])
        print(f"[resume] epoch {start_epoch}", flush=True)

    started = time.perf_counter()
    for epoch in range(start_epoch, int(spec["epochs"]) + 1):
        loader = shared.data_loader(train_dataset, props.batch_size, True, seed + epoch * 1009)
        train_metrics = train_epoch(model, props, train_dataset, loader, optimizer)
        val_rows, val_summary, val_diagnostics = evaluate_checkpoint(
            model, props, val_dataset, val_start
        )
        write_csv(run_dir / f"validation_epoch_{epoch:03d}_metrics.csv", val_rows)
        write_json(
            run_dir / f"validation_epoch_{epoch:03d}_summary.json",
            {"classes": val_summary, "diagnostics": val_diagnostics},
        )
        indexed = summary_index(val_summary)
        rank = checkpoint_rank(val_summary)
        row = {
            "epoch": epoch,
            "high_norm_mean": indexed["High"]["norm_fulfill_mean"],
            "high_norm_p1": indexed["High"]["norm_fulfill_p1"],
            "high_norm_p10": indexed["High"]["norm_fulfill_p10"],
            "medium_norm_mean": indexed["Medium"]["norm_fulfill_mean"],
            "medium_norm_p1": indexed["Medium"]["norm_fulfill_p1"],
            "medium_norm_p10": indexed["Medium"]["norm_fulfill_p10"],
            "low_norm_mean": indexed["Low"]["norm_fulfill_mean"],
            "max_post_admission_mlu": max(
                float(item["max_admitted_capacity_ratio"]) for item in val_summary
            ),
            "max_disabled_flow": max(float(item["max_disabled_flow"]) for item in val_summary),
            **val_diagnostics,
            **train_metrics,
        }
        history = [item for item in history if int(item["epoch"]) < epoch]
        history.append(row)
        write_csv(history_path, history)
        payload = {
            "epoch": epoch,
            "rank": rank,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }
        torch.save(payload, final_path)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            torch.save(payload, best_path)
        print(
            f"[{spec['label']}] seed={seed} epoch={epoch}/{spec['epochs']} "
            f"high={row['high_norm_mean']:.4f} mid={row['medium_norm_mean']:.4f} "
            f"low={row['low_norm_mean']:.4f} best={best_epoch}",
            flush=True,
        )

    evaluations = []
    final_reference = None
    for checkpoint_name, path in (("final", final_path), ("best", best_path)):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        rows, summary, diagnostics = evaluate_checkpoint(model, props, eval_dataset, eval_start)
        for row in rows:
            row.update({"checkpoint": checkpoint_name, "seed": seed, "level": level})
        write_csv(run_dir / f"{checkpoint_name}_evaluation_metrics.csv", rows)
        write_json(
            run_dir / f"{checkpoint_name}_evaluation_summary.json",
            {"classes": summary, "diagnostics": diagnostics},
        )
        evaluations.append(
            {
                "checkpoint": checkpoint_name,
                "epoch": int(checkpoint["epoch"]),
                "classes": summary,
                "diagnostics": diagnostics,
            }
        )
        if checkpoint_name == "final":
            final_reference = rows

    recovery = torch.load(final_path, map_location=device, weights_only=False)
    replay_model = Hattrick(props).to(device=device, dtype=props.dtype)
    replay_optimizer = ADAMOptimizer(replay_model.parameters(), lr=props.lr)
    replay_model.load_state_dict(recovery["model_state_dict"])
    replay_optimizer.load_state_dict(recovery["optimizer_state_dict"])
    replay_rows, _, _ = evaluate_checkpoint(replay_model, props, eval_dataset, eval_start)
    numeric_fields = (
        "admitted_traffic", "fulfill_ratio", "norm_fulfill", "raw_mlu",
        "normalized_mlu", "disabled_flow", "admitted_capacity_ratio",
    )
    max_delta = max(
        abs(float(expected[field]) - float(actual[field]))
        for expected, actual in zip(final_reference, replay_rows)
        for field in numeric_fields
    )
    if max_delta > 1e-5:
        raise RuntimeError(f"Checkpoint recovery mismatch: {max_delta}")

    complete = {
        "best_epoch": best_epoch,
        "best_rank": best_rank,
        "runtime_seconds_this_invocation": time.perf_counter() - started,
        "checkpoint_recovery_max_delta": max_delta,
        "evaluations": evaluations,
        "artifact_sha256": {
            "final_model.pt": sha256(final_path),
            "best_model.pt": sha256(best_path),
            "train_history.csv": sha256(history_path),
        },
    }
    write_json(complete_path, complete)
    print(f"[ok] complete {run_dir}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared-2x Hattrick complete six-objective experiment")
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_one(args.level, args.seed, force=args.force)


if __name__ == "__main__":
    main()
