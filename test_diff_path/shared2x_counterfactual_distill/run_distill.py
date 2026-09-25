from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
LEVEL3_ROOT = ROOT.parent / "shared2x_counterfactual_level3"
SHARED_RUNTIME = TEST_DIR / "shared2x_order_regularizer"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(SHARED_RUNTIME))

import run_experiment as shared  # noqa: E402
from counterfactual_search import (  # noqa: E402
    changed_od_distillation_loss,
    distillation_loss,
    guarded_changed_od_distillation_loss,
)
from frameworks.hattrick_system import Hattrick  # noqa: E402
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster  # noqa: E402


LEVELS = {
    1: {
        "label": "level1_correctness",
        "train": (0, 32),
        "validation": (32, 40),
        "evaluation": (32, 40),
        "epochs": 2,
        "targets": THIS_DIR / "artifacts" / "level0_search" / "train32" / "targets.pt",
    },
    2: {
        "label": "level2_proxy",
        "train": (0, 160),
        "validation": (160, 200),
        "evaluation": (200, 250),
        "epochs": 6,
        "targets": THIS_DIR / "artifacts" / "level0_search" / "train160" / "targets.pt",
    },
    3: {
        "label": "level3_validation_only",
        "train": (0, 350),
        "validation": (350, 400),
        "evaluation": (350, 400),
        "epochs": 15,
        "targets": None,
    },
}


def phase_a_path(level: int, seed: int) -> Path:
    if level == 3:
        label = "level3_validation_only"
        independent = LEVEL3_ROOT / "phase_a" / label / f"seed_{seed}" / "best_model.pt"
        if independent.exists():
            return independent
    else:
        # Preserve the completed Level-1/2 common-start experiments.
        label = "level2_proxy"
        seed = 490
    return (
        TEST_DIR
        / "shared2x_order_epsilon"
        / "artifacts"
        / "phase_a"
        / label
        / f"seed_{seed}"
        / "best_model.pt"
    )


def target_path(level: int, seed: int, spec: dict) -> Path:
    if level == 3:
        return (
            LEVEL3_ROOT
            / "level0_search"
            / "train350_batched"
            / f"seed_{seed}"
            / "targets.pt"
        )
    return spec["targets"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def class_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def paired(rows: list[dict], class_name: str, field: str) -> dict[int, float]:
    return {
        int(row["snapshot"]): float(row[field])
        for row in rows
        if row["class"] == class_name
    }


def checkpoint_rank(
    rows: list[dict],
    summary: list[dict],
    initial_rows: list[dict],
) -> tuple:
    indexed = class_index(summary)
    high = paired(rows, "High", "norm_fulfill")
    initial_high = paired(initial_rows, "High", "norm_fulfill")
    minimum_high_delta = min(high[key] - initial_high[key] for key in high)
    capacity_safe = (
        max(float(row["max_admitted_capacity_ratio"]) for row in summary) <= 1.0001
        and max(float(row["max_disabled_flow"]) for row in summary) <= 1e-8
    )
    high_safe = minimum_high_delta >= -0.005
    return (
        int(capacity_safe and high_safe),
        float(indexed["Medium"]["norm_fulfill_p10"]),
        float(indexed["Medium"]["norm_fulfill_p1"]),
        float(indexed["Medium"]["norm_fulfill_mean"]),
        minimum_high_delta,
    )


def train_epoch(
    model,
    props,
    dataset,
    optimizer,
    targets: dict[int, dict],
    split: tuple[int, int],
    distill_mode: str,
):
    model.train()
    props.mode = "train"
    props.sim_mf_mlu = 0
    props.research_return_policy = True
    path_masks = shared.base.move_dataset_static(dataset, props.device)
    if path_masks is None:
        masks = tuple(
            torch.ones(dataset.pte.shape[0], dtype=torch.bool, device=props.device)
            for _ in range(3)
        )
    else:
        masks = tuple(path_masks[index] for index in range(3))
    loader = shared.data_loader(dataset, props.batch_size, False, 0)
    offset = split[0]
    total_loss = 0.0
    batches = 0
    for inputs in loader:
        values = shared.unpack_to_device(inputs, props)
        prediction, _ = shared.model_forward(model, props, dataset, values, path_masks)
        batch = prediction[0].shape[0]
        ids = list(range(offset, offset + batch))
        if any(index not in targets for index in ids):
            raise KeyError(f"missing counterfactual targets for {ids}")
        teacher = tuple(
            torch.cat([targets[index]["policies"][class_index] for index in ids], dim=0)
            .to(device=props.device, dtype=props.dtype)
            for class_index in range(3)
        )
        if distill_mode in ("changed", "guarded_changed"):
            source = tuple(
                torch.cat(
                    [targets[index]["initial_policies"][class_index] for index in ids],
                    dim=0,
                ).to(device=props.device, dtype=props.dtype)
                for class_index in range(3)
            )
            loss_function = (
                guarded_changed_od_distillation_loss
                if distill_mode == "guarded_changed"
                else changed_od_distillation_loss
            )
            loss = loss_function(
                prediction, teacher, source, masks, int(props.num_paths_per_pair)
            )
        else:
            loss = distillation_loss(prediction, teacher, masks)
        if not torch.isfinite(loss).item():
            raise RuntimeError("non-finite distillation loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.detach().item())
        batches += 1
        offset += batch
    props.research_return_policy = False
    return {"distillation_loss": total_loss / max(batches, 1), "train_batches": batches}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=tuple(LEVELS), required=True)
    parser.add_argument("--seed", type=int, default=490)
    parser.add_argument(
        "--distill-mode",
        choices=("full", "changed", "guarded_changed"),
        default="full",
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--epochs-override", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    spec = LEVELS[args.level]
    if args.train_limit is not None:
        if args.level != 3 or args.train_limit <= 0 or args.train_limit > 350:
            parser.error("--train-limit is only valid for Level 3 and must lie in [1, 350]")
        train_split = (spec["train"][0], spec["train"][0] + args.train_limit)
    else:
        train_split = spec["train"]
    epochs = args.epochs_override if args.epochs_override is not None else spec["epochs"]
    if epochs <= 0:
        parser.error("epochs must be positive")
    targets_file = target_path(args.level, args.seed, spec)
    phase_a = phase_a_path(args.level, args.seed)
    if not targets_file.exists():
        parser.error(f"missing targets: {targets_file}")
    if not phase_a.exists():
        parser.error(f"missing Phase-A checkpoint: {phase_a}")
    artifact_root = LEVEL3_ROOT if args.level == 3 else THIS_DIR / "artifacts"
    label = spec["label"]
    if args.train_limit is not None:
        label += f"_screen_train{args.train_limit}"
    run_dir = artifact_root / "distill" / label / args.distill_mode / f"seed_{args.seed}"
    if args.force and run_dir.exists():
        resolved = run_dir.resolve()
        root = (artifact_root / "distill").resolve()
        if root not in resolved.parents:
            raise RuntimeError("unsafe output removal")
        shutil.rmtree(run_dir)
    if (run_dir / "complete.json").exists():
        print(run_dir, flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = shared.build_props(2, device)
    train = DM_Dataset_within_Cluster(props, 0, *train_split)
    validation = DM_Dataset_within_Cluster(props, 0, *spec["validation"])
    evaluation = DM_Dataset_within_Cluster(props, 0, *spec["evaluation"])
    targets = torch.load(targets_file, map_location="cpu", weights_only=False)
    checkpoint = torch.load(phase_a, map_location=device, weights_only=False)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(checkpoint["model_state_dict"])
    initial_hash = state_sha256(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(props.lr))
    initial_rows, initial_summary, initial_diagnostics = shared.evaluate(
        model, props, evaluation, spec["evaluation"][0]
    )
    initial_val_rows, _, _ = shared.evaluate(
        model, props, validation, spec["validation"][0]
    )
    write_csv(run_dir / "initial_evaluation_metrics.csv", initial_rows)
    write_json(
        run_dir / "initial_evaluation_summary.json",
        {"classes": initial_summary, "diagnostics": initial_diagnostics},
    )
    config = {
        "level": args.level,
        "seed": args.seed,
        "distill_mode": args.distill_mode,
        "train": list(train_split),
        "validation": list(spec["validation"]),
        "evaluation": list(spec["evaluation"]),
        "epochs": epochs,
        "objective": (
            "all High/Low ODs anchored to teacher; Medium KL only on teacher-changed ODs"
            if args.distill_mode == "guarded_changed"
            else (
                "masked KL on teacher-changed OD pairs only"
                if args.distill_mode == "changed"
                else "mean masked KL on all OD pairs"
            )
        ) + "; per-class count normalization; no lambda, order, or MLU penalty",
        "checkpoint_guard": "per-snapshot validation High NormFulFill delta versus Phase-A >= -0.005",
        "phase_a": str(phase_a.resolve()),
        "phase_a_sha256": sha256(phase_a),
        "targets": str(targets_file.resolve()),
        "targets_sha256": sha256(targets_file),
        "search_source_sha256": sha256(THIS_DIR / "counterfactual_search.py"),
        "runner_source_sha256": sha256(Path(__file__)),
    }
    write_json(run_dir / "config.json", config)
    history = []
    best_rank = None
    best_epoch = None
    started = time.perf_counter()
    for epoch in range(1, int(epochs) + 1):
        train_metrics = train_epoch(
            model,
            props,
            train,
            optimizer,
            targets,
            train_split,
            args.distill_mode,
        )
        val_rows, val_summary, val_diagnostics = shared.evaluate(
            model, props, validation, spec["validation"][0]
        )
        rank = checkpoint_rank(val_rows, val_summary, initial_val_rows)
        indexed = class_index(val_summary)
        row = {
            "epoch": epoch,
            **train_metrics,
            "checkpoint_feasible": int(rank[0]),
            "validation_high_mean": indexed["High"]["norm_fulfill_mean"],
            "validation_high_min_delta": rank[-1],
            "validation_medium_mean": indexed["Medium"]["norm_fulfill_mean"],
            "validation_medium_p1": indexed["Medium"]["norm_fulfill_p1"],
            "validation_medium_p10": indexed["Medium"]["norm_fulfill_p10"],
            "validation_inversion_fraction": val_diagnostics["inversion_violation_fraction"],
        }
        history.append(row)
        write_csv(run_dir / "train_history.csv", history)
        payload = {
            "epoch": epoch,
            "rank": rank,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "phase_a_parameter_sha256": initial_hash,
        }
        torch.save(payload, run_dir / "final_model.pt")
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            torch.save(payload, run_dir / "best_model.pt")
        print(
            f"[distill {label}] epoch={epoch}/{epochs} "
            f"loss={row['distillation_loss']:.6f} high_delta_min={rank[-1]:+.5f} "
            f"mid={row['validation_medium_mean']:.4f}/{row['validation_medium_p10']:.4f} "
            f"feasible={bool(rank[0])} best={best_epoch}",
            flush=True,
        )
    evaluations = []
    for name in ("final", "best"):
        saved = torch.load(run_dir / f"{name}_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state_dict"])
        rows, summary, diagnostics = shared.evaluate(
            model, props, evaluation, spec["evaluation"][0]
        )
        write_csv(run_dir / f"{name}_evaluation_metrics.csv", rows)
        write_json(
            run_dir / f"{name}_evaluation_summary.json",
            {"classes": summary, "diagnostics": diagnostics},
        )
        evaluations.append(
            {"checkpoint": name, "epoch": saved["epoch"], "classes": summary, "diagnostics": diagnostics}
        )
    write_json(
        run_dir / "complete.json",
        {
            "status": "COMPLETE",
            "best_epoch": best_epoch,
            "best_rank": list(best_rank),
            "runtime_seconds": time.perf_counter() - started,
            "phase_a_parameter_sha256": initial_hash,
            "all_parameters_trainable": all(parameter.requires_grad for parameter in model.parameters()),
            "evaluations": evaluations,
            "artifacts": {
                "best_model_sha256": sha256(run_dir / "best_model.pt"),
                "history_sha256": sha256(run_dir / "train_history.csv"),
            },
        },
    )
    print(run_dir, flush=True)


if __name__ == "__main__":
    main()
