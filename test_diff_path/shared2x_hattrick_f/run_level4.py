from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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
sys.path.insert(0, str(THIS_DIR))

_CORE_SPEC = importlib.util.spec_from_file_location(
    "hattrick_f_level4_core", THIS_DIR / "run_experiment.py"
)
if _CORE_SPEC is None or _CORE_SPEC.loader is None:
    raise RuntimeError("cannot load Hattrick-f core")
core = importlib.util.module_from_spec(_CORE_SPEC)
sys.modules[_CORE_SPEC.name] = core
_CORE_SPEC.loader.exec_module(core)


LEVEL = 4
SEED = 490
EPOCHS = 30
PHASE_A_PATH = (
    TEST_DIR
    / "shared2x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
    / "best_model.pt"
)
PHASE_A_DIR = PHASE_A_PATH.parent
OUTPUT_DIR = (
    THIS_DIR
    / "artifacts"
    / "level4_confirmation"
    / "fh_release"
    / f"seed_{SEED}"
)
OUTPUT_BASE = THIS_DIR / "artifacts" / "level4_confirmation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_remove(path: Path) -> None:
    resolved = path.resolve()
    root = (THIS_DIR / "artifacts" / "level4_confirmation").resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"unsafe removal target: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def set_epoch_seed(epoch: int) -> None:
    value = SEED + epoch * 104729
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(value)
        torch.cuda.manual_seed_all(value)


def summary_index(summary: list[dict]) -> dict[str, dict]:
    return {row["class"]: row for row in summary}


def high_min(rows: list[dict]) -> float:
    return min(
        float(row["norm_fulfill"]) for row in rows if row["class"] == "High"
    )


def checkpoint_rank(
    rows: list[dict], summary: list[dict], baseline_summary: list[dict]
) -> tuple:
    indexed = summary_index(summary)
    baseline = summary_index(baseline_summary)
    eligible = (
        high_min(rows) >= core.HIGH_HARD_FLOOR - core.HIGH_HARD_TOLERANCE
        and float(indexed["Low"]["norm_fulfill_mean"])
        >= float(baseline["Low"]["norm_fulfill_mean"]) - core.LOW_MEAN_BUDGET
        and max(float(row["max_admitted_capacity_ratio"]) for row in summary)
        <= 1.0001
        and max(float(row["max_disabled_flow"]) for row in summary) <= 1e-8
    )
    return (
        int(eligible),
        float(indexed["Medium"]["norm_fulfill_mean"]),
        float(indexed["Medium"]["norm_fulfill_p10"]),
        float(indexed["Medium"]["norm_fulfill_p1"]),
        float(indexed["High"]["norm_fulfill_mean"]),
        float(indexed["Low"]["norm_fulfill_mean"]),
    )


def paired_bootstrap(
    baseline_rows: list[dict], candidate_rows: list[dict], class_name: str
) -> dict:
    baseline = {
        int(row["snapshot"]): float(row["norm_fulfill"])
        for row in baseline_rows
        if row["class"] == class_name
    }
    candidate = {
        int(row["snapshot"]): float(row["norm_fulfill"])
        for row in candidate_rows
        if row["class"] == class_name
    }
    snapshots = sorted(set(baseline) & set(candidate))
    delta = np.asarray(
        [candidate[snapshot] - baseline[snapshot] for snapshot in snapshots],
        dtype=np.float64,
    )
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(delta), size=(20000, len(delta)))
    bootstrap = delta[samples].mean(axis=1)
    return {
        "n": len(delta),
        "mean_delta": float(delta.mean()),
        "ci95": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "candidate_better_fraction": float(np.mean(delta > 0)),
    }


def class_comparison(
    baseline_summary: list[dict], candidate_summary: list[dict]
) -> dict:
    baseline = summary_index(baseline_summary)
    candidate = summary_index(candidate_summary)
    return {
        class_name: {
            "baseline_norm_mean": float(baseline[class_name]["norm_fulfill_mean"]),
            "hattrick_f_norm_mean": float(
                candidate[class_name]["norm_fulfill_mean"]
            ),
            "delta": float(
                candidate[class_name]["norm_fulfill_mean"]
                - baseline[class_name]["norm_fulfill_mean"]
            ),
            "baseline_norm_p10": float(
                baseline[class_name]["norm_fulfill_p10"]
            ),
            "hattrick_f_norm_p10": float(
                candidate[class_name]["norm_fulfill_p10"]
            ),
            "baseline_norm_p1": float(baseline[class_name]["norm_fulfill_p1"]),
            "hattrick_f_norm_p1": float(
                candidate[class_name]["norm_fulfill_p1"]
            ),
        }
        for class_name in ("High", "Medium", "Low")
    }


def config_for_run(epochs: int, low_budget: float) -> dict:
    source_paths = {
        "run_level4.py": Path(__file__).resolve(),
        "run_experiment.py": THIS_DIR / "run_experiment.py",
        "ordered_projection.py": TEST_DIR
        / "shared2x_full_objectives"
        / "ordered_projection.py",
        "hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
    }
    return {
        "method": "Hattrick-f",
        "level": LEVEL,
        "seed": SEED,
        "train": [0, 350],
        "validation_selection_only": [350, 400],
        "independent_evaluation": [400, 500],
        "phase_f_epochs": epochs,
        "strict_inference": "ESM prediction is the only traffic input to routing policy",
        "training_labels": "actual traffic is used only by differentiable admission/loss",
        "phase_a": {
            "checkpoint": str(PHASE_A_PATH.resolve()),
            "sha256": sha256(PHASE_A_PATH),
            "objectives": ["Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml"],
        },
        "phase_f": {
            "objectives": ["Fh", "Fhm", "Fhml"],
            "optimizer_reset": True,
            "all_parameters_trainable": True,
            "persistent_mlu_minimization": False,
        },
        "selection_gate": {
            "high_each_validation_snapshot": ">= 0.995 with 1e-4 numerical tolerance",
            "low_validation_mean": f">= Phase-A mean - {low_budget}",
            "low_absolute_budget": low_budget,
            "simulator": "capacity <= 1.0001 and disabled flow <= 1e-8",
        },
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
    }


def run(*, epochs: int, force: bool, low_budget: float) -> Path:
    global OUTPUT_DIR
    core.LOW_MEAN_BUDGET = float(low_budget)
    if abs(low_budget - 0.02) <= 1e-12:
        OUTPUT_DIR = OUTPUT_BASE / "fh_release" / f"seed_{SEED}"
    else:
        label = format(low_budget, ".4g").replace(".", "p")
        OUTPUT_DIR = OUTPUT_BASE / f"fh_release_low_budget_{label}" / f"seed_{SEED}"
    if not PHASE_A_PATH.exists():
        raise FileNotFoundError(PHASE_A_PATH)
    complete_path = OUTPUT_DIR / "complete.json"
    if complete_path.exists() and not force:
        print(f"[skip] complete {OUTPUT_DIR}", flush=True)
        return OUTPUT_DIR
    if force:
        safe_remove(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = config_for_run(epochs, low_budget)
    config_path = OUTPUT_DIR / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise RuntimeError("existing Level-4 configuration differs; use --force")
    else:
        write_json(config_path, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core.set_seed(SEED)
    props = core.shared.build_props(LEVEL, device)
    spec = core.shared.LEVELS[LEVEL]
    train_start, train_end = spec["train"]
    val_start, val_end = spec["validation"]
    eval_start, eval_end = spec["evaluation"]
    train_dataset = core.DM_Dataset_within_Cluster(
        props, 0, train_start, train_end
    )
    val_dataset = core.DM_Dataset_within_Cluster(props, 0, val_start, val_end)
    if int(train_dataset.max_source_index_read) != train_end - 1:
        raise RuntimeError("train split audit failed")
    if int(val_dataset.max_source_index_read) != val_end - 1:
        raise RuntimeError("validation split audit failed")

    phase_a = torch.load(PHASE_A_PATH, map_location=device, weights_only=False)
    model = core.Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(phase_a["model_state_dict"])
    teacher = core.Hattrick(props).to(device=device, dtype=props.dtype)
    teacher.load_state_dict(phase_a["model_state_dict"])
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = core.ADAMOptimizer(model.parameters(), lr=props.lr)

    phase_a_epoch = int(phase_a["epoch"])
    baseline_val_payload = json.loads(
        (PHASE_A_DIR / f"validation_epoch_{phase_a_epoch:03d}_summary.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_val_summary = baseline_val_payload["classes"]
    baseline_val_rows = read_csv(
        PHASE_A_DIR / f"validation_epoch_{phase_a_epoch:03d}_metrics.csv"
    )

    final_path = OUTPUT_DIR / "final_model.pt"
    best_path = OUTPUT_DIR / "best_model.pt"
    history_path = OUTPUT_DIR / "train_history.csv"
    history = read_csv(history_path)
    start_epoch = 1
    best_rank = checkpoint_rank(
        baseline_val_rows, baseline_val_summary, baseline_val_summary
    )
    best_epoch = 0
    initial_payload = {
        "epoch": 0,
        "rank": best_rank,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    if not best_path.exists():
        torch.save(initial_payload, best_path)
    if final_path.exists() and not force:
        checkpoint = torch.load(final_path, map_location=device, weights_only=False)
        if checkpoint["config"] != config:
            raise RuntimeError("resume checkpoint configuration mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if best_path.exists():
            best = torch.load(best_path, map_location="cpu", weights_only=False)
            best_rank = tuple(best["rank"])
            best_epoch = int(best["epoch"])
        print(f"[resume] Level-4 epoch {start_epoch}", flush=True)

    started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        set_epoch_seed(epoch)
        loader = core.shared.data_loader(
            train_dataset, props.batch_size, True, SEED + epoch * 1009
        )
        train_metrics = core.train_epoch(
            model,
            teacher,
            props,
            train_dataset,
            loader,
            optimizer,
            core.HIGH_TRAIN_FLOOR,
            None,
            True,
        )
        val_rows, val_summary, val_diagnostics = core.evaluate(
            model, props, val_dataset, val_start
        )
        write_csv(OUTPUT_DIR / f"validation_epoch_{epoch:03d}_metrics.csv", val_rows)
        write_json(
            OUTPUT_DIR / f"validation_epoch_{epoch:03d}_summary.json",
            {"classes": val_summary, "diagnostics": val_diagnostics},
        )
        indexed = summary_index(val_summary)
        rank = checkpoint_rank(val_rows, val_summary, baseline_val_summary)
        row = {
            "epoch": epoch,
            "eligible": int(rank[0]),
            "high_norm_mean": indexed["High"]["norm_fulfill_mean"],
            "high_norm_p10": indexed["High"]["norm_fulfill_p10"],
            "high_norm_min": high_min(val_rows),
            "medium_norm_mean": indexed["Medium"]["norm_fulfill_mean"],
            "medium_norm_p10": indexed["Medium"]["norm_fulfill_p10"],
            "medium_norm_p1": indexed["Medium"]["norm_fulfill_p1"],
            "low_norm_mean": indexed["Low"]["norm_fulfill_mean"],
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
            f"[Hattrick-f level4] epoch={epoch}/{epochs} eligible={rank[0]} "
            f"H={row['high_norm_mean']:.6f} Hmin={row['high_norm_min']:.6f} "
            f"M={row['medium_norm_mean']:.6f} L={row['low_norm_mean']:.6f} "
            f"best={best_epoch}",
            flush=True,
        )

    if best_path.exists() is False:
        raise RuntimeError("no Level-4 checkpoint was produced")
    best = torch.load(best_path, map_location=device, weights_only=False)
    if int(best["rank"][0]) != 1:
        raise RuntimeError("no Level-4 epoch passed the validation safety gate")
    model.load_state_dict(best["model_state_dict"])
    best_val_rows, best_val_summary, best_val_diagnostics = core.evaluate(
        model, props, val_dataset, val_start
    )
    write_csv(OUTPUT_DIR / "best_validation_metrics.csv", best_val_rows)
    write_json(
        OUTPUT_DIR / "best_validation_summary.json",
        {"classes": best_val_summary, "diagnostics": best_val_diagnostics},
    )

    # The independent range is not touched until validation has fixed the epoch.
    evaluation_dataset = core.DM_Dataset_within_Cluster(
        props, 0, eval_start, eval_end
    )
    if int(evaluation_dataset.max_source_index_read) != eval_end - 1:
        raise RuntimeError("evaluation split audit failed")
    evaluation_rows, evaluation_summary, evaluation_diagnostics = core.evaluate(
        model, props, evaluation_dataset, eval_start
    )
    write_csv(OUTPUT_DIR / "evaluation_metrics.csv", evaluation_rows)
    write_json(
        OUTPUT_DIR / "evaluation_summary.json",
        {"classes": evaluation_summary, "diagnostics": evaluation_diagnostics},
    )

    baseline_eval_rows = read_csv(PHASE_A_DIR / "best_evaluation_metrics.csv")
    baseline_eval_payload = json.loads(
        (PHASE_A_DIR / "best_evaluation_summary.json").read_text(encoding="utf-8")
    )
    baseline_eval_summary = baseline_eval_payload["classes"]
    initial_policy = core.capture_high_policy(
        teacher, props, evaluation_dataset, eval_start
    )
    final_policy = core.capture_high_policy(model, props, evaluation_dataset, eval_start)
    route_rows, route_summary = core.route_metrics(initial_policy, final_policy)
    write_csv(OUTPUT_DIR / "evaluation_route_reconstruction.csv", route_rows)

    statistics = {
        class_name: paired_bootstrap(
            baseline_eval_rows, evaluation_rows, class_name
        )
        for class_name in ("High", "Medium", "Low")
    }
    complete = {
        "status": "COMPLETE",
        "selection_used_evaluation": False,
        "best_epoch": int(best["epoch"]),
        "best_rank": list(best["rank"]),
        "validation_comparison": class_comparison(
            baseline_val_summary, best_val_summary
        ),
        "independent_evaluation_comparison": class_comparison(
            baseline_eval_summary, evaluation_summary
        ),
        "independent_evaluation_high_min": high_min(evaluation_rows),
        "paired_bootstrap_vs_six_loss_hattrick": statistics,
        "independent_evaluation_route_reconstruction": route_summary,
        "runtime_seconds_this_invocation": time.perf_counter() - started,
        "best_checkpoint_sha256": sha256(best_path),
    }
    write_json(complete_path, complete)
    print(json.dumps(complete, indent=2, ensure_ascii=False), flush=True)
    return OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Hattrick-f Level-4 confirmation")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--low-budget", type=float, default=0.02)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("epochs must be positive")
    if args.low_budget < 0:
        parser.error("low budget must be non-negative")
    run(epochs=args.epochs, force=args.force, low_budget=args.low_budget)


if __name__ == "__main__":
    main()
