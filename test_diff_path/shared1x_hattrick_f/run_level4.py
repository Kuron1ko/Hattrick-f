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
CORE_PATH = TEST_DIR / "shared2x_hattrick_f" / "run_experiment.py"

spec = importlib.util.spec_from_file_location("hattrick_f_onex_core", CORE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load Hattrick-f core: {CORE_PATH}")
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


TOPOLOGY = "geant_priomask500_shared"
LEVEL = 4
SEED = 490
EPOCHS = 30
LOW_MEAN_BUDGET = 0.03
BASELINE_MODEL_PATH = (
    TEST_DIR
    / "shared1x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
    / f"seed_{SEED}"
    / "best_model.pt"
)
OUTPUT_ROOT = THIS_DIR / "artifacts" / "level4_confirmation"


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
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
    root = OUTPUT_ROOT.resolve()
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
    rows: list[dict], summary: list[dict], baseline_summary: list[dict], low_budget: float
) -> tuple:
    indexed = summary_index(summary)
    baseline = summary_index(baseline_summary)
    eligible = (
        high_min(rows) >= core.HIGH_HARD_FLOOR - core.HIGH_HARD_TOLERANCE
        and float(indexed["Low"]["norm_fulfill_mean"])
        >= float(baseline["Low"]["norm_fulfill_mean"]) - float(low_budget)
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


def comparison(baseline: list[dict], candidate: list[dict]) -> dict:
    left = summary_index(baseline)
    right = summary_index(candidate)
    output = {}
    for class_name in ("High", "Medium", "Low"):
        output[class_name] = {
            "baseline_norm_mean": float(left[class_name]["norm_fulfill_mean"]),
            "hattrick_f_norm_mean": float(right[class_name]["norm_fulfill_mean"]),
            "delta": float(
                right[class_name]["norm_fulfill_mean"]
                - left[class_name]["norm_fulfill_mean"]
            ),
            "baseline_norm_p10": float(left[class_name]["norm_fulfill_p10"]),
            "hattrick_f_norm_p10": float(right[class_name]["norm_fulfill_p10"]),
            "baseline_norm_p1": float(left[class_name]["norm_fulfill_p1"]),
            "hattrick_f_norm_p1": float(right[class_name]["norm_fulfill_p1"]),
        }
    return output


def run_directory(low_budget: float) -> Path:
    label = format(low_budget, ".4g").replace(".", "p")
    return OUTPUT_ROOT / f"fh_release_low_budget_{label}" / f"seed_{SEED}"


def config_for_run(epochs: int, low_budget: float) -> dict:
    sources = {
        "run_level4.py": Path(__file__).resolve(),
        "hattrick_f_core.py": CORE_PATH,
        "ordered_projection.py": TEST_DIR
        / "shared2x_full_objectives"
        / "ordered_projection.py",
        "hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
    }
    return {
        "method": "Hattrick-f",
        "level": LEVEL,
        "load": "1x",
        "topology": TOPOLOGY,
        "seed": SEED,
        "train": [0, 350],
        "validation_selection_only": [350, 400],
        "independent_evaluation": [400, 500],
        "phase_f_epochs": epochs,
        "strict_inference": "ESM prediction is the only traffic input to routing policy",
        "training_labels": "actual traffic is used only by differentiable admission/loss",
        "phase_a": {
            "checkpoint": str(BASELINE_MODEL_PATH.resolve()),
            "sha256": sha256(BASELINE_MODEL_PATH),
            "identity": "independently trained 1x complete-six-objective Hattrick",
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
        "source_sha256": {name: sha256(path) for name, path in sources.items()},
    }


def run(*, epochs: int, low_budget: float, force: bool) -> Path:
    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(BASELINE_MODEL_PATH)
    output_dir = run_directory(low_budget)
    complete_path = output_dir / "complete.json"
    if complete_path.exists() and not force:
        print(f"[skip] complete {output_dir}", flush=True)
        return output_dir
    if force:
        safe_remove(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = config_for_run(epochs, low_budget)
    config_path = output_dir / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise RuntimeError("existing configuration differs; use --force")
    else:
        write_json(config_path, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core.set_seed(SEED)
    core.shared.TOPOLOGY = TOPOLOGY
    props = core.shared.build_props(LEVEL, device)
    split = core.shared.LEVELS[LEVEL]
    train_start, train_end = split["train"]
    val_start, val_end = split["validation"]
    eval_start, eval_end = split["evaluation"]

    train_dataset = core.DM_Dataset_within_Cluster(props, 0, train_start, train_end)
    val_dataset = core.DM_Dataset_within_Cluster(props, 0, val_start, val_end)
    evaluation_dataset = core.DM_Dataset_within_Cluster(
        props, 0, eval_start, eval_end
    )
    if int(train_dataset.max_source_index_read) != train_end - 1:
        raise RuntimeError("train split audit failed")
    if int(val_dataset.max_source_index_read) != val_end - 1:
        raise RuntimeError("validation split audit failed")
    if int(evaluation_dataset.max_source_index_read) != eval_end - 1:
        raise RuntimeError("evaluation split audit failed")

    phase_a = torch.load(
        BASELINE_MODEL_PATH, map_location=device, weights_only=False
    )
    baseline_state = phase_a["model_state_dict"]
    model = core.Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(baseline_state, strict=True)
    teacher = core.Hattrick(props).to(device=device, dtype=props.dtype)
    teacher.load_state_dict(baseline_state, strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    optimizer = core.ADAMOptimizer(model.parameters(), lr=props.lr)

    baseline_val_rows, baseline_val_summary, baseline_val_diagnostics = core.evaluate(
        teacher, props, val_dataset, val_start
    )
    baseline_eval_rows, baseline_eval_summary, baseline_eval_diagnostics = core.evaluate(
        teacher, props, evaluation_dataset, eval_start
    )
    write_csv(output_dir / "baseline_validation_metrics.csv", baseline_val_rows)
    write_json(
        output_dir / "baseline_validation_summary.json",
        {"classes": baseline_val_summary, "diagnostics": baseline_val_diagnostics},
    )
    write_csv(output_dir / "baseline_evaluation_metrics.csv", baseline_eval_rows)
    write_json(
        output_dir / "baseline_evaluation_summary.json",
        {"classes": baseline_eval_summary, "diagnostics": baseline_eval_diagnostics},
    )

    final_path = output_dir / "final_model.pt"
    best_path = output_dir / "best_model.pt"
    history_path = output_dir / "train_history.csv"
    best_rank = checkpoint_rank(
        baseline_val_rows, baseline_val_summary, baseline_val_summary, low_budget
    )
    best_epoch = 0
    initial = {
        "epoch": 0,
        "rank": best_rank,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    if not best_path.exists():
        torch.save(initial, best_path)

    indexed_baseline = summary_index(baseline_val_summary)
    history = read_csv(history_path)
    if not history:
        history = [
            {
                "epoch": 0,
                "eligible": int(best_rank[0]),
                "high_norm_mean": indexed_baseline["High"]["norm_fulfill_mean"],
                "high_norm_p10": indexed_baseline["High"]["norm_fulfill_p10"],
                "high_norm_min": high_min(baseline_val_rows),
                "medium_norm_mean": indexed_baseline["Medium"]["norm_fulfill_mean"],
                "medium_norm_p10": indexed_baseline["Medium"]["norm_fulfill_p10"],
                "medium_norm_p1": indexed_baseline["Medium"]["norm_fulfill_p1"],
                "low_norm_mean": indexed_baseline["Low"]["norm_fulfill_mean"],
            }
        ]
        write_csv(history_path, history)

    start_epoch = 1
    if final_path.exists() and not force:
        checkpoint = torch.load(final_path, map_location=device, weights_only=False)
        if checkpoint["config"] != config:
            raise RuntimeError("resume checkpoint configuration mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        best_rank = tuple(best["rank"])
        best_epoch = int(best["epoch"])
        print(f"[resume] 1x Hattrick-f epoch {start_epoch}", flush=True)

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
        write_csv(output_dir / f"validation_epoch_{epoch:03d}_metrics.csv", val_rows)
        write_json(
            output_dir / f"validation_epoch_{epoch:03d}_summary.json",
            {"classes": val_summary, "diagnostics": val_diagnostics},
        )
        indexed = summary_index(val_summary)
        rank = checkpoint_rank(val_rows, val_summary, baseline_val_summary, low_budget)
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
        if rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            torch.save(payload, best_path)
        print(
            f"[Hattrick-f 1x level4] epoch={epoch}/{epochs} eligible={rank[0]} "
            f"H={row['high_norm_mean']:.6f} Hmin={row['high_norm_min']:.6f} "
            f"M={row['medium_norm_mean']:.6f} L={row['low_norm_mean']:.6f} "
            f"best={best_epoch}",
            flush=True,
        )

    best = torch.load(best_path, map_location=device, weights_only=False)
    if int(best["rank"][0]) != 1:
        raise RuntimeError("no checkpoint passed the validation safety gate")
    model.load_state_dict(best["model_state_dict"])
    best_val_rows, best_val_summary, best_val_diagnostics = core.evaluate(
        model, props, val_dataset, val_start
    )
    evaluation_rows, evaluation_summary, evaluation_diagnostics = core.evaluate(
        model, props, evaluation_dataset, eval_start
    )
    write_csv(output_dir / "best_validation_metrics.csv", best_val_rows)
    write_json(
        output_dir / "best_validation_summary.json",
        {"classes": best_val_summary, "diagnostics": best_val_diagnostics},
    )
    write_csv(output_dir / "evaluation_metrics.csv", evaluation_rows)
    write_json(
        output_dir / "evaluation_summary.json",
        {"classes": evaluation_summary, "diagnostics": evaluation_diagnostics},
    )

    initial_policy = core.capture_high_policy(
        teacher, props, evaluation_dataset, eval_start
    )
    final_policy = core.capture_high_policy(
        model, props, evaluation_dataset, eval_start
    )
    route_rows, route_summary = core.route_metrics(initial_policy, final_policy)
    write_csv(output_dir / "evaluation_route_reconstruction.csv", route_rows)

    complete = {
        "status": "COMPLETE",
        "selection_used_evaluation": False,
        "best_epoch": int(best["epoch"]),
        "best_rank": list(best["rank"]),
        "validation_comparison": comparison(baseline_val_summary, best_val_summary),
        "independent_evaluation_comparison": comparison(
            baseline_eval_summary, evaluation_summary
        ),
        "independent_evaluation_high_min": high_min(evaluation_rows),
        "paired_bootstrap_vs_hattrick": {
            class_name: paired_bootstrap(
                baseline_eval_rows, evaluation_rows, class_name
            )
            for class_name in ("High", "Medium", "Low")
        },
        "independent_evaluation_route_reconstruction": route_summary,
        "runtime_seconds_this_invocation": time.perf_counter() - started,
        "best_checkpoint_sha256": sha256(best_path),
    }
    write_json(complete_path, complete)
    print(json.dumps(complete, indent=2, ensure_ascii=False), flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Hattrick-f 1x Level-4 training")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--low-budget", type=float, default=LOW_MEAN_BUDGET)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("epochs must be positive")
    if args.low_budget < 0:
        parser.error("low budget must be non-negative")
    run(epochs=args.epochs, low_budget=args.low_budget, force=args.force)


if __name__ == "__main__":
    main()
