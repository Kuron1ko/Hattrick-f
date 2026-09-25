from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
DEFAULT_RUN_DIR = (
    THIS_DIR / "artifacts" / "level4_confirmation" / "seed_490"
)
DEFAULT_OUTPUT_DIR = THIS_DIR / "posthoc_level4" / "seed_490"
BASELINE_ROWS = (
    ROOT / "output" / "comparisons" / "dotemc_hattricke" / "comparison_rows_2x.csv"
)
BASELINE_VALIDATION_SUMMARY = (
    TEST_DIR
    / "shared2x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
    / "validation_epoch_060_summary.json"
)
CLASSES = ("High", "Medium", "Low")

HISTORY_SELECTION_COLUMNS = (
    "epoch",
    "high_norm_mean",
    "high_norm_p1",
    "high_norm_p10",
    "medium_norm_mean",
    "medium_norm_p1",
    "medium_norm_p10",
    "low_norm_mean",
    "max_post_admission_mlu",
    "max_disabled_flow",
)
SELECTION_COLUMNS = HISTORY_SELECTION_COLUMNS + (
    "low_norm_p1",
    "low_norm_p10",
)

STRICT_THRESHOLDS = {
    "high_norm_mean": 0.995,
    # The user-specified hard target is High mean >= 0.995.  The two tail
    # guards prevent a mean-safe checkpoint from hiding a severe regression.
    "high_norm_p1": 0.985,
    "high_norm_p10": 0.990,
    "max_post_admission_mlu": 1.0001,
    "max_disabled_flow": 1e-8,
}
LOW_GUARD_DELTA = 0.005


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_history_rows(run_dir: Path) -> list[dict[str, float | int]]:
    history_path = run_dir / "train_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(history_path)
    archived = {
        int(path.stem.split("_")[-1]): path
        for path in (run_dir / "epoch_checkpoints").glob("epoch_*.pt")
    }
    by_epoch: dict[int, dict[str, float | int]] = {}
    for raw in read_csv(history_path):
        try:
            epoch = int(raw["epoch"])
            parsed: dict[str, float | int] = {"epoch": epoch}
            for column in HISTORY_SELECTION_COLUMNS[1:]:
                parsed[column] = float(raw[column])
            summary_path = run_dir / f"validation_epoch_{epoch:03d}_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            low = next(item for item in summary["classes"] if item["class"] == "Low")
            parsed["low_norm_p1"] = float(low["norm_fulfill_p1"])
            parsed["low_norm_p10"] = float(low["norm_fulfill_p10"])
            if not all(math.isfinite(float(parsed[column])) for column in SELECTION_COLUMNS[1:]):
                continue
        except (FileNotFoundError, json.JSONDecodeError, KeyError, StopIteration, TypeError, ValueError):
            # A concurrently-written final line is ignored by preview mode.
            continue
        if epoch in archived:
            by_epoch[epoch] = parsed
    if not by_epoch:
        raise RuntimeError("No complete train_history row has a matching archived checkpoint")
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def network_safe(row: dict[str, float | int]) -> bool:
    return (
        float(row["max_post_admission_mlu"])
        <= STRICT_THRESHOLDS["max_post_admission_mlu"]
        and float(row["max_disabled_flow"])
        <= STRICT_THRESHOLDS["max_disabled_flow"]
    )


def load_fixed_validation_reference() -> dict[str, float]:
    payload = json.loads(BASELINE_VALIDATION_SUMMARY.read_text(encoding="utf-8"))
    indexed = {item["class"]: item for item in payload["classes"]}
    reference: dict[str, float] = {}
    for class_name, prefix in (("High", "high"), ("Medium", "medium"), ("Low", "low")):
        item = indexed[class_name]
        for suffix in ("mean", "p1", "p10"):
            reference[f"{prefix}_norm_{suffix}"] = float(item[f"norm_fulfill_{suffix}"])
    return reference


def strict_feasible(
    row: dict[str, float | int], reference: dict[str, float]
) -> bool:
    return (
        network_safe(row)
        and float(row["high_norm_mean"])
        >= STRICT_THRESHOLDS["high_norm_mean"]
        and float(row["high_norm_p1"])
        >= STRICT_THRESHOLDS["high_norm_p1"]
        and float(row["high_norm_p10"])
        >= STRICT_THRESHOLDS["high_norm_p10"]
        and all(
            float(row[f"low_norm_{suffix}"])
            >= reference[f"low_norm_{suffix}"] - LOW_GUARD_DELTA
            for suffix in ("mean", "p1", "p10")
        )
    )


def choose_pool(
    rows: list[dict[str, float | int]],
    reference: dict[str, float],
) -> tuple[str, list[dict[str, float | int]]]:
    strict = [row for row in rows if strict_feasible(row, reference)]
    if len(strict) >= 2:
        return "task_high_guard_and_relative_low_guard", strict
    high_guard = [
        row
        for row in rows
        if network_safe(row)
        and float(row["high_norm_mean"]) >= STRICT_THRESHOLDS["high_norm_mean"]
        and float(row["high_norm_p1"]) >= STRICT_THRESHOLDS["high_norm_p1"]
        and float(row["high_norm_p10"]) >= STRICT_THRESHOLDS["high_norm_p10"]
    ]
    if len(high_guard) >= 2:
        return "relaxed_high_mean_guard", high_guard
    safe = [row for row in rows if network_safe(row)]
    if len(safe) >= 2:
        return "network_safe_fallback", safe
    if len(rows) >= 2:
        return "all_archived_fallback", rows
    raise RuntimeError("At least two archived epochs are required")


def descending_key(*columns: str):
    return lambda row: tuple(float(row[column]) for column in columns)


def normalized_balanced_score(
    row: dict[str, float | int],
    pool: list[dict[str, float | int]],
) -> float:
    columns = (
        "high_norm_mean",
        "high_norm_p1",
        "high_norm_p10",
        "medium_norm_mean",
        "medium_norm_p1",
        "medium_norm_p10",
    )
    scores = []
    for column in columns:
        values = np.asarray([float(item[column]) for item in pool], dtype=np.float64)
        width = float(values.max() - values.min())
        score = 1.0 if width <= 1e-12 else (float(row[column]) - float(values.min())) / width
        scores.append(score)
    # Maximin avoids hiding a weak High tail behind a strong Medium mean.
    return float(min(scores) + 0.05 * np.mean(scores))


def medium_maximin_delta(
    row: dict[str, float | int], reference: dict[str, float]
) -> float:
    return min(
        float(row[f"medium_norm_{suffix}"])
        - reference[f"medium_norm_{suffix}"]
        for suffix in ("mean", "p1", "p10")
    )


def select_candidates(
    rows: list[dict[str, float | int]],
    maximum: int,
    reference: dict[str, float],
) -> tuple[str, list[dict]]:
    if maximum < 2 or maximum > 4:
        raise ValueError("maximum candidates must be between 2 and 4")
    pool_name, pool = choose_pool(rows, reference)
    full_high_cdf_pool = [
        row
        for row in pool
        if min(
            float(row["high_norm_mean"]),
            float(row["high_norm_p1"]),
            float(row["high_norm_p10"]),
        )
        >= 0.995
    ]
    if not full_high_cdf_pool:
        full_high_cdf_pool = pool

    roles: list[tuple[str, list[dict[str, float | int]]]] = [
        (
            "primary_guarded_medium_maximin",
            sorted(
                pool,
                key=lambda row: (
                    medium_maximin_delta(row, reference),
                    float(row["medium_norm_mean"]),
                    float(row["medium_norm_p1"]),
                    float(row["medium_norm_p10"]),
                    float(row["high_norm_p1"]),
                    -int(row["epoch"]),
                ),
                reverse=True,
            ),
        ),
        (
            "full_high_cdf_guard",
            sorted(
                full_high_cdf_pool,
                key=lambda row: (
                    medium_maximin_delta(row, reference),
                    float(row["high_norm_p1"]),
                    float(row["high_norm_p10"]),
                    float(row["medium_norm_mean"]),
                ),
                reverse=True,
            ),
        ),
        (
            "guarded_medium_mean",
            sorted(
                pool,
                key=descending_key(
                    "medium_norm_mean",
                    "medium_norm_p10",
                    "medium_norm_p1",
                    "high_norm_p1",
                ),
                reverse=True,
            ),
        ),
        (
            "medium_tail_guard",
            sorted(
                pool,
                key=descending_key(
                    "medium_norm_p10",
                    "medium_norm_p1",
                    "medium_norm_mean",
                    "high_norm_mean",
                ),
                reverse=True,
            ),
        ),
    ]
    selected: list[dict] = []
    selected_epochs: set[int] = set()
    for role, ranked in roles:
        for rank, row in enumerate(ranked, start=1):
            epoch = int(row["epoch"])
            if epoch in selected_epochs:
                continue
            selected.append(
                {
                    "selection_role": role,
                    "role_rank": rank,
                    **row,
                }
            )
            selected_epochs.add(epoch)
            break
        if len(selected) >= maximum:
            break
    # Degenerate validation rankings can make several roles choose one epoch.
    if len(selected) < min(2, maximum):
        primary_order = roles[0][1]
        for rank, row in enumerate(primary_order, start=1):
            epoch = int(row["epoch"])
            if epoch in selected_epochs:
                continue
            selected.append(
                {
                    "selection_role": "guarded_medium_runner_up",
                    "role_rank": rank,
                    **row,
                }
            )
            selected_epochs.add(epoch)
            if len(selected) >= min(2, maximum):
                break
    return pool_name, selected[:maximum]


def make_selection_manifest(
    run_dir: Path,
    maximum: int,
    *,
    training_epochs_complete: bool,
    run_complete_marker: bool,
) -> dict:
    rows = load_history_rows(run_dir)
    reference = load_fixed_validation_reference()
    pool_name, selected = select_candidates(rows, maximum, reference)
    checkpoint_dir = run_dir / "epoch_checkpoints"
    for item in selected:
        epoch = int(item["epoch"])
        item["checkpoint"] = str((checkpoint_dir / f"epoch_{epoch:03d}.pt").resolve())
    return {
        "created_utc": utc_now(),
        "selection_frozen_before_test_replay": bool(training_epochs_complete),
        "training_epochs_complete": bool(training_epochs_complete),
        "run_complete_marker_present": bool(run_complete_marker),
        "selection_source": str((run_dir / "train_history.csv").resolve()),
        "selection_source_sha256": sha256(run_dir / "train_history.csv"),
        "selection_uses_test_window": False,
        "test_window_reserved": [400, 500],
        "available_archived_epochs": [int(row["epoch"]) for row in rows],
        "feasibility_pool": pool_name,
        "strict_thresholds": STRICT_THRESHOLDS,
        "fixed_hattrick_validation_reference": {
            "source": str(BASELINE_VALIDATION_SUMMARY.resolve()),
            "source_sha256": sha256(BASELINE_VALIDATION_SUMMARY),
            "metrics": reference,
        },
        "relative_low_guard_delta": LOW_GUARD_DELTA,
        "selection_rules_in_order": [
            "Primary: maximize the weakest of Medium mean/P1/P10 improvements versus fixed Hattrick epoch 60",
            "Full-High-CDF guard: repeat the Medium maximin rule with High mean/P1/P10 all at least 0.995",
            "Medium-mean guard: highest Medium validation mean",
            "Medium-tail guard: highest Medium validation P10",
        ],
        "primary_epoch": int(selected[0]["epoch"]),
        "selected": selected,
        "note": (
            "Low mean/P1/P10 are loaded from each validation summary and must stay within "
            "0.005 of the fixed Hattrick epoch-60 validation reference. Test metrics are "
            "reported but never used to change the preselected primary checkpoint."
        ),
    }


def expected_epochs(run_dir: Path) -> int:
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    value = int(config["epochs"])
    if value <= 0:
        raise RuntimeError(f"Invalid epoch count in {config_path}: {value}")
    return value


def training_epochs_are_archived(run_dir: Path) -> bool:
    try:
        target = expected_epochs(run_dir)
        rows = load_history_rows(run_dir)
    except (FileNotFoundError, RuntimeError, json.JSONDecodeError):
        return False
    return max(int(row["epoch"]) for row in rows) >= target


def load_runtime():
    path = THIS_DIR / "run_experiment.py"
    spec = importlib.util.spec_from_file_location(
        "shared2x_medium_pressure_posthoc_runtime", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def candidate_summary_rows(
    method: str,
    epoch: int,
    role: str,
    summary: list[dict],
) -> list[dict]:
    output = []
    for item in summary:
        output.append(
            {
                "load": "2x",
                "method": method,
                "checkpoint_epoch": epoch,
                "selection_role": role,
                "class": item["class"],
                "n": int(item["n"]),
                "norm_fulfill_mean": float(item["norm_fulfill_mean"]),
                "norm_fulfill_p1": float(item["norm_fulfill_p1"]),
                "norm_fulfill_p10": float(item["norm_fulfill_p10"]),
            }
        )
    return output


def summarize_baseline(method: str, rows: Iterable[dict[str, str]]) -> list[dict]:
    selected = [
        row
        for row in rows
        if row.get("load") == "2x"
        and row.get("method") == method
        and 400 <= int(row["snapshot"]) < 500
    ]
    output = []
    for class_name in CLASSES:
        values = np.asarray(
            [float(row["norm_fulfill"]) for row in selected if row["class"] == class_name],
            dtype=np.float64,
        )
        if values.size != 100:
            raise RuntimeError(
                f"Expected 100 baseline rows for {method}/{class_name}, got {values.size}"
            )
        output.append(
            {
                "load": "2x",
                "method": method,
                "checkpoint_epoch": "",
                "selection_role": "fixed_reference",
                "class": class_name,
                "n": int(values.size),
                "norm_fulfill_mean": float(values.mean()),
                "norm_fulfill_p1": float(np.percentile(values, 1)),
                "norm_fulfill_p10": float(np.percentile(values, 10)),
            }
        )
    return output


def add_reference_deltas(rows: list[dict]) -> list[dict]:
    index = {(row["method"], row["class"]): row for row in rows}
    output = []
    for row in rows:
        enriched = dict(row)
        for reference in ("Hattrick", "Hatrrick-e"):
            baseline = index.get((reference, row["class"]))
            for metric in (
                "norm_fulfill_mean",
                "norm_fulfill_p1",
                "norm_fulfill_p10",
            ):
                enriched[f"delta_{metric}_vs_{reference}"] = (
                    ""
                    if baseline is None or row["method"] in ("Hattrick", "Hatrrick-e")
                    else float(row[metric]) - float(baseline[metric])
                )
        output.append(enriched)
    return output


def evaluate_manifest(
    manifest: dict,
    output_dir: Path,
    device: torch.device,
) -> dict:
    if not manifest.get("selection_frozen_before_test_replay", False):
        raise RuntimeError("Refusing to evaluate a preview selection on the test window")
    runtime = load_runtime()
    full = runtime.full
    props = full.shared.build_props(4, device)
    if not bool(props.pred) or str(props.pred_type).lower() != "esm":
        raise RuntimeError("Strict replay requires pred=1 and pred_type=esm")
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_admitted = False
    dataset = full.DM_Dataset_within_Cluster(props, 0, 400, 500)
    if int(dataset.max_source_index_read) != 499:
        raise RuntimeError("Test split audit failed")

    all_candidate_rows: list[dict] = []
    all_summary_rows: list[dict] = []
    diagnostics: dict[str, dict] = {}
    for selected in manifest["selected"]:
        epoch = int(selected["epoch"])
        role = str(selected["selection_role"])
        checkpoint_path = Path(selected["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if int(checkpoint["epoch"]) != epoch:
            raise RuntimeError(f"Checkpoint epoch mismatch: {checkpoint_path}")
        model = runtime.MediumPressureHattrick(props).to(
            device=device, dtype=props.dtype
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        rows, summary, diagnostic = full.evaluate_checkpoint(model, props, dataset, 400)
        method = f"Hattrick-LA epoch {epoch:03d}"
        for row in rows:
            all_candidate_rows.append(
                {
                    "load": "2x",
                    "method": method,
                    "checkpoint_epoch": epoch,
                    "selection_role": role,
                    **row,
                }
            )
        all_summary_rows.extend(candidate_summary_rows(method, epoch, role, summary))
        diagnostics[method] = {
            **diagnostic,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256(checkpoint_path),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not BASELINE_ROWS.exists():
        raise FileNotFoundError(BASELINE_ROWS)
    baseline_rows = read_csv(BASELINE_ROWS)
    for method in ("Hattrick", "Hatrrick-e"):
        all_summary_rows.extend(summarize_baseline(method, baseline_rows))
    comparison_rows = add_reference_deltas(all_summary_rows)
    atomic_write_csv(output_dir / "candidate_test_rows.csv", all_candidate_rows)
    atomic_write_csv(output_dir / "comparison_summary.csv", comparison_rows)

    primary_epoch = int(manifest["primary_epoch"])
    primary_method = f"Hattrick-LA epoch {primary_epoch:03d}"
    primary_rows = [row for row in comparison_rows if row["method"] == primary_method]
    report = {
        "created_utc": utc_now(),
        "protocol": {
            "selection": "validation-only train_history, frozen before test replay",
            "inference": "strict ESM prediction values",
            "test_window": [400, 500],
            "primary_checkpoint_is_not_reselected_from_test": True,
            "device": str(device),
        },
        "selection_manifest": manifest,
        "primary_method": primary_method,
        "primary_summary": primary_rows,
        "all_summaries": comparison_rows,
        "diagnostics": diagnostics,
        "baseline_source": str(BASELINE_ROWS.resolve()),
        "baseline_source_sha256": sha256(BASELINE_ROWS),
    }
    atomic_write_json(output_dir / "test_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only archived-checkpoint selection followed by strict-ESM "
            "Level-4 replay. The test split can never select or replace the primary epoch."
        )
    )
    parser.add_argument(
        "action",
        choices=("select", "watch-select", "evaluate", "all"),
        help="Protocol phase",
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-candidates", type=int, default=4, choices=(2, 3, 4))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="Polling interval for watch-select",
    )
    parser.add_argument(
        "--allow-incomplete-preview",
        action="store_true",
        help="For action=select only: write selection_preview.json while training is active",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    run_complete = (run_dir / "complete.json").exists()
    manifest_path = output_dir / "selection_manifest.json"

    if args.action == "watch-select":
        if args.poll_seconds <= 0:
            raise ValueError("--poll-seconds must be positive")
        print(
            f"[watching] validation history until epoch {expected_epochs(run_dir)} is archived",
            flush=True,
        )
        while not training_epochs_are_archived(run_dir):
            time.sleep(args.poll_seconds)
        run_complete = (run_dir / "complete.json").exists()

    if args.action in ("select", "watch-select", "all"):
        epochs_complete = training_epochs_are_archived(run_dir)
        if not epochs_complete and not args.allow_incomplete_preview:
            raise RuntimeError(
                "The final train_history epoch is not archived. Use "
                "--allow-incomplete-preview only to inspect a non-testable preview, "
                "or use watch-select."
            )
        manifest = make_selection_manifest(
            run_dir,
            args.max_candidates,
            training_epochs_complete=epochs_complete,
            run_complete_marker=run_complete,
        )
        if epochs_complete:
            if manifest_path.exists() and not args.force:
                raise FileExistsError(
                    f"Frozen selection already exists: {manifest_path}; use --force to replace"
                )
            atomic_write_json(manifest_path, manifest)
            print(f"[selected] {manifest_path}")
        else:
            preview_path = output_dir / "selection_preview.json"
            atomic_write_json(preview_path, manifest)
            print(f"[preview only; test replay disabled] {preview_path}")
            for item in manifest["selected"]:
                print(item["selection_role"], "epoch", item["epoch"])
            return

        if args.action == "watch-select":
            return

    if args.action in ("evaluate", "all"):
        if not (run_dir / "complete.json").exists():
            raise RuntimeError("Refusing to open snapshots 400-499 before training completes")
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Frozen validation selection is required before evaluation: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report_path = output_dir / "test_report.json"
        if report_path.exists() and not args.force:
            raise FileExistsError(f"Test report already exists: {report_path}")
        report = evaluate_manifest(manifest, output_dir, resolve_device(args.device))
        print(json.dumps(report["primary_summary"], ensure_ascii=False, indent=2))
        print(f"[complete] {report_path}")


if __name__ == "__main__":
    main()
