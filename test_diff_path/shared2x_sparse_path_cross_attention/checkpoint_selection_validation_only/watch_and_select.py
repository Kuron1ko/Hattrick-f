from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
METHOD_DIR = THIS_DIR.parent
TEST_DIR = METHOD_DIR.parent
RUN_DIR = METHOD_DIR / "artifacts/level4_confirmation/seed_490"
BASELINE_DIR = TEST_DIR / "shared2x_full_objectives/artifacts/level4_confirmation/seed_490"
BASELINE_METRICS = BASELINE_DIR / "validation_epoch_060_metrics.csv"
POLICY_PATH = THIS_DIR / "selection_policy.json"
ARCHIVE_DIR = THIS_DIR / "archive"
LIVE_MANIFEST = THIS_DIR / "manifest_live.json"
FROZEN_MANIFEST = THIS_DIR / "manifest_frozen_epoch_060.json"
SELECTED_CHECKPOINT = THIS_DIR / "selected_checkpoint.pt"
WATCHER_STATUS = THIS_DIR / "watcher_status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def history_by_epoch() -> dict[int, dict[str, str]]:
    rows = read_csv(RUN_DIR / "train_history.csv")
    return {int(row["epoch"]): row for row in rows}


def validation_paths(epoch: int) -> tuple[Path, Path]:
    return (
        RUN_DIR / f"validation_epoch_{epoch:03d}_summary.json",
        RUN_DIR / f"validation_epoch_{epoch:03d}_metrics.csv",
    )


def checkpoint_epoch(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "epoch" not in payload:
        raise RuntimeError(f"Checkpoint has no epoch: {path}")
    return int(payload["epoch"])


def archive_checkpoint(source: Path, history: dict[int, dict[str, str]]) -> dict | None:
    """Copy a stable runner checkpoint to an immutable validation-only archive."""

    if not source.exists():
        return None
    before = source.stat()
    try:
        epoch = checkpoint_epoch(source)
    except Exception:
        return None
    after_load = source.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after_load.st_size,
        after_load.st_mtime_ns,
    ):
        return None
    summary_path, metrics_path = validation_paths(epoch)
    if epoch not in history or not summary_path.exists() or not metrics_path.exists():
        return None

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    destination = ARCHIVE_DIR / f"epoch_{epoch:03d}.pt"
    if destination.exists():
        if checkpoint_epoch(destination) != epoch:
            raise RuntimeError(f"Archived checkpoint epoch mismatch: {destination}")
        return {
            "epoch": epoch,
            "path": str(destination.resolve()),
            "sha256": sha256(destination),
            "source": source.name,
        }

    temporary = destination.with_suffix(".pt.tmp")
    shutil.copyfile(source, temporary)
    after_copy = source.stat()
    if (after_load.st_size, after_load.st_mtime_ns) != (
        after_copy.st_size,
        after_copy.st_mtime_ns,
    ):
        temporary.unlink(missing_ok=True)
        return None
    if checkpoint_epoch(temporary) != epoch:
        temporary.unlink(missing_ok=True)
        return None
    os.replace(temporary, destination)
    return {
        "epoch": epoch,
        "path": str(destination.resolve()),
        "sha256": sha256(destination),
        "source": source.name,
    }


def archive_available(history: dict[int, dict[str, str]]) -> list[dict]:
    # best_model may preserve an epoch already overwritten in final_model.
    records = []
    for name in ("best_model.pt", "final_model.pt"):
        record = archive_checkpoint(RUN_DIR / name, history)
        if record is not None:
            records.append(record)
    known = {item["epoch"]: item for item in records}
    if ARCHIVE_DIR.exists():
        for path in sorted(ARCHIVE_DIR.glob("epoch_[0-9][0-9][0-9].pt")):
            epoch = checkpoint_epoch(path)
            known[epoch] = {
                "epoch": epoch,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "source": known.get(epoch, {}).get("source", "archive"),
            }
    return [known[epoch] for epoch in sorted(known)]


def low_by_snapshot(path: Path) -> dict[int, float]:
    rows = read_csv(path)
    result = {
        int(row["snapshot"]): float(row["norm_fulfill"])
        for row in rows
        if row["class"] == "Low"
    }
    if len(result) != 50 or set(result) != set(range(350, 400)):
        raise RuntimeError(f"Expected only validation Low snapshots 350-399: {path}")
    return result


def paired_low_audit(
    candidate_metrics: Path,
    direct_mean_delta_min: float,
    fallback_margin: float,
) -> dict[str, float | int | bool]:
    baseline = low_by_snapshot(BASELINE_METRICS)
    candidate = low_by_snapshot(candidate_metrics)
    if set(candidate) != set(baseline):
        raise RuntimeError("Candidate/baseline validation snapshots differ")
    differences = [candidate[index] - baseline[index] for index in sorted(baseline)]
    mean = statistics.fmean(differences)
    standard_error = statistics.stdev(differences) / math.sqrt(len(differences))
    # Student-t critical values for the fixed n=50 paired validation snapshots
    # (df=49): two-sided 95% and one-sided 95%, respectively.
    two_sided_critical = 2.009575
    one_sided_critical = 1.676551
    two_sided_lower = mean - two_sided_critical * standard_error
    two_sided_upper = mean + two_sided_critical * standard_error
    one_sided_lower = mean - one_sided_critical * standard_error
    direct_pass = bool(mean >= direct_mean_delta_min and two_sided_upper >= 0.0)
    fallback_pass = bool(
        mean >= -fallback_margin and one_sided_lower >= -fallback_margin
    )
    return {
        "n": len(differences),
        "mean_delta": mean,
        "standard_error": standard_error,
        "two_sided_95_lower": two_sided_lower,
        "two_sided_95_upper": two_sided_upper,
        "direct_mean_delta_min": direct_mean_delta_min,
        "direct_two_sided_95_upper_min": 0.0,
        "direct_no_significant_decline_pass": direct_pass,
        "one_sided_95_lower": one_sided_lower,
        "fallback_noninferiority_margin": fallback_margin,
        "fallback_noninferiority_pass": fallback_pass,
    }


def dominates(left: dict, right: dict, coordinates: tuple[str, ...]) -> bool:
    return all(left[key] >= right[key] for key in coordinates) and any(
        left[key] > right[key] for key in coordinates
    )


def select(history: dict[int, dict[str, str]], archives: list[dict], policy: dict) -> dict:
    hard = policy["hard_gates"]
    low_direct = policy["preferred_low_no_significant_decline"]
    low_fallback = policy["fallback_low_noninferiority"]
    tails = policy["preferred_high_tail_gates"]
    archive_index = {int(item["epoch"]): item for item in archives}
    candidates = []
    for epoch in sorted(archive_index):
        if epoch not in history:
            continue
        summary_path, metrics_path = validation_paths(epoch)
        if not summary_path.exists() or not metrics_path.exists():
            continue
        row = history[epoch]
        item = {
            "epoch": epoch,
            "checkpoint": archive_index[epoch],
            "high_norm_mean": float(row["high_norm_mean"]),
            "high_norm_p1": float(row["high_norm_p1"]),
            "high_norm_p10": float(row["high_norm_p10"]),
            "medium_norm_mean": float(row["medium_norm_mean"]),
            "medium_norm_p1": float(row["medium_norm_p1"]),
            "medium_norm_p10": float(row["medium_norm_p10"]),
            "low_norm_mean": float(row["low_norm_mean"]),
            "max_post_admission_mlu": float(row["max_post_admission_mlu"]),
            "max_disabled_flow": float(row["max_disabled_flow"]),
            "validation_summary_sha256": sha256(summary_path),
            "validation_metrics_sha256": sha256(metrics_path),
        }
        item["low_paired_audit"] = paired_low_audit(
            metrics_path,
            float(low_direct["mean_delta_min"]),
            float(low_fallback["margin"]),
        )
        item["base_safety_gate_pass"] = bool(
            item["high_norm_mean"] >= float(hard["high_norm_mean_min"])
            and item["max_post_admission_mlu"]
            <= float(hard["max_post_admission_mlu_max"])
            and item["max_disabled_flow"] <= float(hard["max_disabled_flow_max"])
        )
        item["low_direct_pass"] = bool(
            item["base_safety_gate_pass"]
            and item["low_paired_audit"]["direct_no_significant_decline_pass"]
        )
        item["low_fallback_pass"] = bool(
            item["base_safety_gate_pass"]
            and item["low_paired_audit"]["fallback_noninferiority_pass"]
        )
        item["preferred_high_tail_pass"] = bool(
            item["high_norm_p1"] >= float(tails["high_norm_p1_min"])
            and item["high_norm_p10"] >= float(tails["high_norm_p10_min"])
        )
        candidates.append(item)

    base_pool = [item for item in candidates if item["base_safety_gate_pass"]]
    direct_pool = [item for item in candidates if item["low_direct_pass"]]
    fallback_pool = [item for item in candidates if item["low_fallback_pass"]]
    if direct_pool:
        low_pool = direct_pool
        low_tier = "paired-low-no-significant-decline"
    elif fallback_pool:
        low_pool = fallback_pool
        low_tier = "paired-low-0.01-noninferiority-fallback"
    else:
        low_pool = []
        low_tier = "no-low-safe-candidate"
    tail_pool = [item for item in low_pool if item["preferred_high_tail_pass"]]
    pool = tail_pool if tail_pool else low_pool
    tier = (
        f"{low_tier}+preferred-high-tail"
        if tail_pool
        else low_tier
    )
    coordinates = ("medium_norm_mean", "medium_norm_p1", "medium_norm_p10")
    pareto = [
        item
        for item in pool
        if not any(
            dominates(other, item, coordinates)
            for other in pool
            if other is not item
        )
    ]
    selected = None
    if pareto:
        ideal = {key: max(item[key] for item in pool) for key in coordinates}
        for item in pareto:
            ratios = [item[key] / max(ideal[key], 1e-12) for key in coordinates]
            item["medium_normalized_ratios"] = dict(zip(coordinates, ratios))
            item["medium_maximin"] = min(ratios)
            item["medium_normalized_mean"] = statistics.fmean(ratios)
        selected = max(
            pareto,
            key=lambda item: (
                item["medium_maximin"],
                item["medium_normalized_mean"],
                item["medium_norm_mean"],
                item["medium_norm_p1"],
                item["medium_norm_p10"],
                min(item["high_norm_p1"], item["high_norm_p10"]),
                -item["epoch"],
            ),
        )
    return {
        "selection_tier": tier,
        "candidate_count": len(candidates),
        "base_safety_gate_count": len(base_pool),
        "low_direct_tier_count": len(direct_pool),
        "low_fallback_tier_count": len(fallback_pool),
        "selection_pool_count": len(pool),
        "preferred_tail_count": len(tail_pool),
        "pareto_epochs": [item["epoch"] for item in pareto],
        "selected": selected,
        "candidates": candidates,
    }


def build_manifest(target_epoch: int, frozen: bool) -> dict:
    policy = read_policy()
    history = history_by_epoch()
    archives = archive_available(history)
    result = select(history, archives, policy)
    observed_epochs = sorted(history)
    archived_epochs = [int(item["epoch"]) for item in archives]
    manifest = {
        "schema": 1,
        "generated_at_utc": utc_now(),
        "frozen": frozen,
        "target_epoch": target_epoch,
        "latest_history_epoch": max(observed_epochs, default=0),
        "observed_history_epochs": observed_epochs,
        "archived_epochs": archived_epochs,
        "unavailable_past_checkpoints": [
            epoch for epoch in observed_epochs if epoch not in archived_epochs
        ],
        "source_run_directory": str(RUN_DIR.resolve()),
        "source_train_history_sha256": (
            sha256(RUN_DIR / "train_history.csv")
            if (RUN_DIR / "train_history.csv").exists()
            else None
        ),
        "baseline_validation_metrics": str(BASELINE_METRICS.resolve()),
        "baseline_validation_metrics_sha256": sha256(BASELINE_METRICS),
        "policy": policy,
        "test_data_read": False,
        "selection": result,
    }
    return manifest


def freeze_if_ready(target_epoch: int) -> bool:
    history = history_by_epoch()
    archive_available(history)
    target_archive = ARCHIVE_DIR / f"epoch_{target_epoch:03d}.pt"
    if target_epoch not in history or not target_archive.exists():
        atomic_json(LIVE_MANIFEST, build_manifest(target_epoch, frozen=False))
        return False
    manifest = build_manifest(target_epoch, frozen=True)
    if int(manifest["latest_history_epoch"]) != target_epoch:
        raise RuntimeError("History advanced beyond target before validation freeze")
    selected = manifest["selection"]["selected"]
    if selected is None:
        manifest["freeze_error"] = "No checkpoint passed the validation-only gates"
        atomic_json(FROZEN_MANIFEST, manifest)
        return True
    source = Path(selected["checkpoint"]["path"])
    temporary = SELECTED_CHECKPOINT.with_suffix(".pt.tmp")
    shutil.copyfile(source, temporary)
    if checkpoint_epoch(temporary) != int(selected["epoch"]):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Selected checkpoint epoch changed during freeze")
    os.replace(temporary, SELECTED_CHECKPOINT)
    manifest["selected_checkpoint_copy"] = {
        "path": str(SELECTED_CHECKPOINT.resolve()),
        "sha256": sha256(SELECTED_CHECKPOINT),
        "epoch": int(selected["epoch"]),
    }
    atomic_json(FROZEN_MANIFEST, manifest)
    atomic_json(LIVE_MANIFEST, manifest)
    return True


def source_signature() -> tuple:
    """Cheap polling key; expensive checkpoint/metric reads run only on change."""

    values = []
    for path in (
        RUN_DIR / "train_history.csv",
        RUN_DIR / "best_model.pt",
        RUN_DIR / "final_model.pt",
    ):
        if path.exists():
            stat = path.stat()
            values.append((path.name, stat.st_size, stat.st_mtime_ns))
        else:
            values.append((path.name, None, None))
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-only Level-4 checkpoint archive and selector"
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--target-epoch", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    if FROZEN_MANIFEST.exists():
        print(str(FROZEN_MANIFEST), flush=True)
        return

    atomic_json(
        WATCHER_STATUS,
        {
            "pid": os.getpid(),
            "started_at_utc": utc_now(),
            "mode": "watch" if args.watch else "once",
            "target_epoch": args.target_epoch,
            "source": str(RUN_DIR.resolve()),
            "test_data_read": False,
        },
    )
    last_signature = None
    frozen = False
    while True:
        signature = source_signature()
        if signature != last_signature:
            frozen = freeze_if_ready(args.target_epoch)
            last_signature = signature
        if frozen or not args.watch:
            break
        time.sleep(args.poll_seconds)
    status = json.loads(WATCHER_STATUS.read_text(encoding="utf-8"))
    status.update(
        {
            "finished_at_utc": utc_now(),
            "frozen": FROZEN_MANIFEST.exists(),
        }
    )
    atomic_json(WATCHER_STATUS, status)
    print(str(FROZEN_MANIFEST if frozen else LIVE_MANIFEST), flush=True)


if __name__ == "__main__":
    main()
