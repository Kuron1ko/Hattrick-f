from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

import torch


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "model"
CACHE_ROOT = ROOT / "cache"
PICTURE_ROOT = ROOT / "pictures"
DATA_ROOT = ROOT / "data"
BASE_ROOT = ROOT / "baseresult"
RULES_PATH = ROOT / "selection_rules.json"
DATASETS = {
    "1x": {
        "topology": "geant_full_shared_load1x_train",
        "load_factor": 1.0,
        "train": (0, 6000),
        "validation": (6000, 7500),
        "test": (7500, 10200),
    },
    "2x": {
        "topology": "geant_full_shared_load2x_train",
        "load_factor": 2.0,
        "train": (0, 6000),
        "validation": (6000, 7500),
        "test": (7500, 10200),
    },
    "3x": {
        "topology": "geant_full_shared_load3x_train",
        "load_factor": 3.0,
        "train": (0, 6000),
        "validation": (6000, 7500),
        "test": (7500, 10200),
    },
}
METHODS = ("Hattrick", "Hattrick-f3")
CLASSES = ("High", "Medium", "Low")
OBJECTIVE_NAMES = ("Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml")
FLOW_OBJECTIVE_NAMES = ("Fh", "Fhm", "Fhml")


def canonical_dataset(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in DATASETS:
        raise ValueError(f"dataset must be one of {', '.join(DATASETS)}; got {value!r}")
    return normalized


def canonical_method(value: str) -> str:
    normalized = value.strip().casefold().replace("_", "-")
    aliases = {
        "hattrick": "Hattrick",
        "hatrrick": "Hattrick",
        "hattrick-f3": "Hattrick-f3",
    }
    if normalized not in aliases:
        raise ValueError(f"method must be Hattrick or Hattrick-f3; got {value!r}")
    return aliases[normalized]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    # Resumed runs may append rows produced by a newer code version.  Preserve
    # the complete schema of both the old and new rows instead of deriving the
    # header from the first row and rejecting later extra fields.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fieldnames.append(field)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def run_dir(dataset: str, method: str, seed: int) -> Path:
    return MODEL_ROOT / canonical_dataset(dataset) / canonical_method(method) / f"seed_{seed}"


def cache_dir(dataset: str, method: str, seed: int, epoch: int) -> Path:
    return (
        CACHE_ROOT / canonical_dataset(dataset) / canonical_method(method)
        / f"seed_{seed}" / f"epoch_{epoch:03d}"
    )


def checkpoint_epoch(path: Path) -> int | None:
    match = re.fullmatch(r"epoch_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
        return int(value["epoch"])
    except Exception:
        return None


def checkpoint_priority(path: Path) -> int:
    parent = path.parent.name.casefold()
    if parent == "checkpoints":
        return 50
    if parent == "pinned":
        return 40
    if parent == "top5":
        return 30
    return {"best_model.pt": 20, "resume_state.pt": 10, "final_model.pt": 5}.get(
        path.name, 0
    )


def saved_checkpoints(dataset: str, method: str, seed: int = 490) -> dict[int, Path]:
    root = run_dir(dataset, method, seed)
    candidates: list[Path] = []
    for directory in ("checkpoints", "pinned", "top5"):
        candidates.extend(sorted((root / directory).glob("*.pt")))
    for filename in ("best_model.pt", "resume_state.pt", "final_model.pt"):
        path = root / filename
        if path.is_file():
            candidates.append(path)
    result: dict[int, Path] = {}
    for path in candidates:
        epoch = checkpoint_epoch(path)
        if epoch is None:
            continue
        previous = result.get(epoch)
        if previous is None or checkpoint_priority(path) > checkpoint_priority(previous):
            result[epoch] = path.resolve()
    return dict(sorted(result.items()))


def resolve_checkpoint(dataset: str, method: str, epoch: int, seed: int = 490) -> Path:
    checkpoints = saved_checkpoints(dataset, method, seed)
    if epoch not in checkpoints:
        available = ", ".join(str(value) for value in checkpoints) or "none"
        raise FileNotFoundError(
            f"No {dataset} {canonical_method(method)} epoch {epoch}; available: {available}"
        )
    return checkpoints[epoch]


def load_checkpoint(path: Path, *, device: str | torch.device = "cpu") -> dict:
    value = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(value, dict) or "model_state_dict" not in value:
        raise RuntimeError(f"Malformed model checkpoint: {path}")
    return value


def validation_paths(dataset: str, method: str, seed: int, epoch: int) -> tuple[Path, Path]:
    root = run_dir(dataset, method, seed)
    return (
        root / f"validation_epoch_{epoch:03d}_metrics.csv",
        root / f"validation_epoch_{epoch:03d}_summary.json",
    )


def checkpoint_summary(
    dataset: str, method: str, seed: int, epoch: int, checkpoint: dict | None = None
) -> list[dict]:
    if checkpoint is None:
        checkpoint = load_checkpoint(resolve_checkpoint(dataset, method, epoch, seed))
    summary = checkpoint.get("validation_summary")
    if isinstance(summary, list):
        return summary
    _metrics, summary_path = validation_paths(dataset, method, seed, epoch)
    if summary_path.is_file():
        value = read_json(summary_path)
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and isinstance(value.get("classes"), list):
            return value["classes"]
    raise RuntimeError(f"No validation summary for {dataset} {method} epoch {epoch}")


def summary_index(summary: list[dict]) -> dict[str, dict]:
    return {str(row["class"]): row for row in summary}


def load_rules() -> dict:
    value = read_json(RULES_PATH)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RuntimeError(f"Malformed selection rules: {RULES_PATH}")
    return value


def maximum_high_mean(
    dataset: str,
    method: str,
    seed: int,
    checkpoints: dict[int, Path] | None = None,
) -> float:
    """Return the largest validation High mean among the method's saved epochs."""
    checkpoints = checkpoints or saved_checkpoints(dataset, method, seed)
    values: list[float] = []
    for epoch, path in checkpoints.items():
        try:
            checkpoint = load_checkpoint(path)
            indexed = summary_index(
                checkpoint_summary(dataset, method, seed, epoch, checkpoint)
            )
            values.append(float(indexed["High"]["norm_fulfill_mean"]))
        except (KeyError, RuntimeError, ValueError):
            continue
    if not values:
        raise RuntimeError(f"No usable High mean for {dataset} {method} seed {seed}")
    return max(values)


def selection_rank(
    dataset: str,
    method: str,
    seed: int,
    epoch: int,
    checkpoint: dict,
    *,
    high_mean_max: float | None = None,
) -> tuple[float, ...]:
    rules = load_rules()
    indexed = summary_index(checkpoint_summary(dataset, method, seed, epoch, checkpoint))
    if high_mean_max is None:
        high_mean_max = maximum_high_mean(dataset, method, seed)
    high_mean = float(indexed["High"]["norm_fulfill_mean"])
    window = float(rules["high"]["mean_window_from_max"])
    eligible = high_mean > float(high_mean_max) - window

    def configured_value(name: str) -> float:
        try:
            class_key, field = name.split(".", 1)
            class_name = {"high": "High", "medium": "Medium", "low": "Low"}[
                class_key.casefold()
            ]
        except (KeyError, ValueError) as error:
            raise RuntimeError(f"Invalid selection field: {name!r}") from error
        try:
            return float(indexed[class_name][field])
        except KeyError as error:
            raise RuntimeError(f"Unknown selection field: {name!r}") from error

    fields = rules.get("candidate_ranking")
    if not isinstance(fields, list) or not fields:
        raise RuntimeError("selection_rules.json requires candidate_ranking")
    medium_rank = tuple(configured_value(str(name)) for name in fields)
    if eligible:
        return (1.0, *medium_rank, -float(epoch))
    # Ineligible epochs can only fill unused Top-K slots; they can never beat
    # a window candidate.  Sort them deterministically by High then Medium.
    return (0.0, high_mean, *medium_rank, -float(epoch))


def select_checkpoint(dataset: str, method: str, seed: int = 490) -> tuple[int, Path, dict]:
    method = canonical_method(method)
    checkpoints = saved_checkpoints(dataset, method, seed)
    if not checkpoints:
        raise FileNotFoundError(f"No saved models for {dataset} {method} seed {seed}")
    high_mean_max = maximum_high_mean(dataset, method, seed, checkpoints)
    ranked: list[tuple[tuple[float, ...], int, Path, dict]] = []
    errors: list[str] = []
    for epoch, path in checkpoints.items():
        try:
            checkpoint = load_checkpoint(path)
            rank = selection_rank(
                dataset, method, seed, epoch, checkpoint, high_mean_max=high_mean_max
            )
            ranked.append((rank, epoch, path, checkpoint))
        except (KeyError, RuntimeError, ValueError) as error:
            errors.append(f"epoch {epoch}: {error}")
    if not ranked:
        raise RuntimeError("No checkpoint has usable validation data: " + "; ".join(errors))
    _rank, epoch, path, checkpoint = max(ranked, key=lambda item: item[0])
    return epoch, path, checkpoint


def metric_fields(summary: list[dict]) -> dict[str, float]:
    indexed = summary_index(summary)
    result: dict[str, float] = {}
    prefixes = {"High": "h", "Medium": "m", "Low": "l"}
    for class_name, prefix in prefixes.items():
        row = indexed[class_name]
        result[f"{prefix}_mean"] = float(row["norm_fulfill_mean"])
        result[f"{prefix}_p1"] = float(row["norm_fulfill_p1"])
        result[f"{prefix}_p10"] = float(row["norm_fulfill_p10"])
    return result


def validate_test_metrics(path: Path, dataset: str) -> None:
    rows = read_csv(path)
    expected_range = DATASETS[dataset]["test"]
    expected_snapshots = list(range(*expected_range))
    if len(rows) != len(expected_snapshots) * len(CLASSES):
        raise RuntimeError(f"Incomplete inference cache: {path}")
    for class_name in CLASSES:
        snapshots = sorted(
            int(row["snapshot"]) for row in rows if row.get("class") == class_name
        )
        if snapshots != expected_snapshots:
            raise RuntimeError(f"Incomplete {class_name} test cache: {path}")


def save_top_k(
    dataset: str,
    method: str,
    seed: int,
    top_k: int,
) -> list[dict]:
    root = run_dir(dataset, method, seed)
    checkpoints = saved_checkpoints(dataset, method, seed)
    high_mean_max = maximum_high_mean(dataset, method, seed, checkpoints)
    ranked: list[tuple[tuple[float, ...], int, Path]] = []
    for epoch, path in checkpoints.items():
        checkpoint = load_checkpoint(path)
        ranked.append((
            selection_rank(
                dataset,
                method,
                seed,
                epoch,
                checkpoint,
                high_mean_max=high_mean_max,
            ),
            epoch,
            path,
        ))
    ranked.sort(key=lambda item: item[0], reverse=True)
    kept = ranked[:top_k]
    top_dir = root / "top5"
    top_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for rank, epoch, source in kept:
        target = top_dir / f"epoch_{epoch:03d}.pt"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        entries.append({
            "epoch": epoch,
            "eligible": bool(rank[0]),
            "rank": list(rank),
            "path": str(target.relative_to(ROOT)),
            "sha256": sha256(target),
        })
    if entries:
        shutil.copy2(top_dir / f"epoch_{entries[0]['epoch']:03d}.pt", root / "best_model.pt")
    write_json(root / "top5.json", {
        "selection_used_test": False,
        "rules": str(RULES_PATH.relative_to(ROOT)),
        "dataset": dataset,
        "method": canonical_method(method),
        "seed": seed,
        "validation": list(DATASETS[dataset]["validation"]),
        "checkpoints": entries,
    })
    return entries


def ensure_workspace_inputs(dataset: str) -> None:
    dataset = canonical_dataset(dataset)
    data = DATA_ROOT / dataset
    baseline = BASE_ROOT / dataset
    required = [
        data / "topology/t1.json",
        data / "pairs/t1.pkl",
        data / "manifest.txt",
        baseline / "filenames.txt",
        baseline / "gt_optimal_values_mf.txt",
        baseline / "gt_optimal_values_mf_mf.txt",
        baseline / "gt_optimal_values_mf_mf_mf.txt",
        baseline / "gt_optimal_values_mlu.txt",
        baseline / "gt_optimal_values_mlu_mlu.txt",
        baseline / "gt_optimal_values_mlu_mlu_mlu.txt",
    ]
    for priority in (1, 2, 3):
        required.extend((
            data / "traffic" / str(priority),
            data / "traffic" / f"{priority}_esm",
        ))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Workspace inputs are incomplete:\n" + "\n".join(missing))


def finite_summary(summary: Iterable[dict]) -> None:
    for row in summary:
        for key, value in row.items():
            if key in {"class", "n"}:
                continue
            if not math.isfinite(float(value)):
                raise RuntimeError(f"Non-finite summary value {key}={value}")
