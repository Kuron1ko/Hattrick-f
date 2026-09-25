from __future__ import annotations

"""Registry-backed checkpoint discovery and strict-ESM test inference."""

import csv
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch


# Strict-ESM traffic pickles were produced by NumPy 2.x.  Install aliases only
# when inference runs under the older repository NumPy environment.
try:
    import numpy._core  # type: ignore[import-not-found]  # noqa: F401
except ModuleNotFoundError:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent.parent
REGISTRY_PATH = ROOT / "test_diff_path" / "full_geant_2x_methods.json"
RESULT_ROOT = ROOT / "test_diff_path" / "full_geant_2x_registered_results"
CLASSES = ("High", "Medium", "Low")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("Refusing to write an empty inference result")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_registry() -> dict:
    value = read_json(REGISTRY_PATH)
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise RuntimeError(f"Unsupported method registry: {REGISTRY_PATH}")
    datasets = value.get("datasets")
    methods = value.get("methods")
    if not isinstance(datasets, dict) or not datasets or not isinstance(methods, dict):
        raise RuntimeError(f"Malformed method registry: {REGISTRY_PATH}")
    for name, dataset in datasets.items():
        if not isinstance(dataset, dict) or dataset.get("strict_esm") is not True:
            raise RuntimeError(f"Registered dataset {name!r} is not strict ESM")
    return value


def registered_dataset(name: str | None = None) -> tuple[str, dict, dict]:
    registry = load_registry()
    requested = name or str(registry.get("default_dataset", "2x"))
    needle = requested.casefold()
    for canonical, config in registry["datasets"].items():
        aliases = [canonical, *config.get("aliases", [])]
        if needle in {str(alias).casefold() for alias in aliases}:
            return canonical, config, registry
    available = ", ".join(registry["datasets"])
    raise KeyError(f"Unknown dataset {requested!r}; registered datasets: {available}")


def normalize_model_list(value: object) -> list[dict]:
    """Validate the public [{"name": ..., "epoch": ...}, ...] interface."""
    if not isinstance(value, list) or not value:
        raise ValueError("model list must be a non-empty JSON list")
    result: list[dict] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"name", "epoch"}:
            raise ValueError(
                f"model list item {index} must contain exactly name and epoch"
            )
        name = item["name"]
        epoch = item["epoch"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"model list item {index} has an invalid name")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError(f"model list item {index} has an invalid epoch")
        result.append({"name": name.strip(), "epoch": epoch})
    return result


def parse_model_list_argument(value: str) -> list[dict]:
    """Accept a JSON path/string, including PowerShell 5.1 quote stripping."""
    candidate = Path(value)
    try:
        is_file = candidate.is_file()
    except OSError:
        is_file = False
    if is_file:
        raw = read_json(candidate)
    else:
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as error:
            try:
                raw = parse_powershell_model_list(value)
            except ValueError:
                raise ValueError(
                    "--models must be a model-list JSON string or an existing "
                    "UTF-8 JSON file"
                ) from error
    return normalize_model_list(raw)


def parse_powershell_model_list(value: str) -> list[dict]:
    """Parse the narrow [{name:X,epoch:N}, ...] form produced by PS 5.1."""
    text = value.strip()
    if len(text) < 2 or text[0] != "[" or text[-1] != "]":
        raise ValueError("not a model list")
    body = text[1:-1]
    matches = list(re.finditer(r"\{([^{}]*)\}", body))
    if not matches:
        raise ValueError("model list has no objects")

    cursor = 0
    parsed: list[dict] = []
    for match in matches:
        separator = body[cursor : match.start()]
        expected = "" if cursor == 0 else ","
        if separator.strip() != expected:
            raise ValueError("invalid object separator")
        cursor = match.end()

        item: dict[str, object] = {}
        for field in match.group(1).split(","):
            if ":" not in field:
                raise ValueError("invalid model field")
            key, raw_value = field.split(":", 1)
            key = key.strip().strip("\"'")
            raw_value = raw_value.strip().strip("\"'")
            if key == "name":
                if not raw_value:
                    raise ValueError("empty model name")
                item[key] = raw_value
            elif key == "epoch":
                try:
                    item[key] = int(raw_value)
                except ValueError as error:
                    raise ValueError("invalid epoch") from error
            else:
                raise ValueError(f"unknown model field {key!r}")
        parsed.append(item)

    if body[cursor:].strip():
        raise ValueError("trailing model-list content")
    return parsed


def registered_method(
    name: str, dataset_name: str | None = None
) -> tuple[str, dict, dict]:
    dataset_key, dataset, registry = registered_dataset(dataset_name)
    needle = name.casefold()
    for canonical, method in registry["methods"].items():
        aliases = [canonical, *method.get("aliases", [])]
        if needle in {str(alias).casefold() for alias in aliases}:
            dataset_methods = method.get("datasets", {})
            if dataset_key not in dataset_methods:
                supported = ", ".join(dataset_methods) or "none"
                raise KeyError(
                    f"Method {canonical!r} is not registered for dataset "
                    f"{dataset_key!r}; supported datasets: {supported}"
                )
            config = {
                "aliases": list(method.get("aliases", [])),
                **dataset_methods[dataset_key],
            }
            context = dict(registry)
            context["dataset"] = {"key": dataset_key, **dataset}
            context["dataset_name"] = dataset_key
            return canonical, config, context
    available = ", ".join(registry["methods"])
    raise KeyError(f"Unknown method {name!r}; registered methods: {available}")


def format_path(template: str, *, seed: int, epoch: int) -> Path:
    path = ROOT / template.format(seed=seed, epoch=epoch)
    # Windows GetFinalPathNameByHandle cannot resolve a path containing glob
    # metacharacters.  The parent is resolved by the subsequent glob call.
    if any(character in str(path) for character in "*?["):
        return Path(os.path.abspath(path))
    return path.resolve()


def checkpoint_identity(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = ("method", "seed", "epoch", "model_state_dict")
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise RuntimeError(f"Checkpoint {path} is missing fields {missing}")
    result = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "method": str(checkpoint["method"]),
        "seed": int(checkpoint["seed"]),
        "epoch": int(checkpoint["epoch"]),
    }
    if "dataset" in checkpoint:
        result["dataset"] = str(checkpoint["dataset"])
    config = checkpoint.get("config", checkpoint.get("signature"))
    if config is not None:
        if not isinstance(config, dict):
            raise RuntimeError(f"Checkpoint {path} has a non-object config")
        config_sha256 = json_sha256(config)
        declared_hash = checkpoint.get("config_sha256")
        if declared_hash is not None and declared_hash != config_sha256:
            raise RuntimeError(
                f"Checkpoint {path} has a stale config_sha256: "
                f"{declared_hash} != {config_sha256}"
            )
        if config.get("method") not in (None, result["method"]):
            raise RuntimeError(f"Checkpoint {path} config method mismatch")
        if config.get("seed") not in (None, result["seed"]):
            raise RuntimeError(f"Checkpoint {path} config seed mismatch")
        result["config_sha256"] = config_sha256
        result["config_contract"] = {
            field: config[field]
            for field in (
                "dataset", "topology", "load_factor", "strict_esm",
                "train", "validation", "test",
            )
            if field in config
        }
    if "rank" in checkpoint:
        result["rank"] = [float(value) for value in checkpoint["rank"]]
    if "anneal_alpha" in checkpoint:
        result["anneal_alpha"] = float(checkpoint["anneal_alpha"])
    return result


def checkpoint_candidates(config: dict, *, seed: int, epoch: int) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for template in config.get("checkpoint_templates", []):
        path = format_path(template, seed=seed, epoch=epoch)
        if path not in seen:
            seen.add(path)
            result.append(path)
    for pattern in config.get("checkpoint_globs", []):
        formatted = format_path(pattern, seed=seed, epoch=epoch)
        for path in sorted(formatted.parent.glob(formatted.name)):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                result.append(resolved)
    return result


def available_epochs(
    method: str, *, seed: int = 490, dataset_name: str | None = None
) -> list[int]:
    canonical, config, registry = registered_method(method, dataset_name)
    epochs: set[int] = set()
    for path in checkpoint_candidates(config, seed=seed, epoch=0):
        if not path.exists():
            continue
        try:
            identity = checkpoint_identity(path)
        except (KeyError, RuntimeError, OSError):
            continue
        contract = identity.get("config_contract", {})
        expected = registry["dataset"]
        identity_dataset_matches = identity.get("dataset", expected["key"]) == expected["key"]
        contract_matches = all(
            field not in contract or contract[field] == expected[field]
            for field in (
                "topology", "load_factor", "strict_esm",
                "train", "validation", "test",
            )
        )
        contract_matches = contract_matches and (
            "dataset" not in contract or contract["dataset"] == expected["key"]
        )
        if (
            identity["method"] == canonical
            and identity["seed"] == seed
            and identity_dataset_matches
            and contract_matches
        ):
            epochs.add(int(identity["epoch"]))
    return sorted(epochs)


def resolve_checkpoint(
    method: str,
    epoch: int,
    *,
    seed: int = 490,
    dataset_name: str | None = None,
) -> tuple[str, Path, dict]:
    canonical, config, registry = registered_method(method, dataset_name)
    mismatches: list[str] = []
    for path in checkpoint_candidates(config, seed=seed, epoch=epoch):
        if not path.exists():
            continue
        identity = checkpoint_identity(path)
        contract = identity.get("config_contract", {})
        expected_contract = {
            "topology": registry["dataset"]["topology"],
            "load_factor": registry["dataset"]["load_factor"],
            "strict_esm": registry["dataset"]["strict_esm"],
            "train": registry["dataset"]["train"],
            "validation": registry["dataset"]["validation"],
            "test": registry["dataset"]["test"],
        }
        config_matches = all(
            field not in contract or contract[field] == expected
            for field, expected in expected_contract.items()
        )
        config_matches = config_matches and (
            "dataset" not in contract
            or contract["dataset"] == registry["dataset"]["key"]
        )
        identity_dataset_matches = (
            identity.get("dataset", registry["dataset"]["key"])
            == registry["dataset"]["key"]
        )
        if (
            identity["method"] == canonical
            and identity["seed"] == seed
            and identity["epoch"] == epoch
            and identity_dataset_matches
            and config_matches
        ):
            return canonical, path, identity
        mismatches.append(
            f"{path.name}:{identity['method']}/seed{identity['seed']}/"
            f"epoch{identity['epoch']}/config_match={config_matches}"
        )
    found = available_epochs(
        canonical, seed=seed, dataset_name=registry["dataset"]["key"]
    )
    detail = f"; inspected mismatches: {mismatches}" if mismatches else ""
    raise FileNotFoundError(
        f"No registered {registry['dataset']['key']} {canonical} checkpoint "
        f"for seed={seed}, epoch={epoch}. "
        f"Available epochs: {found}{detail}"
    )


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")


def canonical_cache_paths(
    method: str, epoch: int, seed: int, dataset_name: str = "2x"
) -> tuple[Path, Path]:
    # Keep the established 2x cache location stable; isolate every other load.
    dataset_root = RESULT_ROOT if dataset_name == "2x" else RESULT_ROOT / dataset_name
    directory = dataset_root / slug(method) / f"seed_{seed}"
    return (
        directory / f"test_epoch_{epoch:03d}_metrics.csv",
        directory / f"test_epoch_{epoch:03d}_summary.json",
    )


def inference_config_identity(
    *,
    canonical: str,
    checkpoint: dict,
    registry: dict,
    seed: int,
    epoch: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[dict, str]:
    runtime_path = (ROOT / registry["dataset"]["runtime"]).resolve()
    model_path = ROOT / "frameworks" / "hattrick_system.py"
    dataset_path = ROOT / "utils" / "build_dataset_within_cluster.py"
    config = {
        "method": canonical,
        "seed": seed,
        "epoch": epoch,
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_config_sha256": checkpoint.get("config_sha256"),
        "dataset": registry["dataset"],
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "source_sha256": {
            "runtime": sha256(runtime_path),
            "hattrick_system": sha256(model_path),
            "dataset": sha256(dataset_path),
        },
    }
    return config, json_sha256(config)


def audit_test_cache(
    *,
    metrics_path: Path,
    summary_path: Path,
    checkpoint: dict,
    inference_config_sha256: str,
    test_range: tuple[int, int],
) -> tuple[list[dict], dict] | None:
    if not metrics_path.exists() or not summary_path.exists():
        return None
    try:
        summary = read_json(summary_path)
        if not isinstance(summary, dict):
            return None
        if summary.get("strict_esm") is not True:
            return None
        if tuple(int(value) for value in summary.get("test", ())) != test_range:
            return None
        if summary.get("inference_config_sha256") != inference_config_sha256:
            return None
        cached_checkpoint = summary.get("checkpoint")
        if not isinstance(cached_checkpoint, dict):
            return None
        identity_fields = ["method", "seed", "epoch", "sha256"]
        if "config_sha256" in checkpoint:
            identity_fields.append("config_sha256")
        for field in identity_fields:
            if cached_checkpoint.get(field) != checkpoint.get(field):
                return None
        rows = read_csv(metrics_path)
        expected_snapshots = set(range(*test_range))
        if len(rows) != len(expected_snapshots) * len(CLASSES):
            return None
        for class_name in CLASSES:
            snapshots = {
                int(row["snapshot"])
                for row in rows
                if row.get("class") == class_name
            }
            if snapshots != expected_snapshots:
                return None
        classes = summary.get("classes")
        if not isinstance(classes, list) or len(classes) != len(CLASSES):
            return None
        counts = {
            str(item.get("class")): int(item.get("n", -1))
            for item in classes
            if isinstance(item, dict)
        }
        if counts != {name: len(expected_snapshots) for name in CLASSES}:
            return None
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None
    return rows, summary


def cache_options(
    *,
    canonical: str,
    config: dict,
    checkpoint: dict,
    seed: int,
    epoch: int,
    dataset_name: str,
) -> list[tuple[Path, Path]]:
    result = [canonical_cache_paths(canonical, epoch, seed, dataset_name)]
    for item in config.get("legacy_test_caches", []):
        result.append(
            (
                format_path(item["metrics"], seed=seed, epoch=epoch),
                format_path(item["summary"], seed=seed, epoch=epoch),
            )
        )
    return result


def load_base_runner(registry: dict):
    path = (ROOT / registry["dataset"]["runtime"]).resolve()
    module_name = f"registered_full_geant_{slug(registry['dataset']['key'])}_runtime"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import registered inference runtime: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def normalize_summary(
    source: dict,
    *,
    checkpoint: dict,
    dataset: dict,
    inference_config: dict,
    inference_config_sha256: str,
    cache_reused: bool,
    cache_source: str | None,
) -> dict:
    return {
        "status": "COMPLETE",
        "strict_esm": True,
        "test": list(dataset["test"]),
        "checkpoint": checkpoint,
        "inference_config": inference_config,
        "inference_config_sha256": inference_config_sha256,
        "classes": source["classes"],
        "diagnostics": source.get("diagnostics", {}),
        "tail_diagnostics": source.get("tail_diagnostics", {}),
        "cache": {
            "reused": cache_reused,
            "source": cache_source,
        },
    }


def infer_registered_method(
    method: str,
    epoch: int,
    *,
    seed: int = 490,
    batch_size: int = 8,
    learning_rate: float = 0.0005,
    device_name: str = "auto",
    dataset_name: str = "2x",
    force: bool = False,
    base_runner=None,
) -> dict:
    canonical, config, registry = registered_method(method, dataset_name)
    dataset_name = registry["dataset"]["key"]
    canonical, checkpoint_path, checkpoint = resolve_checkpoint(
        canonical, epoch, seed=seed, dataset_name=dataset_name
    )
    inference_config, inference_config_sha256 = inference_config_identity(
        canonical=canonical,
        checkpoint=checkpoint,
        registry=registry,
        seed=seed,
        epoch=epoch,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    test_range = tuple(int(value) for value in registry["dataset"]["test"])
    output_metrics, output_summary = canonical_cache_paths(
        canonical, epoch, seed, dataset_name
    )

    if not force:
        for metrics_path, summary_path in cache_options(
            canonical=canonical,
            config=config,
            checkpoint=checkpoint,
            seed=seed,
            epoch=epoch,
            dataset_name=dataset_name,
        ):
            cached = audit_test_cache(
                metrics_path=metrics_path,
                summary_path=summary_path,
                checkpoint=checkpoint,
                inference_config_sha256=inference_config_sha256,
                test_range=test_range,
            )
            if cached is None:
                continue
            rows, summary = cached
            normalized = normalize_summary(
                summary,
                checkpoint=checkpoint,
                dataset=registry["dataset"],
                inference_config=inference_config,
                inference_config_sha256=inference_config_sha256,
                cache_reused=True,
                cache_source=str(metrics_path.resolve()),
            )
            if metrics_path.resolve() != output_metrics.resolve():
                write_csv(output_metrics, rows)
                write_json(output_summary, normalized)
            return {
                "method": canonical,
                "dataset": dataset_name,
                "seed": seed,
                "epoch": epoch,
                "checkpoint": str(checkpoint_path),
                "metrics": str(output_metrics.resolve()),
                "summary": str(output_summary.resolve()),
                "cache_reused": True,
                "cache_source": str(metrics_path.resolve()),
            }

    runtime = base_runner or load_base_runner(registry)
    shared, _full, _hattrick_f = runtime.load_runtimes()
    from frameworks.hattrick_system import Hattrick
    from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError("device_name must be auto, cpu, or cuda")

    props = runtime.build_props(
        device,
        batch_size=batch_size,
        epochs=1,
        learning_rate=learning_rate,
    )
    test_dataset = DM_Dataset_within_Cluster(props, 0, *test_range)
    if int(test_dataset.max_source_index_read) != test_range[1] - 1:
        raise RuntimeError("Registered test split audit failed")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(payload["model_state_dict"])
    rows, classes, diagnostics = shared.evaluate(
        model, props, test_dataset, test_range[0]
    )
    raw_summary = {
        "classes": classes,
        "diagnostics": diagnostics,
        "tail_diagnostics": runtime.test_diagnostics(rows),
    }
    normalized = normalize_summary(
        raw_summary,
        checkpoint=checkpoint,
        dataset=registry["dataset"],
        inference_config=inference_config,
        inference_config_sha256=inference_config_sha256,
        cache_reused=False,
        cache_source=None,
    )
    write_csv(output_metrics, rows)
    write_json(output_summary, normalized)
    return {
        "method": canonical,
        "dataset": dataset_name,
        "seed": seed,
        "epoch": epoch,
        "checkpoint": str(checkpoint_path),
        "metrics": str(output_metrics.resolve()),
        "summary": str(output_summary.resolve()),
        "cache_reused": False,
        "cache_source": None,
    }


def infer_registered_models(
    models: object,
    *,
    seed: int = 490,
    batch_size: int = 8,
    learning_rate: float = 0.0005,
    device_name: str = "auto",
    dataset_name: str = "2x",
    force: bool = False,
    base_runner=None,
) -> list[dict]:
    """Infer a validated model list, reusing each audited result independently."""
    specs = normalize_model_list(models)
    results: list[dict] = []
    for spec in specs:
        results.append(
            infer_registered_method(
                spec["name"],
                spec["epoch"],
                seed=seed,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device_name=device_name,
                dataset_name=dataset_name,
                force=force,
                base_runner=base_runner,
            )
        )
    return results
