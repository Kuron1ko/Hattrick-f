from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import plot_cdf


REQUIRED_ROLES = plot_cdf.ROLE_ORDER
CLASSES = plot_cdf.CLASSES
STANDARD_FIELDS = (
    "snapshot",
    "class",
    "admitted_traffic",
    "demand",
    "fulfill_ratio",
    "oracle_admitted_traffic",
    "norm_fulfill",
    "raw_mlu",
    "oracle_mlu",
    "normalized_mlu",
    "disabled_flow",
    "admitted_capacity_ratio",
)
NUMERIC_FIELDS = tuple(field for field in STANDARD_FIELDS if field != "class")
ALLOWED_SELECTION_BASES = {"validation", "predeclared_final_epoch"}
EXPECTED_TOPOLOGY = "geant_priomask500_shared_load2x_train"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def check_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    actual = sha256(path)
    if not expected or actual.lower() != str(expected).lower():
        raise RuntimeError(
            f"{label} SHA256 mismatch: expected {expected!r}, got {actual}"
        )
    return actual


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = fieldnames or list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_manifest(manifest: dict, manifest_path: Path) -> dict:
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError("Manifest schema_version must be 1")
    if manifest.get("selection_locked_before_evaluation") is not True:
        raise ValueError("selection_locked_before_evaluation must be true")
    if manifest.get("test_metrics_used_for_selection") is not False:
        raise ValueError("test_metrics_used_for_selection must be false")

    evaluation = manifest.get("evaluation", {})
    if evaluation.get("split") != "test":
        raise ValueError("evaluation.split must be 'test'")
    snapshot_range = tuple(int(value) for value in evaluation.get("snapshots", ()))
    if len(snapshot_range) != 2 or snapshot_range[1] <= snapshot_range[0]:
        raise ValueError("evaluation.snapshots must be [start,end)")
    if float(evaluation.get("load_factor", float("nan"))) != 2.0:
        raise ValueError("This pipeline is frozen to load_factor=2.0")
    if evaluation.get("topology") != EXPECTED_TOPOLOGY:
        raise ValueError(f"evaluation.topology must be {EXPECTED_TOPOLOGY!r}")
    if int(evaluation.get("paths_per_pair", -1)) != 8:
        raise ValueError("evaluation.paths_per_pair must be 8")
    if str(evaluation.get("prediction", "")).lower() != "esm":
        raise ValueError("evaluation.prediction must be 'esm'")
    if evaluation.get("strict_prediction_only_policy") is not True:
        raise ValueError("strict_prediction_only_policy must be true")
    expected_samples = snapshot_range[1] - snapshot_range[0]
    if int(evaluation.get("expected_samples_per_class", -1)) != expected_samples:
        raise ValueError("expected_samples_per_class disagrees with snapshot range")

    methods = manifest.get("methods")
    if not isinstance(methods, list) or len(methods) != 3:
        raise ValueError("Manifest must contain exactly three methods")
    roles = [method.get("role") for method in methods]
    if set(roles) != set(REQUIRED_ROLES) or len(set(roles)) != len(roles):
        raise ValueError(f"Method roles must be exactly {REQUIRED_ROLES}")
    labels = [str(method.get("label", "")).strip() for method in methods]
    if any(not label for label in labels) or len(set(labels)) != 3:
        raise ValueError("Method labels must be non-empty and unique")

    base = manifest_path.parent
    method_audits = []
    for method in methods:
        role = method["role"]
        selection = method.get("selection", {})
        basis = selection.get("basis")
        if basis not in ALLOWED_SELECTION_BASES:
            raise ValueError(
                f"{role}: selection basis must be one of {sorted(ALLOWED_SELECTION_BASES)}"
            )
        if selection.get("test_metrics_read") is not False:
            raise ValueError(f"{role}: selection.test_metrics_read must be false")
        rule = str(selection.get("rule", "")).strip()
        if not rule:
            raise ValueError(f"{role}: selection rule is required")
        if basis == "validation":
            source_range = tuple(int(value) for value in selection.get("source_range", ()))
            if len(source_range) != 2 or source_range[1] > snapshot_range[0]:
                raise ValueError(
                    f"{role}: validation source_range must end before test starts"
                )
        else:
            if "epoch" not in selection:
                raise ValueError(f"{role}: predeclared final selection requires epoch")

        evidence_path = resolve_path(base, selection.get("evidence_path", ""))
        evidence_hash = check_hash(
            evidence_path,
            selection.get("evidence_sha256", ""),
            f"{role} selection evidence",
        )
        source = method.get("input", {})
        kind = source.get("kind")
        if kind not in ("rows", "checkpoint"):
            raise ValueError(f"{role}: input.kind must be rows or checkpoint")
        if kind == "rows":
            input_path = resolve_path(base, source.get("path", ""))
            input_hash = check_hash(
                input_path, source.get("sha256", ""), f"{role} rows"
            )
            source_audit = {
                "kind": kind,
                "path": str(input_path),
                "sha256": input_hash,
            }
        else:
            checkpoint_path = resolve_path(base, source.get("checkpoint", ""))
            checkpoint_hash = check_hash(
                checkpoint_path,
                source.get("checkpoint_sha256", ""),
                f"{role} checkpoint",
            )
            runner_path = resolve_path(base, source.get("runner", ""))
            runner_hash = check_hash(
                runner_path, source.get("runner_sha256", ""), f"{role} runner"
            )
            model_class = str(source.get("model_class", "")).strip()
            if not model_class:
                raise ValueError(f"{role}: checkpoint input requires model_class")
            if "epoch" not in selection:
                raise ValueError(f"{role}: checkpoint input requires frozen selection.epoch")
            source_audit = {
                "kind": kind,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "runner": str(runner_path),
                "runner_sha256": runner_hash,
                "model_class": model_class,
            }
        method_audits.append(
            {
                "role": role,
                "label": method["label"],
                "selection": {
                    "basis": basis,
                    "rule": rule,
                    "test_metrics_read": False,
                    "evidence_path": str(evidence_path),
                    "evidence_sha256": evidence_hash,
                },
                "input": source_audit,
            }
        )
    return {
        "comparison_id": manifest.get("comparison_id"),
        "snapshot_range": list(snapshot_range),
        "evaluation": evaluation,
        "methods": method_audits,
        "selection_algorithm_in_evaluator": None,
        "checkpoint_discovery_or_globbing": False,
        "test_outcomes_used_to_choose_input": False,
    }


def normalize_rows(
    raw_rows: list[dict],
    role: str,
    label: str,
    start: int,
    end: int,
    method_filter: str | None = None,
) -> list[dict]:
    result = []
    for row_index, row in enumerate(raw_rows):
        if method_filter is not None and row.get("method") != method_filter:
            continue
        missing = set(STANDARD_FIELDS) - set(row)
        if missing:
            raise ValueError(f"{role} row {row_index} missing {sorted(missing)}")
        snapshot = int(row["snapshot"])
        if snapshot < start or snapshot >= end:
            continue
        normalized: dict[str, Any] = {"method": label, "role": role}
        for field in STANDARD_FIELDS:
            if field == "class":
                normalized[field] = str(row[field])
            elif field == "snapshot":
                normalized[field] = snapshot
            else:
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError(f"{role} row {row_index} has non-finite {field}")
                normalized[field] = value
        result.append(normalized)
    result.sort(key=lambda row: (int(row["snapshot"]), CLASSES.index(row["class"])))
    expected = (end - start) * len(CLASSES)
    if len(result) != expected:
        raise ValueError(f"{role}: expected {expected} rows, found {len(result)}")
    return result


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def checkpoint_state(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict) and value and all(
                torch.is_tensor(tensor) for tensor in value.values()
            ):
                return value
        if payload and all(torch.is_tensor(tensor) for tensor in payload.values()):
            return payload
    raise KeyError("Checkpoint does not contain a model state dictionary")


def actual_traffic_permutation(values, num_ods: int, paths_per_od: int):
    changed = list(values)
    total_l1 = 0.0
    for index in (2, 4, 6):
        actual = values[index]
        grouped = actual.reshape(actual.shape[0], num_ods, paths_per_od, -1)
        replacement = torch.flip(grouped, dims=(1,)).reshape_as(actual)
        changed[index] = replacement
        total_l1 += float((actual - replacement).abs().sum().item())
    if total_l1 == 0.0:
        raise RuntimeError("Strict-ESM audit permutation changed no actual traffic")
    return tuple(changed), total_l1


def policy_comparison(before, after) -> dict:
    result = {}
    for class_name, left, right in zip(CLASSES, before, after):
        delta = (left - right).abs()
        result[class_name] = {
            "torch_equal": bool(torch.equal(left, right)),
            "different_elements": int(torch.count_nonzero(left != right).item()),
            "max_abs_delta": float(delta.max().item()),
        }
    return result


def audit_strict_esm_policy(model, props, dataset, full_runtime) -> dict:
    shared = full_runtime.shared
    path_masks = shared.base.move_dataset_static(dataset, props.device)
    loader = shared.data_loader(dataset, 1, False, 0)
    values = shared.unpack_to_device(next(iter(loader)), props)
    total_paths = int(dataset.pte.shape[0])
    paths_per_od = int(props.num_paths_per_pair)
    num_ods = total_paths // paths_per_od

    model.eval()
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_policy = True
    props.research_return_admitted = False
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    changed_values, actual_l1 = actual_traffic_permutation(
        values, num_ods, paths_per_od
    )
    with torch.no_grad():
        before, _ = shared.model_forward(model, props, dataset, values, path_masks)
        after, _ = shared.model_forward(
            model, props, dataset, changed_values, path_masks
        )
    comparison = policy_comparison(before, after)
    if any(not row["torch_equal"] for row in comparison.values()):
        raise RuntimeError(f"Actual traffic affects policy: {comparison}")
    props.research_return_policy = False
    return {
        "snapshot": int(values[-1][0]),
        "actual_traffic_permutation_l1": actual_l1,
        "prediction_tensors_modified": False,
        "policy": comparison,
        "passes": True,
    }


def evaluate_checkpoint_input(
    method: dict,
    manifest_base: Path,
    evaluation: dict,
) -> tuple[list[dict], dict]:
    role = method["role"]
    label = method["label"]
    source = method["input"]
    runner_path = resolve_path(manifest_base, source["runner"])
    checkpoint_path = resolve_path(manifest_base, source["checkpoint"])
    module_name = f"strict2x_final_{role}_{sha256(runner_path)[:12]}"
    runtime = load_module(module_name, runner_path)
    full_runtime = getattr(runtime, "full", runtime)
    model_class = getattr(runtime, source["model_class"], None)
    if model_class is None:
        raise AttributeError(
            f"Runner {runner_path} has no model class {source['model_class']}"
        )

    device = torch.device("cpu")
    level = int(evaluation.get("level", 4))
    props = full_runtime.shared.build_props(level, device)
    props.checkpoint = 0
    start, end = (int(value) for value in evaluation["snapshots"])
    dataset = full_runtime.DM_Dataset_within_Cluster(props, 0, start, end)
    if int(dataset.max_source_index_read) != end - 1:
        raise RuntimeError(f"{role}: dataset split audit failed")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selected_epoch = int(method["selection"]["epoch"])
    if isinstance(payload, dict) and "epoch" in payload:
        if int(payload["epoch"]) != selected_epoch:
            raise RuntimeError(
                f"{role}: manifest epoch {selected_epoch} != checkpoint epoch {payload['epoch']}"
            )
    checkpoint_config = payload.get("config", {}) if isinstance(payload, dict) else {}
    if checkpoint_config:
        config_evaluation = tuple(int(value) for value in checkpoint_config.get("evaluation", ()))
        if config_evaluation and config_evaluation != (start, end):
            raise RuntimeError(
                f"{role}: checkpoint evaluation range {config_evaluation} != {(start, end)}"
            )
        config_prediction = checkpoint_config.get("prediction", "esm")
        if str(config_prediction).lower() != "esm":
            raise RuntimeError(f"{role}: checkpoint is not configured for ESM")
        config_topology = checkpoint_config.get("topology")
        if config_topology and config_topology != evaluation["topology"]:
            raise RuntimeError(
                f"{role}: checkpoint topology {config_topology!r} != "
                f"{evaluation['topology']!r}"
            )

    model = model_class(props).to(device=device, dtype=props.dtype)
    load_result = model.load_state_dict(checkpoint_state(payload), strict=True)
    strict_audit = audit_strict_esm_policy(model, props, dataset, full_runtime)
    started = time.perf_counter()
    rows, summary, diagnostics = full_runtime.evaluate_checkpoint(
        model, props, dataset, start
    )
    elapsed = time.perf_counter() - started
    normalized = normalize_rows(rows, role, label, start, end)
    audit = {
        "input_kind": "checkpoint",
        "runner": str(runner_path),
        "runner_sha256": sha256(runner_path),
        "model_class": source["model_class"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": (
            int(payload["epoch"]) if isinstance(payload, dict) and "epoch" in payload else None
        ),
        "missing_state_keys": list(load_result.missing_keys),
        "unexpected_state_keys": list(load_result.unexpected_keys),
        "strict_esm_policy_audit": strict_audit,
        "evaluation_seconds_cpu": elapsed,
        "runner_summary": summary,
        "runner_diagnostics": diagnostics,
    }
    return normalized, audit


def summarize(combined_rows: list[dict]) -> list[dict]:
    output = []
    for role in REQUIRED_ROLES:
        label = next(row["method"] for row in combined_rows if row["role"] == role)
        for class_name in CLASSES:
            values = np.asarray(
                [
                    float(row["norm_fulfill"])
                    for row in combined_rows
                    if row["role"] == role and row["class"] == class_name
                ],
                dtype=np.float64,
            )
            output.append(
                {
                    "method": label,
                    "role": role,
                    "class": class_name,
                    "count": int(values.size),
                    "norm_fulfill_mean": float(values.mean()),
                    "norm_fulfill_std": float(values.std(ddof=1)),
                    "norm_fulfill_p1": float(np.percentile(values, 1)),
                    "norm_fulfill_p10": float(np.percentile(values, 10)),
                    "norm_fulfill_p50": float(np.percentile(values, 50)),
                    "norm_fulfill_min": float(values.min()),
                    "norm_fulfill_max": float(values.max()),
                }
            )
    return output


def paired_bootstrap(
    combined_rows: list[dict],
    repetitions: int,
    seed: int,
) -> list[dict]:
    by_key = {
        (row["role"], row["class"], int(row["snapshot"])): float(row["norm_fulfill"])
        for row in combined_rows
    }
    snapshots = sorted({int(row["snapshot"]) for row in combined_rows})
    rng = np.random.default_rng(seed)
    result = []
    metric_functions = {
        "mean": np.mean,
        "p1": lambda values: np.percentile(values, 1),
        "p10": lambda values: np.percentile(values, 10),
    }
    for candidate_role in ("hattrick_e", "sparse_cross_attention"):
        candidate_label = next(
            row["method"] for row in combined_rows if row["role"] == candidate_role
        )
        for class_name in CLASSES:
            baseline = np.asarray(
                [by_key[("native_hattrick", class_name, snapshot)] for snapshot in snapshots]
            )
            candidate = np.asarray(
                [by_key[(candidate_role, class_name, snapshot)] for snapshot in snapshots]
            )
            sample_indices = rng.integers(
                0, len(snapshots), size=(int(repetitions), len(snapshots))
            )
            baseline_samples = baseline[sample_indices]
            candidate_samples = candidate[sample_indices]
            for metric_name, metric_function in metric_functions.items():
                estimate = float(metric_function(candidate) - metric_function(baseline))
                if metric_name == "mean":
                    draws = candidate_samples.mean(axis=1) - baseline_samples.mean(axis=1)
                else:
                    percentile = 1 if metric_name == "p1" else 10
                    draws = np.percentile(candidate_samples, percentile, axis=1) - np.percentile(
                        baseline_samples, percentile, axis=1
                    )
                result.append(
                    {
                        "candidate": candidate_label,
                        "candidate_role": candidate_role,
                        "baseline": "Hattrick",
                        "class": class_name,
                        "metric": metric_name,
                        "candidate_minus_baseline": estimate,
                        "ci95_low": float(np.percentile(draws, 2.5)),
                        "ci95_high": float(np.percentile(draws, 97.5)),
                        "probability_delta_gt_zero": float(np.mean(draws > 0.0)),
                        "paired_snapshots": len(snapshots),
                        "bootstrap_repetitions": int(repetitions),
                        "bootstrap_seed": int(seed),
                    }
                )
    return result


def run_manifest(
    manifest_path: Path,
    output_dir: Path,
    force: bool = False,
    bootstrap_repetitions: int = 10000,
) -> dict:
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_audit = validate_manifest(manifest, manifest_path)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Output directory is non-empty; pass --force to overwrite named outputs: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "selection_manifest_snapshot.json", manifest)

    evaluation = manifest["evaluation"]
    start, end = (int(value) for value in evaluation["snapshots"])
    combined_rows = []
    input_audits = []
    for method in manifest["methods"]:
        role = method["role"]
        label = method["label"]
        source = method["input"]
        if source["kind"] == "rows":
            rows_path = resolve_path(manifest_path.parent, source["path"])
            raw_rows = read_csv(rows_path)
            rows = normalize_rows(
                raw_rows,
                role,
                label,
                start,
                end,
                method_filter=source.get("method_filter"),
            )
            input_audit = {
                "role": role,
                "input_kind": "rows",
                "path": str(rows_path),
                "sha256": sha256(rows_path),
                "strict_esm_policy_audit": "trusted frozen rows provenance; no model replay",
            }
        else:
            rows, input_audit = evaluate_checkpoint_input(
                method, manifest_path.parent, evaluation
            )
            input_audit["role"] = role
        method_rows_path = output_dir / f"rows_{role}.csv"
        write_csv(method_rows_path, rows)
        input_audit["normalized_rows"] = str(method_rows_path)
        input_audit["normalized_rows_sha256"] = sha256(method_rows_path)
        input_audits.append(input_audit)
        combined_rows.extend(rows)

    class_rank = {name: index for index, name in enumerate(CLASSES)}
    role_rank = {name: index for index, name in enumerate(REQUIRED_ROLES)}
    combined_rows.sort(
        key=lambda row: (
            role_rank[row["role"]],
            int(row["snapshot"]),
            class_rank[row["class"]],
        )
    )
    validation = plot_cdf.validate_rows(
        combined_rows, evaluation_range=(start, end)
    )
    combined_path = output_dir / "comparison_rows.csv"
    write_csv(combined_path, combined_rows)

    summary_rows = summarize(combined_rows)
    summary_path = output_dir / "comparison_summary.csv"
    write_csv(summary_path, summary_rows)
    write_json(output_dir / "comparison_summary.json", summary_rows)
    bootstrap_rows = paired_bootstrap(
        combined_rows,
        repetitions=bootstrap_repetitions,
        seed=int(manifest.get("bootstrap_seed", 20260824)),
    )
    bootstrap_path = output_dir / "paired_bootstrap.csv"
    write_csv(bootstrap_path, bootstrap_rows)
    write_json(output_dir / "paired_bootstrap.json", bootstrap_rows)

    figure_path = output_dir / "cdf_norm_fulfill_strict2x.png"
    plot_metadata_path = output_dir / "cdf_metadata.json"
    plot_metadata = plot_cdf.render_cdf(
        combined_path,
        figure_path,
        plot_metadata_path,
        evaluation_range=(start, end),
    )
    provenance = {
        "status": "complete",
        "device": "cpu",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "selection_contract": manifest_audit,
        "input_audits": input_audits,
        "rows_validation": validation,
        "no_model_or_checkpoint_selection_performed": True,
        "test_rows_are_used_only_for_final_evaluation_and_reporting": True,
        "artifacts": {
            "comparison_rows.csv": sha256(combined_path),
            "comparison_summary.csv": sha256(summary_path),
            "paired_bootstrap.csv": sha256(bootstrap_path),
            "cdf_norm_fulfill_strict2x.png": sha256(figure_path),
            "cdf_metadata.json": sha256(plot_metadata_path),
        },
        "plot": plot_metadata,
    }
    write_json(output_dir / "comparison_provenance.json", provenance)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen-manifest strict-ESM evaluation for Hattrick, Hattrick-e, "
            "and sparse cross-attention"
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    if args.bootstrap_repetitions < 100:
        raise ValueError("bootstrap-repetitions must be at least 100")
    result = run_manifest(
        args.manifest,
        args.output_dir,
        force=args.force,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
