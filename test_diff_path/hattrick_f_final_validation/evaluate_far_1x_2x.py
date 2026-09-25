from __future__ import annotations

"""Strict-ESM far-horizon comparison for the matched seed-490 1x/2x models.

The routing policy sees ESM predictions only.  Actual traffic is introduced
only after policy generation, during sequential class-aware admission.
The reported metric is raw FulfillRatio because no far-horizon oracle is used.
"""

import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
ROOT = TEST_DIR.parent
OUTPUT_DIR = HERE / "artifacts" / "far_1x_2x_seed490"
SEED = 490
START = 9000
STOP = 9500
BATCH_SIZE = 8
CLASSES = ("High", "Medium", "Low")

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import evaluate_holdout as core  # noqa: E402


CHECKPOINTS = {
    ("1x", "hattrick"): TEST_DIR
    / "shared1x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
    / "best_model.pt",
    ("1x", "hattrick_f"): TEST_DIR
    / "shared1x_hattrick_f"
    / "artifacts"
    / "level4_confirmation"
    / "fh_release_low_budget_0p03"
    / "seed_490"
    / "best_model.pt",
    ("2x", "hattrick"): HERE
    / "artifacts"
    / "phase_a_current"
    / "seed_490"
    / "best_model.pt",
    ("2x", "hattrick_f"): HERE
    / "artifacts"
    / "hattrick_f"
    / "fh_release_low_budget_0p03"
    / "seed_490"
    / "best_model.pt",
}
LABELS = {"hattrick": "Hattrick", "hattrick_f": "Hattrick-f"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def method_specs(load: str) -> list[core.MethodSpec]:
    specs = []
    for method in ("hattrick", "hattrick_f"):
        checkpoint = CHECKPOINTS[(load, method)].resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        specs.append(
            core.MethodSpec(
                key=method,
                label=LABELS[method],
                checkpoint=checkpoint,
                artifact_dir=checkpoint.parent,
                artifact_audit={"load": load, "seed": SEED},
            )
        )
    return specs


def evaluate_load(
    load: str,
    load_factor: float,
    runtime,
    snapshot_module,
    cluster_module,
    device: torch.device,
) -> tuple[list[dict], list[dict], float]:
    core.LOAD_FACTOR = float(load_factor)
    props = runtime.build_props(4, device)
    props.topo = core.TOPOLOGY
    props.device = device
    props.dtype = torch.float32
    props.dynamic = 0
    props.checkpoint = 0
    props.path_mask = 0

    specs = method_specs(load)
    models, audits = core.load_models(specs, runtime, props, device)
    dataset = core.RawHoldoutDataset(
        props, START, STOP, snapshot_module, cluster_module
    )
    static = core.move_static(dataset, device)
    rows: list[dict] = []
    maximum_capacity_ratio = 0.0

    for batch in dataset.batches(BATCH_SIZE):
        node_features = batch["node_features"].to(device)
        capacities = batch["capacities"].to(device)
        actual = tuple(value.to(device) for value in batch["actual"])
        predicted = tuple(value.to(device) for value in batch["predicted"])
        for spec in specs:
            policy = core.generate_prediction_only_policy(
                models[spec.key],
                props,
                static,
                node_features,
                capacities,
                predicted,
            )
            admitted, cumulative_admitted, _ = core.sequential_actual_admission(
                models[spec.key], props, static, policy, actual, capacities
            )
            for class_index, class_name in enumerate(CLASSES):
                demand = (
                    actual[class_index].reshape(len(batch["indices"]), -1).sum(dim=1)
                    / float(core.PATHS_PER_OD)
                )
                admitted_total = admitted[class_index].sum(dim=1)
                fulfill = admitted_total / demand.clamp_min(1e-12)
                capacity_ratio = core.link_ratio(
                    cumulative_admitted[class_index], static.pte, capacities
                ).amax(dim=1)
                maximum_capacity_ratio = max(
                    maximum_capacity_ratio, float(capacity_ratio.max().item())
                )
                if float(fulfill.min().item()) < -1e-8:
                    raise RuntimeError("negative FulfillRatio")
                if float(fulfill.max().item()) > 1.0 + core.CAPACITY_TOLERANCE:
                    raise RuntimeError("raw FulfillRatio exceeds one")
                if float(capacity_ratio.max().item()) > 1.0 + core.CAPACITY_TOLERANCE:
                    raise RuntimeError("sequential admission exceeded capacity")
                for local, snapshot in enumerate(batch["indices"]):
                    rows.append(
                        {
                            "load": load,
                            "load_factor": load_factor,
                            "seed": SEED,
                            "snapshot": int(snapshot),
                            "method": spec.key,
                            "method_label": spec.label,
                            "class": class_name,
                            "raw_fulfill_ratio": min(
                                float(fulfill[local].item()), 1.0
                            ),
                            "actual_demand": float(demand[local].item()),
                            "admitted_traffic": float(admitted_total[local].item()),
                            "post_admission_capacity_ratio": float(
                                capacity_ratio[local].item()
                            ),
                        }
                    )
        print(
            f"[far {load}] snapshots={batch['indices'][0]}..{batch['indices'][-1]}",
            flush=True,
        )
    return rows, audits, maximum_capacity_ratio


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for load in ("1x", "2x"):
        for method in ("hattrick", "hattrick_f"):
            for class_name in CLASSES:
                selected = [
                    float(row["raw_fulfill_ratio"])
                    for row in rows
                    if row["load"] == load
                    and row["method"] == method
                    and row["class"] == class_name
                ]
                values = np.asarray(selected, dtype=np.float64)
                if values.size != STOP - START:
                    raise RuntimeError(
                        f"{load}/{method}/{class_name}: expected 500 rows, "
                        f"got {values.size}"
                    )
                output.append(
                    {
                        "load": load,
                        "seed": SEED,
                        "method": method,
                        "method_label": LABELS[method],
                        "class": class_name,
                        "n": int(values.size),
                        "mean": float(values.mean()),
                        "p1": float(np.percentile(values, 1)),
                        "p10": float(np.percentile(values, 10)),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }
                )
    return output


def main() -> None:
    core.verify_frozen_sources()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_dir = str((TEST_DIR / "shared2x_order_regularizer").resolve())
    if shared_dir not in sys.path:
        sys.path.insert(0, shared_dir)
    root_string = str(ROOT.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

    runtime = core.load_module("far_1x_2x_runtime", core.RUNTIME_SOURCE)
    snapshot_module = core.load_module(
        "far_1x_2x_snapshot", ROOT / "utils" / "snapshot_utils.py"
    )
    cluster_module = core.load_module(
        "far_1x_2x_cluster", ROOT / "utils" / "cluster_utils.py"
    )

    started = time.perf_counter()
    all_rows: list[dict] = []
    all_audits: list[dict] = []
    capacity_audit = {}
    for load, factor in (("1x", 1.0), ("2x", 2.0)):
        rows, audits, maximum_ratio = evaluate_load(
            load,
            factor,
            runtime,
            snapshot_module,
            cluster_module,
            device,
        )
        all_rows.extend(rows)
        all_audits.extend(audits)
        capacity_audit[load] = {
            "maximum_post_admission_ratio": maximum_ratio,
            "passes": maximum_ratio <= 1.0 + core.CAPACITY_TOLERANCE,
        }

    summary_rows = summarize(all_rows)
    snapshot_csv = OUTPUT_DIR / "snapshots.csv"
    summary_csv = OUTPUT_DIR / "summary.csv"
    write_csv(snapshot_csv, all_rows)
    write_csv(summary_csv, summary_rows)
    result = {
        "experiment": "strict-ESM far-horizon 1x/2x matched seed-490 comparison",
        "metric": "raw FulfillRatio (not oracle-normalized NormFulFill)",
        "strict_esm_contract": {
            "policy_input": "ESM prediction only",
            "actual_traffic": "used only after policy generation for sequential admission",
            "oracle_loaded": False,
        },
        "window": [START, STOP],
        "seed": SEED,
        "loads": ["1x", "2x"],
        "methods": ["hattrick", "hattrick_f"],
        "device": str(device),
        "runtime_seconds": time.perf_counter() - started,
        "checkpoint_audit": all_audits,
        "checkpoint_sha256": {
            f"{load}_{method}": sha256(path)
            for (load, method), path in CHECKPOINTS.items()
        },
        "capacity_audit": capacity_audit,
        "groups": summary_rows,
        "snapshot_csv": str(snapshot_csv.resolve()),
        "snapshot_csv_sha256": sha256(snapshot_csv),
        "summary_csv": str(summary_csv.resolve()),
        "summary_csv_sha256": sha256(summary_csv),
    }
    write_json(OUTPUT_DIR / "summary.json", result)
    print(OUTPUT_DIR.resolve(), flush=True)


if __name__ == "__main__":
    main()
