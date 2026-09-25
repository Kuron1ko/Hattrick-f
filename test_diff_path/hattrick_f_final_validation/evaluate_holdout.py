from __future__ import annotations

"""Strict-ESM temporal holdout and global prediction-bias evaluation.

The routing-policy function in this file deliberately has no actual-traffic
argument.  It receives only ESM traffic and uses zero placeholders for the
``tm*`` parameters retained by Hattrick's historical forward signature.  The
unmodified actual traffic is introduced later, and only to sequential
admission.  No 1x oracle file is loaded or used by this evaluator.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from frozen_sources import EXPECTED as FROZEN_EXPECTED
from frozen_sources import ROOT as FROZEN_ROOT
from frozen_sources import verify as verify_frozen_sources

# The long GEANT trace was serialized by NumPy 2.x, while the repository's
# CUDA environment currently carries NumPy 1.26.  NumPy only renamed this
# private module; registering the old implementation under the new pickle
# module name preserves the ndarray payload without transforming the data.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
if "numpy._core.multiarray" not in sys.modules:
    sys.modules["numpy._core.multiarray"] = np.core.multiarray


HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
ROOT = TEST_DIR.parent
RUNTIME_SOURCE = TEST_DIR / "shared2x_order_regularizer" / "run_experiment.py"
RESULTS_DIR = ROOT / "results" / "geant" / "8sp" / "0"
OUTPUT_ROOT = HERE / "artifacts" / "holdout"

TOPOLOGY = "geant"
PATHS_PER_OD = 8
LOAD_FACTOR = 2.0
CLASSES = ("High", "Medium", "Low")
DEFAULT_BIASES = (0.8, 0.9, 1.0, 1.1, 1.2)
WINDOWS = {
    "near": (500, 1000),
    "far": (9000, 9500),
}
CAPACITY_TOLERANCE = 1e-4
EXPECTED_SEEDS = (490, 491, 492)
EXPECTED_METHODS = ("phase_a", "hattrick_f", "six_loss_control")
EXPECTED_WINDOW_COUNTS = {name: stop - start for name, (start, stop) in WINDOWS.items()}
PHASE_A_OBJECTIVES = ("Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml")
HATTRICK_F_OBJECTIVES = ("Fh", "Fhm", "Fhml")
PHASE_A_SOURCE_HASHES = {
    "run_experiment.py": "a77c112d25f549e2cbe73b9a41f4ef4cebe10b60061e8683f9faebd86b112a42",
    "ordered_projection.py": "a8ed960f641e9cb0786ab8edee3b4a3c571255c3e76c939f0d6803ce01543e3a",
    "frameworks/hattrick_system.py": "4f7b15dba81b23f0e23d432e0b65f8ae895b459070dd8249a22c1345b15eb862",
    "utils/training_utils.py": "b4b8c08ef353194ae107fc6756bde6f7668f12e5d6e87bf5f26280dbf4057185",
    "shared2x_order_regularizer/run_experiment.py": "317d0b796773963352a5b92652e6273074707bad42b3078a734e569366789310",
}
CONTINUATION_SOURCE_HASHES = {
    "run_level4.py": "dc13339b20a28ae9d6520e5466cd4bc5f0c7ec5ec704d5aabcf072884cb6d8d9",
    "run_experiment.py": "33d9d4d634d6661e935ba5800a90df4b2f5eaeeb0a42fb4465c1528b89025d18",
    "ordered_projection.py": "a8ed960f641e9cb0786ab8edee3b4a3c571255c3e76c939f0d6803ce01543e3a",
    "hattrick_system.py": "4f7b15dba81b23f0e23d432e0b65f8ae895b459070dd8249a22c1345b15eb862",
}
CONTROL_SOURCE_HASHES = {
    f"six_loss_dependency::{key}": value
    for key, value in PHASE_A_SOURCE_HASHES.items()
}
FROZEN_CORE_SOURCE_HASHES = {
    f"frozen::{path.resolve().relative_to(FROZEN_ROOT.resolve()).as_posix()}": digest
    for path, digest in FROZEN_EXPECTED.items()
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, source: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty snapshot CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def checkpoint_state(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict) and value and all(
                torch.is_tensor(tensor) for tensor in value.values()
            ):
                return value
        if payload and all(torch.is_tensor(tensor) for tensor in payload.values()):
            return payload
    raise KeyError("checkpoint contains no model state dictionary")


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    checkpoint: Path
    artifact_dir: Path
    artifact_audit: dict


class RawHoldoutDataset:
    """Load GEANT traffic/topology without loading any optimizer oracle file."""

    def __init__(
        self,
        props,
        start: int,
        stop: int,
        snapshot_module: ModuleType,
        cluster_module: ModuleType,
    ) -> None:
        if start < 0 or stop <= start:
            raise ValueError(f"invalid holdout range [{start}, {stop})")
        filenames = np.loadtxt(
            RESULTS_DIR / "filenames.txt",
            dtype="U",
            delimiter=",",
            skiprows=start,
            max_rows=stop - start,
        ).reshape(-1, 3)
        if len(filenames) != stop - start:
            raise RuntimeError(
                f"requested {stop - start} snapshots, loaded {len(filenames)}"
            )

        self.start = int(start)
        self.stop = int(stop)
        self.indices = list(range(start, stop))
        self.node_features: list[torch.Tensor] = []
        self.capacities: list[torch.Tensor] = []
        self.actual: list[tuple[torch.Tensor, ...]] = []
        self.predicted: list[tuple[torch.Tensor, ...]] = []
        final_snapshot = None
        for topology_file, pairs_file, tm_file in filenames:
            snapshot = snapshot_module.Read_Snapshot(
                props, topology_file, pairs_file, tm_file
            )
            final_snapshot = snapshot
            self.node_features.append(
                torch.as_tensor(snapshot._node_features, dtype=torch.float32)
            )
            self.capacities.append(
                torch.as_tensor(snapshot.capacities, dtype=torch.float32)
            )
            self.actual.append(
                tuple(
                    torch.as_tensor(value, dtype=torch.float32) * LOAD_FACTOR
                    for value in (snapshot.tm1, snapshot.tm2, snapshot.tm3)
                )
            )
            self.predicted.append(
                tuple(
                    torch.as_tensor(value, dtype=torch.float32) * LOAD_FACTOR
                    for value in (
                        snapshot.tm1_pred,
                        snapshot.tm2_pred,
                        snapshot.tm3_pred,
                    )
                )
            )
        if final_snapshot is None:
            raise RuntimeError("empty holdout dataset")

        cluster = cluster_module.Cluster_Info(final_snapshot, props, 0)
        self.directed_edges = list(final_snapshot.graph.edges())
        self.edge_index = cluster.sp.get_edge_index()
        self.pij = cluster.compute_ksp_paths(PATHS_PER_OD, cluster.sp.pairs)
        self.pte = cluster.get_paths_to_edges_matrix(self.pij)
        (
            self.padded_edge_ids_per_path,
            self.edge_ids_dict_tensor,
            self.original_pos_edge_ids_dict_tensor,
        ) = cluster.get_padded_edge_ids_per_path(self.pij, cluster.edges_map)
        self.num_pairs = int(cluster.num_pairs)
        if int(self.pte.shape[0]) != self.num_pairs * PATHS_PER_OD:
            raise RuntimeError("path incidence shape does not match 8 paths per OD")

    def batches(self, batch_size: int):
        for left in range(0, len(self.indices), batch_size):
            right = min(left + batch_size, len(self.indices))
            yield {
                "indices": self.indices[left:right],
                "node_features": torch.stack(self.node_features[left:right]),
                "capacities": torch.stack(self.capacities[left:right]),
                "actual": tuple(
                    torch.stack([row[class_index] for row in self.actual[left:right]])
                    for class_index in range(3)
                ),
                "predicted": tuple(
                    torch.stack(
                        [row[class_index] for row in self.predicted[left:right]]
                    )
                    for class_index in range(3)
                ),
            }


@dataclass(frozen=True)
class StaticTopology:
    edge_index: torch.Tensor
    pte: torch.Tensor
    padded_edge_ids_per_path: torch.Tensor
    edge_ids_dict_tensor: dict
    original_pos_edge_ids_dict_tensor: dict
    pte_info: tuple


def move_static(dataset: RawHoldoutDataset, device: torch.device) -> StaticTopology:
    pte = dataset.pte.to(device=device, dtype=torch.float32).coalesce()
    padded = dataset.padded_edge_ids_per_path.to(device)
    edge_maps = []
    for source in (
        dataset.edge_ids_dict_tensor,
        dataset.original_pos_edge_ids_dict_tensor,
    ):
        edge_maps.append({key: value.to(device) for key, value in source.items()})
    indices = pte.indices()
    return StaticTopology(
        edge_index=dataset.edge_index.to(device),
        pte=pte,
        padded_edge_ids_per_path=padded,
        edge_ids_dict_tensor=edge_maps[0],
        original_pos_edge_ids_dict_tensor=edge_maps[1],
        pte_info=(pte, indices[0], indices[1], pte.values()),
    )


def generate_prediction_only_policy(
    model,
    props,
    static: StaticTopology,
    node_features: torch.Tensor,
    capacities: torch.Tensor,
    predicted_tms: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Generate routing without accepting or observing actual traffic."""

    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    model.eval()
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_admitted = False
    props.research_return_policy = True
    zero_actual = tuple(torch.zeros_like(value) for value in predicted_tms)
    with torch.no_grad():
        policy = model(
            props,
            node_features[:1],
            static.edge_index,
            capacities[:1],
            static.padded_edge_ids_per_path,
            zero_actual[0],
            predicted_tms[0],
            zero_actual[1],
            predicted_tms[1],
            zero_actual[2],
            predicted_tms[2],
            static.pte,
            static.edge_ids_dict_tensor,
            static.original_pos_edge_ids_dict_tensor,
            None,
        )
    props.research_return_policy = False
    return tuple(value.detach() for value in policy)


def sequential_actual_admission(
    model,
    props,
    static: StaticTopology,
    policy: tuple[torch.Tensor, ...],
    actual_tms: tuple[torch.Tensor, ...],
    capacities: torch.Tensor,
) -> tuple[tuple[torch.Tensor, ...], list[torch.Tensor], list[torch.Tensor]]:
    """Replay a fixed policy; actual traffic first appears in this function."""

    batch_size = int(capacities.shape[0])
    with torch.no_grad():
        admitted_ratios = model.simulate(
            list(policy),
            list(actual_tms),
            capacities,
            static.pte_info,
            batch_size,
            props,
            rate_cap=props.rate_cap,
        )[:3]
    admitted = tuple(
        ratio.reshape(batch_size, -1) * actual.reshape(batch_size, -1)
        for ratio, actual in zip(admitted_ratios, actual_tms)
    )
    cumulative_admitted: list[torch.Tensor] = []
    cumulative_offered: list[torch.Tensor] = []
    admitted_sum = torch.zeros_like(admitted[0])
    offered_sum = torch.zeros_like(admitted[0])
    for class_index in range(3):
        admitted_sum = admitted_sum + admitted[class_index]
        offered_sum = offered_sum + (
            policy[class_index].reshape(batch_size, -1)
            * actual_tms[class_index].reshape(batch_size, -1)
        )
        cumulative_admitted.append(admitted_sum.clone())
        cumulative_offered.append(offered_sum.clone())
    return admitted, cumulative_admitted, cumulative_offered


def link_ratio(
    path_traffic: torch.Tensor,
    pte: torch.Tensor,
    capacities: torch.Tensor,
) -> torch.Tensor:
    link_load = torch.sparse.mm(
        pte.t(), path_traffic.to(dtype=torch.float32).t()
    ).t()
    return link_load / capacities.to(dtype=torch.float32).clamp_min(1e-12)


def batch_rows(
    *,
    seed: int,
    method: MethodSpec,
    window: str,
    bias: float,
    indices: list[int],
    actual: tuple[torch.Tensor, ...],
    predicted: tuple[torch.Tensor, ...],
    admitted: tuple[torch.Tensor, ...],
    cumulative_admitted: list[torch.Tensor],
    cumulative_offered: list[torch.Tensor],
    capacities: torch.Tensor,
    static: StaticTopology,
) -> list[dict]:
    rows: list[dict] = []
    k = float(PATHS_PER_OD)
    for class_index, class_name in enumerate(CLASSES):
        demand = actual[class_index].reshape(len(indices), -1).sum(dim=1) / k
        predicted_demand = (
            predicted[class_index].reshape(len(indices), -1).sum(dim=1) / k
        )
        admitted_total = admitted[class_index].sum(dim=1)
        fulfill = admitted_total / demand.clamp_min(1e-12)
        admitted_ratio = link_ratio(
            cumulative_admitted[class_index], static.pte, capacities
        ).amax(dim=1)
        offered_ratio = link_ratio(
            cumulative_offered[class_index], static.pte, capacities
        ).amax(dim=1)
        if float(fulfill.min().item()) < -CAPACITY_TOLERANCE:
            raise RuntimeError("negative FulfillRatio")
        if float(fulfill.max().item()) > 1.0 + CAPACITY_TOLERANCE:
            raise RuntimeError(
                f"raw FulfillRatio exceeds one: {float(fulfill.max().item())}"
            )
        if float(admitted_ratio.max().item()) > 1.0 + CAPACITY_TOLERANCE:
            raise RuntimeError(
                "sequential admission exceeded capacity: "
                f"{float(admitted_ratio.max().item())}"
            )
        for local, snapshot in enumerate(indices):
            rows.append(
                {
                    "seed": seed,
                    "method": method.key,
                    "method_label": method.label,
                    "window": window,
                    "snapshot": snapshot,
                    "esm_global_bias": bias,
                    "class": class_name,
                    "actual_demand_2x": float(demand[local].item()),
                    "predicted_demand_2x_biased": float(
                        predicted_demand[local].item()
                    ),
                    "admitted_traffic": float(admitted_total[local].item()),
                    "raw_fulfill_ratio": float(fulfill[local].item()),
                    "pre_admission_capacity_ratio": float(
                        offered_ratio[local].item()
                    ),
                    "post_admission_capacity_ratio": float(
                        admitted_ratio[local].item()
                    ),
                    "capacity_violation": max(
                        float(admitted_ratio[local].item()) - 1.0, 0.0
                    ),
                }
            )
    return rows


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            int(row["seed"]),
            str(row["method"]),
            str(row["method_label"]),
            str(row["window"]),
            float(row["esm_global_bias"]),
            str(row["class"]),
        )
        groups.setdefault(key, []).append(row)
    output: list[dict] = []
    for key, selected in sorted(groups.items()):
        seed, method, label, window, bias, class_name = key
        fulfill = np.asarray(
            [float(row["raw_fulfill_ratio"]) for row in selected],
            dtype=np.float64,
        )
        admitted = np.asarray(
            [float(row["admitted_traffic"]) for row in selected], dtype=np.float64
        )
        output.append(
            {
                "seed": seed,
                "method": method,
                "method_label": label,
                "window": window,
                "esm_global_bias": bias,
                "class": class_name,
                "n": int(fulfill.size),
                "raw_fulfill_mean": float(fulfill.mean()),
                "raw_fulfill_p1": percentile(fulfill, 1),
                "raw_fulfill_p10": percentile(fulfill, 10),
                "raw_fulfill_min": float(fulfill.min()),
                "raw_fulfill_max": float(fulfill.max()),
                "admitted_traffic_mean": float(admitted.mean()),
                "post_admission_capacity_ratio_max": max(
                    float(row["post_admission_capacity_ratio"])
                    for row in selected
                ),
                "capacity_violation_max": max(
                    float(row["capacity_violation"]) for row in selected
                ),
            }
        )
    return output


def paired_differences(rows: list[dict]) -> list[dict]:
    values = {
        (
            int(row["seed"]),
            str(row["window"]),
            float(row["esm_global_bias"]),
            str(row["class"]),
            int(row["snapshot"]),
            str(row["method"]),
        ): row
        for row in rows
    }
    candidates = sorted({str(row["method"]) for row in rows} - {"phase_a"})
    combinations = sorted(
        {
            (
                int(row["seed"]),
                str(row["window"]),
                float(row["esm_global_bias"]),
                str(row["class"]),
            )
            for row in rows
        }
    )
    output = []
    for seed, window, bias, class_name in combinations:
        snapshots = sorted(
            {
                int(row["snapshot"])
                for row in rows
                if int(row["seed"]) == seed
                and str(row["window"]) == window
                and float(row["esm_global_bias"]) == bias
                and str(row["class"]) == class_name
            }
        )
        for candidate in candidates:
            deltas = []
            admitted_deltas = []
            for snapshot in snapshots:
                prefix = (seed, window, bias, class_name, snapshot)
                baseline = values.get((*prefix, "phase_a"))
                proposed = values.get((*prefix, candidate))
                if baseline is None or proposed is None:
                    continue
                deltas.append(
                    float(proposed["raw_fulfill_ratio"])
                    - float(baseline["raw_fulfill_ratio"])
                )
                admitted_deltas.append(
                    float(proposed["admitted_traffic"])
                    - float(baseline["admitted_traffic"])
                )
            if deltas:
                delta = np.asarray(deltas, dtype=np.float64)
                output.append(
                    {
                        "seed": seed,
                        "candidate": candidate,
                        "baseline": "phase_a",
                        "window": window,
                        "esm_global_bias": bias,
                        "class": class_name,
                        "n": int(delta.size),
                        "raw_fulfill_mean_delta": float(delta.mean()),
                        "admitted_traffic_mean_delta": float(
                            np.mean(admitted_deltas)
                        ),
                        "candidate_better_fraction": float(np.mean(delta > 0)),
                    }
                )
    return output


def require_source_hashes(
    config: dict, expected: dict[str, str], artifact_name: str
) -> None:
    recorded = config.get("source_sha256", {})
    mismatches = {
        key: {"expected": digest, "actual": recorded.get(key)}
        for key, digest in expected.items()
        if recorded.get(key) != digest
    }
    if mismatches:
        raise RuntimeError(
            f"{artifact_name} was not produced by the frozen current sources: "
            f"{mismatches}"
        )


def require_phase_a_training_lineage(config: dict) -> None:
    expected = {
        "level": 4,
        "label": "level4_confirmation",
        "epochs": 60,
        "topology": "geant_priomask500_shared_load2x_train",
        "paths_per_pair": 8,
        "load_factor": 2.0,
        "shared_paths": True,
        "train": [0, 350],
        "validation": [350, 400],
        "evaluation": [400, 500],
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if tuple(config.get("objectives", ())) != PHASE_A_OBJECTIVES:
        mismatches["objectives"] = {
            "expected": list(PHASE_A_OBJECTIVES),
            "actual": config.get("objectives"),
        }
    if mismatches:
        raise RuntimeError(
            f"phase_a is not the frozen six-loss 60-epoch Level-4 run: {mismatches}"
        )
    require_source_hashes(config, PHASE_A_SOURCE_HASHES, "phase_a")


def require_continuation_lineage(config: dict, key: str) -> None:
    objectives = (
        HATTRICK_F_OBJECTIVES if key == "hattrick_f" else PHASE_A_OBJECTIVES
    )
    persistent_mlu = key == "six_loss_control"
    expected = {
        "level": 4,
        "phase_f_epochs": 30,
        "train": [0, 350],
        "validation_selection_only": [350, 400],
        "independent_evaluation": [400, 500],
    }
    mismatches = {
        field: {"expected": value, "actual": config.get(field)}
        for field, value in expected.items()
        if config.get(field) != value
    }
    phase_f = config.get("phase_f")
    if not isinstance(phase_f, dict):
        mismatches["phase_f"] = {"expected": "dictionary", "actual": phase_f}
    else:
        phase_f_expected = {
            "objectives": list(objectives),
            "optimizer_reset": True,
            "all_parameters_trainable": True,
            "persistent_mlu_minimization": persistent_mlu,
        }
        for field, value in phase_f_expected.items():
            if phase_f.get(field) != value:
                mismatches[f"phase_f.{field}"] = {
                    "expected": value,
                    "actual": phase_f.get(field),
                }
    phase_a = config.get("phase_a")
    if not isinstance(phase_a, dict) or tuple(
        phase_a.get("objectives", ())
    ) != PHASE_A_OBJECTIVES:
        mismatches["phase_a.objectives"] = {
            "expected": list(PHASE_A_OBJECTIVES),
            "actual": phase_a.get("objectives") if isinstance(phase_a, dict) else None,
        }
    selection = config.get("selection_gate")
    if not isinstance(selection, dict) or float(
        selection.get("low_absolute_budget", float("nan"))
    ) != 0.03:
        mismatches["selection_gate.low_absolute_budget"] = {
            "expected": 0.03,
            "actual": selection.get("low_absolute_budget")
            if isinstance(selection, dict)
            else None,
        }
    if mismatches:
        raise RuntimeError(f"{key} continuation lineage mismatch: {mismatches}")
    require_source_hashes(config, CONTINUATION_SOURCE_HASHES, key)
    require_source_hashes(config, FROZEN_CORE_SOURCE_HASHES, key)
    if key == "six_loss_control":
        require_source_hashes(config, CONTROL_SOURCE_HASHES, key)


def validate_completed_artifact(
    key: str,
    label: str,
    directory: Path,
    seed: int,
    phase_a_sha256: str | None,
) -> MethodSpec:
    """Validate completion, identity and checkpoint provenance before use."""
    directory = directory.resolve()
    checkpoint = directory / "best_model.pt"
    config_path = directory / "config.json"
    complete_path = directory / "complete.json"
    required = {
        "best_model.pt": checkpoint,
        "config.json": config_path,
        "complete.json": complete_path,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"incomplete {key} artifact for seed {seed} at {directory}: "
            f"missing {missing}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = payload.get("config") if isinstance(payload, dict) else None
    if int(config.get("seed", -1)) != int(seed):
        raise RuntimeError(
            f"{key} config seed mismatch: expected {seed}, got {config.get('seed')}"
        )
    if not isinstance(checkpoint_config, dict):
        raise RuntimeError(f"{key} checkpoint has no config dictionary")
    if int(checkpoint_config.get("seed", -1)) != int(seed):
        raise RuntimeError(
            f"{key} checkpoint config seed mismatch: expected {seed}, "
            f"got {checkpoint_config.get('seed')}"
        )
    if checkpoint_config != config:
        raise RuntimeError(f"{key} config.json and checkpoint config differ")
    if key == "phase_a":
        if complete.get("status") not in (None, "COMPLETE"):
            raise RuntimeError(
                f"{key} completion status is not COMPLETE: {complete.get('status')!r}"
            )
    elif complete.get("status") != "COMPLETE":
        raise RuntimeError(
            f"{key} completion status is not COMPLETE: {complete.get('status')!r}"
        )
    for possible_seed in (
        complete.get("seed"),
        (complete.get("config") or {}).get("seed")
        if isinstance(complete.get("config"), dict)
        else None,
    ):
        if possible_seed is not None and int(possible_seed) != int(seed):
            raise RuntimeError(f"{key} complete.json seed mismatch")

    if key == "phase_a":
        declared_method = config.get("method")
        if declared_method not in (None, "Hattrick", "Hattrick (Phase-A)"):
            raise RuntimeError(
                f"phase_a method mismatch: {declared_method!r}"
            )
        if tuple(config.get("objectives", ())) != PHASE_A_OBJECTIVES:
            raise RuntimeError(
                "phase_a method identity failed: expected the six original objectives"
            )
        require_phase_a_training_lineage(config)
    else:
        expected_method = {
            "hattrick_f": "Hattrick-f",
            "six_loss_control": "Hattrick six-loss continuation control",
        }[key]
        if config.get("method") != expected_method:
            raise RuntimeError(
                f"{key} method mismatch: expected {expected_method!r}, "
                f"got {config.get('method')!r}"
            )
        require_continuation_lineage(config, key)

    actual_hash = sha256(checkpoint)
    recorded_hash = complete.get("artifact_sha256", {}).get("best_model.pt")
    if recorded_hash is None:
        recorded_hash = complete.get("best_checkpoint_sha256")
    if not recorded_hash or recorded_hash != actual_hash:
        raise RuntimeError(
            f"{key} complete/checkpoint SHA-256 mismatch: "
            f"recorded={recorded_hash!r}, actual={actual_hash}"
        )
    if int(complete.get("best_epoch", -1)) != int(payload.get("epoch", -2)):
        raise RuntimeError(f"{key} complete/checkpoint best_epoch mismatch")
    if list(complete.get("best_rank", [])) != list(payload.get("rank", [])):
        raise RuntimeError(f"{key} complete/checkpoint best_rank mismatch")

    recorded_phase_a = None
    if key != "phase_a":
        phase_a = config.get("phase_a")
        if not isinstance(phase_a, dict) or not phase_a.get("sha256"):
            raise RuntimeError(f"{key} config has no Phase-A SHA-256 provenance")
        recorded_phase_a = str(phase_a["sha256"])
        if phase_a_sha256 is None or recorded_phase_a != phase_a_sha256:
            raise RuntimeError(
                f"{key} Phase-A provenance mismatch for seed {seed}: "
                f"recorded={recorded_phase_a}, selected={phase_a_sha256}"
            )
        integrity = config.get("phase_a_integrity")
        if isinstance(integrity, dict) and integrity.get("checkpoint_sha256") not in (
            None,
            phase_a_sha256,
        ):
            raise RuntimeError(f"{key} phase_a_integrity SHA-256 mismatch")

    audit = {
        "artifact_dir": str(directory),
        "complete_json": str(complete_path),
        "config_json": str(config_path),
        "checkpoint_sha256": actual_hash,
        "recorded_phase_a_sha256": recorded_phase_a,
        "seed_verified": True,
        "method_verified": True,
        "completion_verified": True,
        "complete_checkpoint_consistent": True,
        "frozen_source_provenance_verified": True,
        "training_lineage_verified": True,
    }
    return MethodSpec(key, label, checkpoint, directory, audit)


def select_completed_artifact(
    key: str,
    label: str,
    directories: list[Path],
    seed: int,
    phase_a_sha256: str | None,
) -> MethodSpec:
    # If a higher-priority artifact directory exists, it must be valid.  An
    # incomplete current run is never silently replaced with an older run.
    for directory in directories:
        if directory.exists():
            return validate_completed_artifact(
                key, label, directory, seed, phase_a_sha256
            )
    raise FileNotFoundError(
        f"no completed {key} artifact directory for seed {seed}; checked "
        f"{[str(path.resolve()) for path in directories]}"
    )


def default_methods(
    seed: int, allow_missing: bool = True
) -> tuple[list[MethodSpec], list[dict]]:
    candidates = {
        "phase_a": [
            HERE / "artifacts" / "phase_a_current" / f"seed_{seed}",
            TEST_DIR
            / "shared2x_full_objectives"
            / "artifacts"
            / "level4_confirmation"
            / f"seed_{seed}",
        ],
        "hattrick_f": [
            HERE
            / "artifacts"
            / "hattrick_f"
            / "fh_release_low_budget_0p03"
            / f"seed_{seed}",
            TEST_DIR
            / "shared2x_hattrick_f"
            / "artifacts"
            / "level4_confirmation"
            / "fh_release_low_budget_0p03"
            / f"seed_{seed}",
        ],
        "six_loss_control": [
            HERE
            / "artifacts"
            / "six_loss_continuation_control"
            / "fh_release_low_budget_0p03"
            / f"seed_{seed}"
        ],
    }
    labels = {
        "phase_a": "Hattrick (Phase-A)",
        "hattrick_f": "Hattrick-f",
        "six_loss_control": "Hattrick six-loss continuation control",
    }
    methods: list[MethodSpec] = []
    skipped: list[dict] = []
    # Phase-A is the provenance anchor and is mandatory even in smoke mode.
    phase_a = select_completed_artifact(
        "phase_a", labels["phase_a"], candidates["phase_a"], seed, None
    )
    methods.append(phase_a)
    phase_a_sha = phase_a.artifact_audit["checkpoint_sha256"]
    for key in ("hattrick_f", "six_loss_control"):
        try:
            methods.append(
                select_completed_artifact(
                    key, labels[key], candidates[key], seed, phase_a_sha
                )
            )
        except FileNotFoundError:
            missing = {
                "seed": seed,
                "method": key,
                "reason": "completed_artifact_missing",
                "searched": [str(path.resolve()) for path in candidates[key]],
            }
            if not allow_missing:
                raise FileNotFoundError(json.dumps(missing, ensure_ascii=False))
            skipped.append(missing)
    return methods, skipped


def load_models(
    methods: list[MethodSpec], runtime: ModuleType, props, device: torch.device
) -> tuple[dict[str, torch.nn.Module], list[dict]]:
    models = {}
    audit = []
    for method in methods:
        payload = torch.load(method.checkpoint, map_location=device, weights_only=False)
        model = runtime.Hattrick(props).to(device=device, dtype=props.dtype)
        load_result = model.load_state_dict(checkpoint_state(payload), strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models[method.key] = model
        audit.append(
            {
                "method": method.key,
                "label": method.label,
                "checkpoint": str(method.checkpoint),
                "checkpoint_sha256": sha256(method.checkpoint),
                **method.artifact_audit,
                "checkpoint_epoch": (
                    int(payload["epoch"])
                    if isinstance(payload, dict) and "epoch" in payload
                    else None
                ),
                "missing_state_keys": list(load_result.missing_keys),
                "unexpected_state_keys": list(load_result.unexpected_keys),
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            }
        )
    return models, audit


def formal_shape_reasons(args: argparse.Namespace) -> list[str]:
    reasons = []
    if tuple(args.seeds) != EXPECTED_SEEDS:
        reasons.append(
            f"seeds must be exactly {list(EXPECTED_SEEDS)}, got {list(args.seeds)}"
        )
    if set(args.windows) != set(WINDOWS) or len(args.windows) != len(WINDOWS):
        reasons.append(
            f"windows must be exactly {list(WINDOWS)}, got {list(args.windows)}"
        )
    if tuple(float(value) for value in args.biases) != DEFAULT_BIASES:
        reasons.append(
            f"biases must be exactly {list(DEFAULT_BIASES)} for a formal result"
        )
    if args.max_snapshots is not None:
        reasons.append("--max-snapshots always makes the result non-formal")
    if args.allow_missing:
        reasons.append("--allow-missing is smoke-only")
    return reasons


def evaluate(args: argparse.Namespace) -> Path:
    verify_frozen_sources()
    output_dir = args.output_dir.resolve()
    shape_reasons = formal_shape_reasons(args)
    if output_dir == OUTPUT_ROOT.resolve() and shape_reasons:
        raise ValueError(
            "refusing to write a non-formal run into the default formal output "
            f"directory; choose --output-dir under a smoke/exploratory name: {shape_reasons}"
        )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    shared_dir = str((TEST_DIR / "shared2x_order_regularizer").resolve())
    if shared_dir not in sys.path:
        sys.path.insert(0, shared_dir)
    root_string = str(ROOT.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    runtime = load_module("hattrick_f_holdout_runtime", RUNTIME_SOURCE)
    snapshot_module = load_module(
        "hattrick_f_holdout_snapshot", ROOT / "utils" / "snapshot_utils.py"
    )
    cluster_module = load_module(
        "hattrick_f_holdout_cluster", ROOT / "utils" / "cluster_utils.py"
    )

    all_rows: list[dict] = []
    all_checkpoint_audits: list[dict] = []
    skipped: list[dict] = []
    started = time.perf_counter()
    resolved_windows = {}
    for name in args.windows:
        left, right = WINDOWS[name]
        if args.max_snapshots is not None:
            right = min(right, left + args.max_snapshots)
        resolved_windows[name] = (left, right)

    methods_by_seed: dict[int, list[MethodSpec]] = {}
    for seed in args.seeds:
        methods, missing = default_methods(seed, args.allow_missing)
        skipped.extend(missing)
        if not methods:
            raise FileNotFoundError(f"no checkpoint exists for seed {seed}")
        methods_by_seed[int(seed)] = methods

    # Complete the artifact preflight for every requested seed before loading
    # any holdout traffic or doing policy inference.
    for seed in args.seeds:
        methods = methods_by_seed[int(seed)]
        props = runtime.build_props(4, device)
        props.topo = TOPOLOGY
        props.device = device
        props.dtype = torch.float32
        props.dynamic = 0
        props.checkpoint = 0
        props.path_mask = 0
        models, model_audit = load_models(methods, runtime, props, device)
        all_checkpoint_audits.extend(
            [{"seed": seed, **row} for row in model_audit]
        )

        for window, (left, right) in resolved_windows.items():
            dataset = RawHoldoutDataset(
                props, left, right, snapshot_module, cluster_module
            )
            static = move_static(dataset, device)
            for batch in dataset.batches(args.batch_size):
                node_features = batch["node_features"].to(device)
                capacities = batch["capacities"].to(device)
                actual = tuple(value.to(device) for value in batch["actual"])
                base_prediction = tuple(
                    value.to(device) for value in batch["predicted"]
                )
                for bias in args.biases:
                    predicted = tuple(value * float(bias) for value in base_prediction)
                    for method in methods:
                        policy = generate_prediction_only_policy(
                            models[method.key],
                            props,
                            static,
                            node_features,
                            capacities,
                            predicted,
                        )
                        admitted, cumulative_admitted, cumulative_offered = (
                            sequential_actual_admission(
                                models[method.key],
                                props,
                                static,
                                policy,
                                actual,
                                capacities,
                            )
                        )
                        all_rows.extend(
                            batch_rows(
                                seed=seed,
                                method=method,
                                window=window,
                                bias=float(bias),
                                indices=batch["indices"],
                                actual=actual,
                                predicted=predicted,
                                admitted=admitted,
                                cumulative_admitted=cumulative_admitted,
                                cumulative_offered=cumulative_offered,
                                capacities=capacities,
                                static=static,
                            )
                        )
                        print(
                            f"[holdout] seed={seed} method={method.key} "
                            f"window={window} bias={bias:g} "
                            f"snapshots={batch['indices'][0]}..{batch['indices'][-1]}",
                            flush=True,
                        )

    if any(
        not math.isfinite(float(value))
        for row in all_rows
        for key, value in row.items()
        if key not in {"method", "method_label", "window", "class"}
    ):
        raise RuntimeError("evaluation produced NaN or Inf")
    csv_path = output_dir / "snapshots.csv"
    atomic_csv(csv_path, all_rows)
    expected_method_seed_pairs = [
        {"seed": seed, "method": method}
        for seed in EXPECTED_SEEDS
        for method in EXPECTED_METHODS
    ]
    actual_method_seed_pairs = sorted(
        {
            (int(row["seed"]), str(row["method"]))
            for row in all_checkpoint_audits
        }
    )
    observed_window_counts = []
    for seed, method in actual_method_seed_pairs:
        for window in args.windows:
            snapshots = {
                int(row["snapshot"])
                for row in all_rows
                if int(row["seed"]) == seed
                and str(row["method"]) == method
                and str(row["window"]) == window
            }
            observed_window_counts.append(
                {
                    "seed": seed,
                    "method": method,
                    "window": window,
                    "expected": EXPECTED_WINDOW_COUNTS.get(window),
                    "configured": resolved_windows[window][1]
                    - resolved_windows[window][0],
                    "observed_unique_snapshots": len(snapshots),
                    "complete": (
                        window in EXPECTED_WINDOW_COUNTS
                        and len(snapshots) == EXPECTED_WINDOW_COUNTS[window]
                    ),
                }
            )
    completeness_reasons = list(shape_reasons)
    expected_pairs = {
        (item["seed"], item["method"]) for item in expected_method_seed_pairs
    }
    if set(actual_method_seed_pairs) != expected_pairs:
        completeness_reasons.append(
            "method/seed set is incomplete: "
            f"expected={sorted(expected_pairs)}, actual={actual_method_seed_pairs}"
        )
    if skipped:
        completeness_reasons.append("one or more methods were skipped")
    if any(not row["complete"] for row in observed_window_counts):
        completeness_reasons.append("one or more method/seed/window groups is not 500 snapshots")
    formal_result = not completeness_reasons

    summary = {
        "schema_version": 1,
        "experiment": "Hattrick-f method-new temporal holdout and global ESM bias",
        "formal_result": formal_result,
        "result_class": "formal" if formal_result else "exploratory",
        "formal_incompleteness_reasons": completeness_reasons,
        "topology": TOPOLOGY,
        "paths_per_od": PATHS_PER_OD,
        "load_factor": LOAD_FACTOR,
        "windows": {key: list(value) for key, value in resolved_windows.items()},
        "biases": [float(value) for value in args.biases],
        "seeds": [int(value) for value in args.seeds],
        "completeness": {
            "expected_seeds": list(EXPECTED_SEEDS),
            "actual_seeds": sorted({int(value) for value in args.seeds}),
            "expected_methods": list(EXPECTED_METHODS),
            "actual_method_seed_pairs": [
                {"seed": seed, "method": method}
                for seed, method in actual_method_seed_pairs
            ],
            "expected_method_seed_pairs": expected_method_seed_pairs,
            "expected_window_snapshot_counts": EXPECTED_WINDOW_COUNTS,
            "configured_window_snapshot_counts": {
                key: stop - start
                for key, (start, stop) in resolved_windows.items()
            },
            "observed_method_seed_window_counts": observed_window_counts,
            "all_methods_and_seeds_complete": set(actual_method_seed_pairs)
            == expected_pairs
            and not skipped,
            "all_windows_complete": all(
                row["complete"] for row in observed_window_counts
            ),
        },
        "strict_esm_contract": {
            "policy_actual_input": "all-zero placeholder; true actual is absent from function signature",
            "policy_traffic_input": "2x ESM prediction multiplied by declared global bias",
            "admission_traffic_input": "unmodified 2x actual traffic",
            "oracle_files_loaded": False,
            "norm_fulfill_reported": False,
            "primary_metrics": [
                "raw_fulfill_ratio",
                "admitted_traffic",
                "post_admission_capacity_ratio",
            ],
        },
        "runtime": {
            "device": str(device),
            "torch": torch.__version__,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "source": {
            "evaluate_holdout.py": sha256(Path(__file__).resolve()),
            "runtime": str(RUNTIME_SOURCE.resolve()),
            "runtime_sha256": sha256(RUNTIME_SOURCE),
        },
        "checkpoints": all_checkpoint_audits,
        "skipped_methods": skipped,
        "classes": summarize(all_rows),
        "paired_vs_phase_a": paired_differences(all_rows),
        "capacity_audit": {
            "tolerance": CAPACITY_TOLERANCE,
            "maximum_post_admission_ratio": max(
                float(row["post_admission_capacity_ratio"]) for row in all_rows
            ),
            "maximum_violation": max(
                float(row["capacity_violation"]) for row in all_rows
            ),
            "passes": all(
                float(row["post_admission_capacity_ratio"])
                <= 1.0 + CAPACITY_TOLERANCE
                for row in all_rows
            ),
        },
        "snapshot_csv": str(csv_path.resolve()),
        "snapshot_csv_sha256": sha256(csv_path),
    }
    atomic_json(output_dir / "summary.json", summary)
    return output_dir


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict-ESM 2x GEANT holdout/global-bias evaluation"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(EXPECTED_SEEDS)
    )
    parser.add_argument(
        "--windows", choices=tuple(WINDOWS), nargs="+", default=list(WINDOWS)
    )
    parser.add_argument(
        "--biases", type=float, nargs="+", default=list(DEFAULT_BIASES)
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-snapshots",
        type=int,
        help="limit each window; intended only for smoke/debug runs",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="allow missing Hattrick-f/control artifacts for an explicit smoke run",
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch size must be positive")
    if args.max_snapshots is not None and args.max_snapshots <= 0:
        parser.error("max snapshots must be positive")
    if any(value <= 0 for value in args.biases):
        parser.error("all ESM biases must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    return args


def main() -> None:
    args = parse_cli()
    print(evaluate(args), flush=True)


if __name__ == "__main__":
    main()
