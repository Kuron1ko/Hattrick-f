from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
RUNTIME_DIR = TEST_DIR / "shared2x_medium_adapter"
for item in (str(ROOT), str(TEST_DIR), str(RUNTIME_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

spec = importlib.util.spec_from_file_location(
    "shared2x_active_set_runtime", RUNTIME_DIR / "run_experiment.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load frozen Hattrick runtime")
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)


def predicted_admitted(model, props, cache: runtime.PolicyCache) -> tuple[torch.Tensor, ...]:
    buckets = [[], [], []]
    for offset in range(0, len(cache), 32):
        ids = torch.arange(offset, min(offset + 32, len(cache)), device=props.device)
        batch = runtime.select_cache(cache, ids)
        batch["tms"] = tuple(value.index_select(0, ids) for value in cache.predicted_tms)
        admitted, _ = runtime.simulate_admission(model, props, cache.dataset, batch, None)
        for class_index in range(3):
            buckets[class_index].append(admitted[class_index].detach())
    return tuple(torch.cat(values, dim=0) for values in buckets)


def utilization_states(model, props, cache: runtime.PolicyCache) -> np.ndarray:
    admitted = predicted_admitted(model, props, cache)
    pte = cache.dataset.pte.coalesce()
    high = torch.sparse.mm(pte.t(), admitted[0].t()).t()
    high_medium = high + torch.sparse.mm(pte.t(), admitted[1].t()).t()
    capacity = cache.capacities.clamp_min(1e-9)
    return torch.cat([high / capacity, high_medium / capacity], dim=1).cpu().numpy()


def kmeans(train: np.ndarray, clusters: int, seed: int = 20260820) -> np.ndarray:
    rng = np.random.default_rng(seed + clusters)
    centers = train[rng.choice(train.shape[0], size=clusters, replace=False)].copy()
    for _ in range(100):
        distance = ((train[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2)
        assignment = distance.argmin(axis=1)
        updated = centers.copy()
        for cluster in range(clusters):
            members = train[assignment == cluster]
            if members.size:
                updated[cluster] = members.mean(axis=0)
        if np.allclose(updated, centers, atol=1e-8):
            break
        centers = updated
    return centers


def quantization(states: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = ((states[:, None, :] - centers[None, :, :]) ** 2).mean(axis=2)
    assignment = distance.argmin(axis=1)
    return assignment, distance[np.arange(states.shape[0]), assignment]


def top_signature(states: np.ndarray, edges: int, top_k: int) -> list[tuple[int, ...]]:
    # The second half is the predicted High+Medium utilization, which defines
    # the active set immediately before Low admission.
    values = states[:, edges:]
    return [tuple(sorted(np.argpartition(row, -top_k)[-top_k:].tolist())) for row in values]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(3, device)
    model, checkpoint = runtime.load_backbone(3, 490, props, device)
    train_cache = runtime.build_policy_cache(model, props, 0, 350, batch_size=32)
    validation_cache = runtime.build_policy_cache(model, props, 350, 400, batch_size=32)
    train = utilization_states(model, props, train_cache)
    validation = utilization_states(model, props, validation_cache)
    edge_count = train.shape[1] // 2
    regimes = []
    for count in (2, 4, 8, 16, 24, 32):
        centers = kmeans(train, count)
        train_assignment, train_error = quantization(train, centers)
        val_assignment, val_error = quantization(validation, centers)
        occupancy = np.bincount(val_assignment, minlength=count)
        regimes.append(
            {
                "clusters": count,
                "train_rmse": float(np.sqrt(train_error.mean())),
                "validation_rmse": float(np.sqrt(val_error.mean())),
                "validation_used_clusters": int((occupancy > 0).sum()),
                "validation_min_occupancy": int(occupancy[occupancy > 0].min()),
                "validation_max_occupancy": int(occupancy.max()),
            }
        )
    signatures = {}
    for top_k in (1, 2, 3, 4, 6):
        train_signatures = top_signature(train, edge_count, top_k)
        validation_signatures = top_signature(validation, edge_count, top_k)
        train_set = set(train_signatures)
        validation_set = set(validation_signatures)
        signatures[str(top_k)] = {
            "train_unique": len(train_set),
            "validation_unique": len(validation_set),
            "validation_seen_in_train_fraction": float(
                np.mean([signature in train_set for signature in validation_signatures])
            ),
            "largest_validation_signature_count": max(
                validation_signatures.count(signature) for signature in validation_set
            ),
        }
    payload = {
        "checkpoint": str(checkpoint),
        "input_contract": "ESM-predicted TMs and frozen Hattrick policies only",
        "train_range": [0, 350],
        "validation_range": [350, 400],
        "edge_count": edge_count,
        "state_dimension": int(train.shape[1]),
        "utilization_quantization": regimes,
        "top_edge_signatures": signatures,
    }
    THIS_DIR.mkdir(parents=True, exist_ok=True)
    (THIS_DIR / "active_set_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
