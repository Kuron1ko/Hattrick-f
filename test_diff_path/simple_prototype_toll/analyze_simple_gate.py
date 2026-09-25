from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


screen = load_module("simple_gate_screen", HERE / "run_screen.py")
trainer = screen.trainer
runtime = screen.runtime


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def row_index(rows: list[dict]) -> dict[tuple[int, str], float]:
    return {
        (int(row["snapshot"]), str(row["class"])): float(row["norm_fulfill"])
        for row in rows
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props, library = screen.load_backbone(2, device)
    prototypes = screen.fit_prototypes(library, 3)
    cache = runtime.build_policy_cache(model, props, 318, 400, batch_size=32)
    baseline_rows, _ = trainer.evaluate(model, props, cache)
    features = trainer.edge_features(cache)
    normalized = (features - library["mean"]) / library["std"]
    flat = normalized.reshape(len(cache), -1)
    distance = screen.squared_distance(flat, prototypes["centroids"])
    nearest_distance, prototype = distance.min(dim=1)
    tolls = screen.predict(library, prototypes, features).clone()
    tolls[:, 0].mul_(0.20)
    tolls[:, 1].mul_(0.30)
    routed = trainer.route_with_tolls(cache, tolls, 1.0)
    candidate_rows, _ = trainer.evaluate(model, props, cache, routed)
    baseline = row_index(baseline_rows)
    candidate = row_index(candidate_rows)
    records = []
    snapshots = sorted({key[0] for key in baseline})
    for offset, snapshot in enumerate(snapshots):
        high = features[offset, :, 0]
        medium = features[offset, :, 1]
        low = features[offset, :, 2]
        high_medium = features[offset, :, 3]
        total = features[offset, :, 4]
        sorted_total = total.sort(descending=True).values
        records.append(
            {
                "snapshot": snapshot,
                "prototype": int(prototype[offset].item()),
                "nearest_distance": float(nearest_distance[offset].item()),
                "high_max": float(high.max().item()),
                "medium_max": float(medium.max().item()),
                "low_max": float(low.max().item()),
                "high_medium_max": float(high_medium.max().item()),
                "total_max": float(total.max().item()),
                "total_mean": float(total.mean().item()),
                "total_std": float(total.std().item()),
                "total_top_gap": float((sorted_total[0] - sorted_total[1]).item()),
                "over_one_fraction": float((total > 1.0).float().mean().item()),
                "medium_delta": candidate[(snapshot, "Medium")]
                - baseline[(snapshot, "Medium")],
                "low_delta": candidate[(snapshot, "Low")]
                - baseline[(snapshot, "Low")],
            }
        )
    scalar_names = [
        key
        for key in records[0]
        if key not in {"snapshot", "prototype", "medium_delta", "low_delta"}
    ]
    correlations = {}
    medium_delta = np.asarray([row["medium_delta"] for row in records])
    for name in scalar_names:
        value = np.asarray([row[name] for row in records])
        correlations[name] = float(np.corrcoef(value, medium_delta)[0, 1])
    payload = {
        "method": "strict-ESM scalar gate diagnostic for P3",
        "actual_values_used_only_as_offline_delta_labels": True,
        "policy_features": scalar_names + ["prototype"],
        "correlations_with_medium_delta": correlations,
        "records": records,
    }
    write_json(HERE / "simple_gate_diagnostic.json", payload)
    print(json.dumps({"correlations": correlations}, indent=2), flush=True)


if __name__ == "__main__":
    main()
