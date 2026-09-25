from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer = load_module("final_export_trainer", "train_linear_toll.py")
asymmetric = load_module(
    "final_export_asymmetric", "train_asymmetric_knn_toll.py"
)
runtime = trainer.runtime


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def on_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: on_device(item, device) for key, item in value.items()}
    return value


def compact_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "snapshot": int(row["snapshot"]),
            "class": str(row["class"]),
            "norm_fulfill": float(row["norm_fulfill"]),
        }
        for row in rows
    ]


def ecdf(values: np.ndarray, left: float, right: float):
    ordered = np.sort(values)
    x = np.concatenate([[left], ordered, [right]])
    y = np.concatenate([[0.0], np.arange(1, len(ordered) + 1) / len(ordered), [1.0]])
    return x, y


def plot_cdf(rows: dict[str, list[dict]], path: Path) -> None:
    import matplotlib.pyplot as plt

    classes = ("High", "Medium", "Low")
    ranges = {"High": (0.985, 1.001), "Medium": (0.84, 1.01), "Low": (0.88, 1.42)}
    colors = {"Hattrick": "#3478d4", "Asymmetric kNN toll": "#28a45b"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2), sharey=True)
    for axis, class_name in zip(axes, classes):
        left, right = ranges[class_name]
        for label, source in (("Hattrick", "baseline"), ("Asymmetric kNN toll", "candidate")):
            values = np.asarray(
                [row["norm_fulfill"] for row in rows[source] if row["class"] == class_name]
            )
            x, y = ecdf(values, left, right)
            axis.step(x, y, where="post", linewidth=2.1, color=colors[label], label=label)
        axis.set_xlim(left, right)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel(class_name, fontsize=12)
        axis.grid(True, color="#dfe3e8", linewidth=0.7)
        axis.tick_params(labelsize=10)
    axes[0].set_ylabel("CDF", fontsize=12)
    fig.suptitle("GEANT 2× strict-ESM: CDF of NormFulfill", fontsize=18, x=0.08, ha="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.08, 0.005), ncol=2, frameon=False)
    fig.supxlabel("NormFulfill", y=0.06, fontsize=12)
    fig.tight_layout(rect=(0.02, 0.11, 1.0, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    saved = torch.load(
        HERE / "artifacts" / "asymmetric_knn_level4" / "model.pt",
        map_location=device,
        weights_only=False,
    )
    library = on_device(saved["library"], device)
    medium_k = int(saved["medium_neighbors"])
    low_k = int(saved["low_neighbors"])
    cache = runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
    features = trainer.edge_features(cache)
    tolls, retrieval = asymmetric.predict(library, features, medium_k, low_k)
    candidate_cache = trainer.route_with_tolls(cache, tolls, 1.0)
    baseline_rows, baseline_summary = trainer.evaluate(model, props, cache)
    candidate_rows, candidate_summary = trainer.evaluate(
        model, props, cache, candidate_cache
    )

    for _ in range(10):
        trial_tolls, _ = asymmetric.predict(library, features, medium_k, low_k)
        trainer.route_with_tolls(cache, trial_tolls, 1.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    repeats = 100
    for _ in range(repeats):
        trial_tolls, _ = asymmetric.predict(library, features, medium_k, low_k)
        trainer.route_with_tolls(cache, trial_tolls, 1.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    artifact_dir = HERE / "artifacts" / "asymmetric_knn_level4"
    rows = {
        "baseline": compact_rows(baseline_rows),
        "candidate": compact_rows(candidate_rows),
    }
    payload = {
        "method": "priority-asymmetric local case retrieval for edge tolls",
        "range": [400, 500],
        "strict_esm_inference": True,
        "checkpoint": str(checkpoint),
        "medium_neighbors": medium_k,
        "low_neighbors": low_k,
        "retrieval": retrieval,
        "baseline_summary": trainer.probe.compact(baseline_summary),
        "candidate_summary": trainer.probe.compact(candidate_summary),
        "delta": trainer.probe.gaps(candidate_summary, baseline_summary),
        "timing_seconds": {
            "retrieval_and_routing_per_100_snapshots": elapsed / repeats,
            "per_snapshot": elapsed / repeats / len(cache),
        },
        "rows": rows,
    }
    write_json(artifact_dir / "final_rows.json", payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
