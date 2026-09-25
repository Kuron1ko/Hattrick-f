from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
PROBE_PATH = THIS_DIR / "probe_onex_transfer.py"
spec = importlib.util.spec_from_file_location("shared1x_gate_probe", PROBE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load 1x transfer probe")
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)
method = probe.method
runtime = probe.runtime


THRESHOLDS = (0.70, 0.75, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 1.0)


def predicted_medium_fulfillment(model, props, cache) -> torch.Tensor:
    with torch.no_grad():
        values = method.predicted_fulfillment(
            model, props, cache, [value.clone() for value in cache.policies]
        )
    return values[:, 1].detach()


def gated_cache(cache, fully_corrected, predicted_medium, threshold: float):
    path_count = int(cache.policies[0].shape[1])
    identity = torch.cat(
        [cache.policies[1].squeeze(-1), cache.policies[2].squeeze(-1)], dim=1
    )
    active = predicted_medium < float(threshold)
    features = torch.where(active[:, None], fully_corrected.path_features, identity)
    return replace(cache, path_features=features), active


def summary_index(summary):
    return {row["class"]: row for row in summary}


def is_safe(summary, baseline, delta) -> bool:
    high = summary_index(summary)["High"]
    base_high = summary_index(baseline)["High"]
    return (
        float(high["norm_fulfill_mean"]) >= float(base_high["norm_fulfill_mean"]) - 1e-7
        and delta["Low.norm_fulfill_mean"] >= -0.0005
        and delta["Low.norm_fulfill_p1"] >= -0.001
        and delta["Low.norm_fulfill_p10"] >= -0.001
    )


def score(delta) -> float:
    return (
        delta["Medium.norm_fulfill_mean"]
        + 0.5 * delta["Medium.norm_fulfill_p1"]
        + 0.5 * delta["Medium.norm_fulfill_p10"]
    )


def evaluate_thresholds(model, props, start: int, stop: int, thresholds) -> dict:
    cache = runtime.build_policy_cache(model, props, start, stop, batch_size=32)
    _, baseline = runtime.evaluate_cache(model, props, cache, None, batch_size=32)
    corrected = probe.correct_strict(model, props, cache)
    predicted = predicted_medium_fulfillment(model, props, cache)
    adapter = method.CorrectedPolicyAdapter(int(cache.policies[0].shape[1]))
    rows = []
    for threshold in thresholds:
        gated, active = gated_cache(cache, corrected, predicted, threshold)
        _, summary = runtime.evaluate_cache(model, props, gated, adapter, batch_size=32)
        delta = method.gaps(summary, baseline)
        rows.append(
            {
                "threshold": threshold,
                "active_count": int(active.sum().item()),
                "active_fraction": float(active.float().mean().item()),
                "safe": is_safe(summary, baseline, delta),
                "score": score(delta),
                "candidate": method.compact(summary),
                "delta": delta,
            }
        )
    return {
        "range": [start, stop],
        "baseline": method.compact(baseline),
        "predicted_medium_fulfillment": {
            "min": float(predicted.min().item()),
            "mean": float(predicted.mean().item()),
            "p10": float(np.percentile(predicted.cpu().numpy(), 10)),
            "max": float(predicted.max().item()),
        },
        "results": rows,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props = probe.load_onex(device)
    small = evaluate_thresholds(model, props, 350, 358, THRESHOLDS)
    safe_small = sorted(
        (row for row in small["results"] if row["safe"]),
        key=lambda row: row["score"], reverse=True,
    )
    selected = [float(row["threshold"]) for row in safe_small[:5]]
    # The original 1x training protocol owns the full 350:400 interval as its
    # validation set. Evaluate the predeclared threshold grid on that interval;
    # 400:500 remains untouched until one threshold is frozen.
    validation = evaluate_thresholds(model, props, 350, 400, THRESHOLDS)
    payload = {
        "method": "1x congestion-gated ESM-SAR",
        "correction_config": {
            "steps": probe.TRANSFER_CONFIG[0],
            "learning_rate": probe.TRANSFER_CONFIG[1],
            "low_weight": probe.TRANSFER_CONFIG[2],
            "anchor_weight": probe.TRANSFER_CONFIG[3],
        },
        "small": small,
        "selected_from_small": selected,
        "validation": validation,
    }
    output = THIS_DIR / "onex_gate_tuning.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
