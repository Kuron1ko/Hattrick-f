from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "run_experiment.py"
SAVED = HERE / "model.pt"
SCALES = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("shared1x_low_scale_base", SOURCE)
trainer = base.trainer
runtime = base.runtime
method = base.method


def on_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: on_device(item, device) for key, item in value.items()}
    return value


def route(cache, tolls, medium_gate, low_scale: float):
    gated = tolls.clone()
    gated[:, 0] = gated[:, 0] * medium_gate[:, None]
    gated[:, 1] = gated[:, 1] * float(low_scale)
    return trainer.route_with_tolls(cache, gated, 1.0)


def evaluate(model, props, cache, library, advantages, low_scale: float):
    features = trainer.edge_features(cache)
    tolls, distance = base.retrieve_tolls(library, features)
    medium_gate, gate_stats = base.advantage_gate(distance, advantages, 32)
    baseline_rows, baseline_summary = trainer.evaluate(model, props, cache)
    candidate_rows, candidate_summary = trainer.evaluate(
        model,
        props,
        cache,
        route(cache, tolls, medium_gate, low_scale),
    )
    result = {
        "low_scale": float(low_scale),
        "medium_gate": gate_stats,
        "summary": trainer.probe.compact(candidate_summary),
        "baseline": trainer.probe.compact(baseline_summary),
        "delta": trainer.probe.gaps(candidate_summary, baseline_summary),
        "bootstrap": method.large.paired_bootstrap(
            baseline_rows, candidate_rows
        ),
    }
    return result, baseline_rows, candidate_rows


def safe(result: dict) -> bool:
    delta = result["delta"]
    return (
        result["summary"]["High"]["norm_fulfill_mean"] >= 0.995
        and delta["Medium.norm_fulfill_mean"] >= -1e-6
        and delta["Medium.norm_fulfill_p1"] >= -1e-6
        and delta["Medium.norm_fulfill_p10"] >= -1e-6
        and delta["Low.norm_fulfill_mean"] >= -0.003
        and delta["Low.norm_fulfill_p1"] >= -0.01
        and delta["Low.norm_fulfill_p10"] >= -0.01
    )


def rank(result: dict) -> tuple[float, ...]:
    delta = result["delta"]
    low = [
        delta[f"Low.norm_fulfill_{metric}"] for metric in ("mean", "p1", "p10")
    ]
    return min(low), sum(low), delta["Medium.norm_fulfill_mean"]


def main() -> None:
    runtime.set_seed(20260826)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props = base.load_backbone(device)
    saved = torch.load(SAVED, map_location=device, weights_only=False)
    library = on_device(saved["library"], device)
    advantages = saved["advantages"].to(device)

    development = runtime.build_policy_cache(model, props, 318, 400, batch_size=32)
    trials = []
    for scale in SCALES:
        result, _, _ = evaluate(
            model, props, development, library, advantages, scale
        )
        result["safe"] = safe(result)
        trials.append(result)
        print(
            f"[development] scale={scale:.2f} safe={result['safe']} "
            f"dM={result['delta']['Medium.norm_fulfill_mean']:+.6f} "
            f"dL={result['delta']['Low.norm_fulfill_mean']:+.6f}",
            flush=True,
        )
    eligible = [result for result in trials if result["safe"]]
    selected = max(eligible, key=rank) if eligible else None
    final = None
    final_rows = None
    if selected:
        cache = runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
        final, baseline_rows, candidate_rows = evaluate(
            model,
            props,
            cache,
            library,
            advantages,
            selected["low_scale"],
        )
        final["safe"] = safe(final)
        final_rows = {
            "baseline": base.compact_rows(baseline_rows),
            "candidate": base.compact_rows(candidate_rows),
        }
        print(
            f"[final] scale={selected['low_scale']:.2f} safe={final['safe']} "
            f"dM={final['delta']['Medium.norm_fulfill_mean']:+.6f} "
            f"dL={final['delta']['Low.norm_fulfill_mean']:+.6f}",
            flush=True,
        )
    payload = {
        "method": "strict-ESM advantage-gated Medium plus development-selected Low toll scale",
        "strict_esm_inference": True,
        "current_actual_tm_used_for_policy": False,
        "protocol": {
            "library_and_advantage_training": [0, 318],
            "low_scale_development": [318, 400],
            "level4": [400, 500],
        },
        "scale_trials": trials,
        "selected_scale": None if selected is None else selected["low_scale"],
        "evaluation": final,
        "rows": final_rows,
        "seconds": time.perf_counter() - started,
    }
    base.write_json(HERE / "low_scale_report.json", payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
