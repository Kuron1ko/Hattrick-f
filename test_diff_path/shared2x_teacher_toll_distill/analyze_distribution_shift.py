from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TRAINER_PATH = HERE / "train_linear_toll.py"
LARGE_PATH = HERE / "run_large_linear_toll.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer = load_module("shift_analysis_trainer", TRAINER_PATH)
large = load_module("shift_analysis_large", LARGE_PATH)
runtime = trainer.runtime


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def cpu_state(state: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in state.items()
    }


def feature_shift(features: torch.Tensor, state: dict) -> dict:
    normalized = (features - state["feature_mean"]) / state["feature_std"]
    absolute = normalized.abs()
    return {
        "mean_abs_z": float(absolute.mean().item()),
        "p95_abs_z": float(torch.quantile(absolute, 0.95).item()),
        "p99_abs_z": float(torch.quantile(absolute, 0.99).item()),
        "max_abs_z": float(absolute.max().item()),
        "fraction_abs_z_over_3": float((absolute > 3.0).float().mean().item()),
        "feature_abs_max": float(features.abs().max().item()),
    }


def analyze(model, props, cache, state: dict) -> dict:
    features = trainer.edge_features(cache)
    targets = trainer.teacher_tolls(model, props, cache)
    predictions = trainer.predict_tolls(state, features)
    baseline_rows, baseline_summary = trainer.evaluate(model, props, cache)
    result = trainer.metrics(model, props, cache, state, 1.0, baseline_summary)
    return {
        "feature_shift": feature_shift(features, state),
        "prediction": trainer.target_prediction_stats(predictions, targets),
        "target_abs_mean": float(targets.abs().mean().item()),
        "target_abs_max": float(targets.abs().max().item()),
        "prediction_abs_mean": float(predictions.abs().mean().item()),
        "prediction_abs_max": float(predictions.abs().max().item()),
        "result": large.public(result),
        "bootstrap": large.paired_bootstrap(baseline_rows, result["rows"]),
    }


def main() -> None:
    torch.manual_seed(20260822)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(
        HERE / "artifacts" / "linear_toll_large" / "model.pt",
        map_location=device,
        weights_only=False,
    )
    state = cpu_state(saved["state"], device)

    props3 = runtime.build_props(3, device)
    model3, checkpoint3 = runtime.load_backbone(3, 490, props3, device)
    props4 = runtime.build_props(4, device)
    model4, checkpoint4 = runtime.load_backbone(4, 490, props4, device)

    conditions = {
        "level3_validation_350_400": (model3, props3, 350, 400),
        "level4_validation_350_400": (model4, props4, 350, 400),
        "level3_final_400_500": (model3, props3, 400, 500),
        "level4_final_400_500": (model4, props4, 400, 500),
    }
    result = {}
    for name, (model, props, start, stop) in conditions.items():
        cache = runtime.build_policy_cache(
            model, props, start, stop, batch_size=16
        )
        result[name] = analyze(model, props, cache, state)
        print(name, json.dumps(result[name], indent=2), flush=True)

    correlations = {
        key: value["prediction"]["correlation"] for key, value in result.items()
    }
    payload = {
        "question": "Does failure come from time shift, backbone shift, or their interaction?",
        "trained_on": "level3 snapshots 0:318",
        "checkpoint3": str(checkpoint3),
        "checkpoint4": str(checkpoint4),
        "conditions": result,
        "correlations": correlations,
    }
    write_json(
        HERE / "artifacts" / "linear_toll_large" / "shift_analysis.json",
        payload,
    )


if __name__ == "__main__":
    main()
