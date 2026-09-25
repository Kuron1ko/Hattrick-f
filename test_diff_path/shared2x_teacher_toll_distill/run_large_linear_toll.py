from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TRAINER_PATH = HERE / "train_linear_toll.py"
spec = importlib.util.spec_from_file_location("large_linear_toll_runtime", TRAINER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {TRAINER_PATH}")
trainer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trainer
spec.loader.exec_module(trainer)
runtime = trainer.runtime
probe = trainer.probe


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def public(value: dict | None) -> dict | None:
    if value is None:
        return None
    return {key: item for key, item in value.items() if key != "rows"}


def paired_bootstrap(baseline_rows: list[dict], candidate_rows: list[dict]) -> dict:
    rng = np.random.default_rng(20260822)
    result = {}
    for class_name in ("High", "Medium", "Low"):
        baseline = np.asarray(
            [float(row["norm_fulfill"]) for row in baseline_rows if row["class"] == class_name]
        )
        candidate = np.asarray(
            [float(row["norm_fulfill"]) for row in candidate_rows if row["class"] == class_name]
        )
        if baseline.shape != candidate.shape:
            raise RuntimeError("Paired row mismatch")
        difference = candidate - baseline
        indices = rng.integers(0, len(difference), size=(20000, len(difference)))
        means = difference[indices].mean(axis=1)
        result[class_name] = {
            "mean_delta": float(difference.mean()),
            "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)),
            "positive_probability": float(np.mean(means > 0.0)),
            "paired_positive_fraction": float(np.mean(difference > 0.0)),
        }
    return result


def build_result(model, props, cache, state: dict, scale: float = 1.0):
    baseline_rows, baseline_summary = trainer.evaluate(model, props, cache)
    result = trainer.metrics(model, props, cache, state, scale, baseline_summary)
    result["baseline"] = probe.compact(baseline_summary)
    result["bootstrap"] = paired_bootstrap(baseline_rows, result["rows"])
    return result, baseline_rows


def state_to_cpu(state: dict) -> dict:
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in state.items()
    }


def main() -> None:
    torch.manual_seed(20260822)
    total_started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Frozen after the small experiment: no Level-3/4 hyperparameter search.
    target_projection_ridge = 0.01
    linear_ridge = 0.03
    scale = 1.0

    props3 = runtime.build_props(3, device)
    model3, checkpoint3 = runtime.load_backbone(3, 490, props3, device)
    train = runtime.build_policy_cache(model3, props3, 0, 318, batch_size=16)
    safety = runtime.build_policy_cache(model3, props3, 318, 350, batch_size=16)
    validation = runtime.build_policy_cache(model3, props3, 350, 400, batch_size=16)

    label_started = time.perf_counter()
    train_features = trainer.edge_features(train)
    train_targets = trainer.teacher_tolls(model3, props3, train)
    label_seconds = time.perf_counter() - label_started
    fit_started = time.perf_counter()
    state = trainer.fit_linear(train_features, train_targets, ridge=linear_ridge)
    fit_seconds = time.perf_counter() - fit_started

    safety_features = trainer.edge_features(safety)
    safety_targets = trainer.teacher_tolls(model3, props3, safety)
    prediction_stats = {
        "train": trainer.target_prediction_stats(
            trainer.predict_tolls(state, train_features), train_targets
        ),
        "safety": trainer.target_prediction_stats(
            trainer.predict_tolls(state, safety_features), safety_targets
        ),
    }
    safety_result, _ = build_result(model3, props3, safety, state, scale)
    if safety_result["safe"]:
        validation_result, _ = build_result(
            model3, props3, validation, state, scale
        )
    else:
        validation_result = None

    final_result = None
    checkpoint4 = None
    inference_seconds = None
    if validation_result is not None and validation_result["safe"]:
        props4 = runtime.build_props(4, device)
        model4, checkpoint4 = runtime.load_backbone(4, 490, props4, device)
        final_cache = runtime.build_policy_cache(
            model4, props4, 400, 500, batch_size=16
        )
        final_result, _ = build_result(model4, props4, final_cache, state, scale)
        inference_started = time.perf_counter()
        with torch.no_grad():
            final_features = trainer.edge_features(final_cache)
            final_tolls = trainer.predict_tolls(state, final_features)
            trainer.route_with_tolls(final_cache, final_tolls, scale)
        inference_seconds = time.perf_counter() - inference_started

    payload = {
        "method": "flow-weighted teacher edge-toll distillation with affine ESM predictor",
        "status": "complete",
        "architecture": {
            "parameters": int(state["coefficients"].numel()),
            "features_per_edge": 7,
            "outputs": "Medium and Low toll per directed edge",
            "inference": "one affine evaluation plus path-cost softmax",
            "high_policy": "exact Hattrick bypass",
        },
        "frozen_configuration": {
            "teacher_steps": probe.TEACHER_CONFIG[0],
            "target_projection": "ESM predicted-flow weighted log-policy ridge",
            "target_projection_ridge": target_projection_ridge,
            "linear_ridge": linear_ridge,
            "scale": scale,
        },
        "strict_esm_inference": True,
        "current_actual_tm_used_for_policy": False,
        "protocol": {
            "train": [0, 318],
            "safety": [318, 350],
            "validation": [350, 400],
            "final": [400, 500],
            "final_open_rule": "safety and validation must both pass before final cache is built",
        },
        "training_checkpoint": str(checkpoint3),
        "final_checkpoint": str(checkpoint4) if checkpoint4 else None,
        "toll_prediction": prediction_stats,
        "safety": public(safety_result),
        "validation": public(validation_result),
        "final": public(final_result),
        "final_opened": final_result is not None,
        "final_success": bool(final_result and final_result["safe"]),
        "timing_seconds": {
            "teacher_label_generation_318": label_seconds,
            "linear_fit": fit_seconds,
            "adapter_inference_100": inference_seconds,
            "total_experiment": time.perf_counter() - total_started,
        },
    }
    artifact_dir = HERE / "artifacts" / "linear_toll_large"
    write_json(artifact_dir / "report.json", payload)
    torch.save(
        {
            "state": state_to_cpu(state),
            "scale": scale,
            "training_checkpoint": str(checkpoint3),
            "final_checkpoint": str(checkpoint4) if checkpoint4 else None,
        },
        artifact_dir / "model.pt",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
