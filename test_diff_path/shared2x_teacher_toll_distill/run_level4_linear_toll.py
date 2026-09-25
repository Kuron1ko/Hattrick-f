from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

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


trainer = load_module("level4_linear_toll_trainer", TRAINER_PATH)
large = load_module("level4_linear_toll_large", LARGE_PATH)
runtime = trainer.runtime
probe = trainer.probe


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def state_to_cpu(state: dict) -> dict:
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in state.items()
    }


def build_result(model, props, cache, state: dict):
    baseline_rows, baseline_summary = trainer.evaluate(model, props, cache)
    result = trainer.metrics(model, props, cache, state, 1.0, baseline_summary)
    result["baseline"] = probe.compact(baseline_summary)
    result["bootstrap"] = large.paired_bootstrap(baseline_rows, result["rows"])
    return result


def prediction_report(state: dict, features: torch.Tensor, targets: torch.Tensor):
    return trainer.target_prediction_stats(
        trainer.predict_tolls(state, features), targets
    )


def main() -> None:
    torch.manual_seed(20260822)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same frozen student and hyperparameters; only the deployment backbone is
    # used consistently for label generation, fitting, validation and inference.
    linear_ridge = 0.03
    scale = 1.0
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    train = runtime.build_policy_cache(model, props, 0, 318, batch_size=16)
    safety = runtime.build_policy_cache(model, props, 318, 350, batch_size=16)
    validation = runtime.build_policy_cache(model, props, 350, 400, batch_size=16)

    train_features = trainer.edge_features(train)
    train_targets = trainer.teacher_tolls(model, props, train)
    state = trainer.fit_linear(train_features, train_targets, ridge=linear_ridge)

    safety_features = trainer.edge_features(safety)
    safety_targets = trainer.teacher_tolls(model, props, safety)
    validation_features = trainer.edge_features(validation)
    validation_targets = trainer.teacher_tolls(model, props, validation)
    prediction = {
        "train": prediction_report(state, train_features, train_targets),
        "safety": prediction_report(state, safety_features, safety_targets),
        "validation": prediction_report(
            state, validation_features, validation_targets
        ),
    }

    safety_result = build_result(model, props, safety, state)
    validation_result = None
    final_result = None
    inference_seconds = None
    if safety_result["safe"]:
        validation_result = build_result(model, props, validation, state)
    if validation_result is not None and validation_result["safe"]:
        final_cache = runtime.build_policy_cache(
            model, props, 400, 500, batch_size=16
        )
        final_result = build_result(model, props, final_cache, state)
        inference_started = time.perf_counter()
        with torch.no_grad():
            features = trainer.edge_features(final_cache)
            tolls = trainer.predict_tolls(state, features)
            trainer.route_with_tolls(final_cache, tolls, scale)
        inference_seconds = time.perf_counter() - inference_started

    payload = {
        "method": "backbone-matched affine edge-toll distillation",
        "status": "complete",
        "architecture": {
            "parameters": int(state["coefficients"].numel()),
            "features_per_edge": 7,
            "outputs": "Medium and Low toll per directed edge",
            "high_policy": "exact Hattrick bypass",
            "inference": "one affine edge pass plus one path softmax",
        },
        "strict_esm_inference": True,
        "current_actual_tm_used_for_policy": False,
        "checkpoint": str(checkpoint),
        "configuration": {
            "teacher_steps": probe.TEACHER_CONFIG[0],
            "target_projection": "ESM predicted-flow weighted log-policy ridge",
            "target_projection_ridge": 0.01,
            "linear_ridge": linear_ridge,
            "scale": scale,
        },
        "protocol": {
            "train": [0, 318],
            "safety": [318, 350],
            "validation": [350, 400],
            "followup_final": [400, 500],
            "note": "Final was already opened by the mismatched-backbone run; this is a causal repair confirmation, not a new untouched holdout.",
        },
        "toll_prediction": prediction,
        "safety": large.public(safety_result),
        "validation": large.public(validation_result),
        "final": large.public(final_result),
        "final_opened": final_result is not None,
        "final_success": bool(final_result and final_result["safe"]),
        "timing_seconds": {
            "adapter_inference_100": inference_seconds,
            "total_experiment": time.perf_counter() - started,
        },
    }
    artifact_dir = HERE / "artifacts" / "level4_linear_toll"
    write_json(artifact_dir / "report.json", payload)
    torch.save(
        {"state": state_to_cpu(state), "scale": scale, "checkpoint": str(checkpoint)},
        artifact_dir / "model.pt",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
