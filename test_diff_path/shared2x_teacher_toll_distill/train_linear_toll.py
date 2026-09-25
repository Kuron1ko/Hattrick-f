from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
ROOT = TEST_DIR.parent
PROBE_PATH = HERE / "probe_toll_projection.py"
for item in (str(ROOT), str(TEST_DIR), str(HERE)):
    if item not in sys.path:
        sys.path.insert(0, item)

spec = importlib.util.spec_from_file_location("linear_toll_probe_runtime", PROBE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {PROBE_PATH}")
probe = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probe
spec.loader.exec_module(probe)
runtime = probe.runtime


K = 8


class LinearTollAdapter:
    def __init__(self, path_count: int):
        self.path_count = int(path_count)

    def adapt_batch(self, policies, batch):
        values = batch["path_features"]
        policies[1] = values[:, : self.path_count].unsqueeze(-1)
        policies[2] = values[:, self.path_count :].unsqueeze(-1)
        return policies


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def edge_features(cache) -> torch.Tensor:
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    capacities = cache.capacities.to(dtype=torch.float32).clamp_min(1e-9)
    loads = []
    for policy, demand in zip(cache.policies, cache.predicted_tms):
        path_flow = policy.squeeze(-1).to(dtype=torch.float32) * demand.squeeze(
            -1
        ).to(dtype=torch.float32)
        loads.append(torch.sparse.mm(pte.t(), path_flow.t()).t() / capacities)
    high, medium, low = loads
    return torch.stack(
        [
            high,
            medium,
            low,
            high + medium,
            high + medium + low,
            torch.relu(1.0 - high),
            torch.relu(1.0 - high - medium),
        ],
        dim=-1,
    ).detach()


def teacher_tolls(model, props, cache) -> torch.Tensor:
    _, _, projection = probe.project_cache(
        model, props, cache, ridge=0.01, weighting="target_flow"
    )
    return torch.stack(
        [projection["medium_tolls"], projection["low_tolls"]], dim=1
    ).detach()


def fit_linear(features: torch.Tensor, targets: torch.Tensor, ridge: float = 0.03):
    # Each edge gets a tiny affine regressor over seven strict-ESM features.
    feature_mean = features.mean(dim=0)
    feature_std = features.std(dim=0).clamp_min(1e-4)
    normalized = (features - feature_mean) / feature_std
    sample_count, edge_count, feature_count = normalized.shape
    coefficients = torch.zeros(
        2,
        edge_count,
        feature_count + 1,
        device=features.device,
        dtype=torch.float64,
    )
    eye = torch.eye(feature_count + 1, device=features.device, dtype=torch.float64)
    eye[0, 0] = 0.0
    for edge in range(edge_count):
        design = torch.cat(
            [
                torch.ones(sample_count, 1, device=features.device),
                normalized[:, edge],
            ],
            dim=1,
        ).to(dtype=torch.float64)
        gram = design.transpose(0, 1) @ design + float(ridge) * eye
        rhs = design.transpose(0, 1) @ targets[:, :, edge].to(dtype=torch.float64)
        coefficients[:, edge] = torch.linalg.solve(gram, rhs).transpose(0, 1)
    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "coefficients": coefficients.to(dtype=torch.float32),
        "ridge": ridge,
    }


def predict_tolls(model_state: dict, features: torch.Tensor) -> torch.Tensor:
    normalized = (features - model_state["feature_mean"]) / model_state["feature_std"]
    design = torch.cat(
        [torch.ones_like(normalized[..., :1]), normalized], dim=-1
    )
    return torch.einsum("nef,cef->nce", design, model_state["coefficients"])


def route_with_tolls(cache, tolls: torch.Tensor, scale: float):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    routed = []
    for class_index, base_policy in enumerate(cache.policies[1:]):
        path_cost = torch.sparse.mm(
            pte, tolls[:, class_index].transpose(0, 1)
        ).transpose(0, 1)
        base = base_policy.squeeze(-1)
        grouped_base = base.reshape(len(cache), -1, K)
        grouped_cost = path_cost.reshape(len(cache), -1, K)
        valid = grouped_base > 0.0
        logits = torch.log(grouped_base.clamp_min(1e-12)) - float(scale) * grouped_cost
        logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
        policy = torch.softmax(logits, dim=-1) * grouped_base.sum(
            dim=-1, keepdim=True
        )
        routed.append(torch.where(valid, policy, torch.zeros_like(policy)).reshape_as(base))
    return replace(cache, path_features=torch.cat(routed, dim=1))


def evaluate(model, props, cache, candidate=None):
    adapter = None
    target = cache
    if candidate is not None:
        target = candidate
        adapter = LinearTollAdapter(int(cache.policies[0].shape[1]))
    rows, summary = runtime.evaluate_cache(model, props, target, adapter, batch_size=16)
    return rows, summary


def metrics(model, props, cache, model_state: dict, scale: float, baseline_summary):
    features = edge_features(cache)
    tolls = predict_tolls(model_state, features)
    candidate_cache = route_with_tolls(cache, tolls, scale)
    rows, summary = evaluate(model, props, cache, candidate_cache)
    delta = probe.gaps(summary, baseline_summary)
    safe = (
        probe.summary_index(summary)["High"]["norm_fulfill_mean"] >= 0.995
        and delta["Medium.norm_fulfill_mean"] > 0.0
        and delta["Medium.norm_fulfill_p1"] > 0.0
        and delta["Medium.norm_fulfill_p10"] > 0.0
        and delta["Low.norm_fulfill_mean"] >= -0.003
        and delta["Low.norm_fulfill_p10"] >= -0.01
    )
    return {
        "scale": scale,
        "safe": safe,
        "summary": probe.compact(summary),
        "delta": delta,
        "toll_abs_mean": float(tolls.abs().mean().item()),
        "toll_abs_max": float(tolls.abs().max().item()),
        "rows": rows,
    }


def target_prediction_stats(predicted: torch.Tensor, target: torch.Tensor) -> dict:
    difference = predicted - target
    left = predicted.reshape(-1).detach().cpu().numpy()
    right = target.reshape(-1).detach().cpu().numpy()
    return {
        "mae": float(difference.abs().mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "correlation": float(np.corrcoef(left, right)[0, 1]),
        "target_std": float(target.std().item()),
        "prediction_std": float(predicted.std().item()),
    }


def public(value: dict) -> dict:
    return {key: item for key, item in value.items() if key != "rows"}


def main() -> None:
    torch.manual_seed(20260822)
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(2, device)
    model, checkpoint = runtime.load_backbone(2, 490, props, device)
    train = runtime.build_policy_cache(model, props, 0, 128, batch_size=16)
    safety = runtime.build_policy_cache(model, props, 128, 160, batch_size=16)
    validation = runtime.build_policy_cache(model, props, 160, 200, batch_size=16)
    evaluation = runtime.build_policy_cache(model, props, 200, 250, batch_size=16)

    train_features = edge_features(train)
    train_targets = teacher_tolls(model, props, train)
    state = fit_linear(train_features, train_targets)
    train_prediction = predict_tolls(state, train_features)
    safety_features = edge_features(safety)
    safety_targets = teacher_tolls(model, props, safety)
    safety_prediction = predict_tolls(state, safety_features)

    _, safety_baseline = evaluate(model, props, safety)
    scale_results = [
        metrics(model, props, safety, state, scale, safety_baseline)
        for scale in (0.1, 0.25, 0.5, 0.75, 1.0)
    ]
    eligible = [row for row in scale_results if row["safe"]]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: min(
                row["delta"]["Medium.norm_fulfill_mean"],
                row["delta"]["Medium.norm_fulfill_p1"],
                row["delta"]["Medium.norm_fulfill_p10"],
            ),
        )
        scale = float(selected["scale"])
        _, validation_baseline = evaluate(model, props, validation)
        validation_result = metrics(
            model, props, validation, state, scale, validation_baseline
        )
        if validation_result["safe"]:
            _, evaluation_baseline = evaluate(model, props, evaluation)
            evaluation_result = metrics(
                model, props, evaluation, state, scale, evaluation_baseline
            )
        else:
            evaluation_result = None
    else:
        scale = None
        validation_result = None
        evaluation_result = None

    payload = {
        "method": "per-edge affine ESM-to-teacher-toll regression",
        "parameters": int(state["coefficients"].numel()),
        "inference": "one affine edge pass plus one path softmax",
        "strict_esm_inference": True,
        "actual_tm_used_for_policy": False,
        "checkpoint": str(checkpoint),
        "protocol": {
            "train": [0, 128],
            "safety": [128, 160],
            "validation": [160, 200],
            "evaluation": [200, 250],
        },
        "teacher": {
            "steps": probe.TEACHER_CONFIG[0],
            "toll_projection": "target-flow weighted ridge",
        },
        "train_toll_prediction": target_prediction_stats(
            train_prediction, train_targets
        ),
        "safety_toll_prediction": target_prediction_stats(
            safety_prediction, safety_targets
        ),
        "safety_scale_results": [public(row) for row in scale_results],
        "selected_scale": scale,
        "validation": public(validation_result) if validation_result else None,
        "evaluation": public(evaluation_result) if evaluation_result else None,
        "small_hypothesis_pass": bool(
            evaluation_result is not None and evaluation_result["safe"]
        ),
        "seconds": time.perf_counter() - started,
    }
    artifact_dir = HERE / "artifacts" / "linear_toll_small"
    write_json(artifact_dir / "report.json", payload)
    torch.save(
        {
            "state": {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in state.items()},
            "scale": scale,
            "checkpoint": str(checkpoint),
        },
        artifact_dir / "model.pt",
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
