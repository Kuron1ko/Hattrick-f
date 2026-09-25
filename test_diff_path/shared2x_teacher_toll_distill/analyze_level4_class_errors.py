from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import sys

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


trainer = load_module("class_error_trainer", "train_linear_toll.py")
large = load_module("class_error_large", "run_large_linear_toll.py")
probe = trainer.probe
runtime = trainer.runtime


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def state_on_device(saved: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in saved["state"].items()
    }


def class_prediction_stats(predicted: torch.Tensor, target: torch.Tensor) -> dict:
    result = {}
    for class_index, class_name in enumerate(("Medium", "Low")):
        left = predicted[:, class_index].reshape(-1).detach().cpu().numpy()
        right = target[:, class_index].reshape(-1).detach().cpu().numpy()
        difference = left - right
        result[class_name] = {
            "correlation": float(np.corrcoef(left, right)[0, 1]),
            "mae": float(np.mean(np.abs(difference))),
            "prediction_std": float(np.std(left)),
            "target_std": float(np.std(right)),
        }
    return result


def evaluate_variant(model, props, cache, adapted, baseline_summary) -> dict:
    rows, summary = trainer.evaluate(model, props, cache, adapted)
    return {
        "summary": probe.compact(summary),
        "delta": probe.gaps(summary, baseline_summary),
        "rows": rows,
    }


def public(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "rows"}


def main() -> None:
    torch.manual_seed(20260822)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    saved = torch.load(
        HERE / "artifacts" / "level4_linear_toll" / "model.pt",
        map_location=device,
        weights_only=False,
    )
    state = state_on_device(saved, device)
    cache = runtime.build_policy_cache(model, props, 350, 400, batch_size=16)
    baseline_rows, baseline_summary = trainer.evaluate(model, props, cache)
    teacher, projected, projection = probe.project_cache(
        model, props, cache, ridge=0.01, weighting="target_flow"
    )
    targets = torch.stack(
        [projection["medium_tolls"], projection["low_tolls"]], dim=1
    )
    predictions = trainer.predict_tolls(state, trainer.edge_features(cache))
    learned = trainer.route_with_tolls(cache, predictions, 1.0)
    path_count = int(cache.policies[0].shape[1])

    base_features = torch.cat(
        [cache.policies[1].squeeze(-1), cache.policies[2].squeeze(-1)], dim=1
    )
    variants = {
        "teacher_24_step": teacher,
        "exact_projected_tolls": projected,
        "predicted_both": learned,
        "predicted_medium_exact_low": replace(
            cache,
            path_features=torch.cat(
                [learned.path_features[:, :path_count], projected.path_features[:, path_count:]],
                dim=1,
            ),
        ),
        "exact_medium_predicted_low": replace(
            cache,
            path_features=torch.cat(
                [projected.path_features[:, :path_count], learned.path_features[:, path_count:]],
                dim=1,
            ),
        ),
        "predicted_medium_base_low": replace(
            cache,
            path_features=torch.cat(
                [learned.path_features[:, :path_count], base_features[:, path_count:]],
                dim=1,
            ),
        ),
        "base_medium_predicted_low": replace(
            cache,
            path_features=torch.cat(
                [base_features[:, :path_count], learned.path_features[:, path_count:]],
                dim=1,
            ),
        ),
    }
    results = {}
    for name, adapted in variants.items():
        value = evaluate_variant(model, props, cache, adapted, baseline_summary)
        value["bootstrap"] = large.paired_bootstrap(baseline_rows, value["rows"])
        results[name] = public(value)
        print(name, json.dumps(results[name], indent=2), flush=True)

    payload = {
        "range": [350, 400],
        "checkpoint": str(checkpoint),
        "strict_esm": True,
        "class_prediction": class_prediction_stats(predictions, targets),
        "projection_stats": {
            "Medium": projection["medium"],
            "Low": projection["low"],
        },
        "baseline": probe.compact(baseline_summary),
        "variants": results,
    }
    write_json(
        HERE / "artifacts" / "level4_linear_toll" / "class_error_analysis.json",
        payload,
    )


if __name__ == "__main__":
    main()
