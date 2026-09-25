from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module("tm_calibration_base", HERE / "run_tradeoff_screen.py")


def unique_tm(values):
    array = values.detach().cpu().numpy().squeeze(-1)
    grouped = array.reshape(len(array), -1, 8)
    repeat_error = float(np.max(np.abs(grouped - grouped[:, :, :1])))
    if repeat_error > 1e-5:
        raise RuntimeError(f"TM is not repeated per candidate path: {repeat_error}")
    return grouped[:, :, 0]


def arrays(cache):
    predicted_parts = [unique_tm(value) for value in cache.predicted_tms]
    actual_parts = [unique_tm(value) for value in cache.tms]
    widths = [value.shape[1] for value in predicted_parts]
    return np.concatenate(predicted_parts, axis=1), np.concatenate(actual_parts, axis=1), widths


def metric(predicted, actual):
    error = predicted - actual
    scale = float(np.mean(actual)) + 1e-12
    total_predicted = predicted.sum(axis=1)
    total_actual = actual.sum(axis=1)
    return {
        "nmae": float(np.mean(np.abs(error)) / scale),
        "nrmse": float(np.sqrt(np.mean(error ** 2)) / scale),
        "snapshot_total_correlation": float(
            np.corrcoef(total_predicted, total_actual)[0, 1]
        ),
        "mean_ratio": float(np.mean(predicted) / scale),
    }


def per_feature_scale(train_x, train_y, test_x):
    ratio = (train_x * train_y).sum(axis=0) / (
        (train_x ** 2).sum(axis=0) + 1e-9
    )
    return np.maximum(test_x * ratio, 0.0)


def per_feature_affine(train_x, train_y, test_x):
    mean_x = train_x.mean(axis=0)
    mean_y = train_y.mean(axis=0)
    centered = train_x - mean_x
    slope = (centered * (train_y - mean_y)).sum(axis=0) / (
        (centered ** 2).sum(axis=0) + 1e-9
    )
    return np.maximum(mean_y + (test_x - mean_x) * slope, 0.0)


def ridge_log(train_x, train_y, test_x, alpha):
    x = np.log1p(train_x)
    y = np.log1p(train_y)
    z = np.log1p(test_x)
    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    xc = x - x_mean
    yc = y - y_mean
    zc = z - x_mean
    feature_scale = np.sqrt(np.mean(xc ** 2, axis=0, keepdims=True) + 1e-8)
    xs = xc / feature_scale
    zs = zc / feature_scale
    kernel = xs @ xs.T / xs.shape[1]
    regularizer = alpha * float(np.trace(kernel) / len(kernel))
    dual = np.linalg.solve(
        kernel + regularizer * np.eye(len(kernel)), yc
    )
    prediction = y_mean + (zs @ xs.T / xs.shape[1]) @ dual
    return np.maximum(np.expm1(prediction), 0.0)


def knn_residual(train_x, train_y, test_x, k):
    x = np.log1p(train_x)
    z = np.log1p(test_x)
    mean = x.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean((x - mean) ** 2, axis=0, keepdims=True) + 1e-8)
    xs = (x - mean) / scale
    zs = (z - mean) / scale
    train_norm = (xs ** 2).sum(axis=1)
    test_norm = (zs ** 2).sum(axis=1)
    distances = test_norm[:, None] + train_norm[None, :] - 2 * zs @ xs.T
    neighbors = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    residual = np.log1p(train_y) - np.log1p(train_x)
    correction = residual[neighbors].mean(axis=1)
    return np.maximum(np.expm1(np.log1p(test_x) + correction), 0.0)


def main():
    base.runtime.set_seed(20260823)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"strict_esm": True, "training_bounds": [0, 318], "loads": {}}
    for load in (2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        caches = {
            "train": base.runtime.build_policy_cache(model, props, 0, 318, batch_size=32),
            "development": base.runtime.build_policy_cache(model, props, 318, 400, batch_size=32),
            "evaluation": base.runtime.build_policy_cache(model, props, 400, 500, batch_size=32),
        }
        values = {name: arrays(cache) for name, cache in caches.items()}
        train_x, train_y, widths = values["train"]
        load_report = {"od_widths": widths, "splits": {}}
        for split in ("development", "evaluation"):
            test_x, test_y, _ = values[split]
            predictions = {
                "raw": test_x,
                "per_feature_scale": per_feature_scale(train_x, train_y, test_x),
                "per_feature_affine": per_feature_affine(train_x, train_y, test_x),
            }
            for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
                predictions[f"ridge_log_{alpha:g}"] = ridge_log(
                    train_x, train_y, test_x, alpha
                )
            for k in (1, 4, 8, 16, 32):
                predictions[f"knn_residual_{k}"] = knn_residual(
                    train_x, train_y, test_x, k
                )
            metrics = {
                name: metric(prediction, test_y)
                for name, prediction in predictions.items()
            }
            load_report["splits"][split] = metrics
            best = min(metrics, key=lambda name: metrics[name]["nrmse"])
            print(
                f"[{load}x/{split}] raw={metrics['raw']['nrmse']:.4f} "
                f"best={best}:{metrics[best]['nrmse']:.4f} "
                f"corr={metrics[best]['snapshot_total_correlation']:.3f}",
                flush=True,
            )
        report["loads"][str(load)] = load_report
    path = HERE / "tm_calibrators.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
