from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
RUNTIME_DIR = TEST_DIR / "shared2x_medium_adapter"
for item in (str(ROOT), str(TEST_DIR), str(RUNTIME_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

spec = importlib.util.spec_from_file_location(
    "shared2x_esm_calibration_runtime", RUNTIME_DIR / "run_experiment.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load frozen-policy runtime")
runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime
spec.loader.exec_module(runtime)


K = 8


def od_values(tensor: torch.Tensor) -> np.ndarray:
    values = tensor.detach().cpu().numpy().squeeze(-1)
    return values.reshape(values.shape[0], -1, K)[:, :, 0].astype(np.float64)


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    scale = max(float(np.mean(actual)), 1e-9)
    return {
        "mean_actual": float(np.mean(actual)),
        "mean_predicted": float(np.mean(predicted)),
        "mean_ratio": float(np.mean(predicted) / scale),
        "nmae": float(np.mean(np.abs(error)) / scale),
        "nrmse": float(np.sqrt(np.mean(error**2)) / scale),
    }


def lag_one_correlation(residual: np.ndarray) -> float:
    left = residual[:-1].reshape(-1)
    right = residual[1:].reshape(-1)
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def calibrated_validation(
    train_actual: np.ndarray,
    train_predicted: np.ndarray,
    val_actual: np.ndarray,
    val_predicted: np.ndarray,
    shrink: float,
    beta: float,
) -> np.ndarray:
    eps = 1e-6
    train_log_error = np.log((train_actual + eps) / (train_predicted + eps))
    static = float(shrink) * np.mean(train_log_error, axis=0)
    state = static.copy()
    corrected = np.empty_like(val_predicted)
    for index in range(val_predicted.shape[0]):
        corrected[index] = val_predicted[index] * np.exp(np.clip(state, -0.7, 0.7))
        observed = np.log((val_actual[index] + eps) / (val_predicted[index] + eps))
        state = (1.0 - float(beta)) * state + float(beta) * observed
    return corrected


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    train = runtime.build_policy_cache(model, props, 0, 350, batch_size=16)
    validation = runtime.build_policy_cache(model, props, 350, 400, batch_size=16)
    rows: list[dict] = []
    for class_index, class_name in enumerate(runtime.CLASSES):
        train_actual = od_values(train.tms[class_index])
        train_predicted = od_values(train.predicted_tms[class_index])
        val_actual = od_values(validation.tms[class_index])
        val_predicted = od_values(validation.predicted_tms[class_index])
        raw = score(val_actual, val_predicted)
        log_residual = np.log((train_actual + 1e-6) / (train_predicted + 1e-6))
        for shrink in (0.0, 0.5, 1.0):
            for beta in (0.0, 0.1, 0.3, 0.5, 0.8, 1.0):
                corrected = calibrated_validation(
                    train_actual,
                    train_predicted,
                    val_actual,
                    val_predicted,
                    shrink,
                    beta,
                )
                item = {
                    "class": class_name,
                    "shrink": shrink,
                    "beta": beta,
                    "train_log_error_lag1": lag_one_correlation(log_residual),
                    "raw": raw,
                    "corrected": score(val_actual, corrected),
                }
                rows.append(item)
    best = {}
    for class_name in runtime.CLASSES:
        candidates = [row for row in rows if row["class"] == class_name]
        best[class_name] = min(candidates, key=lambda row: row["corrected"]["nrmse"])
    output = {
        "checkpoint": str(checkpoint),
        "protocol": "validation correction at t uses training residuals and residuals observed only before t",
        "best_by_validation_nrmse": best,
        "all": rows,
    }
    THIS_DIR.mkdir(parents=True, exist_ok=True)
    with (THIS_DIR / "prediction_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(best, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
