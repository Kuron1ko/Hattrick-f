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


base = load_module("contextual_oracle_base", HERE / "run_tradeoff_screen.py")
WEIGHTS = (None, 0.0, 0.05, 0.10, 0.15, 0.20, 0.24, 0.30, 0.40)
LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20, 2.0, 4.0, 8.0)


def class_values(rows, class_name):
    return np.asarray(
        [float(row["norm_fulfill"]) for row in rows if row["class"] == class_name],
        dtype=np.float64,
    )


def stats(values):
    return {
        "mean": float(values.mean()),
        "p1": float(np.quantile(values, 0.01)),
        "p10": float(np.quantile(values, 0.10)),
    }


def gaps(values, baseline):
    left, right = stats(values), stats(baseline)
    return {key: left[key] - right[key] for key in left}


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "purpose": "oracle selector feasibility over strict-ESM candidates",
        "selector_uses_actual_and_is_not_deployable": True,
        "weights": list(WEIGHTS),
        "loads": {},
    }
    for load in (2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        cache = base.runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
        baseline_rows, _ = base.trainer.evaluate(model, props, cache)
        base_m = class_values(baseline_rows, "Medium")
        base_l = class_values(baseline_rows, "Low")
        medium_candidates = [base_m]
        low_candidates = [base_l]
        candidate_reports = []
        for weight in WEIGHTS[1:]:
            candidate = base.correction.correct_cache(
                model, props, cache, 24, 0.06, weight, 0.01
            )
            rows, _ = base.trainer.evaluate(model, props, cache, candidate)
            medium = class_values(rows, "Medium")
            low = class_values(rows, "Low")
            medium_candidates.append(medium)
            low_candidates.append(low)
            candidate_reports.append(
                {
                    "weight": weight,
                    "medium_delta": gaps(medium, base_m),
                    "low_delta": gaps(low, base_l),
                }
            )
        medium_matrix = np.stack(medium_candidates, axis=1)
        low_matrix = np.stack(low_candidates, axis=1)
        selector_rows = []
        for penalty in LAMBDAS:
            utility = medium_matrix + penalty * low_matrix
            choice = utility.argmax(axis=1)
            selected_m = medium_matrix[np.arange(len(choice)), choice]
            selected_l = low_matrix[np.arange(len(choice)), choice]
            value = {
                "low_penalty": penalty,
                "choice_counts": {
                    str(WEIGHTS[index]): int((choice == index).sum())
                    for index in range(len(WEIGHTS))
                    if bool((choice == index).any())
                },
                "medium_delta": gaps(selected_m, base_m),
                "low_delta": gaps(selected_l, base_l),
            }
            selector_rows.append(value)
            print(
                f"[{load}x] lambda={penalty:g} "
                f"M={value['medium_delta']['mean']:+.6f}/"
                f"{value['medium_delta']['p1']:+.6f}/"
                f"{value['medium_delta']['p10']:+.6f} "
                f"L={value['low_delta']['mean']:+.6f}/"
                f"{value['low_delta']['p1']:+.6f}/"
                f"{value['low_delta']['p10']:+.6f}",
                flush=True,
            )
        report["loads"][str(load)] = {
            "candidates": candidate_reports,
            "oracle_selectors": selector_rows,
        }
    path = HERE / "contextual_oracle.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
