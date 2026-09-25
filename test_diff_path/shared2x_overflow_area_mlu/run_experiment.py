from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EDGE_RUNNER_DIR = HERE.parent / "shared2x_learned_edge_cost"
SHADOW_CONTROL_DIR = (
    HERE.parent
    / "shared2x_shadow_price_weighted_mlu"
    / "artifacts"
    / "small_q0p75_delta0p003"
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EDGE_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_RUNNER_DIR))

import run_experiment as edge_runner  # noqa: E402


def number_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def paired_bootstrap(control: np.ndarray, candidate: np.ndarray) -> dict:
    rng = np.random.default_rng(490)
    difference = candidate - control
    indices = rng.integers(0, len(difference), size=(20000, len(difference)))
    means = difference[indices].mean(axis=1)
    return {
        class_name: {
            "mean_delta": float(difference[:, class_index].mean()),
            "ci95_low": float(np.percentile(means[:, class_index], 2.5)),
            "ci95_high": float(np.percentile(means[:, class_index], 97.5)),
            "positive_probability": float(np.mean(means[:, class_index] > 0.0)),
        }
        for class_index, class_name in enumerate(edge_runner.CLASSES)
    }


def load_reusable_small_control() -> tuple[str, str, np.ndarray]:
    report = json.loads(
        (SHADOW_CONTROL_DIR / "report.json").read_text(encoding="utf-8")
    )
    label = next(name for name in report["models"] if "control" in name)
    model = report["models"][label]
    values = np.loadtxt(
        SHADOW_CONTROL_DIR / f"{label}_norm_fulfill.csv",
        delimiter=",",
        skiprows=1,
    )
    return label, model, values


def train_and_evaluate(phase: str, label: str, overflow_lambda: float):
    model_path, elapsed = edge_runner.train_model(
        phase,
        label,
        None,
        "weighted_mlu_actual",
        4.0,
        0.0,
        False,
        0.0,
        0.0,
        hm_overflow_lambda=overflow_lambda,
    )
    values = edge_runner.evaluate(phase, label, model_path, False)
    return model_path, elapsed, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=edge_runner.PHASES, default="small")
    parser.add_argument("--lambdas", type=float, nargs="+", default=[0.01, 0.03])
    args = parser.parse_args()
    if any(value <= 0.0 or value > 1.0 for value in args.lambdas):
        raise SystemExit("Every lambda must be in (0, 1]")

    output_dir = HERE / "artifacts" / args.phase
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "small":
        control_label, control_model, control_values = load_reusable_small_control()
        control_seconds = 0.0
        control_reused = True
    else:
        control_label = f"overflow_area_{args.phase}_control"
        control_path, control_seconds, control_values = train_and_evaluate(
            args.phase, control_label, 0.0
        )
        control_model = str(control_path)
        control_reused = False
    np.savetxt(
        output_dir / "control_norm_fulfill.csv",
        control_values,
        delimiter=",",
        header=",".join(edge_runner.CLASSES),
        comments="",
    )

    control_summary = edge_runner.summarize(control_values)
    candidates = {}
    for overflow_lambda in args.lambdas:
        tag = number_tag(overflow_lambda)
        label = f"overflow_area_{args.phase}_lambda{tag}"
        model_path, elapsed, values = train_and_evaluate(
            args.phase, label, overflow_lambda
        )
        np.savetxt(
            output_dir / f"lambda{tag}_norm_fulfill.csv",
            values,
            delimiter=",",
            header=",".join(edge_runner.CLASSES),
            comments="",
        )
        summary = edge_runner.summarize(values)
        delta = {
            class_name: {
                metric: summary[class_name][metric]
                - control_summary[class_name][metric]
                for metric in ("mean", "p1", "p10", "median")
            }
            for class_name in edge_runner.CLASSES
        }
        bootstrap = paired_bootstrap(control_values, values)
        candidates[tag] = {
            "lambda": overflow_lambda,
            "model": str(model_path),
            "training_seconds": elapsed,
            "summary": summary,
            "candidate_minus_control": delta,
            "paired_bootstrap": bootstrap,
            "small_screen_pass": bool(
                delta["High"]["mean"] >= -0.0005
                and delta["Low"]["mean"] >= -0.003
                and bootstrap["Medium"]["ci95_low"] > 0.0
            ),
        }

    report = {
        "method": "High+Medium overflow-area MLU tie-break",
        "formula": (
            "L_HM = native_MLU_HM + lambda * "
            "mean_e(ReLU(real_train_util_HM,e - 1)^2)"
        ),
        "strict_esm_inference": True,
        "inference_change": None,
        "high_mlu_changed": False,
        "total_mlu_changed": False,
        "protocol": edge_runner.PHASES[args.phase],
        "phase": args.phase,
        "same_initial_model": str(edge_runner.BASE_MODEL),
        "identity_check": str(
            HERE / "artifacts" / "overflow_area_identity.json"
        ),
        "control": {
            "label": control_label,
            "model": control_model,
            "training_seconds": control_seconds,
            "reused_deterministic_control": control_reused,
            "summary": control_summary,
        },
        "candidates": candidates,
        "expandable_lambdas": [
            value["lambda"]
            for value in candidates.values()
            if value["small_screen_pass"]
        ],
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
