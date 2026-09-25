from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EDGE_RUNNER_DIR = HERE.parent / "shared2x_learned_edge_cost"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EDGE_RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(EDGE_RUNNER_DIR))

import run_experiment as edge_runner  # noqa: E402


SOURCE_ARTIFACT = HERE / "artifacts" / "small_q0p75_delta0p003"


def tag_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def paired_bootstrap(control: np.ndarray, candidate: np.ndarray) -> dict:
    rng = np.random.default_rng(490)
    differences = candidate - control
    indices = rng.integers(0, len(differences), size=(20000, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    return {
        class_name: {
            "mean_delta": float(differences[:, class_index].mean()),
            "ci95_low": float(np.percentile(bootstrap[:, class_index], 2.5)),
            "ci95_high": float(np.percentile(bootstrap[:, class_index], 97.5)),
            "positive_probability": float(
                np.mean(bootstrap[:, class_index] > 0.0)
            ),
        }
        for class_index, class_name in enumerate(edge_runner.CLASSES)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strength", type=float, default=0.003)
    args = parser.parse_args()
    if not 0.0 <= args.strength <= 0.05:
        raise SystemExit("--strength must be in [0, 0.05]")

    source_report = json.loads(
        (SOURCE_ARTIFACT / "report.json").read_text(encoding="utf-8")
    )
    source_payload = torch.load(
        SOURCE_ARTIFACT / "shadow_price_weights.pt",
        map_location="cpu",
        weights_only=False,
    )
    score = source_payload["high_shadow_score"].to(dtype=torch.float32)
    scores = torch.zeros(3, len(score), dtype=torch.float32)
    scores[0] = score

    strength_tag = tag_number(args.strength)
    output_dir = HERE / "artifacts" / f"small_smooth_tiebreak_beta{strength_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / "shadow_scores.pt"
    torch.save(
        {
            "method": "Medium shadow-price smooth High-MLU tie-break",
            "strict_esm_inference": True,
            "weights": scores,
            "source": str(SOURCE_ARTIFACT / "shadow_price_weights.pt"),
        },
        score_path,
    )

    # The control is deterministic and was trained with the identical seed,
    # initialization, split, optimizer, and 12-epoch protocol in the immediately
    # preceding experiment.  Reuse avoids another redundant control fit.
    control_label = next(
        label for label in source_report["models"] if "control" in label
    )
    control_csv = SOURCE_ARTIFACT / f"{control_label}_norm_fulfill.csv"
    control_values = np.loadtxt(control_csv, delimiter=",", skiprows=1)
    control_summary = edge_runner.summarize(control_values)

    candidate_label = f"shadow_smooth_beta{strength_tag}"
    model_path, elapsed = edge_runner.train_model(
        "small",
        candidate_label,
        score_path,
        "shadow_tiebreak_mlu_actual",
        4.0,
        args.strength,
        False,
        0.0,
        0.0,
    )
    candidate_values = edge_runner.evaluate(
        "small", candidate_label, model_path, False
    )
    np.savetxt(
        output_dir / f"{candidate_label}_norm_fulfill.csv",
        candidate_values,
        delimiter=",",
        header=",".join(edge_runner.CLASSES),
        comments="",
    )
    candidate_summary = edge_runner.summarize(candidate_values)
    delta = {
        class_name: {
            metric: candidate_summary[class_name][metric]
            - control_summary[class_name][metric]
            for metric in ("mean", "p1", "p10", "median")
        }
        for class_name in edge_runner.CLASSES
    }
    bootstrap = paired_bootstrap(control_values, candidate_values)
    report = {
        "method": "native High MLU plus smooth Medium-shadow occupancy tie-break",
        "formula": (
            "L_H=max_e(real_util_H,e) + beta * "
            "sum_e(score_e*real_util_H,e)/sum_e(score_e)"
        ),
        "strict_esm_inference": True,
        "inference_change": None,
        "only_high_mlu_changed": True,
        "strength": args.strength,
        "score_source": str(SOURCE_ARTIFACT / "shadow_price_weights.pt"),
        "protocol": edge_runner.PHASES["small"],
        "control_reused": {
            "label": control_label,
            "model": source_report["models"][control_label],
            "reason": "deterministic identical control protocol",
        },
        "candidate_model": str(model_path),
        "training_seconds": elapsed,
        "summaries": {
            "control": control_summary,
            "candidate": candidate_summary,
        },
        "candidate_minus_control": delta,
        "paired_bootstrap": bootstrap,
        "small_screen_pass": bool(
            delta["High"]["mean"] >= -0.0005
            and delta["Low"]["mean"] >= -0.003
            and bootstrap["Medium"]["ci95_low"] > 0.0
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
