from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
LEVEL3_ROOT = ROOT.parent / "shared2x_counterfactual_level3"
SHARED_RUNTIME = TEST_DIR / "shared2x_order_regularizer"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(SHARED_RUNTIME))

import run_experiment as shared  # noqa: E402
from frameworks.hattrick_system import Hattrick  # noqa: E402
from run_level0 import phase_a_path, sha256  # noqa: E402
from utils.build_dataset_within_cluster import DM_Dataset_within_Cluster  # noqa: E402


ALPHAS = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)


def class_rows(rows: list[dict], name: str) -> dict[int, dict]:
    return {int(row["snapshot"]): row for row in rows if row["class"] == name}


def main() -> None:
    seed = 490
    output = LEVEL3_ROOT / "line_search" / "guarded_changed_seed_490"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared.set_seed(seed)
    props = shared.build_props(3, device)
    dataset = DM_Dataset_within_Cluster(props, 0, 350, 400)
    phase_path = phase_a_path("train350", seed)
    final_path = (
        LEVEL3_ROOT
        / "distill"
        / "level3_validation_only_screen_train32"
        / "guarded_changed"
        / "seed_490"
        / "final_model.pt"
    )
    phase = torch.load(phase_path, map_location="cpu", weights_only=False)[
        "model_state_dict"
    ]
    final = torch.load(final_path, map_location="cpu", weights_only=False)[
        "model_state_dict"
    ]
    model = Hattrick(props).to(device=device, dtype=props.dtype)
    records = []
    baseline_rows = None
    for alpha in ALPHAS:
        state = {}
        for name in phase:
            left = phase[name]
            right = final[name]
            state[name] = left + float(alpha) * (right - left) if left.is_floating_point() else left
        model.load_state_dict(state)
        if hasattr(model, "transformer_output"):
            delattr(model, "transformer_output")
        rows, summary, diagnostics = shared.evaluate(model, props, dataset, 350)
        if baseline_rows is None:
            baseline_rows = rows
        baseline_high = class_rows(baseline_rows, "High")
        current_high = class_rows(rows, "High")
        high_delta = np.asarray(
            [
                float(current_high[key]["norm_fulfill"])
                - float(baseline_high[key]["norm_fulfill"])
                for key in sorted(baseline_high)
            ]
        )
        baseline_medium = class_rows(baseline_rows, "Medium")
        current_medium = class_rows(rows, "Medium")
        medium_delta = np.asarray(
            [
                float(current_medium[key]["norm_fulfill"])
                - float(baseline_medium[key]["norm_fulfill"])
                for key in sorted(baseline_medium)
            ]
        )
        indexed = {row["class"]: row for row in summary}
        baseline_indexed = {
            name: np.asarray(
                [float(value["norm_fulfill"]) for value in class_rows(baseline_rows, name).values()]
            )
            for name in ("High", "Medium", "Low")
        }
        low_base = class_rows(baseline_rows, "Low")
        low_now = class_rows(rows, "Low")
        low_fulfill_delta = np.mean(
            [
                float(low_now[key]["fulfill_ratio"])
                - float(low_base[key]["fulfill_ratio"])
                for key in low_base
            ]
        )
        records.append(
            {
                "alpha": alpha,
                "high_delta_mean": float(high_delta.mean()),
                "high_delta_min": float(high_delta.min()),
                "medium_delta_mean": float(medium_delta.mean()),
                "medium_delta_p1": float(np.percentile(medium_delta, 1)),
                "medium_delta_p10": float(np.percentile(medium_delta, 10)),
                "medium_positive_fraction": float((medium_delta > 0).mean()),
                "medium_mean": indexed["Medium"]["norm_fulfill_mean"],
                "medium_p1": indexed["Medium"]["norm_fulfill_p1"],
                "medium_p10": indexed["Medium"]["norm_fulfill_p10"],
                "low_fulfill_delta_mean": float(low_fulfill_delta),
                "inversion_fraction": diagnostics["inversion_violation_fraction"],
                "max_capacity_ratio": max(
                    float(value["max_admitted_capacity_ratio"]) for value in summary
                ),
                "eligible": bool(high_delta.min() >= -0.005),
            }
        )
        print(
            f"alpha={alpha:.3f} high_min={high_delta.min():+.5f} "
            f"medium_mean_gap={medium_delta.mean():+.5f} "
            f"medium_p10_gap={np.percentile(medium_delta, 10):+.5f}",
            flush=True,
        )
    eligible = [row for row in records if row["eligible"]]
    selected = max(
        eligible,
        key=lambda row: (row["medium_p10"], row["medium_p1"], row["medium_mean"]),
    )
    with (output / "line_search.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    payload = {
        "status": "PASS" if selected["medium_delta_p10"] >= 0.01 else "NO_GO",
        "selected": selected,
        "phase_a": str(phase_path.resolve()),
        "phase_a_sha256": sha256(phase_path),
        "final": str(final_path.resolve()),
        "final_sha256": sha256(final_path),
        "note": "validation-only checkpoint interpolation; no test window used",
    }
    (output / "complete.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
