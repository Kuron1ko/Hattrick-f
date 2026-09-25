from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
METHOD_PATH = THIS_DIR / "run_causal_esm_low.py"
for item in (str(THIS_DIR), str(THIS_DIR.parent), str(THIS_DIR.parent.parent)):
    if item not in sys.path:
        sys.path.insert(0, item)
spec = importlib.util.spec_from_file_location("fine_response_runtime", METHOD_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {METHOD_PATH}")
method = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = method
spec.loader.exec_module(method)
edge = method.edge
runtime = method.runtime


def public(result):
    return {k: v for k, v in result.items() if k != "rows"}


def main():
    runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    saved = torch.load(THIS_DIR / "best_edge_toll_head.pt", map_location=device)
    head = edge.EdgeTollHead(saved["edge_count"], saved["feature_count"]).to(device)
    head.load_state_dict(saved["state_dict"])
    head.eval()

    history = runtime.build_policy_cache(model, props, 0, edge.SAFETY[0], batch_size=64)
    development = runtime.build_policy_cache(
        model, props, edge.SAFETY[0], edge.VALIDATION[1], batch_size=64
    )
    dev_rows, _ = runtime.evaluate_cache(model, props, development, None, batch_size=32)
    ordinary_dev = edge.edge_features(development)
    corrected_features = {}
    causal_configs = [
        (strength, decay, window)
        for strength in (0.5, 0.75, 1.0)
        for decay in (0.8, 0.95)
        for window in (8, 32)
    ]
    for config in causal_configs:
        corrected = method.causal_predictions(history, development, *config)
        corrected_features[config] = edge.edge_features(
            replace(development, predicted_tms=corrected)
        )

    reference_config = (0.75, 0.8, 32)
    medium_sweep = []
    for medium_scale in np.linspace(0.0, 3.0, 31):
        result = method.evaluate(
            model,
            props,
            head,
            development,
            ordinary_dev,
            corrected_features[reference_config],
            1.2,
            dev_rows,
            medium_scale=float(medium_scale),
        )
        result["medium_scale"] = float(medium_scale)
        medium_sweep.append(result)
    medium_winner = max(
        medium_sweep,
        key=lambda x: (
            -x["diagnostics"]["Medium"]["ecdf_max_upward_violation"],
            x["diagnostics"]["Medium"]["mean_gain"],
            x["diagnostics"]["Medium"]["p10_gain"],
        ),
    )

    low_sweep = []
    for config in causal_configs:
        for low_scale in np.linspace(0.6, 1.6, 21):
            result = method.evaluate(
                model,
                props,
                head,
                development,
                ordinary_dev,
                corrected_features[config],
                float(low_scale),
                dev_rows,
                medium_scale=medium_winner["medium_scale"],
            )
            result.update(
                {
                    "strength": config[0],
                    "decay": config[1],
                    "window": config[2],
                    "low_scale": float(low_scale),
                }
            )
            low_sweep.append(result)
    low_winner = max(
        low_sweep,
        key=lambda x: (
            -x["diagnostics"]["Low"]["ecdf_max_upward_violation"],
            x["diagnostics"]["Low"]["p1_gain"],
            x["diagnostics"]["Low"]["p10_gain"],
            x["diagnostics"]["Low"]["mean_gain"],
        ),
    )

    final_history = runtime.build_policy_cache(model, props, 0, edge.FINAL[0], batch_size=64)
    final = runtime.build_policy_cache(model, props, *edge.FINAL, batch_size=64)
    final_rows, final_summary = runtime.evaluate_cache(
        model, props, final, None, batch_size=32
    )
    final_corrected = method.causal_predictions(
        final_history,
        final,
        low_winner["strength"],
        low_winner["decay"],
        low_winner["window"],
    )
    final_result = method.evaluate(
        model,
        props,
        head,
        final,
        edge.edge_features(final),
        edge.edge_features(replace(final, predicted_tms=final_corrected)),
        low_winner["low_scale"],
        final_rows,
        medium_scale=medium_winner["medium_scale"],
    )
    payload = {
        "method": "Fine-calibrated causal asymmetric edge-toll head",
        "checkpoint": str(checkpoint),
        "development_range": [edge.SAFETY[0], edge.VALIDATION[1]],
        "level4_range": list(edge.FINAL),
        "current_actual_tm_used": False,
        "medium_winner": public(medium_winner),
        "low_winner": public(low_winner),
        "medium_sweep": [public(x) for x in medium_sweep],
        "level4_baseline": runtime.summary_index(final_summary),
        "level4_candidate": public(final_result),
    }
    runtime.write_json(THIS_DIR / "fine_response_level4.json", payload)
    runtime.write_csv(THIS_DIR / "fine_level4_baseline_rows.csv", final_rows)
    runtime.write_csv(THIS_DIR / "fine_level4_candidate_rows.csv", final_result["rows"])
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
