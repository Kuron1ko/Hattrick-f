from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
RISK_PATH = THIS_DIR / "run_risk_controller.py"
for item in (str(THIS_DIR), str(THIS_DIR.parent), str(THIS_DIR.parent.parent)):
    if item not in sys.path:
        sys.path.insert(0, item)
spec = importlib.util.spec_from_file_location("causal_esm_runtime", RISK_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {RISK_PATH}")
risk = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = risk
spec.loader.exec_module(risk)
edge = risk.edge
runtime = risk.runtime


def ratio(actual, predicted):
    value = actual / predicted.clamp_min(1e-6)
    return value.clamp(0.25, 4.0)


def causal_predictions(history, evaluation, strength, decay, window):
    states = []
    for class_index in range(3):
        history_ratio = ratio(
            history.tms[class_index][-window:],
            history.predicted_tms[class_index][-window:],
        )
        states.append(history_ratio.mean(dim=0))
    corrected = [[] for _ in range(3)]
    for sample in range(len(evaluation)):
        for class_index in range(3):
            prediction = evaluation.predicted_tms[class_index][sample]
            correction = (1.0 - float(strength)) + float(strength) * states[class_index]
            corrected[class_index].append(prediction * correction)
            observed_ratio = ratio(
                evaluation.tms[class_index][sample],
                evaluation.predicted_tms[class_index][sample],
            )
            states[class_index] = (
                float(decay) * states[class_index]
                + (1.0 - float(decay)) * observed_ratio
            )
    return tuple(torch.stack(values, dim=0) for values in corrected)


def candidate_cache(
    head, cache, ordinary_features, corrected_features, low_scale, medium_scale=1.0
):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    medium = []
    low = []
    with torch.no_grad():
        for start in range(0, len(cache), 32):
            stop = min(start + 32, len(cache))
            indices = torch.arange(start, stop, device=ordinary_features.device)
            base = [
                cache.policies[c].index_select(0, indices).squeeze(-1)
                for c in (1, 2)
            ]
            _, ordinary_tolls, _ = head(
                ordinary_features.index_select(0, indices), base, pte
            )
            _, corrected_tolls, _ = head(
                corrected_features.index_select(0, indices), base, pte
            )
            medium.append(
                risk.route_from_tolls(
                    base[0], ordinary_tolls[:, :, 0] * float(medium_scale), pte
                )
            )
            low.append(
                risk.route_from_tolls(
                    base[1], corrected_tolls[:, :, 1] * float(low_scale), pte
                )
            )
    return replace(
        cache,
        path_features=torch.cat(
            [torch.cat(medium, dim=0), torch.cat(low, dim=0)], dim=1
        ),
    )


def evaluate(
    model,
    props,
    head,
    cache,
    ordinary_features,
    corrected_features,
    low_scale,
    baseline_rows,
    medium_scale=1.0,
):
    adapted = candidate_cache(
        head,
        cache,
        ordinary_features,
        corrected_features,
        low_scale,
        medium_scale,
    )
    adapter = edge.TwoPolicyAdapter(int(cache.policies[0].shape[1]))
    rows, summary = runtime.evaluate_cache(model, props, adapted, adapter, batch_size=32)
    baseline_by_key = {(int(r["snapshot"]), r["class"]): r for r in baseline_rows}
    for row in rows:
        if row["class"] == "High":
            row.update(baseline_by_key[(int(row["snapshot"]), "High")])
    return {
        "summary": runtime.summary_index(summary),
        "diagnostics": edge.diagnostics(baseline_rows, rows),
        "rows": rows,
    }


def public(result):
    return {k: v for k, v in result.items() if k != "rows"}


def rank(result):
    diag = result["diagnostics"]
    worst = max(
        diag["Medium"]["ecdf_max_upward_violation"],
        diag["Low"]["ecdf_max_upward_violation"],
    )
    return (
        -worst,
        diag["Low"]["p1_gain"],
        diag["Low"]["p10_gain"],
        diag["Low"]["mean_gain"],
    )


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

    train = runtime.build_policy_cache(model, props, *edge.TRAIN, batch_size=64)
    safety = runtime.build_policy_cache(model, props, *edge.SAFETY, batch_size=32)
    history = runtime.build_policy_cache(model, props, 0, edge.VALIDATION[0], batch_size=64)
    validation = runtime.build_policy_cache(model, props, *edge.VALIDATION, batch_size=32)
    safety_rows, _ = runtime.evaluate_cache(model, props, safety, None, batch_size=32)
    validation_rows, _ = runtime.evaluate_cache(model, props, validation, None, batch_size=32)
    ordinary_safety = edge.edge_features(safety)
    ordinary_validation = edge.edge_features(validation)

    configs = [
        (strength, decay, window, low_scale)
        for strength in (0.5, 0.75, 1.0)
        for decay in (0.8, 0.95)
        for window in (8, 32)
        for low_scale in (1.0, 1.1, 1.2, 1.3, 1.4)
    ]
    safety_candidates = []
    for strength, decay, window, low_scale in configs:
        corrected_tms = causal_predictions(train, safety, strength, decay, window)
        corrected_cache = replace(safety, predicted_tms=corrected_tms)
        result = evaluate(
            model,
            props,
            head,
            safety,
            ordinary_safety,
            edge.edge_features(corrected_cache),
            low_scale,
            safety_rows,
        )
        result.update(
            {
                "strength": strength,
                "decay": decay,
                "window": window,
                "low_scale": low_scale,
            }
        )
        safety_candidates.append(result)
    winner = max(safety_candidates, key=rank)

    corrected_validation_tms = causal_predictions(
        history,
        validation,
        winner["strength"],
        winner["decay"],
        winner["window"],
    )
    corrected_validation_cache = replace(
        validation, predicted_tms=corrected_validation_tms
    )
    validation_result = evaluate(
        model,
        props,
        head,
        validation,
        ordinary_validation,
        edge.edge_features(corrected_validation_cache),
        winner["low_scale"],
        validation_rows,
    )
    validation_result.update(
        {
            "strength": winner["strength"],
            "decay": winner["decay"],
            "window": winner["window"],
            "low_scale": winner["low_scale"],
        }
    )
    # Diagnostics for method iteration; final 400-499 remains untouched.
    validation_sweep = []
    for strength, decay, window, low_scale in configs:
        corrected_tms = causal_predictions(history, validation, strength, decay, window)
        corrected_cache = replace(validation, predicted_tms=corrected_tms)
        result = evaluate(
            model,
            props,
            head,
            validation,
            ordinary_validation,
            edge.edge_features(corrected_cache),
            low_scale,
            validation_rows,
        )
        result.update(
            {
                "strength": strength,
                "decay": decay,
                "window": window,
                "low_scale": low_scale,
            }
        )
        validation_sweep.append(result)
    payload = {
        "method": "Causal rolling ESM-error calibration for Low edge tolls",
        "online_information": "current ESM plus prior realized TMs only",
        "current_actual_tm_used": False,
        "checkpoint": str(checkpoint),
        "safety_candidates": [public(x) for x in safety_candidates],
        "safety_winner": public(winner),
        "validation": public(validation_result),
        "validation_diagnostic_sweep": [public(x) for x in validation_sweep],
    }
    runtime.write_json(THIS_DIR / "causal_esm_low_validation.json", payload)
    runtime.write_csv(THIS_DIR / "causal_esm_low_validation_rows.csv", validation_result["rows"])
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
