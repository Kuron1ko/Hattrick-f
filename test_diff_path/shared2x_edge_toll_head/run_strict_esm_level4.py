from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
SOURCE_PATH = THIS_DIR / "run_experiment.py"
for item in (str(THIS_DIR), str(THIS_DIR.parent), str(THIS_DIR.parent.parent)):
    if item not in sys.path:
        sys.path.insert(0, item)
spec = importlib.util.spec_from_file_location("strict_esm_2x_source", SOURCE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {SOURCE_PATH}")
edge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = edge
spec.loader.exec_module(edge)
runtime = edge.runtime


def route_from_toll(base, toll, pte):
    batch = int(base.shape[0])
    path_cost = torch.sparse.mm(pte, toll.transpose(0, 1)).transpose(0, 1)
    grouped_base = base.reshape(batch, -1, edge.K)
    grouped_cost = path_cost.reshape(batch, -1, edge.K)
    valid = grouped_base > 0
    logits = torch.log(grouped_base.clamp_min(1e-12)) - grouped_cost
    logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
    routed = torch.softmax(logits, dim=-1) * grouped_base.sum(dim=-1, keepdim=True)
    return torch.where(valid, routed, torch.zeros_like(routed)).reshape_as(base)


def build_scaled_cache(head, cache, features, medium_scale, low_scale, batch_size=32):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    medium = []
    low = []
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            stop = min(start + batch_size, len(cache))
            indices = torch.arange(start, stop, device=features.device)
            base = [
                cache.policies[class_index].index_select(0, indices).squeeze(-1)
                for class_index in (1, 2)
            ]
            _, tolls, _ = head(features.index_select(0, indices), base, pte)
            outputs = []
            for class_index, scale in enumerate((medium_scale, low_scale)):
                if abs(float(scale)) <= 1e-12:
                    outputs.append(base[class_index])
                else:
                    outputs.append(
                        route_from_toll(
                            base[class_index],
                            tolls[:, :, class_index] * float(scale),
                            pte,
                        )
                    )
            medium.append(outputs[0])
            low.append(outputs[1])
    return replace(
        cache,
        path_features=torch.cat(
            [torch.cat(medium, dim=0), torch.cat(low, dim=0)], dim=1
        ),
    )


def evaluate_scaled(
    model,
    props,
    head,
    cache,
    features,
    baseline_rows,
    medium_scale,
    low_scale,
):
    candidate_cache = build_scaled_cache(
        head, cache, features, medium_scale, low_scale
    )
    adapter = edge.TwoPolicyAdapter(int(cache.policies[0].shape[1]))
    rows, summary = runtime.evaluate_cache(
        model, props, candidate_cache, adapter, batch_size=32
    )
    baseline_by_key = {
        (int(row["snapshot"]), row["class"]): row for row in baseline_rows
    }
    exact_classes = {"High"}
    if abs(float(medium_scale)) <= 1e-12:
        exact_classes.add("Medium")
    if (
        abs(float(medium_scale)) <= 1e-12
        and abs(float(low_scale)) <= 1e-12
    ):
        exact_classes.add("Low")
    for row in rows:
        if row["class"] in exact_classes:
            row.update(baseline_by_key[(int(row["snapshot"]), row["class"])])
    return {
        "medium_scale": float(medium_scale),
        "low_scale": float(low_scale),
        "summary": runtime.summary_index(summary),
        "diagnostics": edge.diagnostics(baseline_rows, rows),
        "rows": rows,
    }


def public(result):
    return {key: value for key, value in result.items() if key != "rows"}


def one_step_safe(result, class_name, count):
    violation = result["diagnostics"][class_name]["ecdf_max_upward_violation"]
    return int(violation <= 1.0 / float(count) + 1e-12)


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

    started = time.perf_counter()
    development = runtime.build_policy_cache(
        model, props, edge.SAFETY[0], edge.VALIDATION[1], batch_size=64
    )
    final = runtime.build_policy_cache(model, props, *edge.FINAL, batch_size=64)
    dev_features = edge.edge_features(development)
    final_features = edge.edge_features(final)
    dev_rows, dev_summary = runtime.evaluate_cache(
        model, props, development, None, batch_size=32
    )
    final_rows, final_summary = runtime.evaluate_cache(
        model, props, final, None, batch_size=32
    )

    raw_development = evaluate_scaled(
        model, props, head, development, dev_features, dev_rows, 1.0, 1.0
    )
    raw_final = evaluate_scaled(
        model, props, head, final, final_features, final_rows, 1.0, 1.0
    )

    scale_grid = np.linspace(0.0, 3.0, 31)
    medium_sweep = [
        evaluate_scaled(
            model,
            props,
            head,
            development,
            dev_features,
            dev_rows,
            float(scale),
            1.0,
        )
        for scale in scale_grid
    ]
    medium_winner = max(
        medium_sweep,
        key=lambda item: (
            one_step_safe(item, "Medium", len(development)),
            item["diagnostics"]["Medium"]["mean_gain"],
            item["diagnostics"]["Medium"]["p10_gain"],
            item["diagnostics"]["Medium"]["p1_gain"],
            -item["diagnostics"]["Medium"]["ecdf_max_upward_violation"],
        ),
    )
    low_sweep = [
        evaluate_scaled(
            model,
            props,
            head,
            development,
            dev_features,
            dev_rows,
            medium_winner["medium_scale"],
            float(scale),
        )
        for scale in scale_grid
    ]
    low_winner = max(
        low_sweep,
        key=lambda item: (
            one_step_safe(item, "Low", len(development)),
            item["diagnostics"]["Low"]["p1_gain"],
            item["diagnostics"]["Low"]["p10_gain"],
            item["diagnostics"]["Low"]["mean_gain"],
            -item["diagnostics"]["Low"]["ecdf_max_upward_violation"],
        ),
    )
    selected_final = evaluate_scaled(
        model,
        props,
        head,
        final,
        final_features,
        final_rows,
        medium_winner["medium_scale"],
        low_winner["low_scale"],
    )

    final_policy = build_scaled_cache(
        head,
        final,
        final_features,
        medium_winner["medium_scale"],
        low_winner["low_scale"],
    )
    counterfactual = replace(
        final, tms=tuple(torch.zeros_like(value) for value in final.tms)
    )
    counterfactual_features = edge.edge_features(counterfactual)
    counterfactual_policy = build_scaled_cache(
        head,
        counterfactual,
        counterfactual_features,
        medium_winner["medium_scale"],
        low_winner["low_scale"],
    )
    repeated_features = edge.edge_features(final)
    repeated_policy = build_scaled_cache(
        head,
        final,
        repeated_features,
        medium_winner["medium_scale"],
        low_winner["low_scale"],
    )
    payload = {
        "method": "2x strict-ESM-only asymmetric edge-toll residual head",
        "checkpoint": str(checkpoint),
        "training_checkpoint": str(THIS_DIR / "best_edge_toll_head.pt"),
        "protocol": {
            "train": list(edge.TRAIN),
            "checkpoint_selection": list(edge.SAFETY),
            "fixed_scale_development": [edge.SAFETY[0], edge.VALIDATION[1]],
            "level4": list(edge.FINAL),
            "note": "Level-4 range was used by earlier experiments and is not a pristine never-before-seen holdout.",
        },
        "information_contract": {
            "policy_inputs": "current doubled ESM predictions, topology, capacities, frozen Hattrick policies",
            "current_actual_tm": "evaluation only",
            "historical_actual_tm": "not used",
            "offline_training_actual_tm": "loss and checkpoint/scale selection only",
            "counterfactual_feature_max_abs_diff": float(
                (final_features - counterfactual_features).abs().max().item()
            ),
            "same_input_repeat_feature_max_abs_diff": float(
                (final_features - repeated_features).abs().max().item()
            ),
            "counterfactual_policy_max_abs_diff": float(
                (final_policy.path_features - counterfactual_policy.path_features)
                .abs()
                .max()
                .item()
            ),
            "same_input_repeat_policy_max_abs_diff": float(
                (final_policy.path_features - repeated_policy.path_features)
                .abs()
                .max()
                .item()
            ),
        },
        "selection_rule": "fixed global scales selected on 318-399; at most one empirical CDF step is treated as nonsignificant",
        "reported_primary": "native trained head at medium_scale=1 and low_scale=1; the development-tuned scales are retained as a failed generalization check",
        "elapsed_seconds": time.perf_counter() - started,
        "development_baseline": runtime.summary_index(dev_summary),
        "raw_development": public(raw_development),
        "medium_winner": public(medium_winner),
        "low_winner": public(low_winner),
        "level4_baseline": runtime.summary_index(final_summary),
        "raw_level4": public(raw_final),
        "primary_level4": public(raw_final),
        "selected_level4": public(selected_final),
        "medium_sweep": [public(item) for item in medium_sweep],
        "low_sweep": [public(item) for item in low_sweep],
    }
    runtime.write_json(THIS_DIR / "strict_esm_2x_level4.json", payload)
    runtime.write_csv(THIS_DIR / "strict_esm_2x_hattrick_rows.csv", final_rows)
    runtime.write_csv(THIS_DIR / "strict_esm_2x_candidate_rows.csv", raw_final["rows"])
    runtime.write_csv(
        THIS_DIR / "strict_esm_2x_devscaled_candidate_rows.csv",
        selected_final["rows"],
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
