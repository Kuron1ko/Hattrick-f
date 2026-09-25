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
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
EDGE_PATH = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
ONEX_PATH = TEST_DIR / "shared1x_esm_sar" / "probe_onex_transfer.py"
for item in (str(ROOT), str(TEST_DIR), str(EDGE_PATH.parent), str(ONEX_PATH.parent)):
    if item not in sys.path:
        sys.path.insert(0, item)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


edge = load_module("shared1x_edge_runtime", EDGE_PATH)
onex = load_module("shared1x_base_runtime", ONEX_PATH)
runtime = edge.runtime

TRAIN = (0, 318)
SAFETY = (318, 350)
VALIDATION = (350, 400)
FINAL = (400, 500)


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
                cache.policies[c].index_select(0, indices).squeeze(-1)
                for c in (1, 2)
            ]
            _, tolls, _ = head(features.index_select(0, indices), base, pte)
            scales = (float(medium_scale), float(low_scale))
            outputs = []
            for class_index, scale in enumerate(scales):
                if abs(scale) <= 1e-12:
                    outputs.append(base[class_index])
                else:
                    outputs.append(
                        route_from_toll(
                            base[class_index], tolls[:, :, class_index] * scale, pte
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
    for row in rows:
        if row["class"] == "High":
            row.update(baseline_by_key[(int(row["snapshot"]), "High")])
    return {
        "medium_scale": float(medium_scale),
        "low_scale": float(low_scale),
        "summary": runtime.summary_index(summary),
        "diagnostics": edge.diagnostics(baseline_rows, rows),
        "rows": rows,
    }


def public(result):
    return {key: value for key, value in result.items() if key not in ("rows", "state")}


def train_rank(result):
    diag = result["diagnostics"]
    worst = max(
        diag["Medium"]["ecdf_max_upward_violation"],
        diag["Low"]["ecdf_max_upward_violation"],
    )
    return (
        -worst,
        diag["Medium"]["mean_gain"],
        diag["Medium"]["p10_gain"],
        diag["Low"]["mean_gain"],
    )


def main():
    edge.runtime.set_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props = onex.load_onex(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    train_cache = runtime.build_policy_cache(model, props, *TRAIN, batch_size=64)
    safety_cache = runtime.build_policy_cache(model, props, *SAFETY, batch_size=32)
    validation_cache = runtime.build_policy_cache(
        model, props, *VALIDATION, batch_size=32
    )
    development_cache = runtime.build_policy_cache(
        model, props, SAFETY[0], VALIDATION[1], batch_size=64
    )
    final_cache = runtime.build_policy_cache(model, props, *FINAL, batch_size=64)

    caches = {
        "train": train_cache,
        "safety": safety_cache,
        "validation": validation_cache,
        "development": development_cache,
        "final": final_cache,
    }
    features = {name: edge.edge_features(cache) for name, cache in caches.items()}
    baseline_rows = {}
    baseline_summary = {}
    for name in ("safety", "validation", "development", "final"):
        rows, summary = runtime.evaluate_cache(
            model, props, caches[name], None, batch_size=32
        )
        baseline_rows[name] = rows
        baseline_summary[name] = runtime.summary_index(summary)
    train_baseline_norm = edge.baseline_normalized(model, props, train_cache)

    started = time.perf_counter()
    candidates = []
    histories = []
    for index, (safety_weight, low_reward) in enumerate(
        ((10.0, 0.25), (25.0, 0.25), (50.0, 0.5))
    ):
        current, history = edge.train_one(
            model,
            props,
            train_cache,
            features["train"],
            train_baseline_norm,
            safety_cache,
            features["safety"],
            baseline_rows["safety"],
            safety_weight,
            low_reward,
            20262001 + index,
        )
        candidates.extend(current)
        histories.append(
            {
                "safety_weight": safety_weight,
                "low_reward": low_reward,
                "history": history,
            }
        )
    trained_winner = max(candidates, key=train_rank)
    head = edge.EdgeTollHead(
        int(train_cache.capacities.shape[1]), int(features["train"].shape[-1])
    ).to(device)
    head.load_state_dict(trained_winner["state"])
    head.eval()

    raw_validation = evaluate_scaled(
        model,
        props,
        head,
        validation_cache,
        features["validation"],
        baseline_rows["validation"],
        1.0,
        1.0,
    )

    scale_grid = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    medium_sweep = []
    for scale in scale_grid:
        result = evaluate_scaled(
            model,
            props,
            head,
            development_cache,
            features["development"],
            baseline_rows["development"],
            scale,
            1.0,
        )
        medium_sweep.append(result)
    medium_winner = max(
        medium_sweep,
        key=lambda x: (
            -x["diagnostics"]["Medium"]["ecdf_max_upward_violation"],
            x["diagnostics"]["Medium"]["mean_gain"],
            x["diagnostics"]["Medium"]["p10_gain"],
            x["diagnostics"]["Medium"]["p1_gain"],
        ),
    )
    low_sweep = []
    for scale in scale_grid:
        result = evaluate_scaled(
            model,
            props,
            head,
            development_cache,
            features["development"],
            baseline_rows["development"],
            medium_winner["medium_scale"],
            scale,
        )
        low_sweep.append(result)
    low_winner = max(
        low_sweep,
        key=lambda x: (
            -x["diagnostics"]["Low"]["ecdf_max_upward_violation"],
            x["diagnostics"]["Low"]["mean_gain"],
            x["diagnostics"]["Low"]["p10_gain"],
            x["diagnostics"]["Low"]["p1_gain"],
        ),
    )

    final_result = evaluate_scaled(
        model,
        props,
        head,
        final_cache,
        features["final"],
        baseline_rows["final"],
        medium_winner["medium_scale"],
        low_winner["low_scale"],
    )
    # Counterfactual information-flow audit: changing every actual traffic matrix
    # must not change a single policy value produced at inference time.
    final_policy_cache = build_scaled_cache(
        head,
        final_cache,
        features["final"],
        medium_winner["medium_scale"],
        low_winner["low_scale"],
    )
    counterfactual_cache = replace(
        final_cache,
        tms=tuple(torch.zeros_like(tm) for tm in final_cache.tms),
    )
    counterfactual_features = edge.edge_features(counterfactual_cache)
    counterfactual_policy_cache = build_scaled_cache(
        head,
        counterfactual_cache,
        counterfactual_features,
        medium_winner["medium_scale"],
        low_winner["low_scale"],
    )
    repeated_features = edge.edge_features(final_cache)
    repeated_policy_cache = build_scaled_cache(
        head,
        final_cache,
        repeated_features,
        medium_winner["medium_scale"],
        low_winner["low_scale"],
    )
    feature_invariance = float(
        (features["final"] - counterfactual_features).abs().max().item()
    )
    policy_invariance = float(
        (
            final_policy_cache.path_features
            - counterfactual_policy_cache.path_features
        )
        .abs()
        .max()
        .item()
    )
    repeat_feature_noise = float(
        (features["final"] - repeated_features).abs().max().item()
    )
    repeat_policy_noise = float(
        (final_policy_cache.path_features - repeated_policy_cache.path_features)
        .abs()
        .max()
        .item()
    )
    payload = {
        "method": "1x strict-ESM-only asymmetric edge-toll residual head",
        "base_model": str(onex.MODEL_PATH),
        "topology": onex.TOPOLOGY,
        "protocol": {
            "train": list(TRAIN),
            "safety": list(SAFETY),
            "validation": list(VALIDATION),
            "development_for_fixed_scales": [SAFETY[0], VALIDATION[1]],
            "level4": list(FINAL),
        },
        "information_contract": {
            "policy_inputs": "current ESM predictions, topology, capacities, frozen Hattrick base policies",
            "current_actual_tm": "evaluation only",
            "historical_actual_tm": "not used",
            "offline_training_actual_tm": "loss and checkpoint evaluation only",
            "counterfactual_feature_max_abs_diff": feature_invariance,
            "counterfactual_policy_max_abs_diff": policy_invariance,
            "same_input_repeat_feature_max_abs_diff": repeat_feature_noise,
            "same_input_repeat_policy_max_abs_diff": repeat_policy_noise,
        },
        "architecture": {
            "high": "exact frozen original Hattrick policy",
            "medium_low": "new edge-toll head trained from zero initialization on 1x",
            "learned_edge_features": int(features["train"].shape[-1]),
            "edge_count": int(train_cache.capacities.shape[1]),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "trained_winner": public(trained_winner),
        "raw_validation": public(raw_validation),
        "medium_winner": public(medium_winner),
        "low_winner": public(low_winner),
        "level4_baseline": baseline_summary["final"],
        "level4_candidate": public(final_result),
        "histories": histories,
    }
    THIS_DIR.mkdir(parents=True, exist_ok=True)
    runtime.write_json(THIS_DIR / "strict_esm_onex_level4.json", payload)
    runtime.write_csv(THIS_DIR / "onex_level4_hattrick_rows.csv", baseline_rows["final"])
    runtime.write_csv(THIS_DIR / "onex_level4_edge_toll_rows.csv", final_result["rows"])
    torch.save(
        {
            "state_dict": trained_winner["state"],
            "feature_count": int(features["train"].shape[-1]),
            "edge_count": int(train_cache.capacities.shape[1]),
            "medium_scale": medium_winner["medium_scale"],
            "low_scale": low_winner["low_scale"],
            "information_contract": payload["information_contract"],
        },
        THIS_DIR / "strict_esm_onex_edge_toll.pt",
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
