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
EDGE_SOURCE = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
for item in (str(ROOT), str(TEST_DIR), str(EDGE_SOURCE.parent)):
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


edge = load_module("shared3x_edge_toll_source", EDGE_SOURCE)
runtime = edge.runtime

TOPOLOGY = "geant_priomask500_shared_load3x_train"
MODEL_PATH = ROOT / f"hattrick_{TOPOLOGY}_8sp.pkl"
TRAIN = (0, 318)
SAFETY = (318, 350)
VALIDATION = (350, 400)
DEVELOPMENT = (318, 400)
FINAL = (400, 500)


def load_hattrick3(device: torch.device):
    runtime.shared.TOPOLOGY = TOPOLOGY
    props = runtime.build_props(4, device)
    props.mode = "test"
    props.sim_mf_mlu = 0
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = model.to(device=device, dtype=props.dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    return model, props


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
    medium: list[torch.Tensor] = []
    low: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            stop = min(start + batch_size, len(cache))
            indices = torch.arange(start, stop, device=features.device)
            base = [
                cache.policies[class_index].index_select(0, indices).squeeze(-1)
                for class_index in (1, 2)
            ]
            _, tolls, _ = head(features.index_select(0, indices), base, pte)
            outputs: list[torch.Tensor] = []
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


def evaluate_scaled(model, props, head, cache, features, baseline_rows, medium_scale, low_scale):
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
    if abs(float(medium_scale)) <= 1e-12 and abs(float(low_scale)) <= 1e-12:
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


def public(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in ("rows", "state")}


def one_step_safe(result: dict, class_name: str, count: int) -> int:
    violation = result["diagnostics"][class_name]["ecdf_max_upward_violation"]
    return int(float(violation) <= 1.0 / float(count) + 1e-12)


def train_rank(result: dict):
    diagnostics = result["diagnostics"]
    worst = max(
        diagnostics["Medium"]["ecdf_max_upward_violation"],
        diagnostics["Low"]["ecdf_max_upward_violation"],
    )
    return (
        -worst,
        diagnostics["Medium"]["mean_gain"],
        diagnostics["Medium"]["p10_gain"],
        diagnostics["Low"]["mean_gain"],
    )


def main() -> None:
    runtime.set_seed(20260822)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props = load_hattrick3(device)

    train_cache = runtime.build_policy_cache(model, props, *TRAIN, batch_size=64)
    safety_cache = runtime.build_policy_cache(model, props, *SAFETY, batch_size=32)
    validation_cache = runtime.build_policy_cache(model, props, *VALIDATION, batch_size=32)
    development_cache = runtime.build_policy_cache(model, props, *DEVELOPMENT, batch_size=64)
    final_cache = runtime.build_policy_cache(model, props, *FINAL, batch_size=64)
    caches = {
        "train": train_cache,
        "safety": safety_cache,
        "validation": validation_cache,
        "development": development_cache,
        "final": final_cache,
    }
    features = {name: edge.edge_features(cache) for name, cache in caches.items()}
    baseline_rows: dict[str, list[dict]] = {}
    baseline_summary: dict[str, dict] = {}
    for name in ("safety", "validation", "development", "final"):
        rows, summary = runtime.evaluate_cache(
            model, props, caches[name], None, batch_size=32
        )
        baseline_rows[name] = rows
        baseline_summary[name] = runtime.summary_index(summary)
    train_baseline_norm = edge.baseline_normalized(model, props, train_cache)

    started = time.perf_counter()
    candidates: list[dict] = []
    histories: list[dict] = []
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
            20263001 + index,
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

    raw_development = evaluate_scaled(
        model,
        props,
        head,
        development_cache,
        features["development"],
        baseline_rows["development"],
        1.0,
        1.0,
    )
    raw_final = evaluate_scaled(
        model,
        props,
        head,
        final_cache,
        features["final"],
        baseline_rows["final"],
        1.0,
        1.0,
    )

    scale_grid = np.linspace(0.0, 3.0, 31)
    medium_sweep = [
        evaluate_scaled(
            model,
            props,
            head,
            development_cache,
            features["development"],
            baseline_rows["development"],
            float(scale),
            1.0,
        )
        for scale in scale_grid
    ]
    medium_winner = max(
        medium_sweep,
        key=lambda item: (
            one_step_safe(item, "Medium", len(development_cache)),
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
            development_cache,
            features["development"],
            baseline_rows["development"],
            medium_winner["medium_scale"],
            float(scale),
        )
        for scale in scale_grid
    ]
    low_winner = max(
        low_sweep,
        key=lambda item: (
            one_step_safe(item, "Low", len(development_cache)),
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
        final_cache,
        features["final"],
        baseline_rows["final"],
        medium_winner["medium_scale"],
        low_winner["low_scale"],
    )

    final_policy = build_scaled_cache(
        head,
        final_cache,
        features["final"],
        medium_winner["medium_scale"],
        low_winner["low_scale"],
    )
    counterfactual_cache = replace(
        final_cache, tms=tuple(torch.zeros_like(tm) for tm in final_cache.tms)
    )
    counterfactual_features = edge.edge_features(counterfactual_cache)
    counterfactual_policy = build_scaled_cache(
        head,
        counterfactual_cache,
        counterfactual_features,
        medium_winner["medium_scale"],
        low_winner["low_scale"],
    )
    repeated_features = edge.edge_features(final_cache)
    repeated_policy = build_scaled_cache(
        head,
        final_cache,
        repeated_features,
        medium_winner["medium_scale"],
        low_winner["low_scale"],
    )
    information_contract = {
        "policy_inputs": "current tripled ESM predictions, topology, capacities, frozen Hattrick3 policies",
        "current_actual_tm": "evaluation only",
        "historical_actual_tm": "not used",
        "offline_training_actual_tm": "loss and checkpoint/scale selection only",
        "counterfactual_feature_max_abs_diff": float(
            (features["final"] - counterfactual_features).abs().max().item()
        ),
        "counterfactual_policy_max_abs_diff": float(
            (final_policy.path_features - counterfactual_policy.path_features)
            .abs()
            .max()
            .item()
        ),
        "same_input_repeat_feature_max_abs_diff": float(
            (features["final"] - repeated_features).abs().max().item()
        ),
        "same_input_repeat_policy_max_abs_diff": float(
            (final_policy.path_features - repeated_policy.path_features)
            .abs()
            .max()
            .item()
        ),
    }
    payload = {
        "method": "Hattrick-neo3: strict-ESM asymmetric edge-toll residual head",
        "base_model": str(MODEL_PATH),
        "topology": TOPOLOGY,
        "load_factor": 3.0,
        "protocol": {
            "neo_train": list(TRAIN),
            "checkpoint_selection": list(SAFETY),
            "fixed_scale_development": list(DEVELOPMENT),
            "level4": list(FINAL),
        },
        "information_contract": information_contract,
        "architecture": {
            "high": "exact frozen Hattrick3 policy",
            "medium_low": "new edge-toll head trained from zero initialization on 3x",
            "edge_features": int(features["train"].shape[-1]),
            "edge_count": int(train_cache.capacities.shape[1]),
        },
        "selection_rule": "checkpoint selected on 318-349; global class scales selected on 318-399 before one Level-4 evaluation",
        "elapsed_seconds": time.perf_counter() - started,
        "trained_winner": public(trained_winner),
        "raw_development": public(raw_development),
        "medium_winner": public(medium_winner),
        "low_winner": public(low_winner),
        "level4_hattrick3": baseline_summary["final"],
        "raw_level4_scale1": public(raw_final),
        "level4_hattrick_neo3": public(selected_final),
        "histories": histories,
    }
    THIS_DIR.mkdir(parents=True, exist_ok=True)
    runtime.write_json(THIS_DIR / "strict_esm_3x_level4.json", payload)
    runtime.write_csv(THIS_DIR / "strict_esm_3x_hattrick3_rows.csv", baseline_rows["final"])
    runtime.write_csv(THIS_DIR / "strict_esm_3x_neo3_rows.csv", selected_final["rows"])
    runtime.write_csv(THIS_DIR / "strict_esm_3x_neo3_native_rows.csv", raw_final["rows"])
    torch.save(
        {
            "state_dict": trained_winner["state"],
            "feature_count": int(features["train"].shape[-1]),
            "edge_count": int(train_cache.capacities.shape[1]),
            "medium_scale": medium_winner["medium_scale"],
            "low_scale": low_winner["low_scale"],
            "information_contract": information_contract,
        },
        THIS_DIR / "hattrick_neo3_edge_toll.pt",
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
