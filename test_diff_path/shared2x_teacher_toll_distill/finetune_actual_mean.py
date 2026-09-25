from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer = load_module("actual_mean_trainer", "train_linear_toll.py")
large = load_module("actual_mean_large", "run_large_linear_toll.py")
probe = trainer.probe
runtime = trainer.runtime
K = 8


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def state_on_device(saved: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in saved["state"].items()
    }


def copy_state(state: dict) -> dict:
    return {
        key: value.detach().clone() if torch.is_tensor(value) else value
        for key, value in state.items()
    }


def pte_info(cache):
    pte = cache.dataset.pte.coalesce()
    indices = pte.indices()
    return pte, indices[0], indices[1], pte.values()


def route_batch(cache, indices: torch.Tensor, tolls: torch.Tensor):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    routed = []
    for class_index in range(2):
        base = cache.policies[class_index + 1].index_select(0, indices).squeeze(-1)
        path_cost = torch.sparse.mm(
            pte, tolls[:, class_index].transpose(0, 1)
        ).transpose(0, 1)
        grouped_base = base.reshape(indices.numel(), -1, K)
        grouped_cost = path_cost.reshape(indices.numel(), -1, K)
        valid = grouped_base > 0.0
        logits = torch.log(grouped_base.clamp_min(1e-12)) - grouped_cost
        logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
        policy = torch.softmax(logits, dim=-1) * grouped_base.sum(
            dim=-1, keepdim=True
        )
        routed.append(
            torch.where(valid, policy, torch.zeros_like(policy))
            .reshape_as(base)
            .unsqueeze(-1)
        )
    return routed


def actual_normalized(model, props, cache, indices, routed):
    selected = runtime.select_cache(cache, indices)
    policies = [
        cache.policies[0].index_select(0, indices),
        routed[0],
        routed[1],
    ]
    ratios = model.simulate(
        policies,
        list(selected["tms"]),
        selected["capacities"],
        pte_info(cache),
        int(indices.numel()),
        props,
        rate_cap=props.rate_cap,
    )[:3]
    admitted = [
        ratio.reshape(indices.numel(), -1) * tm.squeeze(-1)
        for ratio, tm in zip(ratios, selected["tms"])
    ]
    return torch.stack(
        [
            flow.sum(dim=1) / oracle.clamp_min(1e-9)
            for flow, oracle in zip(admitted, selected["oracle_flows"])
        ],
        dim=1,
    )


def train_one(
    model,
    props,
    train_cache,
    train_features,
    initial_state: dict,
    low_weight: float,
    seed: int,
    epochs: int,
    learning_rate: float,
    anchor_weight: float,
):
    torch.manual_seed(seed)
    coefficients = torch.nn.Parameter(initial_state["coefficients"].detach().clone())
    initial_coefficients = initial_state["coefficients"].detach().clone()
    optimizer = torch.optim.Adam([coefficients], lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = 32
    history = []
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(train_cache), generator=generator)
        total = 0.0
        for start in range(0, len(train_cache), batch_size):
            indices = permutation[start : start + batch_size].to(props.device)
            features = train_features.index_select(0, indices)
            state = dict(initial_state)
            state["coefficients"] = coefficients
            tolls = trainer.predict_tolls(state, features)
            routed = route_batch(train_cache, indices, tolls)
            normalized = actual_normalized(
                model, props, train_cache, indices, routed
            )
            utility = normalized[:, 1].mean() + float(low_weight) * normalized[:, 2].mean()
            anchor = (coefficients - initial_coefficients).square().mean()
            loss = -utility + float(anchor_weight) * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([coefficients], 2.0)
            optimizer.step()
            total += float(loss.item()) * int(indices.numel())
        history.append({"epoch": epoch, "loss": total / len(train_cache)})
    state = dict(initial_state)
    state["coefficients"] = coefficients.detach()
    return state, history


def evaluate(model, props, cache, state: dict):
    baseline_rows, baseline_summary = trainer.evaluate(model, props, cache)
    result = trainer.metrics(model, props, cache, state, 1.0, baseline_summary)
    result["baseline"] = probe.compact(baseline_summary)
    result["bootstrap"] = large.paired_bootstrap(baseline_rows, result["rows"])
    return result


def candidate_safe(result: dict) -> bool:
    delta = result["delta"]
    return (
        result["summary"]["High"]["norm_fulfill_mean"] >= 0.995
        and delta["Medium.norm_fulfill_mean"] > 0.0
        and delta["Medium.norm_fulfill_p1"] > 0.0
        and delta["Medium.norm_fulfill_p10"] > 0.0
        and delta["Low.norm_fulfill_mean"] >= -0.003
        and delta["Low.norm_fulfill_p10"] >= -0.01
    )


def score(result: dict) -> float:
    delta = result["delta"]
    return min(
        delta["Medium.norm_fulfill_mean"],
        delta["Medium.norm_fulfill_p1"],
        delta["Medium.norm_fulfill_p10"],
    )


def public(result: dict | None):
    if result is None:
        return None
    return {key: value for key, value in result.items() if key != "rows"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.level == 2:
        splits = {"train": (0, 128), "safety": (128, 160), "validation": (160, 200), "evaluation": (200, 250)}
        initial_path = HERE / "artifacts" / "linear_toll_small" / "model.pt"
        epochs = 30
        low_weights = (0.25, 0.5, 1.0, 2.0)
    else:
        splits = {"train": (0, 318), "safety": (318, 350), "validation": (350, 400), "evaluation": (400, 500)}
        initial_path = HERE / "artifacts" / "level4_linear_toll" / "model.pt"
        # Frozen from the Level-2 experiment: the large run performs no search.
        epochs = 30
        low_weights = (0.25,)

    props = runtime.build_props(args.level, device)
    model, checkpoint = runtime.load_backbone(args.level, 490, props, device)
    caches = {
        name: runtime.build_policy_cache(model, props, *interval, batch_size=32)
        for name, interval in splits.items()
        if name != "evaluation"
    }
    initial = state_on_device(
        torch.load(initial_path, map_location=device, weights_only=False), device
    )
    train_features = trainer.edge_features(caches["train"])

    trials = []
    for trial_index, low_weight in enumerate(low_weights):
        state, history = train_one(
            model,
            props,
            caches["train"],
            train_features,
            initial,
            low_weight=low_weight,
            seed=20260823 + trial_index,
            epochs=epochs,
            learning_rate=0.001,
            anchor_weight=0.01,
        )
        safety = evaluate(model, props, caches["safety"], state)
        trials.append(
            {
                "low_weight": low_weight,
                "history": history,
                "safety": safety,
                "state": copy_state(state),
            }
        )
        print(
            f"weight={low_weight:g} safe={candidate_safe(safety)} "
            f"M={safety['delta']['Medium.norm_fulfill_mean']:+.6f} "
            f"L={safety['delta']['Low.norm_fulfill_mean']:+.6f}",
            flush=True,
        )

    eligible = [trial for trial in trials if candidate_safe(trial["safety"])]
    selected = max(eligible, key=lambda trial: score(trial["safety"])) if eligible else None
    validation = None
    evaluation = None
    if selected is not None:
        validation = evaluate(model, props, caches["validation"], selected["state"])
    if validation is not None and candidate_safe(validation):
        evaluation_cache = runtime.build_policy_cache(
            model, props, *splits["evaluation"], batch_size=32
        )
        evaluation = evaluate(model, props, evaluation_cache, selected["state"])

    artifact_dir = HERE / "artifacts" / f"actual_mean_level{args.level}"
    payload = {
        "method": "teacher-initialized affine edge toll with historical-actual mean fine-tuning",
        "level": args.level,
        "strict_esm_inference": True,
        "current_actual_tm_used_for_policy": False,
        "historical_actual_tm_used_for_training": True,
        "objective": "mean Medium normalized fulfillment + low_weight * mean Low normalized fulfillment",
        "no_tail_selection": True,
        "checkpoint": str(checkpoint),
        "initial_model": str(initial_path),
        "splits": splits,
        "configuration": {
            "epochs": epochs,
            "learning_rate": 0.001,
            "anchor_weight": 0.01,
        },
        "trials": [
            {
                "low_weight": trial["low_weight"],
                "history": trial["history"],
                "safety": public(trial["safety"]),
            }
            for trial in trials
        ],
        "selected_low_weight": selected["low_weight"] if selected else None,
        "validation": public(validation),
        "evaluation": public(evaluation),
        "pass": bool(evaluation and candidate_safe(evaluation)),
        "seconds": time.perf_counter() - started,
    }
    write_json(artifact_dir / "report.json", payload)
    if selected is not None:
        torch.save(
            {"state": copy_state(selected["state"]), "checkpoint": str(checkpoint)},
            artifact_dir / "model.pt",
        )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
