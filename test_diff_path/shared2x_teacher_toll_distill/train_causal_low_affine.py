from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import replace
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


trainer = load_module("causal_low_trainer", "train_linear_toll.py")
large = load_module("causal_low_large", "run_large_linear_toll.py")
actual = load_module("causal_low_actual", "finetune_actual_mean.py")
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


def medium_policy(cache, medium_state: dict) -> torch.Tensor:
    features = trainer.edge_features(cache)
    tolls = trainer.predict_tolls(medium_state, features)
    tolls = torch.stack([tolls[:, 0], torch.zeros_like(tolls[:, 1])], dim=1)
    routed = trainer.route_with_tolls(cache, tolls, 1.0)
    path_count = int(cache.policies[0].shape[1])
    return routed.path_features[:, :path_count].detach()


def post_medium_features(cache, medium: torch.Tensor) -> torch.Tensor:
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    capacities = cache.capacities.to(dtype=torch.float32).clamp_min(1e-9)
    policies = [
        cache.policies[0].squeeze(-1).to(dtype=torch.float32),
        medium.to(dtype=torch.float32),
        cache.policies[2].squeeze(-1).to(dtype=torch.float32),
    ]
    loads = []
    for policy, demand in zip(policies, cache.predicted_tms):
        path_flow = policy * demand.squeeze(-1).to(dtype=torch.float32)
        loads.append(torch.sparse.mm(pte.t(), path_flow.t()).t() / capacities)
    high, mid, low = loads
    return torch.stack(
        [
            high,
            mid,
            low,
            high + mid,
            high + mid + low,
            torch.relu(1.0 - high),
            torch.relu(1.0 - high - mid),
        ],
        dim=-1,
    ).detach()


def reparameterize_low(state: dict, new_features: torch.Tensor) -> dict:
    new_mean = new_features.mean(dim=0)
    new_std = new_features.std(dim=0).clamp_min(1e-4)
    old_mean = state["feature_mean"]
    old_std = state["feature_std"]
    old = state["coefficients"][1]
    raw_weight = old[:, 1:] / old_std
    raw_bias = old[:, 0] - (raw_weight * old_mean).sum(dim=1)
    coefficients = torch.cat(
        [
            (raw_bias + (raw_weight * new_mean).sum(dim=1)).unsqueeze(1),
            raw_weight * new_std,
        ],
        dim=1,
    )
    return {
        "feature_mean": new_mean,
        "feature_std": new_std,
        "coefficients": coefficients,
    }


def predict_low(state: dict, features: torch.Tensor) -> torch.Tensor:
    normalized = (features - state["feature_mean"]) / state["feature_std"]
    design = torch.cat(
        [torch.ones_like(normalized[..., :1]), normalized], dim=-1
    )
    return torch.einsum("nef,ef->ne", design, state["coefficients"])


def route_low(cache, indices: torch.Tensor, toll: torch.Tensor) -> torch.Tensor:
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    base = cache.policies[2].index_select(0, indices).squeeze(-1)
    path_cost = torch.sparse.mm(pte, toll.transpose(0, 1)).transpose(0, 1)
    grouped_base = base.reshape(indices.numel(), -1, K)
    grouped_cost = path_cost.reshape(indices.numel(), -1, K)
    valid = grouped_base > 0.0
    logits = torch.log(grouped_base.clamp_min(1e-12)) - grouped_cost
    logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
    policy = torch.softmax(logits, dim=-1) * grouped_base.sum(
        dim=-1, keepdim=True
    )
    return torch.where(valid, policy, torch.zeros_like(policy)).reshape_as(base)


def train_low(
    model,
    props,
    cache,
    medium: torch.Tensor,
    features: torch.Tensor,
    initial: dict,
    epochs: int,
):
    coefficients = torch.nn.Parameter(initial["coefficients"].detach().clone())
    initial_coefficients = initial["coefficients"].detach().clone()
    optimizer = torch.optim.Adam([coefficients], lr=0.001)
    generator = torch.Generator(device="cpu").manual_seed(20260823)
    checkpoints = {}
    history = []
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(cache), generator=generator)
        total = 0.0
        for start in range(0, len(cache), 32):
            indices = permutation[start : start + 32].to(props.device)
            state = dict(initial)
            state["coefficients"] = coefficients
            toll = predict_low(state, features.index_select(0, indices))
            low = route_low(cache, indices, toll)
            normalized = actual.actual_normalized(
                model,
                props,
                cache,
                indices,
                [medium.index_select(0, indices).unsqueeze(-1), low.unsqueeze(-1)],
            )
            anchor = (coefficients - initial_coefficients).square().mean()
            loss = -normalized[:, 2].mean() + 0.01 * anchor
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([coefficients], 2.0)
            optimizer.step()
            total += float(loss.item()) * int(indices.numel())
        history.append({"epoch": epoch, "loss": total / len(cache)})
        if epoch % 5 == 0:
            checkpoints[epoch] = coefficients.detach().clone()
    return checkpoints, history


def build_candidate(cache, medium, features, state: dict):
    with torch.no_grad():
        indices = torch.arange(len(cache), device=features.device)
        low = route_low(cache, indices, predict_low(state, features))
    return replace(cache, path_features=torch.cat([medium, low], dim=1))


def evaluate(model, props, cache, medium, features, state: dict):
    baseline_rows, baseline_summary = trainer.evaluate(model, props, cache)
    candidate = build_candidate(cache, medium, features, state)
    rows, summary = trainer.evaluate(model, props, cache, candidate)
    result = {
        "summary": probe.compact(summary),
        "baseline": probe.compact(baseline_summary),
        "delta": probe.gaps(summary, baseline_summary),
        "bootstrap": large.paired_bootstrap(baseline_rows, rows),
    }
    return result


def safe(result: dict) -> bool:
    delta = result["delta"]
    return (
        result["summary"]["High"]["norm_fulfill_mean"] >= 0.995
        and delta["Medium.norm_fulfill_mean"] > 0.0
        and delta["Medium.norm_fulfill_p1"] > 0.0
        and delta["Medium.norm_fulfill_p10"] > 0.0
        and delta["Low.norm_fulfill_mean"] >= -0.003
        and delta["Low.norm_fulfill_p1"] >= -0.01
        and delta["Low.norm_fulfill_p10"] >= -0.01
    )


def prepare(cache, medium_state):
    medium = medium_policy(cache, medium_state)
    features = post_medium_features(cache, medium)
    return medium, features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits = (
        {"train": (0, 128), "safety": (128, 160), "validation": (160, 200), "evaluation": (200, 250)}
        if args.level == 2
        else {"train": (0, 318), "safety": (318, 350), "validation": (350, 400), "evaluation": (400, 500)}
    )
    source = HERE / "artifacts" / f"actual_mean_level{args.level}" / "model.pt"
    props = runtime.build_props(args.level, device)
    model, checkpoint = runtime.load_backbone(args.level, 490, props, device)
    medium_state = state_on_device(
        torch.load(source, map_location=device, weights_only=False), device
    )
    caches = {
        name: runtime.build_policy_cache(model, props, *interval, batch_size=32)
        for name, interval in splits.items()
        if name != "evaluation"
    }
    prepared = {
        name: prepare(cache, medium_state) for name, cache in caches.items()
    }
    initial = reparameterize_low(medium_state, prepared["train"][1])
    max_epochs = 30 if args.level == 2 else int(
        json.loads(
            (HERE / "artifacts" / "causal_low_level2" / "report.json").read_text(encoding="utf-8")
        )["selected_epoch"]
    )
    checkpoints, history = train_low(
        model,
        props,
        caches["train"],
        prepared["train"][0],
        prepared["train"][1],
        initial,
        max_epochs,
    )

    safety_trials = []
    epochs = sorted(checkpoints) if args.level == 2 else [max_epochs]
    for epoch in epochs:
        state = dict(initial)
        state["coefficients"] = checkpoints[epoch]
        result = evaluate(
            model, props, caches["safety"], *prepared["safety"], state
        )
        safety_trials.append({"epoch": epoch, "result": result, "state": state})
        print(
            f"epoch={epoch} safe={safe(result)} "
            f"M={result['delta']['Medium.norm_fulfill_mean']:+.6f} "
            f"L={result['delta']['Low.norm_fulfill_mean']:+.6f}",
            flush=True,
        )
    eligible = [row for row in safety_trials if safe(row["result"])]
    selected = max(
        eligible,
        key=lambda row: min(
            row["result"]["delta"]["Low.norm_fulfill_mean"],
            row["result"]["delta"]["Low.norm_fulfill_p1"],
            row["result"]["delta"]["Low.norm_fulfill_p10"],
        ),
    ) if eligible else None
    validation = None
    evaluation = None
    if selected:
        validation = evaluate(
            model, props, caches["validation"], *prepared["validation"], selected["state"]
        )
    if validation is not None and safe(validation):
        final_cache = runtime.build_policy_cache(
            model, props, *splits["evaluation"], batch_size=32
        )
        final_prepared = prepare(final_cache, medium_state)
        evaluation = evaluate(
            model, props, final_cache, *final_prepared, selected["state"]
        )

    artifact_dir = HERE / "artifacts" / f"causal_low_level{args.level}"
    payload = {
        "method": "two-pass causal affine edge toll",
        "level": args.level,
        "architecture": "Medium affine toll, recompute strict-ESM residual capacity, Low affine toll",
        "parameters": 1152,
        "strict_esm_inference": True,
        "current_actual_tm_used_for_policy": False,
        "historical_actual_tm_used_for_training": True,
        "low_objective": "mean Low normalized fulfillment only",
        "no_tail_selection": True,
        "checkpoint": str(checkpoint),
        "source_medium_model": str(source),
        "splits": splits,
        "configuration": {"learning_rate": 0.001, "anchor_weight": 0.01},
        "history": history,
        "safety_trials": [
            {"epoch": row["epoch"], "result": row["result"]}
            for row in safety_trials
        ],
        "selected_epoch": selected["epoch"] if selected else None,
        "validation": validation,
        "evaluation": evaluation,
        "pass": bool(evaluation and safe(evaluation)),
        "seconds": time.perf_counter() - started,
    }
    write_json(artifact_dir / "report.json", payload)
    if selected:
        torch.save(
            {
                "medium_state": medium_state,
                "low_state": selected["state"],
                "checkpoint": str(checkpoint),
            },
            artifact_dir / "model.pt",
        )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
