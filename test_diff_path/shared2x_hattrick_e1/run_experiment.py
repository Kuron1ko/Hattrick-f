from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
EDGE_DIR = TEST_DIR / "shared2x_edge_toll_head"
EDGE_PATH = EDGE_DIR / "run_experiment.py"
EDGE_CHECKPOINT = EDGE_DIR / "best_edge_toll_head.pt"

for item in (str(ROOT), str(TEST_DIR), str(EDGE_DIR), str(THIS_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


edge = load_module("hattrick_e1_edge_runtime", EDGE_PATH)
runtime = edge.runtime

K = edge.K
TRAIN = (0, 318)
SAFETY = (318, 350)
VALIDATION = (350, 400)
DEVELOPMENT = (318, 400)
FINAL = (400, 500)


class ThreePolicyAdapter:
    def __init__(self, path_count: int):
        self.path_count = int(path_count)

    def adapt_batch(self, policies, batch):
        features = batch["path_features"]
        for class_index in range(3):
            start = class_index * self.path_count
            stop = start + self.path_count
            policies[class_index] = features[:, start:stop].unsqueeze(-1)
        return policies


def route_from_toll(base, toll, pte):
    batch = int(base.shape[0])
    path_cost = torch.sparse.mm(pte, toll.transpose(0, 1)).transpose(0, 1)
    grouped_base = base.reshape(batch, -1, K)
    grouped_cost = path_cost.reshape(batch, -1, K)
    valid = grouped_base > 0
    logits = torch.log(grouped_base.clamp_min(1e-12)) - grouped_cost
    logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
    routed = torch.softmax(logits, dim=-1) * grouped_base.sum(dim=-1, keepdim=True)
    return torch.where(valid, routed, torch.zeros_like(routed)).reshape_as(base)


class HighEdgeTollHead(nn.Module):
    """Prediction-only High recourse with an explicit zero-initialized trust gate."""

    def __init__(
        self,
        edge_count: int,
        feature_count: int,
        max_toll: float,
        load_radius: float,
    ):
        super().__init__()
        self.edge_count = int(edge_count)
        self.feature_count = int(feature_count)
        self.max_toll = float(max_toll)
        self.load_radius = float(load_radius)
        self.edge_embedding = nn.Parameter(torch.zeros(edge_count, 6))
        self.local = nn.Sequential(
            nn.Linear(feature_count + 6, 24),
            nn.SiLU(),
            nn.Linear(24, 12),
            nn.SiLU(),
            nn.Linear(12, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(feature_count * 2, 12),
            nn.SiLU(),
            nn.Linear(12, 1),
        )
        nn.init.zeros_(self.local[-1].weight)
        nn.init.zeros_(self.local[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, features, base_high, pte, scale: float = 1.0):
        batch = int(features.shape[0])
        embedding = self.edge_embedding.unsqueeze(0).expand(batch, -1, -1)
        raw = self.local(torch.cat([features, embedding], dim=-1)).squeeze(-1)
        pooled = torch.cat(
            [features.mean(dim=1), features.amax(dim=1)], dim=-1
        )
        gate = torch.sigmoid(self.gate(pooled)).squeeze(-1)
        toll = self.max_toll * torch.tanh(raw) * gate.unsqueeze(1)
        routed = route_from_toll(base_high, toll * float(scale), pte)
        return routed, toll, gate


def selected(values, indices):
    return value_select(values, indices)


def value_select(values, indices):
    return values.index_select(0, indices)


def predicted_loads(cache, indices, policies):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    capacities = cache.capacities.index_select(0, indices).to(
        dtype=torch.float32
    ).clamp_min(1e-9)
    loads = []
    for class_index, policy in enumerate(policies):
        demand = cache.predicted_tms[class_index].index_select(0, indices)
        flow = policy.squeeze(-1).to(dtype=torch.float32) * demand.squeeze(-1).to(
            dtype=torch.float32
        )
        loads.append(torch.sparse.mm(pte.t(), flow.t()).t() / capacities)
    return loads


def predicted_high_load(cache, indices, high_policy):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    capacities = cache.capacities.index_select(0, indices).to(
        dtype=torch.float32
    ).clamp_min(1e-9)
    demand = cache.predicted_tms[0].index_select(0, indices)
    flow = high_policy.squeeze(-1).to(dtype=torch.float32) * demand.squeeze(-1).to(
        dtype=torch.float32
    )
    return torch.sparse.mm(pte.t(), flow.t()).t() / capacities


def features_from_loads(loads):
    high, medium, low = loads
    return torch.stack(
        [
            high,
            medium,
            low,
            high + medium,
            high + medium + low,
            torch.relu(1.0 - high),
            torch.relu(1.0 - high - medium),
        ],
        dim=-1,
    )


def e_policies(e_head, cache, indices, features):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    high = cache.policies[0].index_select(0, indices)
    base_ml = [
        cache.policies[class_index].index_select(0, indices).squeeze(-1)
        for class_index in (1, 2)
    ]
    routed, _, _ = e_head(features.index_select(0, indices), base_ml, pte)
    return [high, routed[0].unsqueeze(-1), routed[1].unsqueeze(-1)]


def e1_policies(high_head, e_head, cache, indices, base_features, scale: float):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    base = [cache.policies[c].index_select(0, indices) for c in range(3)]
    raw_high, high_toll, high_gate = high_head(
        base_features.index_select(0, indices),
        base[0].squeeze(-1),
        pte,
        scale=1.0,
    )
    base_high_load = predicted_high_load(cache, indices, base[0])
    raw_high_load = predicted_high_load(cache, indices, raw_high)
    max_load_change = (raw_high_load - base_high_load).abs().amax(
        dim=1, keepdim=True
    )
    # ``scale`` explores progressively larger moves inside the same learned
    # direction.  Never extrapolate beyond the head's raw proposal.
    trust_alpha = torch.clamp(
        float(scale)
        * torch.clamp(
            high_head.load_radius / max_load_change.clamp_min(1e-9), max=1.0
        ),
        max=1.0,
    )
    base_high = base[0].squeeze(-1)
    high = base_high + trust_alpha * (raw_high - base_high)
    provisional = [high.unsqueeze(-1), base[1], base[2]]
    causal_features = features_from_loads(
        predicted_loads(cache, indices, provisional)
    )
    base_ml = [base[1].squeeze(-1), base[2].squeeze(-1)]
    routed_ml, _, _ = e_head(causal_features, base_ml, pte)
    policies = [
        high.unsqueeze(-1),
        routed_ml[0].unsqueeze(-1),
        routed_ml[1].unsqueeze(-1),
    ]
    return policies, high_toll, high_gate, causal_features


def reference_normalized(model, props, e_head, cache, features, batch_size=32):
    chunks = []
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            indices = torch.arange(
                start, min(start + batch_size, len(cache)), device=props.device
            )
            policies = e_policies(e_head, cache, indices, features)
            chunks.append(
                edge.normalized_fulfillment(model, props, cache, indices, policies)
            )
    return torch.cat(chunks, dim=0)


def build_policy_cache(
    high_head, e_head, cache, features, scale: float, batch_size=32
):
    all_policies = [[], [], []]
    gates = []
    tolls = []
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            indices = torch.arange(
                start, min(start + batch_size, len(cache)), device=features.device
            )
            policies, toll, gate, _ = e1_policies(
                high_head, e_head, cache, indices, features, scale
            )
            for class_index in range(3):
                all_policies[class_index].append(policies[class_index].squeeze(-1))
            tolls.append(toll)
            gates.append(gate)
    joined = [torch.cat(parts, dim=0) for parts in all_policies]
    return (
        replace(cache, path_features=torch.cat(joined, dim=1)),
        torch.cat(tolls, dim=0),
        torch.cat(gates, dim=0),
        joined,
    )


def build_e_cache(e_head, cache, features, batch_size=32):
    all_policies = [[], [], []]
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            indices = torch.arange(
                start, min(start + batch_size, len(cache)), device=features.device
            )
            policies = e_policies(e_head, cache, indices, features)
            for class_index in range(3):
                all_policies[class_index].append(policies[class_index].squeeze(-1))
    joined = [torch.cat(parts, dim=0) for parts in all_policies]
    return replace(cache, path_features=torch.cat(joined, dim=1)), joined


def evaluate_policy_cache(model, props, cache):
    adapter = ThreePolicyAdapter(int(cache.policies[0].shape[1]))
    return runtime.evaluate_cache(model, props, cache, adapter, batch_size=32)


def evaluate_reference(model, props, e_head, cache, features):
    policy_cache, policies = build_e_cache(e_head, cache, features)
    rows, summary = evaluate_policy_cache(model, props, policy_cache)
    return rows, runtime.summary_index(summary), policies


def evaluate_candidate(
    model,
    props,
    high_head,
    e_head,
    cache,
    features,
    reference_rows,
    scale: float,
):
    policy_cache, tolls, gates, policies = build_policy_cache(
        high_head, e_head, cache, features, scale
    )
    rows, summary = evaluate_policy_cache(model, props, policy_cache)
    path_delta = float(
        (policies[0] - cache.policies[0].squeeze(-1)).abs().mean().item()
    )
    return {
        "scale": float(scale),
        "summary": runtime.summary_index(summary),
        "diagnostics_vs_e": edge.diagnostics(reference_rows, rows),
        "high_path_mean_abs_delta": path_delta,
        "high_toll_abs_mean": float(tolls.abs().mean().item()),
        "high_gate_mean": float(gates.mean().item()),
        "rows": rows,
        "state": {
            key: value.detach().cpu().clone()
            for key, value in high_head.state_dict().items()
        },
    }


def public(result):
    return {key: value for key, value in result.items() if key not in ("rows", "state")}


def top_fraction(values, fraction=0.25):
    count = max(1, int(math.ceil(float(values.numel()) * float(fraction))))
    return torch.topk(values, count).values.mean()


def policy_kl(candidate, baseline):
    batch = int(candidate.shape[0])
    q = candidate.reshape(batch, -1, K).clamp_min(1e-12)
    p = baseline.reshape(batch, -1, K).clamp_min(1e-12)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return (q * (torch.log(q) - torch.log(p))).sum(dim=-1).mean()


def train_head(
    model,
    props,
    e_head,
    cache,
    features,
    reference_norm,
    config,
    epochs,
    seed,
):
    torch.manual_seed(seed)
    head = HighEdgeTollHead(
        int(cache.capacities.shape[1]),
        int(features.shape[-1]),
        float(config["max_toll"]),
        float(config["load_radius"]),
    ).to(props.device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(config["lr"]), weight_decay=1e-4
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch_size = int(config.get("batch_size", 32))
    history = []

    for epoch in range(1, int(epochs) + 1):
        permutation = torch.randperm(len(cache), generator=generator)
        epoch_loss = 0.0
        epoch_high = 0.0
        epoch_medium = 0.0
        for start in range(0, len(cache), batch_size):
            indices = permutation[start : start + batch_size].to(props.device)
            policies, toll, gate, causal_features = e1_policies(
                head, e_head, cache, indices, features, 1.0
            )
            candidate = edge.normalized_fulfillment(
                model, props, cache, indices, policies
            )
            reference = reference_norm.index_select(0, indices)
            gain = candidate - reference

            high_violation = top_fraction(
                torch.relu(-float(config["high_tolerance"]) - gain[:, 0])
            )
            medium_violation = top_fraction(
                torch.relu(-float(config["medium_tolerance"]) - gain[:, 1])
            )
            low_violation = top_fraction(
                torch.relu(-float(config["low_tolerance"]) - gain[:, 2])
            )
            safety = (
                high_violation
                + float(config["medium_safety_ratio"]) * medium_violation
                + float(config["low_safety_ratio"]) * low_violation
            )

            loads = predicted_loads(cache, indices, policies)
            base_policies = [
                cache.policies[c].index_select(0, indices) for c in range(3)
            ]
            base_loads = predicted_loads(cache, indices, base_policies)
            contention = (
                loads[0] * (loads[1] + float(config["low_contention"]) * loads[2])
            ).mean()
            base_contention = (
                base_loads[0]
                * (
                    base_loads[1]
                    + float(config["low_contention"]) * base_loads[2]
                )
            ).mean()

            utility = (
                float(config["high_reward"]) * gain[:, 0]
                + gain[:, 1]
                + float(config["low_reward"]) * gain[:, 2]
            ).mean()
            trust = policy_kl(
                policies[0].squeeze(-1), base_policies[0].squeeze(-1)
            )
            # The trust region and KL term already bound the High move.  A
            # direct penalty on the gate collapses it toward zero and makes
            # e1 indistinguishable from e, so regularize only toll magnitude.
            regularizer = toll.square().mean()
            loss = (
                -utility
                + float(config["safety_weight"]) * safety
                + float(config["contention_weight"]) * (contention - base_contention)
                + float(config["trust_weight"]) * trust
                + 2e-4 * regularizer
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * int(indices.numel())
            epoch_high += float(gain[:, 0].mean().item()) * int(indices.numel())
            epoch_medium += float(gain[:, 1].mean().item()) * int(indices.numel())

        if epoch == 1 or epoch % 10 == 0 or epoch == int(epochs):
            history.append(
                {
                    "epoch": epoch,
                    "loss": epoch_loss / len(cache),
                    "train_high_gain": epoch_high / len(cache),
                    "train_medium_gain": epoch_medium / len(cache),
                }
            )
    return head, history


def summary_class(summary, name):
    return summary[name]


def development_feasible(result, reference_summary):
    summary = result["summary"]
    high = summary["High"]
    medium = summary["Medium"]
    low = summary["Low"]
    ref_h = reference_summary["High"]
    ref_m = reference_summary["Medium"]
    ref_l = reference_summary["Low"]
    return (
        high["norm_fulfill_mean"] >= max(0.995, ref_h["norm_fulfill_mean"] - 5e-4)
        and high["norm_fulfill_p1"] >= ref_h["norm_fulfill_p1"] - 5e-4
        and high["norm_fulfill_p10"] >= ref_h["norm_fulfill_p10"] - 5e-4
        and medium["norm_fulfill_mean"] >= ref_m["norm_fulfill_mean"] - 2e-4
        and medium["norm_fulfill_p1"] >= ref_m["norm_fulfill_p1"] - 0.001
        and medium["norm_fulfill_p10"] >= ref_m["norm_fulfill_p10"] - 0.001
        and low["norm_fulfill_mean"] >= ref_l["norm_fulfill_mean"] - 0.003
        and low["norm_fulfill_p1"] >= ref_l["norm_fulfill_p1"] - 0.005
        and low["norm_fulfill_p10"] >= ref_l["norm_fulfill_p10"] - 0.010
        # Mean OD-level L1 = K times the reported mean per-path change.
        and result["high_path_mean_abs_delta"] * K <= 0.015
    )


def development_rank(result, reference_summary):
    diag = result["diagnostics_vs_e"]
    return (
        int(float(result["scale"]) > 0 and development_feasible(result, reference_summary)),
        diag["High"]["mean_gain"],
        diag["High"]["p10_gain"],
        diag["Medium"]["mean_gain"],
        diag["Medium"]["p10_gain"],
        diag["Low"]["mean_gain"],
        -diag["Low"]["ecdf_max_upward_violation"],
    )


def config_grid():
    common = {
        "lr": 0.002,
        "max_toll": 0.75,
        "load_radius": 0.02,
        "high_reward": 0.5,
        "low_reward": 0.20,
        "high_tolerance": 0.0001,
        "medium_tolerance": 0.0005,
        "low_tolerance": 0.002,
        "medium_safety_ratio": 0.35,
        "low_safety_ratio": 0.20,
        "low_contention": 0.20,
        "trust_weight": 0.01,
        "batch_size": 32,
    }
    return [
        dict(common, name="balanced", safety_weight=50.0, contention_weight=0.01),
        dict(common, name="high_safe", safety_weight=150.0, contention_weight=0.01),
        dict(common, name="contention", safety_weight=100.0, contention_weight=0.04),
        dict(
            common,
            name="high_focus",
            max_toll=1.25,
            load_radius=0.04,
            high_reward=5.0,
            trust_weight=0.001,
            safety_weight=150.0,
            contention_weight=0.01,
        ),
        dict(
            common,
            name="high_tail_safe",
            max_toll=1.25,
            load_radius=0.03,
            high_reward=3.0,
            trust_weight=0.001,
            safety_weight=300.0,
            contention_weight=0.01,
        ),
    ]


def load_frozen(device):
    runtime.set_seed(20260824)
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    saved = torch.load(EDGE_CHECKPOINT, map_location=device, weights_only=False)
    e_head = edge.EdgeTollHead(saved["edge_count"], saved["feature_count"]).to(device)
    e_head.load_state_dict(saved["state_dict"])
    e_head.eval()
    for parameter in e_head.parameters():
        parameter.requires_grad_(False)
    return model, props, checkpoint, e_head


def build_cache_bundle(model, props, ranges):
    bundle = {}
    for name, bounds in ranges.items():
        cache = runtime.build_policy_cache(model, props, *bounds, batch_size=64)
        bundle[name] = (cache, edge.edge_features(cache))
    return bundle


def run_screen(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props, checkpoint, e_head = load_frozen(device)
    bundle = build_cache_bundle(
        model,
        props,
        {
            "screen_train": (0, int(args.screen_train)),
            "safety": SAFETY,
            "validation": VALIDATION,
        },
    )
    train_cache, train_features = bundle["screen_train"]
    safety_cache, safety_features = bundle["safety"]
    validation_cache, validation_features = bundle["validation"]
    train_reference = reference_normalized(
        model, props, e_head, train_cache, train_features
    )
    safety_rows, safety_summary, _ = evaluate_reference(
        model, props, e_head, safety_cache, safety_features
    )
    validation_rows, validation_summary, _ = evaluate_reference(
        model, props, e_head, validation_cache, validation_features
    )

    started = time.perf_counter()
    results = []
    for index, config in enumerate(config_grid()):
        head, history = train_head(
            model,
            props,
            e_head,
            train_cache,
            train_features,
            train_reference,
            config,
            int(args.screen_epochs),
            20262401 + index,
        )
        safety_result = evaluate_candidate(
            model,
            props,
            head,
            e_head,
            safety_cache,
            safety_features,
            safety_rows,
            1.0,
        )
        validation_result = evaluate_candidate(
            model,
            props,
            head,
            e_head,
            validation_cache,
            validation_features,
            validation_rows,
            1.0,
        )
        results.append(
            {
                "config": config,
                "history": history,
                "safety": public(safety_result),
                "validation": public(validation_result),
                "combined_rank": (
                    int(development_feasible(safety_result, safety_summary))
                    + int(development_feasible(validation_result, validation_summary)),
                    validation_result["diagnostics_vs_e"]["High"]["mean_gain"],
                    validation_result["diagnostics_vs_e"]["High"]["p10_gain"],
                    validation_result["diagnostics_vs_e"]["Medium"]["mean_gain"],
                    validation_result["diagnostics_vs_e"]["Medium"]["p10_gain"],
                ),
            }
        )
        print(
            f"screen {config['name']}: "
            f"H={validation_result['diagnostics_vs_e']['High']['mean_gain']:+.6f} "
            f"M={validation_result['diagnostics_vs_e']['Medium']['mean_gain']:+.6f} "
            f"L={validation_result['diagnostics_vs_e']['Low']['mean_gain']:+.6f}",
            flush=True,
        )

    winner = max(results, key=lambda item: tuple(item["combined_rank"]))
    payload = {
        "method": "Hattrick-e1 screen: causal High edge-toll recourse",
        "checkpoint": str(checkpoint),
        "protocol": {
            "screen_train": [0, int(args.screen_train)],
            "safety": list(SAFETY),
            "validation": list(VALIDATION),
            "strict_esm_inference": True,
        },
        "seconds": time.perf_counter() - started,
        "reference": {
            "safety": safety_summary,
            "validation": validation_summary,
        },
        "candidates": results,
        "winner_config": winner["config"],
    }
    THIS_DIR.mkdir(parents=True, exist_ok=True)
    runtime.write_json(THIS_DIR / "screen.json", payload)
    print(json.dumps({"winner_config": winner["config"]}, ensure_ascii=False))


def run_final(args):
    screen_path = THIS_DIR / "screen.json"
    if not screen_path.exists():
        raise FileNotFoundError("Run --stage screen first")
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    config = screen["winner_config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props, checkpoint, e_head = load_frozen(device)
    bundle = build_cache_bundle(
        model,
        props,
        {"train": TRAIN, "development": DEVELOPMENT, "final": FINAL},
    )
    train_cache, train_features = bundle["train"]
    development_cache, development_features = bundle["development"]
    final_cache, final_features = bundle["final"]
    train_reference = reference_normalized(
        model, props, e_head, train_cache, train_features
    )
    development_rows, development_summary, _ = evaluate_reference(
        model, props, e_head, development_cache, development_features
    )
    final_reference_rows, final_reference_summary, final_reference_policies = (
        evaluate_reference(model, props, e_head, final_cache, final_features)
    )
    final_hattrick_rows, final_hattrick_summary = runtime.evaluate_cache(
        model, props, final_cache, None, batch_size=32
    )

    started = time.perf_counter()
    head, history = train_head(
        model,
        props,
        e_head,
        train_cache,
        train_features,
        train_reference,
        config,
        int(args.final_epochs),
        20262501,
    )
    scale_grid = (
        0.0,
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
        0.75,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
    )
    development_candidates = [
        evaluate_candidate(
            model,
            props,
            head,
            e_head,
            development_cache,
            development_features,
            development_rows,
            scale,
        )
        for scale in scale_grid
    ]
    positive_feasible = [
        item
        for item in development_candidates
        if item["scale"] > 0 and development_feasible(item, development_summary)
    ]
    if positive_feasible:
        winner = max(
            positive_feasible,
            key=lambda item: development_rank(item, development_summary),
        )
    else:
        winner = next(
            item for item in development_candidates if abs(item["scale"]) <= 1e-12
        )
    head.load_state_dict(winner["state"])
    final_result = evaluate_candidate(
        model,
        props,
        head,
        e_head,
        final_cache,
        final_features,
        final_reference_rows,
        winner["scale"],
    )

    # Strict information-flow audit: actual test TMs must not affect any policy.
    original_policy_cache, _, _, original_policies = build_policy_cache(
        head, e_head, final_cache, final_features, winner["scale"]
    )
    counterfactual = replace(
        final_cache, tms=tuple(torch.zeros_like(value) for value in final_cache.tms)
    )
    counterfactual_features = edge.edge_features(counterfactual)
    counterfactual_policy_cache, _, _, counterfactual_policies = build_policy_cache(
        head, e_head, counterfactual, counterfactual_features, winner["scale"]
    )
    policy_invariance = max(
        float((a - b).abs().max().item())
        for a, b in zip(original_policies, counterfactual_policies)
    )
    high_path_change = float(
        (
            original_policies[0]
            - final_cache.policies[0].squeeze(-1)
        ).abs().mean().item()
    )

    payload = {
        "method": "Hattrick-e1: causal High edge-toll recourse followed by Hattrick-e",
        "architecture": {
            "high": "learned ESM-conditioned edge toll with bounded residual reweighting",
            "medium_low": "frozen Hattrick-e edge-toll head recomputed after High",
            "causal_order": "High reroute -> predicted residual recomputation -> Medium/Low reroute",
        },
        "protocol": {
            "train": list(TRAIN),
            "development_selection": list(DEVELOPMENT),
            "level4": list(FINAL),
            "strict_esm_inference": True,
            "actual_tm": "offline loss and evaluation only",
            "level4_status": "follow-up confirmation; this thread previously inspected 400-499",
        },
        "checkpoint": str(checkpoint),
        "config": config,
        "selected_high_scale": winner["scale"],
        "elapsed_seconds": time.perf_counter() - started,
        "information_audit": {
            "actual_tm_counterfactual_policy_max_abs_diff": policy_invariance,
            "high_path_mean_abs_change": high_path_change,
        },
        "development_reference_hattrick_e": development_summary,
        "development_candidates": [public(item) for item in development_candidates],
        "development_winner": public(winner),
        "level4_hattrick": runtime.summary_index(final_hattrick_summary),
        "level4_hattrick_e": final_reference_summary,
        "level4_hattrick_e1": public(final_result),
        "diagnostics_e1_vs_hattrick": edge.diagnostics(
            final_hattrick_rows, final_result["rows"]
        ),
        "training_history": history,
    }
    runtime.write_json(THIS_DIR / "level4.json", payload)
    runtime.write_csv(THIS_DIR / "level4_hattrick_rows.csv", final_hattrick_rows)
    runtime.write_csv(THIS_DIR / "level4_hattrick_e_rows.csv", final_reference_rows)
    runtime.write_csv(THIS_DIR / "level4_hattrick_e1_rows.csv", final_result["rows"])
    torch.save(
        {
            "state_dict": head.state_dict(),
            "feature_count": int(train_features.shape[-1]),
            "edge_count": int(train_cache.capacities.shape[1]),
            "max_toll": float(config["max_toll"]),
            "load_radius": float(config["load_radius"]),
            "high_scale": float(winner["scale"]),
            "config": config,
        },
        THIS_DIR / "best_hattrick_e1_high_head.pt",
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen", "final"), required=True)
    parser.add_argument("--screen-train", type=int, default=96)
    parser.add_argument("--screen-epochs", type=int, default=30)
    parser.add_argument("--final-epochs", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.stage == "screen":
        run_screen(args)
    else:
        run_final(args)
