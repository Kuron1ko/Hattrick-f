from __future__ import annotations

"""Train a prediction-only Low rerouting guard for the persistent Level-2 model.

High and Medium tensors are never replaced.  The guard acts only on the final
Low policy, after the formal serial High -> High+Medium stages, and preserves
each Low OD's total split mass.  Real traffic is used solely by the offline
admission loss/evaluator after prediction-only policies have been cached.
"""

import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
MODEL_RUNNER = THIS_DIR / "run_experiment.py"
EDGE_RUNNER = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
STRICT_EVALUATOR = (
    TEST_DIR
    / "shared2x_sparse_path_cross_attention"
    / "evaluate_strict_esm_sequential.py"
)
MODEL_CHECKPOINT = (
    THIS_DIR / "artifacts" / "level2_proxy" / "seed_490" / "final_model.pt"
)
OUTPUT_DIR = THIS_DIR / "artifacts" / "level2_low_guard"

EXPECTED_HASHES = {
    MODEL_RUNNER: "c4061d15495d4d5ecfdf4e3ca7dd77fa0a64a732bfb0c1c1d0ceefb61a1df14f",
    MODEL_CHECKPOINT: "4ad40dca89d14c86d954e7a90cc75c8a47da77f56c4a1c10c19a22e279032aa5",
    EDGE_RUNNER: "839c6a8b2a091553fe817abedc97016b35e498980a4eb2d8f936cb5cbae4c866",
    STRICT_EVALUATOR: "4afdbd4fb2ccc4d8b6bd77297cf0dc24485f34b922719283f31121f1f741f98b",
}
K = 8


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs() -> dict[str, str]:
    result = {}
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen input changed: {path}: {actual} != {expected}")
        result[str(path.resolve())] = actual
    return result


class LowGuard(nn.Module):
    def __init__(self, edge_count: int, feature_count: int, max_toll: float):
        super().__init__()
        self.edge_count = int(edge_count)
        self.feature_count = int(feature_count)
        self.max_toll = float(max_toll)
        self.edge_embedding = nn.Parameter(torch.zeros(edge_count, 6))
        self.net = nn.Sequential(
            nn.Linear(feature_count + 6, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        embedding = self.edge_embedding.unsqueeze(0).expand(len(features), -1, -1)
        values = self.net(torch.cat((features, embedding), dim=-1)).squeeze(-1)
        return self.max_toll * torch.tanh(values)


class LowOnlyAdapter:
    def adapt_batch(self, policies, batch):
        policies[2] = batch["path_features"].unsqueeze(-1)
        return policies


def route_low(base: torch.Tensor, edge_toll: torch.Tensor, pte) -> torch.Tensor:
    path_cost = torch.sparse.mm(pte, edge_toll.transpose(0, 1)).transpose(0, 1)
    grouped_base = base.reshape(len(base), -1, K)
    grouped_cost = path_cost.reshape(len(base), -1, K)
    valid = grouped_base > 0
    logits = torch.log(grouped_base.clamp_min(1e-12)) - grouped_cost
    logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
    routed = torch.softmax(logits, dim=-1) * grouped_base.sum(dim=-1, keepdim=True)
    return torch.where(valid, routed, torch.zeros_like(routed)).reshape_as(base)


def build_candidate_cache(head, cache, features, batch_size=32):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    chunks = []
    tolls = []
    with torch.no_grad():
        for start in range(0, len(cache), batch_size):
            stop = min(start + batch_size, len(cache))
            current = head(features[start:stop])
            low = route_low(
                cache.policies[2][start:stop].squeeze(-1), current, pte
            )
            chunks.append(low)
            tolls.append(current)
    return (
        replace(cache, path_features=torch.cat(chunks, dim=0)),
        torch.cat(tolls, dim=0),
    )


def evaluate(runtime, edge, model, props, head, cache, features, baseline_rows):
    candidate_cache, tolls = build_candidate_cache(head, cache, features)
    probe_count = min(4, len(cache))
    probe_indices = torch.arange(probe_count, device=props.device)
    probe = runtime.select_cache(candidate_cache, probe_indices)
    before = list(probe["policies"])
    after = LowOnlyAdapter().adapt_batch(list(before), probe)
    policy_preservation = {
        "High": bool(torch.equal(before[0], after[0])),
        "Medium": bool(torch.equal(before[1], after[1])),
    }
    if not all(policy_preservation.values()):
        raise RuntimeError(f"Low guard changed an upstream policy: {policy_preservation}")
    rows, summary = runtime.evaluate_cache(
        model, props, candidate_cache, LowOnlyAdapter(), batch_size=32
    )
    baseline_by_key = {
        (int(row["snapshot"]), row["class"]): row for row in baseline_rows
    }
    numeric = (
        "admitted_traffic",
        "demand",
        "fulfill_ratio",
        "oracle_admitted_traffic",
        "norm_fulfill",
        "raw_mlu",
        "oracle_mlu",
        "normalized_mlu",
        "disabled_flow",
        "admitted_capacity_ratio",
    )
    preservation = {}
    for class_name in ("High", "Medium"):
        maximum = max(
            abs(
                float(row[field])
                - float(baseline_by_key[(int(row["snapshot"]), class_name)][field])
            )
            for row in rows
            if row["class"] == class_name
            for field in numeric
        )
        preservation[class_name] = maximum
        # CUDA sparse admission replay can differ by a few ulps even when the
        # exact same upstream tensor is reused.  Policy equality above is the
        # hard structural check; this bound covers only repeated simulation.
        if maximum > 1e-5:
            raise RuntimeError(
                f"Low guard changed {class_name} beyond replay tolerance: {maximum}"
            )
    return {
        "summary": runtime.summary_index(summary),
        "diagnostics": edge.diagnostics(baseline_rows, rows),
        "preservation_max_delta": preservation,
        "upstream_policy_torch_equal": policy_preservation,
        "admission_replay_tolerance": 1e-5,
        "toll_abs_mean": float(tolls.abs().mean().item()),
        "toll_abs_max": float(tolls.abs().max().item()),
        "rows": rows,
        "state": {
            key: value.detach().cpu().clone() for key, value in head.state_dict().items()
        },
    }


def public(result: dict) -> dict:
    return {key: value for key, value in result.items() if key not in ("rows", "state")}


def rank(result: dict) -> tuple:
    low = result["diagnostics"]["Low"]
    return (
        int(low["ecdf_max_upward_violation"] <= 1e-12),
        -float(low["ecdf_max_upward_violation"]),
        float(low["p1_gain"]),
        float(low["p10_gain"]),
        float(low["mean_gain"]),
        -float(result["toll_abs_mean"]),
    )


def train_one(
    edge,
    model,
    props,
    train_cache,
    train_features,
    train_baseline,
    safety_cache,
    safety_features,
    safety_rows,
    max_toll,
    safety_weight,
    seed,
):
    torch.manual_seed(seed)
    head = LowGuard(
        int(train_cache.capacities.shape[1]),
        int(train_features.shape[-1]),
        max_toll,
    ).to(props.device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=2e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pte = train_cache.dataset.pte.coalesce().to(dtype=torch.float32)
    candidates = []
    history = []
    for epoch in range(1, 61):
        permutation = torch.randperm(len(train_cache), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(train_cache), 32):
            indices = permutation[start : start + 32].to(props.device)
            toll = head(train_features.index_select(0, indices))
            low = route_low(
                train_cache.policies[2]
                .index_select(0, indices)
                .squeeze(-1),
                toll,
                pte,
            )
            policies = [
                train_cache.policies[0].index_select(0, indices),
                train_cache.policies[1].index_select(0, indices),
                low.unsqueeze(-1),
            ]
            candidate = edge.normalized_fulfillment(
                model, props, train_cache, indices, policies
            )[:, 2]
            baseline = train_baseline.index_select(0, indices)[:, 2]
            gain = candidate - baseline
            shortfall = torch.relu(0.0002 - gain)
            hard_count = max(1, int(math.ceil(len(indices) * 0.30)))
            safety = shortfall.mean() + torch.topk(
                shortfall, hard_count
            ).values.mean()
            loss = (
                -gain.mean()
                + float(safety_weight) * safety
                + 3e-4 * toll.square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            optimizer.step()
            epoch_loss += float(loss.item()) * len(indices)
        if epoch == 1 or epoch % 10 == 0:
            result = evaluate(
                edge.runtime,
                edge,
                model,
                props,
                head,
                safety_cache,
                safety_features,
                safety_rows,
            )
            result.update(
                {
                    "epoch": epoch,
                    "max_toll": float(max_toll),
                    "safety_weight": float(safety_weight),
                }
            )
            candidates.append(result)
            history.append(
                {
                    "epoch": epoch,
                    "loss": epoch_loss / len(train_cache),
                    "low": result["diagnostics"]["Low"],
                }
            )
    return candidates, history


def main() -> None:
    frozen_hashes = verify_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_module = load_module("persistent_low_guard_model", MODEL_RUNNER)
    edge = load_module("persistent_low_guard_edge", EDGE_RUNNER)
    strict = load_module("persistent_low_guard_strict", STRICT_EVALUATOR)
    runtime = edge.runtime
    runtime.set_seed(20260824)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(2, device)
    payload = torch.load(MODEL_CHECKPOINT, map_location=device, weights_only=False)
    if int(payload.get("epoch", -1)) != 12:
        raise RuntimeError("Frozen persistent Level-2 checkpoint is not epoch 12")
    model = model_module.PersistentStage2SparseAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    load_result = model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    caches = {}
    information_audits = {}
    features = {}
    rows = {}
    summaries = {}
    for name, bounds, batch_size in (
        ("train", (0, 128), 32),
        ("safety", (128, 160), 32),
        ("validation", (160, 200), 20),
    ):
        cache, audit = strict.build_strict_policy_cache(
            runtime,
            model,
            props,
            *bounds,
            batch_size,
            actual_input_mode="zero",
        )
        caches[name] = cache
        information_audits[name] = audit
        features[name] = edge.edge_features(cache)
        rows[name], summaries[name] = runtime.evaluate_cache(
            model, props, cache, None, batch_size=32
        )

    train_baseline = edge.baseline_normalized(model, props, caches["train"])
    started = time.perf_counter()
    all_candidates = []
    histories = []
    settings = ((0.75, 10.0), (1.25, 30.0), (2.0, 60.0))
    for index, (max_toll, safety_weight) in enumerate(settings):
        candidates, history = train_one(
            edge,
            model,
            props,
            caches["train"],
            features["train"],
            train_baseline,
            caches["safety"],
            features["safety"],
            rows["safety"],
            max_toll,
            safety_weight,
            20261001 + index,
        )
        all_candidates.extend(candidates)
        histories.append(
            {
                "max_toll": max_toll,
                "safety_weight": safety_weight,
                "history": history,
            }
        )

    winner = max(all_candidates, key=rank)
    head = LowGuard(
        int(caches["train"].capacities.shape[1]),
        int(features["train"].shape[-1]),
        winner["max_toll"],
    ).to(device)
    head.load_state_dict(winner["state"], strict=True)
    head.eval()
    validation = evaluate(
        runtime,
        edge,
        model,
        props,
        head,
        caches["validation"],
        features["validation"],
        rows["validation"],
    )
    low_validation = validation["diagnostics"]["Low"]
    validation_passed = (
        low_validation["mean_gain"] >= 0.0
        and low_validation["p1_gain"] >= 0.0
        and low_validation["p10_gain"] >= 0.0
    )

    checkpoint_path = OUTPUT_DIR / "best_low_guard.pt"
    torch.save(
        {
            "state_dict": winner["state"],
            "edge_count": int(caches["train"].capacities.shape[1]),
            "feature_count": int(features["train"].shape[-1]),
            "max_toll": winner["max_toll"],
            "selection": public(winner),
            "backbone_sha256": EXPECTED_HASHES[MODEL_CHECKPOINT],
            "test_data_read": False,
        },
        checkpoint_path,
    )

    evaluation = None
    if validation_passed:
        evaluation_cache, evaluation_audit = strict.build_strict_policy_cache(
            runtime,
            model,
            props,
            200,
            250,
            25,
            actual_input_mode="zero",
        )
        evaluation_features = edge.edge_features(evaluation_cache)
        evaluation_rows, evaluation_summary = runtime.evaluate_cache(
            model, props, evaluation_cache, None, batch_size=25
        )
        evaluation = evaluate(
            runtime,
            edge,
            model,
            props,
            head,
            evaluation_cache,
            evaluation_features,
            evaluation_rows,
        )
        evaluation["information_audit"] = evaluation_audit
        runtime.write_csv(OUTPUT_DIR / "evaluation_baseline_rows.csv", evaluation_rows)
        runtime.write_csv(OUTPUT_DIR / "evaluation_candidate_rows.csv", evaluation["rows"])

    report = {
        "method": "prediction-only Low guard after persistent formal cascade",
        "status": "level2 validation passed" if validation_passed else "rejected",
        "protocol": {
            "train": [0, 128],
            "safety_selection": [128, 160],
            "independent_validation": [160, 200],
            "evaluation_if_validation_passes": [200, 250],
            "settings": [list(value) for value in settings],
            "actual_tm_policy_input": False,
            "actual_tm_offline_training_loss": True,
            "high_medium_policy_modified": False,
        },
        "frozen_hashes": frozen_hashes,
        "information_audits": information_audits,
        "state_load": {
            "missing": list(load_result.missing_keys),
            "unexpected": list(load_result.unexpected_keys),
        },
        "safety_baseline": runtime.summary_index(summaries["safety"]),
        "safety_candidates": [public(value) for value in all_candidates],
        "safety_winner": public(winner),
        "validation_baseline": runtime.summary_index(summaries["validation"]),
        "validation": public(validation),
        "validation_passed_before_evaluation": validation_passed,
        "evaluation": None if evaluation is None else public(evaluation),
        "histories": histories,
        "runtime": {"device": str(device), "seconds": time.perf_counter() - started},
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
        },
    }
    runtime.write_json(OUTPUT_DIR / "report.json", report)
    runtime.write_csv(OUTPUT_DIR / "validation_baseline_rows.csv", rows["validation"])
    runtime.write_csv(OUTPUT_DIR / "validation_candidate_rows.csv", validation["rows"])
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
