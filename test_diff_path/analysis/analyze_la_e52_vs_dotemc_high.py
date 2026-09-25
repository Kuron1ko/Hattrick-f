from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "test_diff_path"
OUTPUT = (
    ROOT
    / "output"
    / "comparisons"
    / "hattrick_la_e52_vs_dotemc"
    / "high_policy_similarity_2x_400_499.json"
)
LA_CHECKPOINT = (
    TEST_DIR
    / "shared2x_medium_pressure_lookahead"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
    / "epoch_checkpoints"
    / "epoch_052.pt"
)
DOTE_CHECKPOINT = (
    TEST_DIR
    / "results_load2x_retrain_shared_strict"
    / "outputs"
    / "shared"
    / "w_1_0p1_0p01"
    / "best_model.pt"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
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


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    return float((values * weights).sum().item() / weights.sum().item())


def weighted_quantile(values: torch.Tensor, weights: torch.Tensor, q: float) -> float:
    values = values.reshape(-1).detach().cpu().numpy().astype(np.float64)
    weights = weights.reshape(-1).detach().cpu().numpy().astype(np.float64)
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    target = float(q) * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(values) - 1)
    return float(values[index])


def distribution_stats(policy: torch.Tensor, weights: torch.Tensor) -> dict:
    safe = policy.clamp_min(1e-12)
    pmax = policy.max(dim=-1).values
    entropy = -(safe * safe.log()).sum(dim=-1)
    effective = entropy.exp()
    return {
        "unweighted": {
            "pmax_mean": float(pmax.mean().item()),
            "pmax_median": float(pmax.median().item()),
            "pmax_ge_0p9_fraction": float((pmax >= 0.9).float().mean().item()),
            "entropy_mean": float(entropy.mean().item()),
            "normalized_entropy_mean": float(entropy.mean().item() / math.log(policy.shape[-1])),
            "effective_paths_mean": float(effective.mean().item()),
            "effective_paths_median": float(effective.median().item()),
        },
        "actual_high_demand_weighted": {
            "pmax_mean": weighted_mean(pmax, weights),
            "pmax_median": weighted_quantile(pmax, weights, 0.5),
            "pmax_ge_0p9_fraction": weighted_mean((pmax >= 0.9).float(), weights),
            "entropy_mean": weighted_mean(entropy, weights),
            "normalized_entropy_mean": weighted_mean(entropy, weights) / math.log(policy.shape[-1]),
            "effective_paths_mean": weighted_mean(effective, weights),
            "effective_paths_median": weighted_quantile(effective, weights, 0.5),
        },
    }


def similarity_stats(la: torch.Tensor, dote: torch.Tensor, weights: torch.Tensor) -> dict:
    tv = 0.5 * (la - dote).abs().sum(dim=-1)
    same = (la.argmax(dim=-1) == dote.argmax(dim=-1)).float()

    def section(weighted: bool) -> dict:
        mean = (lambda x: weighted_mean(x, weights)) if weighted else (lambda x: float(x.mean().item()))
        median = (
            (lambda x: weighted_quantile(x, weights, 0.5))
            if weighted
            else (lambda x: float(x.median().item()))
        )
        return {
            "argmax_agreement": mean(same),
            "argmax_changed": 1.0 - mean(same),
            "tv_mean": mean(tv),
            "tv_median": median(tv),
            "tv_ge_0p25_fraction": mean((tv >= 0.25).float()),
            "tv_ge_0p50_fraction": mean((tv >= 0.50).float()),
            "tv_ge_0p75_fraction": mean((tv >= 0.75).float()),
        }

    return {
        "unweighted": section(False),
        "actual_high_demand_weighted": section(True),
    }


def extract_with_zero_actual_tm(runtime, model, props, dataset, batch_size: int) -> torch.Tensor:
    """Counterfactual audit: retain ESM predictions but zero every actual TM."""
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    path_masks = runtime.shared.base.move_dataset_static(dataset, props.device)
    loader = runtime.shared.data_loader(dataset, batch_size, False, 0)
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_policy = True
    chunks = []
    with torch.no_grad():
        for inputs in loader:
            values = list(runtime.shared.unpack_to_device(inputs, props))
            for index in (2, 4, 6):
                values[index] = torch.zeros_like(values[index])
            policies = runtime.cached_policy_forward(
                model, props, dataset, tuple(values), path_masks
            )
            chunks.append(policies[0].detach())
    props.research_return_policy = False
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    return torch.cat(chunks, dim=0)


def main() -> None:
    for path in (
        ROOT,
        TEST_DIR,
        TEST_DIR / "shared2x_medium_adapter",
        TEST_DIR / "shared2x_order_regularizer",
        TEST_DIR / "shared2x_full_objectives",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    runtime = load_module(
        "la_e52_policy_runtime",
        TEST_DIR / "shared2x_medium_adapter" / "run_experiment.py",
    )
    lookahead = load_module(
        "la_e52_model_runtime",
        TEST_DIR / "shared2x_medium_pressure_lookahead" / "run_experiment.py",
    )
    dote_runtime = load_module(
        "la_e52_dote_runtime",
        TEST_DIR / "run_dotemc_priority_mask_experiment.py",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model = lookahead.MediumPressureHattrick(props).to(device=device, dtype=props.dtype)
    checkpoint = torch.load(LA_CHECKPOINT, map_location=device, weights_only=False)
    if int(checkpoint["epoch"]) != 52:
        raise RuntimeError("The requested Hattrick-LA checkpoint is not epoch 52")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    cache = runtime.build_policy_cache(model, props, 400, 500, batch_size=20)
    num_pairs = int(cache.dataset.num_pairs)
    la_high = cache.policies[0].squeeze(-1).reshape(100, num_pairs, 8).float()
    zero_actual_la_high = extract_with_zero_actual_tm(
        runtime, model, props, cache.dataset, batch_size=20
    ).squeeze(-1).reshape_as(la_high).float()

    dote_model, mean, std = dote_runtime.load_checkpoint(DOTE_CHECKPOINT, device)
    dote_chunks = []
    with torch.no_grad():
        for start in range(0, 100, 50):
            predicted = [value[start : start + 50] for value in cache.predicted_tms]
            features = dote_runtime.make_inputs(
                predicted[0].float(),
                predicted[1].float(),
                predicted[2].float(),
                num_pairs,
                mean,
                std,
            )
            dote_chunks.append(dote_model(features, None)[:, 0].detach())
    dote_high = torch.cat(dote_chunks, dim=0).float()

    # Only the weighting below uses real High demand. Neither policy receives it.
    actual_high_demand = (
        cache.tms[0]
        .reshape(100, num_pairs, 8, 1)[:, :, 0, 0]
        .float()
        .clamp_min(0.0)
    )
    if float(actual_high_demand.sum().item()) <= 0.0:
        raise RuntimeError("Actual High demand weights are empty")

    payload = {
        "comparison": "Hattrick-LA epoch052 versus DOTE-MC High policy",
        "window": [400, 500],
        "load": "2x",
        "paths_per_od": 8,
        "od_snapshot_count": int(la_high.shape[0] * la_high.shape[1]),
        "information_contract": {
            "policy_inputs": "ESM-predicted High/Medium/Low traffic, topology, capacities, paths",
            "real_high_demand": "used only as an offline aggregation weight, never as policy input",
            "policies_are_pre_admission": True,
        },
        "similarity": similarity_stats(la_high, dote_high, actual_high_demand),
        "hattrick_la_epoch052": distribution_stats(la_high, actual_high_demand),
        "dote_mc": distribution_stats(dote_high, actual_high_demand),
        "audits": {
            "la_actual_tm_counterfactual_max_abs_policy_diff": float(
                (la_high - zero_actual_la_high).abs().max().item()
            ),
            "la_max_od_probability_sum_error": float(
                (la_high.sum(dim=-1) - 1.0).abs().max().item()
            ),
            "dote_max_od_probability_sum_error": float(
                (dote_high.sum(dim=-1) - 1.0).abs().max().item()
            ),
            "actual_high_demand_sum": float(actual_high_demand.sum().item()),
        },
        "artifacts": {
            "script": str(Path(__file__).resolve()),
            "la_checkpoint": str(LA_CHECKPOINT.resolve()),
            "la_checkpoint_sha256": sha256(LA_CHECKPOINT),
            "dote_checkpoint": str(DOTE_CHECKPOINT.resolve()),
            "dote_checkpoint_sha256": sha256(DOTE_CHECKPOINT),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
