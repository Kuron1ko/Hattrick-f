from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
DEFAULT_DOTE_CHECKPOINT = (
    TEST_DIR
    / "results_load2x_retrain_shared_strict"
    / "outputs"
    / "shared"
    / "w_1_0p1_0p01"
    / "best_model.pt"
)
PATHS_PER_OD = 8


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


def checkpoint_state(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint must be a mapping")
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise RuntimeError("Checkpoint has no model_state_dict")
    return checkpoint, state


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    denominator = weights.sum()
    if float(denominator.item()) <= 0.0:
        raise RuntimeError("Aggregation weights are empty")
    return float((values * weights).sum().item() / denominator.item())


def weighted_quantile(values: torch.Tensor, weights: torch.Tensor, q: float) -> float:
    flat_values = values.reshape(-1).detach().cpu().numpy().astype(np.float64)
    flat_weights = weights.reshape(-1).detach().cpu().numpy().astype(np.float64)
    positive = flat_weights > 0.0
    flat_values = flat_values[positive]
    flat_weights = flat_weights[positive]
    if flat_values.size == 0:
        raise RuntimeError("Weighted quantile has no positive weights")
    order = np.argsort(flat_values, kind="stable")
    flat_values = flat_values[order]
    flat_weights = flat_weights[order]
    cumulative = np.cumsum(flat_weights)
    target = float(q) * float(cumulative[-1])
    index = min(
        int(np.searchsorted(cumulative, target, side="left")),
        len(flat_values) - 1,
    )
    return float(flat_values[index])


def aggregation_sections(values: dict[str, torch.Tensor], weights: dict[str, torch.Tensor]):
    result = {}
    for section_name, section_weights in weights.items():
        if section_weights is None:
            mean = lambda value: float(value.mean().item())
            median = lambda value: float(value.median().item())
            q90 = lambda value: float(torch.quantile(value, 0.9).item())
        else:
            mean = lambda value, w=section_weights: weighted_mean(value, w)
            median = lambda value, w=section_weights: weighted_quantile(value, w, 0.5)
            q90 = lambda value, w=section_weights: weighted_quantile(value, w, 0.9)
        result[section_name] = {
            key: operation(value)
            for key, value, operation in (
                *( (key + "_mean", value, mean) for key, value in values.items() ),
                *( (key + "_median", value, median) for key, value in values.items() ),
                *( (key + "_p90", value, q90) for key, value in values.items() ),
            )
        }
    return result


def distribution_stats(policy: torch.Tensor, weights: dict[str, torch.Tensor]) -> dict:
    safe = policy.clamp_min(1e-12)
    sorted_policy = torch.sort(policy, dim=-1, descending=True).values
    entropy = -(safe * safe.log()).sum(dim=-1)
    metrics = {
        "pmax": sorted_policy[..., 0],
        "top2_mass": sorted_policy[..., :2].sum(dim=-1),
        "hhi": policy.square().sum(dim=-1),
        "entropy": entropy,
        "normalized_entropy": entropy / math.log(policy.shape[-1]),
        "effective_paths": entropy.exp(),
        "support_ge_0p01": (policy >= 0.01).float().sum(dim=-1),
    }
    sections = aggregation_sections(metrics, weights)
    for section_name, section_weights in weights.items():
        if section_weights is None:
            sections[section_name]["pmax_ge_0p9_fraction"] = float(
                (metrics["pmax"] >= 0.9).float().mean().item()
            )
        else:
            sections[section_name]["pmax_ge_0p9_fraction"] = weighted_mean(
                (metrics["pmax"] >= 0.9).float(), section_weights
            )
    return sections


def similarity_stats(
    candidate: torch.Tensor,
    dote: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> dict:
    tv = 0.5 * (candidate - dote).abs().sum(dim=-1)
    same = (candidate.argmax(dim=-1) == dote.argmax(dim=-1)).float()
    l1 = (candidate - dote).abs().sum(dim=-1)
    base = aggregation_sections({"tv": tv, "l1": l1}, weights)
    for section_name, section_weights in weights.items():
        mean = (
            (lambda value: float(value.mean().item()))
            if section_weights is None
            else (lambda value, w=section_weights: weighted_mean(value, w))
        )
        base[section_name].update(
            {
                "argmax_agreement": mean(same),
                "argmax_changed": 1.0 - mean(same),
                "tv_ge_0p10_fraction": mean((tv >= 0.10).float()),
                "tv_ge_0p25_fraction": mean((tv >= 0.25).float()),
                "tv_ge_0p50_fraction": mean((tv >= 0.50).float()),
                "tv_ge_0p75_fraction": mean((tv >= 0.75).float()),
            }
        )
    return base


def remove_topology_cache(model) -> None:
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    if hasattr(model, "_spc_transformer_cache"):
        model._spc_transformer_cache = None


def transform_actual(values, mode: str, num_ods: int, paths_per_od: int):
    changed = list(values)
    l1 = 0.0
    for index in (2, 4, 6):
        original = values[index]
        if mode == "zero":
            replacement = torch.zeros_like(original)
        elif mode == "permute":
            grouped = original.reshape(
                original.shape[0], num_ods, paths_per_od, -1
            )
            replacement = torch.flip(grouped, dims=(1,)).reshape_as(original)
        else:
            raise ValueError(mode)
        changed[index] = replacement
        l1 += float((original - replacement).abs().sum().item())
    return tuple(changed), l1


def extract_counterfactual(
    runtime,
    model,
    props,
    dataset,
    path_masks,
    batch_size: int,
    mode: str,
):
    remove_topology_cache(model)
    loader = runtime.shared.data_loader(dataset, batch_size, False, seed=0)
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_policy = True
    props.research_return_admitted = False
    num_ods = int(dataset.num_pairs)
    chunks = [[], [], []]
    total_l1 = 0.0
    with torch.no_grad():
        for inputs in loader:
            values = runtime.shared.unpack_to_device(inputs, props)
            changed, batch_l1 = transform_actual(
                values, mode, num_ods, int(props.num_paths_per_pair)
            )
            total_l1 += batch_l1
            policies = runtime.cached_policy_forward(
                model, props, dataset, changed, path_masks
            )
            for class_index in range(3):
                chunks[class_index].append(policies[class_index].detach())
    props.research_return_policy = False
    remove_topology_cache(model)
    return tuple(torch.cat(value, dim=0) for value in chunks), total_l1


def counterfactual_audit(
    baseline: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    changed: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
):
    result = {}
    for class_name, before, after in zip(("High", "Medium", "Low"), baseline, changed):
        delta = (before - after).abs()
        result[class_name] = {
            "torch_equal": bool(torch.equal(before, after)),
            "different_elements": int(torch.count_nonzero(before != after).item()),
            "max_abs_policy_delta": float(delta.max().item()),
            "mean_abs_policy_delta": float(delta.mean().item()),
        }
    return result


def demand_per_od(values: torch.Tensor, snapshots: int, num_ods: int) -> torch.Tensor:
    grouped = values.reshape(snapshots, num_ods, PATHS_PER_OD, -1)
    return grouped.mean(dim=(2, 3)).float().clamp_min(0.0)


def top_tv_examples(
    candidate: torch.Tensor,
    dote: torch.Tensor,
    predicted_high: torch.Tensor,
    actual_high: torch.Tensor,
    pair_list: list[tuple[int, int]],
    source_start: int,
    count: int,
) -> list[dict]:
    tv = 0.5 * (candidate - dote).abs().sum(dim=-1)
    count = min(int(count), int(tv.numel()))
    flat_indices = torch.topk(tv.reshape(-1), k=count).indices.cpu().tolist()
    examples = []
    num_ods = int(candidate.shape[1])
    for flat_index in flat_indices:
        snapshot_offset, od_index = divmod(int(flat_index), num_ods)
        candidate_row = candidate[snapshot_offset, od_index]
        dote_row = dote[snapshot_offset, od_index]
        src, dst = pair_list[od_index]
        examples.append(
            {
                "source_snapshot": source_start + snapshot_offset,
                "od_index": od_index,
                "od": f"{src}->{dst}",
                "tv": float(tv[snapshot_offset, od_index].item()),
                "candidate_argmax": int(candidate_row.argmax().item()),
                "dote_argmax": int(dote_row.argmax().item()),
                "esm_high_demand": float(predicted_high[snapshot_offset, od_index].item()),
                "actual_high_demand_offline_only": float(
                    actual_high[snapshot_offset, od_index].item()
                ),
                "candidate_policy": [float(value) for value in candidate_row.tolist()],
                "dote_policy": [float(value) for value in dote_row.tolist()],
            }
        )
    return examples


def write_output(destination: str, payload: dict) -> None:
    serialized = json.dumps(payload, indent=2, ensure_ascii=False)
    if destination != "-":
        path = Path(destination).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(path)
    print(serialized, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Strict-ESM policy/mechanism audit for one explicitly supplied "
            "sparse-path-cross-attention checkpoint. This script never ranks "
            "or selects checkpoints."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dote-checkpoint", type=Path, default=DEFAULT_DOTE_CHECKPOINT)
    parser.add_argument("--start", type=int, default=400)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--level", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--top-tv-examples", type=int, default=10)
    parser.add_argument(
        "--output",
        default="-",
        help="JSON file, or '-' to print without writing a file",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    dote_checkpoint_path = args.dote_checkpoint.resolve()
    for path in (checkpoint_path, dote_checkpoint_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.end <= args.start:
        raise ValueError("--end must be greater than --start")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested explicitly but is unavailable")

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
        "spc_strict_esm_policy_cache_runtime",
        TEST_DIR / "shared2x_medium_adapter" / "run_experiment.py",
    )
    spc = load_module(
        "spc_strict_esm_model_runtime",
        THIS_DIR / "run_experiment.py",
    )
    dote_runtime = load_module(
        "spc_strict_esm_dote_runtime",
        TEST_DIR / "run_dotemc_priority_mask_experiment.py",
    )

    torch.set_num_threads(1 if args.device == "cpu" else torch.get_num_threads())
    device = torch.device(args.device)
    props = runtime.build_props(args.level, device)
    checkpoint, state = checkpoint_state(checkpoint_path, device)
    required_keys = {
        "spc_query.weight": (spc.ATTENTION_DIM, None),
        "spc_medium_kv.weight": (spc.ATTENTION_DIM, None),
        "medium_pressure_init_adapter.2.weight": (1, 32),
        "medium_pressure_rau_adapter.2.weight": (1, 32),
    }
    for name, expected in required_keys.items():
        if name not in state:
            raise RuntimeError(f"Checkpoint is not sparse path cross-attention: missing {name}")
        shape = tuple(state[name].shape)
        if shape[0] != expected[0] or (
            expected[1] is not None and shape[1] != expected[1]
        ):
            raise RuntimeError(f"Unexpected {name} shape: {shape}")

    model = spc.SparsePathCrossAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    cache = runtime.build_policy_cache(
        model, props, args.start, args.end, batch_size=args.batch_size
    )
    baseline = cache.policies
    snapshots = args.end - args.start
    num_ods = int(cache.dataset.num_pairs)
    candidate_high = baseline[0].squeeze(-1).reshape(
        snapshots, num_ods, PATHS_PER_OD
    ).float()

    zero_policies, zero_actual_l1 = extract_counterfactual(
        runtime,
        model,
        props,
        cache.dataset,
        cache.path_masks,
        args.batch_size,
        "zero",
    )
    permuted_policies, permuted_actual_l1 = extract_counterfactual(
        runtime,
        model,
        props,
        cache.dataset,
        cache.path_masks,
        args.batch_size,
        "permute",
    )

    dote_model, dote_mean, dote_std = dote_runtime.load_checkpoint(
        dote_checkpoint_path, device
    )
    dote_chunks = []
    with torch.no_grad():
        for start in range(0, snapshots, args.batch_size):
            stop = min(start + args.batch_size, snapshots)
            predicted = [value[start:stop] for value in cache.predicted_tms]
            features = dote_runtime.make_inputs(
                predicted[0].float(),
                predicted[1].float(),
                predicted[2].float(),
                num_ods,
                dote_mean,
                dote_std,
            )
            dote_chunks.append(dote_model(features, None)[:, 0].detach())
    dote_high = torch.cat(dote_chunks, dim=0).float()
    if tuple(dote_high.shape) != tuple(candidate_high.shape):
        raise RuntimeError(
            f"DOTE/candidate shape mismatch: {dote_high.shape} vs {candidate_high.shape}"
        )

    actual_high = demand_per_od(cache.tms[0], snapshots, num_ods)
    predicted_high = demand_per_od(cache.predicted_tms[0], snapshots, num_ods)
    weights = {
        "unweighted": None,
        "esm_high_demand_weighted": predicted_high,
        "actual_high_demand_weighted_offline_only": actual_high,
    }
    pair_list = [
        tuple(map(int, pair)) for pair in cache.dataset.list_snapshots[0].pairs
    ]
    if len(pair_list) != num_ods:
        raise RuntimeError("OD metadata count does not match policy")

    zero_audit = counterfactual_audit(baseline, zero_policies)
    permutation_audit = counterfactual_audit(baseline, permuted_policies)
    strict_counterfactual_pass = all(
        row["torch_equal"]
        for audit in (zero_audit, permutation_audit)
        for row in audit.values()
    )
    payload = {
        "comparison": "Sparse path cross-attention versus DOTE-MC High policy",
        "window": [args.start, args.end],
        "load": "2x",
        "paths_per_od": PATHS_PER_OD,
        "snapshot_count": snapshots,
        "od_count": num_ods,
        "od_snapshot_count": snapshots * num_ods,
        "checkpoint_selection_contract": {
            "candidate_checkpoint": "explicit CLI input",
            "checkpoint_discovery_or_ranking_in_this_script": False,
            "test_metrics_used_to_select_checkpoint": False,
        },
        "information_contract": {
            "candidate_policy_inputs": (
                "ESM-predicted traffic, topology/capacity path embeddings, "
                "PTE, capacities, and path masks"
            ),
            "dote_policy_inputs": "ESM-predicted High/Medium/Low traffic",
            "actual_traffic_policy_input": False,
            "actual_high_demand_use": (
                "offline aggregation weight and example annotation only; never inference"
            ),
            "policies_are_pre_admission": True,
        },
        "architecture_load_audit": {
            "model_class": type(model).__name__,
            "attention_dim": int(spc.ATTENTION_DIM),
            "neighbor_ods": int(spc.NEIGHBOR_ODS),
            "feature_count": int(spc.TOTAL_FEATURES),
            "strict_state_dict_load": True,
            "checkpoint_epoch": checkpoint.get("epoch"),
        },
        "similarity": similarity_stats(candidate_high, dote_high, weights),
        "candidate_distribution": distribution_stats(candidate_high, weights),
        "dote_distribution": distribution_stats(dote_high, weights),
        "strict_esm_counterfactual": {
            "pass_all_classes_bitwise": strict_counterfactual_pass,
            "actual_tm_zero_replacement_l1": zero_actual_l1,
            "actual_tm_zero_policy_audit": zero_audit,
            "actual_tm_od_permutation_l1": permuted_actual_l1,
            "actual_tm_od_permutation_policy_audit": permutation_audit,
        },
        "probability_simplex_audit": {
            "candidate_max_error": float(
                (candidate_high.sum(dim=-1) - 1.0).abs().max().item()
            ),
            "dote_max_error": float(
                (dote_high.sum(dim=-1) - 1.0).abs().max().item()
            ),
        },
        "top_tv_examples": top_tv_examples(
            candidate_high,
            dote_high,
            predicted_high,
            actual_high,
            pair_list,
            args.start,
            args.top_tv_examples,
        ),
        "artifacts": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "candidate_checkpoint": str(checkpoint_path),
            "candidate_checkpoint_sha256": sha256(checkpoint_path),
            "dote_checkpoint": str(dote_checkpoint_path),
            "dote_checkpoint_sha256": sha256(dote_checkpoint_path),
            "output": None if args.output == "-" else str(Path(args.output).resolve()),
        },
    }
    if not strict_counterfactual_pass:
        raise RuntimeError(
            "Strict ESM audit failed: actual-TM replacement changed at least one policy"
        )
    write_output(args.output, payload)


if __name__ == "__main__":
    main()
