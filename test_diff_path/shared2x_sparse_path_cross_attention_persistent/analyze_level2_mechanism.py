from __future__ import annotations

"""Read-only mechanism audit for the persistent Stage-2 Level-2 checkpoint."""

import csv
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
PARENT_RUNNER = TEST_DIR / "shared2x_sparse_path_cross_attention" / "run_experiment.py"
PERSISTENT_RUNNER = THIS_DIR / "run_experiment.py"
STRICT_RUNNER = (
    TEST_DIR
    / "shared2x_sparse_path_cross_attention"
    / "evaluate_strict_esm_sequential.py"
)
EDGE_RUNNER = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
LOW_GUARD_RUNNER = THIS_DIR / "train_low_guard.py"
PARENT_CHECKPOINT = (
    TEST_DIR
    / "shared2x_sparse_path_cross_attention"
    / "artifacts"
    / "level2_proxy"
    / "seed_490"
    / "final_model.pt"
)
PERSISTENT_CHECKPOINT = (
    THIS_DIR / "artifacts" / "level2_proxy" / "seed_490" / "final_model.pt"
)
LOW_GUARD_CHECKPOINT = THIS_DIR / "artifacts" / "level2_low_guard" / "best_low_guard.pt"
LOW_GUARD_REPORT = THIS_DIR / "artifacts" / "level2_low_guard" / "report.json"
OUTPUT_DIR = ROOT / "output" / "analysis" / "persistent_stage2_level2_mechanism"
START, STOP = 200, 250
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


def describe(values) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float((values * weights).sum() / max(weights.sum(), 1e-12))


def correlation(left, right) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.std() <= 1e-15 or right.std() <= 1e-15:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def grouped_policy(policy: torch.Tensor) -> torch.Tensor:
    grouped = policy.squeeze(-1).to(torch.float32).reshape(policy.shape[0], -1, K)
    return grouped / grouped.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def policy_arrays(policy: torch.Tensor) -> dict[str, np.ndarray]:
    probability = grouped_policy(policy)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=-1)
    return {
        "effective_paths": entropy.exp().cpu().numpy(),
        "pmax": probability.amax(dim=-1).cpu().numpy(),
        "argmax": probability.argmax(dim=-1).cpu().numpy(),
        "probability": probability.cpu().numpy(),
    }


def policy_summary(policy: torch.Tensor, demand_weight: np.ndarray) -> dict:
    arrays = policy_arrays(policy)
    return {
        "effective_paths": describe(arrays["effective_paths"]),
        "effective_paths_demand_weighted_mean": weighted_mean(
            arrays["effective_paths"], demand_weight
        ),
        "pmax": describe(arrays["pmax"]),
        "pmax_demand_weighted_mean": weighted_mean(arrays["pmax"], demand_weight),
    }


def policy_comparison(
    before: torch.Tensor, after: torch.Tensor, demand_weight: np.ndarray
) -> tuple[dict, dict[str, np.ndarray]]:
    before_probability = grouped_policy(before)
    after_probability = grouped_policy(after)
    tv = 0.5 * (after_probability - before_probability).abs().sum(dim=-1)
    argmax_changed = (
        before_probability.argmax(dim=-1) != after_probability.argmax(dim=-1)
    ).to(torch.float32)
    group_mass_delta = (
        before.squeeze(-1).reshape(before.shape[0], -1, K).sum(dim=-1)
        - after.squeeze(-1).reshape(after.shape[0], -1, K).sum(dim=-1)
    ).abs()
    tv_np = tv.cpu().numpy()
    changed_np = argmax_changed.cpu().numpy()
    result = {
        "torch_equal": bool(torch.equal(before, after)),
        "different_elements": int(torch.count_nonzero(before != after).item()),
        "max_abs_path_delta": float((before - after).abs().max().item()),
        "max_od_group_mass_delta": float(group_mass_delta.max().item()),
        "total_variation": describe(tv_np),
        "total_variation_demand_weighted_mean": weighted_mean(tv_np, demand_weight),
        "argmax_changed_count": int(argmax_changed.sum().item()),
        "argmax_changed_fraction": float(argmax_changed.mean().item()),
        "argmax_changed_demand_weighted_fraction": weighted_mean(
            changed_np, demand_weight
        ),
    }
    return result, {"tv": tv_np, "argmax_changed": changed_np}


def load_model(model_class, props, checkpoint_path: Path):
    payload = torch.load(checkpoint_path, map_location=props.device, weights_only=False)
    model = model_class(props).to(device=props.device, dtype=props.dtype)
    loaded = model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload, {
        "missing_keys": list(loaded.missing_keys),
        "unexpected_keys": list(loaded.unexpected_keys),
    }


def medium_counterfactual(values: tuple, mode: str, num_ods: int) -> tuple:
    changed = list(values)
    medium = changed[5]
    grouped = medium.reshape(medium.shape[0], num_ods, K, -1)
    if mode == "baseline":
        replacement = grouped
    elif mode == "reverse_od":
        replacement = torch.flip(grouped, dims=(1,))
    elif mode == "zero":
        replacement = torch.zeros_like(grouped)
    elif mode == "scale_1.10":
        replacement = grouped * 1.10
    else:
        raise ValueError(mode)
    changed[5] = replacement.reshape_as(medium)
    return tuple(changed)


def infer_high_modes(runtime, strict, model, props) -> dict:
    dataset = runtime.DM_Dataset_within_Cluster(props, 0, START, STOP)
    if int(dataset.max_source_index_read) != STOP - 1:
        raise RuntimeError("Split-safe reader audit failed")
    path_masks = runtime.shared.base.move_dataset_static(dataset, props.device)
    loader = runtime.shared.data_loader(dataset, 10, False, seed=0)
    modes = ("baseline", "reverse_od", "zero", "scale_1.10")
    outputs = {mode: [] for mode in modes}
    high_demand = []
    medium_demand = []
    actual_nonzero = 0
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_policy = True
    props.research_return_admitted = False
    strict.remove_topology_cache(model)
    model.eval()
    with torch.no_grad():
        for inputs in loader:
            original = runtime.shared.unpack_to_device(inputs, props)
            strict_values, _ = strict.policy_actual_inputs(
                original, "zero", int(dataset.num_pairs), K
            )
            actual_nonzero += sum(
                int(torch.count_nonzero(strict_values[index]).item())
                for index in (2, 4, 6)
            )
            high_group = original[3].reshape(
                original[3].shape[0], int(dataset.num_pairs), K, -1
            )
            medium_group = original[5].reshape(
                original[5].shape[0], int(dataset.num_pairs), K, -1
            )
            high_demand.append(high_group[:, :, 0, 0].detach().cpu())
            medium_demand.append(medium_group[:, :, 0, 0].detach().cpu())
            for mode in modes:
                policy_values = medium_counterfactual(
                    strict_values, mode, int(dataset.num_pairs)
                )
                policies = runtime.cached_policy_forward(
                    model, props, dataset, policy_values, path_masks
                )
                outputs[mode].append(policies[0].detach().cpu())
    props.research_return_policy = False
    strict.remove_topology_cache(model)
    if actual_nonzero != 0:
        raise RuntimeError("Strict actual-TM zero audit failed")
    return {
        "policies": {mode: torch.cat(chunks, dim=0) for mode, chunks in outputs.items()},
        "predicted_high_od": torch.cat(high_demand, dim=0).numpy(),
        "predicted_medium_od": torch.cat(medium_demand, dim=0).numpy(),
        "strict_actual_nonzero_count": actual_nonzero,
        "snapshot_count": STOP - START,
        "od_count": int(dataset.num_pairs),
    }


def response_report(run: dict) -> dict:
    baseline = run["policies"]["baseline"]
    high_weight = run["predicted_high_od"]
    medium_original = run["predicted_medium_od"]
    result = {}
    for mode in ("reverse_od", "zero", "scale_1.10"):
        comparison, arrays = policy_comparison(
            baseline, run["policies"][mode], high_weight
        )
        if mode == "reverse_od":
            medium_change = np.abs(medium_original - medium_original[:, ::-1])
        elif mode == "zero":
            medium_change = np.abs(medium_original)
        else:
            medium_change = np.abs(medium_original * 0.10)
        comparison["response_tv_vs_medium_change_correlation"] = correlation(
            arrays["tv"], medium_change
        )
        result[mode] = comparison
    return result


def adapter_norms(model) -> dict:
    result = {}
    for name in ("stage2_high_init_adapter", "stage2_high_rau_adapter"):
        adapter = getattr(model, name)
        result[name] = {
            "input_weight_l2": float(adapter[0].weight.norm().item()),
            "output_weight_l2": float(adapter[2].weight.norm().item()),
            "output_bias_abs": float(adapter[2].bias.abs().max().item()),
        }
    return result


def disable_stage2_adapters(model) -> None:
    with torch.no_grad():
        for name in ("stage2_high_init_adapter", "stage2_high_rau_adapter"):
            adapter = getattr(model, name)
            adapter[2].weight.zero_()
            adapter[2].bias.zero_()


def csv_replay_delta(before_path: Path, after_path: Path, class_name: str) -> dict:
    with before_path.open(newline="", encoding="utf-8") as handle:
        before_rows = list(csv.DictReader(handle))
    with after_path.open(newline="", encoding="utf-8") as handle:
        after_rows = list(csv.DictReader(handle))
    before = {
        (int(row["snapshot"]), row["class"]): row
        for row in before_rows
        if row["class"] == class_name
    }
    after = {
        (int(row["snapshot"]), row["class"]): row
        for row in after_rows
        if row["class"] == class_name
    }
    numeric = [key for key in next(iter(before.values())) if key not in ("snapshot", "class")]
    deltas = {
        key: max(
            abs(float(after[index][key]) - float(before[index][key]))
            for index in before
        )
        for key in numeric
    }
    return {"all_keys_equal": set(before) == set(after), "max_abs_delta": deltas}


def main() -> None:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    edge = load_module("persistent_l2_mechanism_edge", EDGE_RUNNER)
    strict = load_module("persistent_l2_mechanism_strict", STRICT_RUNNER)
    parent = load_module("persistent_l2_mechanism_parent", PARENT_RUNNER)
    persistent = load_module("persistent_l2_mechanism_model", PERSISTENT_RUNNER)
    low_guard_module = load_module("persistent_l2_mechanism_low_guard", LOW_GUARD_RUNNER)
    runtime = edge.runtime

    parent_props = runtime.build_props(2, device)
    persistent_props = runtime.build_props(2, device)
    adapter_off_props = runtime.build_props(2, device)
    parent_model, parent_payload, parent_load = load_model(
        parent.SparsePathCrossAttentionHattrick, parent_props, PARENT_CHECKPOINT
    )
    persistent_model, persistent_payload, persistent_load = load_model(
        persistent.PersistentStage2SparseAttentionHattrick,
        persistent_props,
        PERSISTENT_CHECKPOINT,
    )
    adapter_off_model, _, adapter_off_load = load_model(
        persistent.PersistentStage2SparseAttentionHattrick,
        adapter_off_props,
        PERSISTENT_CHECKPOINT,
    )
    learned_adapter_norms = adapter_norms(persistent_model)
    disable_stage2_adapters(adapter_off_model)

    parent_run = infer_high_modes(runtime, strict, parent_model, parent_props)
    persistent_run = infer_high_modes(
        runtime, strict, persistent_model, persistent_props
    )
    adapter_off_run = infer_high_modes(
        runtime, strict, adapter_off_model, adapter_off_props
    )
    for key in ("predicted_high_od", "predicted_medium_od"):
        if not np.array_equal(parent_run[key], persistent_run[key]) or not np.array_equal(
            parent_run[key], adapter_off_run[key]
        ):
            raise RuntimeError(f"Demand mismatch across model runs: {key}")

    high_weight = parent_run["predicted_high_od"]
    parent_policy = parent_run["policies"]["baseline"]
    persistent_policy = persistent_run["policies"]["baseline"]
    adapter_off_policy = adapter_off_run["policies"]["baseline"]
    parent_arrays = policy_arrays(parent_policy)
    persistent_arrays = policy_arrays(persistent_policy)
    adapter_off_arrays = policy_arrays(adapter_off_policy)
    parent_vs_persistent, comparison_arrays = policy_comparison(
        parent_policy, persistent_policy, high_weight
    )
    full_vs_adapter_off, _ = policy_comparison(
        adapter_off_policy, persistent_policy, high_weight
    )
    concentration_delta = {
        "effective_paths_persistent_minus_parent": describe(
            persistent_arrays["effective_paths"] - parent_arrays["effective_paths"]
        ),
        "effective_paths_delta_demand_weighted_mean": weighted_mean(
            persistent_arrays["effective_paths"] - parent_arrays["effective_paths"],
            high_weight,
        ),
        "pmax_persistent_minus_parent": describe(
            persistent_arrays["pmax"] - parent_arrays["pmax"]
        ),
        "pmax_delta_demand_weighted_mean": weighted_mean(
            persistent_arrays["pmax"] - parent_arrays["pmax"], high_weight
        ),
        "persistent_more_concentrated_od_fraction": float(
            np.mean(
                persistent_arrays["effective_paths"]
                < parent_arrays["effective_paths"] - 1e-7
            )
        ),
        "persistent_more_concentrated_demand_weighted_fraction": weighted_mean(
            (
                persistent_arrays["effective_paths"]
                < parent_arrays["effective_paths"] - 1e-7
            ).astype(np.float64),
            high_weight,
        ),
    }
    adapter_direct_concentration_delta = {
        "effective_paths_full_minus_adapters_disabled": describe(
            persistent_arrays["effective_paths"]
            - adapter_off_arrays["effective_paths"]
        ),
        "effective_paths_delta_demand_weighted_mean": weighted_mean(
            persistent_arrays["effective_paths"]
            - adapter_off_arrays["effective_paths"],
            high_weight,
        ),
        "pmax_full_minus_adapters_disabled": describe(
            persistent_arrays["pmax"] - adapter_off_arrays["pmax"]
        ),
        "pmax_delta_demand_weighted_mean": weighted_mean(
            persistent_arrays["pmax"] - adapter_off_arrays["pmax"], high_weight
        ),
        "full_more_concentrated_demand_weighted_fraction": weighted_mean(
            (
                persistent_arrays["effective_paths"]
                < adapter_off_arrays["effective_paths"] - 1e-7
            ).astype(np.float64),
            high_weight,
        ),
    }

    # Exact Low-only guard policy audit on the same strict evaluation window.
    guard_props = runtime.build_props(2, device)
    guard_model, _, guard_model_load = load_model(
        persistent.PersistentStage2SparseAttentionHattrick,
        guard_props,
        PERSISTENT_CHECKPOINT,
    )
    guard_cache, guard_information_audit = strict.build_strict_policy_cache(
        runtime,
        guard_model,
        guard_props,
        START,
        STOP,
        10,
        actual_input_mode="zero",
    )
    edge_features = edge.edge_features(guard_cache)
    guard_payload = torch.load(
        LOW_GUARD_CHECKPOINT, map_location=device, weights_only=False
    )
    guard = low_guard_module.LowGuard(
        int(guard_payload["edge_count"]),
        int(guard_payload["feature_count"]),
        float(guard_payload["max_toll"]),
    ).to(device)
    guard_loaded = guard.load_state_dict(guard_payload["state_dict"], strict=True)
    guard.eval()
    candidate_cache, tolls = low_guard_module.build_candidate_cache(
        guard, guard_cache, edge_features, batch_size=10
    )
    candidate_low = candidate_cache.path_features.unsqueeze(-1)
    adapted = low_guard_module.LowOnlyAdapter().adapt_batch(
        list(guard_cache.policies), {"path_features": candidate_cache.path_features}
    )
    low_weight = guard_cache.predicted_tms[2].squeeze(-1).reshape(
        len(guard_cache), -1, K
    )[:, :, 0].cpu().numpy()
    low_comparison, _ = policy_comparison(
        guard_cache.policies[2], candidate_low, low_weight
    )
    guard_report = json.loads(LOW_GUARD_REPORT.read_text(encoding="utf-8"))
    guard_policy_audit = {
        "adapter_high_torch_equal": bool(
            torch.equal(adapted[0], guard_cache.policies[0])
        ),
        "adapter_medium_torch_equal": bool(
            torch.equal(adapted[1], guard_cache.policies[1])
        ),
        "adapter_low_torch_equal": bool(
            torch.equal(adapted[2], guard_cache.policies[2])
        ),
        "adapter_high_same_storage": adapted[0] is guard_cache.policies[0],
        "adapter_medium_same_storage": adapted[1] is guard_cache.policies[1],
        "high_max_abs_policy_delta": float(
            (adapted[0] - guard_cache.policies[0]).abs().max().item()
        ),
        "medium_max_abs_policy_delta": float(
            (adapted[1] - guard_cache.policies[1]).abs().max().item()
        ),
        "low": low_comparison,
        "low_base": policy_summary(guard_cache.policies[2], low_weight),
        "low_guarded": policy_summary(candidate_low, low_weight),
        "disabled_low_path_nonzero_count": int(
            torch.count_nonzero(
                (guard_cache.policies[2] == 0) & (candidate_low != 0)
            ).item()
        ),
        "toll_abs_mean": float(tolls.abs().mean().item()),
        "toll_abs_max": float(tolls.abs().max().item()),
        "strict_information_audit": guard_information_audit,
        "validation_report_upstream_policy_torch_equal": guard_report["validation"][
            "upstream_policy_torch_equal"
        ],
        "evaluation_report_upstream_policy_torch_equal": guard_report["evaluation"][
            "upstream_policy_torch_equal"
        ],
        "evaluation_report_preservation_max_delta": guard_report["evaluation"][
            "preservation_max_delta"
        ],
        "evaluation_csv_replay_delta": {
            class_name: csv_replay_delta(
                THIS_DIR
                / "artifacts"
                / "level2_low_guard"
                / "evaluation_baseline_rows.csv",
                THIS_DIR
                / "artifacts"
                / "level2_low_guard"
                / "evaluation_candidate_rows.csv",
                class_name,
            )
            for class_name in ("High", "Medium")
        },
    }

    report = {
        "scope": {
            "level": 2,
            "seed": 490,
            "window": [START, STOP],
            "strict_esm": True,
            "parent_checkpoint": str(PARENT_CHECKPOINT.resolve()),
            "persistent_checkpoint": str(PERSISTENT_CHECKPOINT.resolve()),
            "low_guard_checkpoint": str(LOW_GUARD_CHECKPOINT.resolve()),
            "checkpoint_sha256": {
                "parent": sha256(PARENT_CHECKPOINT),
                "persistent": sha256(PERSISTENT_CHECKPOINT),
                "low_guard": sha256(LOW_GUARD_CHECKPOINT),
            },
            "checkpoint_epochs": {
                "parent": int(parent_payload["epoch"]),
                "persistent": int(persistent_payload["epoch"]),
            },
        },
        "state_load": {
            "parent": parent_load,
            "persistent": persistent_load,
            "persistent_adapter_off": adapter_off_load,
            "guard_backbone": guard_model_load,
            "low_guard": {
                "missing_keys": list(guard_loaded.missing_keys),
                "unexpected_keys": list(guard_loaded.unexpected_keys),
            },
        },
        "high_policy": {
            "parent": policy_summary(parent_policy, high_weight),
            "persistent": policy_summary(persistent_policy, high_weight),
            "persistent_stage2_adapters_disabled": policy_summary(
                adapter_off_policy, high_weight
            ),
            "parent_vs_persistent": parent_vs_persistent,
            "concentration_change": concentration_delta,
            "stage2_adapter_direct_concentration_change": (
                adapter_direct_concentration_delta
            ),
        },
        "medium_response": {
            "counterfactuals": {
                "reverse_od": "reverse Medium OD order; aggregate/distribution preserved",
                "zero": "set predicted Medium demand to literal zero",
                "scale_1.10": "multiply predicted Medium demand by 1.10",
            },
            "parent": response_report(parent_run),
            "persistent": response_report(persistent_run),
            "persistent_stage2_adapters_disabled": response_report(adapter_off_run),
            "learned_stage2_adapter_norms": learned_adapter_norms,
            "full_persistent_vs_same_checkpoint_adapters_disabled_baseline": (
                full_vs_adapter_off
            ),
        },
        "low_guard": guard_policy_audit,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    snapshots = np.repeat(np.arange(START, STOP), high_weight.shape[1])
    ods = np.tile(np.arange(high_weight.shape[1]), STOP - START)
    rows = []
    flat = lambda value: np.asarray(value).reshape(-1)
    for index in range(len(snapshots)):
        rows.append(
            {
                "snapshot": int(snapshots[index]),
                "od": int(ods[index]),
                "predicted_high_demand": float(flat(high_weight)[index]),
                "predicted_medium_demand": float(
                    flat(parent_run["predicted_medium_od"])[index]
                ),
                "parent_effective_paths": float(
                    flat(parent_arrays["effective_paths"])[index]
                ),
                "persistent_effective_paths": float(
                    flat(persistent_arrays["effective_paths"])[index]
                ),
                "parent_pmax": float(flat(parent_arrays["pmax"])[index]),
                "persistent_pmax": float(flat(persistent_arrays["pmax"])[index]),
                "parent_argmax": int(flat(parent_arrays["argmax"])[index]),
                "persistent_argmax": int(flat(persistent_arrays["argmax"])[index]),
                "parent_persistent_tv": float(flat(comparison_arrays["tv"])[index]),
            }
        )
    with (OUTPUT_DIR / "high_od_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "report": str((OUTPUT_DIR / "report.json").resolve()),
                "rows": str((OUTPUT_DIR / "high_od_rows.csv").resolve()),
                "parent_persistent_weighted_tv": parent_vs_persistent[
                    "total_variation_demand_weighted_mean"
                ],
                "parent_effective_paths": report["high_policy"]["parent"][
                    "effective_paths_demand_weighted_mean"
                ],
                "persistent_effective_paths": report["high_policy"]["persistent"][
                    "effective_paths_demand_weighted_mean"
                ],
                "low_guard_high_equal": guard_policy_audit[
                    "adapter_high_torch_equal"
                ],
                "low_guard_medium_equal": guard_policy_audit[
                    "adapter_medium_torch_equal"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
