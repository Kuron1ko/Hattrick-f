from __future__ import annotations

"""Read-only 0-349 audit of Level-2 versus rejected Level-4 Low guards.

The only dataset window constructed by this program is 300-349.  Existing
Level-2 200-249 rows are read for reporting.  No 350-499 artifact is opened.
"""

import csv
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
TEST_DIR = ROOT / "test_diff_path"
PERSISTENT_DIR = TEST_DIR / "shared2x_sparse_path_cross_attention_persistent"
PERSISTENT_RUNNER = PERSISTENT_DIR / "run_experiment.py"
STRICT_RUNNER = (
    TEST_DIR
    / "shared2x_sparse_path_cross_attention"
    / "evaluate_strict_esm_sequential.py"
)
EDGE_RUNNER = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
GUARD_RUNNER = PERSISTENT_DIR / "train_low_guard.py"
LEVEL2_CORE = (
    PERSISTENT_DIR / "artifacts" / "level2_proxy" / "seed_490" / "final_model.pt"
)
LEVEL4_CORE = (
    PERSISTENT_DIR / "level4_selection_validation_only" / "selected_checkpoint.pt"
)
LEVEL2_HEAD = PERSISTENT_DIR / "artifacts" / "level2_low_guard" / "best_low_guard.pt"
LEVEL4_HEAD = (
    PERSISTENT_DIR
    / "artifacts"
    / "level4_low_guard_frozen"
    / "frozen_low_guard_prevalidation.pt"
)
LEVEL2_REPORT = PERSISTENT_DIR / "artifacts" / "level2_low_guard" / "report.json"
LEVEL4_REJECTION = (
    PERSISTENT_DIR
    / "artifacts"
    / "level4_low_guard_frozen"
    / "rejected_internal_gate.json"
)
OUTPUT_DIR = Path(__file__).resolve().parent
START, STOP = 300, 350
K = 8


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
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
        "p1": float(np.quantile(values, 0.01)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def correlation(left, right, rank: bool = False) -> float | None:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if rank:
        left, right = rankdata(left), rankdata(right)
    if left.std() <= 1e-15 or right.std() <= 1e-15:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            key: value if key == "class" else int(value) if key == "snapshot" else float(value)
            for key, value in row.items()
        }
        for row in rows
    ]


def low_by_snapshot(rows: list[dict]) -> dict[int, float]:
    return {
        int(row["snapshot"]): float(row["norm_fulfill"])
        for row in rows
        if row["class"] == "Low"
    }


def gain_summary(baseline_rows: list[dict], candidate_rows: list[dict]) -> dict:
    baseline = low_by_snapshot(baseline_rows)
    candidate = low_by_snapshot(candidate_rows)
    if set(baseline) != set(candidate):
        raise RuntimeError("Low row keys differ")
    snapshots = sorted(baseline)
    before = np.asarray([baseline[index] for index in snapshots])
    after = np.asarray([candidate[index] for index in snapshots])
    gain = after - before
    return {
        "baseline": describe(before),
        "candidate": describe(after),
        "gain": describe(gain),
        "negative_count": int(np.count_nonzero(gain < -1e-7)),
        "positive_count": int(np.count_nonzero(gain > 1e-7)),
        "near_zero_count": int(np.count_nonzero(np.abs(gain) <= 1e-7)),
        "snapshots": snapshots,
        "gain_values": gain,
    }


def public_gain(value: dict) -> dict:
    return {key: item for key, item in value.items() if key not in ("snapshots", "gain_values")}


def load_model(model_module, runtime, level: int, checkpoint: Path, device):
    props = runtime.build_props(level, device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = model_module.PersistentStage2SparseAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    result = model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return props, model, {
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
    }


def load_head(guard_module, checkpoint: Path, device):
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    head = guard_module.LowGuard(
        int(payload["edge_count"]),
        int(payload["feature_count"]),
        float(payload["max_toll"]),
    ).to(device)
    result = head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head, payload, {
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
    }


def evaluate_cache(runtime, model, props, cache, adapter):
    rows, summary = runtime.evaluate_cache(
        model, props, cache, adapter, batch_size=10
    )
    return rows, runtime.summary_index(summary)


def candidate_from_tolls(guard_module, cache, tolls, scale: float):
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    low = guard_module.route_low(
        cache.policies[2].squeeze(-1), tolls * float(scale), pte
    )
    return replace(cache, path_features=low)


def low_utilization(cache, policy: torch.Tensor) -> torch.Tensor:
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    demand = cache.predicted_tms[2].squeeze(-1).to(torch.float32)
    flow = policy.to(torch.float32) * demand
    return (
        torch.sparse.mm(pte.t(), flow.t()).t()
        / cache.capacities.to(torch.float32).clamp_min(1e-9)
    )


def congestion_score(high_medium: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
    total = high_medium + low
    overload = torch.relu(total - 1.0).square().mean(dim=1)
    remaining = torch.relu(1.0 - high_medium) + 0.05
    scarcity = (low / remaining).square().mean(dim=1)
    soft_barrier = (
        torch.nn.functional.softplus((total - 1.0) / 0.05) * 0.05
    ).mean(dim=1)
    return overload + 0.05 * scarcity + soft_barrier


def actual_high_medium_util(runtime, model, props, cache):
    indices = torch.arange(len(cache), device=props.device)
    batch = runtime.select_cache(cache, indices)
    admitted, _ = runtime.simulate_admission(
        model, props, cache.dataset, batch, adapter=None
    )
    pte = cache.dataset.pte.coalesce().to(dtype=torch.float32)
    flow = admitted[0].to(torch.float32) + admitted[1].to(torch.float32)
    return (
        torch.sparse.mm(pte.t(), flow.t()).t()
        / cache.capacities.to(torch.float32).clamp_min(1e-9)
    )


def policy_tv(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    b = before.reshape(len(before), -1, K)
    a = after.reshape(len(after), -1, K)
    return 0.5 * (a - b).abs().sum(dim=-1).mean(dim=1)


def per_snapshot_edge_correlation(left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    values = []
    for index in range(len(left)):
        current = correlation(
            left[index].detach().cpu().numpy(),
            right[index].detach().cpu().numpy(),
            rank=True,
        )
        values.append(float("nan") if current is None else current)
    return np.asarray(values)


def main() -> None:
    if STOP > 350:
        raise RuntimeError("Forbidden window requested")
    torch.set_num_threads(4)
    torch.manual_seed(20260824)
    np.random.seed(20260824)
    device = torch.device("cpu")

    # Match the module-load order used by the formal Low-guard trainer.
    guard_module = load_module("l4_guard_shift_guard", GUARD_RUNNER)
    model_module = load_module("l4_guard_shift_model", PERSISTENT_RUNNER)
    edge = load_module("l4_guard_shift_edge", EDGE_RUNNER)
    strict = load_module("l4_guard_shift_strict", STRICT_RUNNER)
    runtime = edge.runtime

    l2_props, l2_model, l2_model_load = load_model(
        model_module, runtime, 2, LEVEL2_CORE, device
    )
    l4_props, l4_model, l4_model_load = load_model(
        model_module, runtime, 4, LEVEL4_CORE, device
    )
    native_props = runtime.build_props(4, device)
    native_model, native_checkpoint = runtime.load_backbone(
        4, 490, native_props, device
    )
    native_model.eval()
    for parameter in native_model.parameters():
        parameter.requires_grad_(False)
    l2_head, l2_payload, l2_head_load = load_head(guard_module, LEVEL2_HEAD, device)
    l4_head, l4_payload, l4_head_load = load_head(guard_module, LEVEL4_HEAD, device)

    l2_cache, l2_info = strict.build_strict_policy_cache(
        runtime, l2_model, l2_props, START, STOP, 10, actual_input_mode="zero"
    )
    l4_cache, l4_info = strict.build_strict_policy_cache(
        runtime, l4_model, l4_props, START, STOP, 10, actual_input_mode="zero"
    )
    native_cache, native_info = strict.build_strict_policy_cache(
        runtime,
        native_model,
        native_props,
        START,
        STOP,
        10,
        actual_input_mode="zero",
    )
    l2_features = edge.edge_features(l2_cache)
    l4_features = edge.edge_features(l4_cache)
    l2_base_rows, l2_base_summary = evaluate_cache(
        runtime, l2_model, l2_props, l2_cache, None
    )
    l4_base_rows, l4_base_summary = evaluate_cache(
        runtime, l4_model, l4_props, l4_cache, None
    )
    native_rows, native_summary = evaluate_cache(
        runtime, native_model, native_props, native_cache, None
    )

    combinations = {}
    combination_rows = []
    for method_name, source_rows in (
        ("native_hattrick", native_rows),
        ("level4_core", l4_base_rows),
    ):
        for row in source_rows:
            copy = dict(row)
            copy["method"] = method_name
            combination_rows.append(copy)
    built = {}
    for core_name, model, props, cache, features, base_rows in (
        ("level2_core", l2_model, l2_props, l2_cache, l2_features, l2_base_rows),
        ("level4_core", l4_model, l4_props, l4_cache, l4_features, l4_base_rows),
    ):
        for head_name, head in (("level2_head", l2_head), ("level4_head", l4_head)):
            candidate, tolls = guard_module.build_candidate_cache(
                head, cache, features, batch_size=10
            )
            rows, summary = evaluate_cache(
                runtime,
                model,
                props,
                candidate,
                guard_module.LowOnlyAdapter(),
            )
            name = f"{core_name}__{head_name}"
            gains = gain_summary(base_rows, rows)
            tv = policy_tv(
                cache.policies[2].squeeze(-1), candidate.path_features
            )
            combinations[name] = {
                "gain": public_gain(gains),
                "summary": summary,
                "toll_abs": describe(tolls.abs().cpu().numpy()),
                "toll_signed": describe(tolls.cpu().numpy()),
                "toll_structure": {
                    "temporal_std_across_snapshots_mean": float(
                        tolls.std(dim=0).mean().item()
                    ),
                    "between_edge_std_of_temporal_mean": float(
                        tolls.mean(dim=0).std().item()
                    ),
                    "dynamic_energy_fraction": float(
                        (tolls - tolls.mean(dim=0, keepdim=True)).square().sum().sqrt().item()
                        / max(float(tolls.square().sum().sqrt().item()), 1e-12)
                    ),
                },
                "policy_tv_per_snapshot": describe(tv.cpu().numpy()),
            }
            built[name] = (candidate, tolls, gains, tv)
            for row in rows:
                copy = dict(row)
                copy["method"] = name
                combination_rows.append(copy)

    l4_actual_hm = actual_high_medium_util(runtime, l4_model, l4_props, l4_cache)
    l4_pred_hm = l4_features[:, :, 3]
    l4_pred_actual_gap = (l4_pred_hm - l4_actual_hm).abs()
    l4_base_low_util = low_utilization(l4_cache, l4_cache.policies[2].squeeze(-1))
    l4_candidate, l4_tolls, l4_gain, l4_tv = built[
        "level4_core__level4_head"
    ]
    l4_candidate_low_util = low_utilization(l4_cache, l4_candidate.path_features)
    base_score = congestion_score(l4_pred_hm, l4_base_low_util)
    candidate_score = congestion_score(l4_pred_hm, l4_candidate_low_util)
    proxy_improvement = base_score - candidate_score
    actual_gain = l4_gain["gain_values"]
    toll_pred_corr = per_snapshot_edge_correlation(l4_tolls, l4_pred_hm)
    toll_actual_corr = per_snapshot_edge_correlation(l4_tolls, l4_actual_hm)

    actual_tm_error = []
    for actual, predicted in zip(l4_cache.tms, l4_cache.predicted_tms):
        actual_od = actual.squeeze(-1).reshape(len(l4_cache), -1, K)[:, :, 0]
        predicted_od = predicted.squeeze(-1).reshape(len(l4_cache), -1, K)[:, :, 0]
        actual_tm_error.append(
            (actual_od - predicted_od).abs().sum(dim=1)
            / actual_od.abs().sum(dim=1).clamp_min(1e-9)
        )
    actual_tm_error = torch.stack(actual_tm_error, dim=1)

    proxy_fields = {
        "core_low_norm_fulfill": np.asarray(
            [low_by_snapshot(l4_base_rows)[index] for index in range(START, STOP)]
        ),
        "predicted_low_demand": (
            l4_cache.predicted_tms[2]
            .squeeze(-1)
            .reshape(len(l4_cache), -1, K)[:, :, 0]
            .sum(dim=1)
            .cpu()
            .numpy()
        ),
        "predicted_hm_util_mean": l4_pred_hm.mean(dim=1).cpu().numpy(),
        "predicted_hm_util_max": l4_pred_hm.max(dim=1).values.cpu().numpy(),
        "predicted_hm_edges_ge_1": (l4_pred_hm >= 1.0).sum(dim=1).cpu().numpy(),
        "predicted_total_base_max": (
            l4_pred_hm + l4_base_low_util
        ).max(dim=1).values.cpu().numpy(),
        "predicted_base_congestion_score": base_score.cpu().numpy(),
        "predicted_proxy_improvement": proxy_improvement.cpu().numpy(),
        "predicted_vs_actual_hm_mae": l4_pred_actual_gap.mean(dim=1).cpu().numpy(),
        "predicted_vs_actual_hm_max_error": l4_pred_actual_gap.max(dim=1).values.cpu().numpy(),
        "high_tm_relative_l1_error": actual_tm_error[:, 0].cpu().numpy(),
        "medium_tm_relative_l1_error": actual_tm_error[:, 1].cpu().numpy(),
        "low_tm_relative_l1_error": actual_tm_error[:, 2].cpu().numpy(),
        "toll_abs_mean": l4_tolls.abs().mean(dim=1).cpu().numpy(),
        "toll_abs_max": l4_tolls.abs().max(dim=1).values.cpu().numpy(),
        "low_policy_tv": l4_tv.cpu().numpy(),
        "toll_vs_predicted_hm_spearman": toll_pred_corr,
        "toll_vs_actual_hm_spearman": toll_actual_corr,
    }
    correlations = {
        name: {
            "pearson_with_actual_gain": correlation(value, actual_gain),
            "spearman_with_actual_gain": correlation(value, actual_gain, rank=True),
        }
        for name, value in proxy_fields.items()
    }

    signs = np.where(actual_gain < -1e-7, "negative", np.where(actual_gain > 1e-7, "positive", "zero"))
    strata = {}
    for label in ("negative", "positive", "zero"):
        mask = signs == label
        if not mask.any():
            continue
        strata[label] = {
            "count": int(mask.sum()),
            "gain": describe(actual_gain[mask]),
            "proxies": {name: describe(value[mask]) for name, value in proxy_fields.items()},
        }

    scale_results = {}
    scale_policies = []
    scale_scores = []
    scales = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    for scale in scales:
        candidate = candidate_from_tolls(guard_module, l4_cache, l4_tolls, scale)
        rows, _ = evaluate_cache(
            runtime,
            l4_model,
            l4_props,
            candidate,
            guard_module.LowOnlyAdapter(),
        )
        gains = gain_summary(l4_base_rows, rows)
        low_util = low_utilization(l4_cache, candidate.path_features)
        score = congestion_score(l4_pred_hm, low_util)
        scale_results[str(scale)] = {
            "gain": public_gain(gains),
            "proxy_improvement": describe((base_score - score).cpu().numpy()),
            "proxy_actual_gain_pearson": correlation(
                (base_score - score).cpu().numpy(), gains["gain_values"]
            ),
            "proxy_actual_gain_spearman": correlation(
                (base_score - score).cpu().numpy(), gains["gain_values"], rank=True
            ),
        }
        scale_policies.append(candidate.path_features)
        scale_scores.append(score)

    # Diagnostic only: choose a per-snapshot scale using the predeclared ESM
    # congestion score.  This is not proposed as an already validated result.
    score_matrix = torch.stack(scale_scores, dim=0)
    policy_matrix = torch.stack(scale_policies, dim=0)
    selected_scale_index = score_matrix.argmin(dim=0)
    snapshot_index = torch.arange(len(l4_cache))
    selected_policy = policy_matrix[selected_scale_index, snapshot_index]
    selected_cache = replace(l4_cache, path_features=selected_policy)
    selected_rows, _ = evaluate_cache(
        runtime,
        l4_model,
        l4_props,
        selected_cache,
        guard_module.LowOnlyAdapter(),
    )
    selected_gain = gain_summary(l4_base_rows, selected_rows)
    selected_counts = {
        str(scale): int((selected_scale_index == index).sum().item())
        for index, scale in enumerate(scales)
    }

    l2_report = json.loads(LEVEL2_REPORT.read_text(encoding="utf-8"))
    l4_rejection = json.loads(LEVEL4_REJECTION.read_text(encoding="utf-8"))
    official = {
        "level2_200_249": {
            "gain": l2_report["evaluation"]["diagnostics"]["Low"],
            "guard_summary": l2_report["evaluation"]["summary"]["Low"],
            "baseline_mean": float(
                l2_report["evaluation"]["summary"]["Low"]["norm_fulfill_mean"]
                - l2_report["evaluation"]["diagnostics"]["Low"]["mean_gain"]
            ),
            "toll_abs_mean": l2_report["evaluation"]["toll_abs_mean"],
            "toll_abs_max": l2_report["evaluation"]["toll_abs_max"],
        },
        "level4_300_349": {
            "gain": l4_rejection["internal_gate"]["diagnostics"]["Low"],
            "guard_summary": l4_rejection["internal_gate"]["summary"]["Low"],
            "baseline_mean": float(
                l4_rejection["internal_gate"]["summary"]["Low"]["norm_fulfill_mean"]
                - l4_rejection["internal_gate"]["diagnostics"]["Low"]["mean_gain"]
            ),
            "toll_abs_mean": l4_rejection["internal_gate"]["toll_abs_mean"],
            "toll_abs_max": l4_rejection["internal_gate"]["toll_abs_max"],
        },
    }

    report = {
        "scope": {
            "constructed_dataset_window": [START, STOP],
            "existing_rows_window": [200, 250],
            "forbidden_data_read": [350, 500],
            "strict_esm": True,
            "device": "cpu",
        },
        "input_sha256": {
            "level2_core": sha256(LEVEL2_CORE),
            "level4_core": sha256(LEVEL4_CORE),
            "level2_head": sha256(LEVEL2_HEAD),
            "level4_head": sha256(LEVEL4_HEAD),
            "persistent_runner": sha256(PERSISTENT_RUNNER),
            "strict_runner": sha256(STRICT_RUNNER),
            "edge_runner": sha256(EDGE_RUNNER),
            "guard_runner": sha256(GUARD_RUNNER),
        },
        "state_load": {
            "level2_core": l2_model_load,
            "level4_core": l4_model_load,
            "level2_head": l2_head_load,
            "level4_head": l4_head_load,
        },
        "checkpoint_contract": {
            "level2_head_max_toll": float(l2_payload["max_toll"]),
            "level4_head_max_toll": float(l4_payload["max_toll"]),
            "level2_head_selection_epoch": int(l2_payload["selection"]["epoch"]),
            "level4_head_selection_epoch": int(l4_payload["selection"]["epoch"]),
            "level2_head_safety_weight": float(l2_payload["selection"]["safety_weight"]),
            "level4_head_safety_weight": float(l4_payload["selection"]["safety_weight"]),
        },
        "strict_information_audit": {
            "level2_core": l2_info,
            "level4_core": l4_info,
            "native_hattrick": native_info,
        },
        "official_window_comparison": official,
        "common_300_349": {
            "level2_core_summary": l2_base_summary,
            "level4_core_summary": l4_base_summary,
            "native_hattrick_summary": native_summary,
            "core_head_cross": combinations,
        },
        "level4_head_mechanism": {
            "correlations": correlations,
            "gain_strata": strata,
            "scale_grid": scale_results,
            "diagnostic_esm_scale_selector": {
                "selected_scale_counts": selected_counts,
                "gain": public_gain(selected_gain),
                "warning": "In-sample 300-349 diagnosis only; not validation evidence",
            },
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    snapshot_rows = []
    for offset, snapshot in enumerate(range(START, STOP)):
        row = {
            "snapshot": snapshot,
            "level4_guard_gain": float(actual_gain[offset]),
            "gain_sign": str(signs[offset]),
        }
        row.update({name: float(value[offset]) for name, value in proxy_fields.items()})
        snapshot_rows.append(row)
    snapshot_path = OUTPUT_DIR / "level4_snapshot_proxies.csv"
    with snapshot_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(snapshot_rows[0]))
        writer.writeheader()
        writer.writerows(snapshot_rows)

    rows_path = OUTPUT_DIR / "common_window_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combination_rows[0]))
        writer.writeheader()
        writer.writerows(combination_rows)

    print(
        json.dumps(
            {
                "report": str(report_path),
                "snapshot_rows": str(snapshot_path),
                "cross_rows": str(rows_path),
                "official": official,
                "cross_gain": {
                    key: value["gain"]["gain"]["mean"]
                    for key, value in combinations.items()
                },
                "esm_selector_gain": selected_gain["gain"]["mean"],
                "esm_selector_negative_count": selected_gain["negative_count"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
