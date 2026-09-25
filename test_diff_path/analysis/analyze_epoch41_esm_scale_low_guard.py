from __future__ import annotations

"""Prediction-only self-calibration audit for the rejected Level-4 LowGuard.

The only dataset window this program is able to construct is 300--349.  The
scale selector receives ESM predictions, policies, capacities, and topology;
actual traffic and oracle values are not arguments to the selector.  Actual
traffic is used only after all per-snapshot scale choices have been frozen.
"""

import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

# Traffic snapshots were serialized by NumPy 2.x (``numpy._core``), while the
# frozen training runtime uses NumPy 1.26 (``numpy.core``).  Alias only the
# module names needed by pickle; tensor values are unchanged.
if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "test_diff_path"
PERSISTENT_DIR = TEST_DIR / "shared2x_sparse_path_cross_attention_persistent"
MODEL_RUNNER = PERSISTENT_DIR / "run_experiment.py"
GUARD_LIBRARY = PERSISTENT_DIR / "train_low_guard.py"
EDGE_RUNNER = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
STRICT_EVALUATOR = (
    TEST_DIR
    / "shared2x_sparse_path_cross_attention"
    / "evaluate_strict_esm_sequential.py"
)
CORE = (
    PERSISTENT_DIR
    / "level4_selection_validation_only"
    / "selected_checkpoint.pt"
)
HEAD = (
    PERSISTENT_DIR
    / "artifacts"
    / "level4_low_guard_frozen"
    / "frozen_low_guard_prevalidation.pt"
)
PRECOMMIT = PERSISTENT_DIR / "level4_precommit_manifest.json"
OUTPUT_DIR = (
    PERSISTENT_DIR
    / "artifacts"
    / "analysis_epoch41_esm_scale_low_guard_300_349"
)

START = 300
STOP = 350
K = 8
SCALES = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
EXPECTED_CORE_SHA256 = (
    "e8f7df484aff6e13fc45bbd26a055deb368e90cb9a1bb1167b1f72ac3416fe14"
)
EXPECTED_HEAD_SHA256 = (
    "8282edf00b2624a7831c6cfa3054012319a5da581f674e6f449c21a9f6271d86"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def precommit_snapshot() -> tuple[dict, dict[str, str]]:
    manifest = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
    items = manifest.get("files", [])
    if manifest.get("test_data_read") is not False or len(items) != 27:
        raise RuntimeError("Unexpected frozen precommit manifest")
    expected = {str(Path(item["path"]).resolve()): item["sha256"] for item in items}
    actual = {path: sha256(Path(path)) for path in expected}
    changed = {path: [expected[path], value] for path, value in actual.items() if value != expected[path]}
    if changed:
        raise RuntimeError(f"Frozen precommit input changed before audit: {changed}")
    return manifest, actual


def pte_info(cache):
    pte = cache.dataset.pte.coalesce()
    indices = pte.indices()
    return pte, indices[0], indices[1], pte.values()


def build_scaled_low_policies(guard, head, policies, features, pte, batch_size=25):
    """Build candidates without exposing actual traffic to this function."""
    chunks = [[] for _ in SCALES]
    toll_chunks = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            toll = head(features[start:stop])
            toll_chunks.append(toll)
            base_low = policies[2][start:stop].squeeze(-1)
            for index, scale in enumerate(SCALES):
                if scale == 0.0:
                    low = base_low
                else:
                    low = guard.route_low(base_low, toll * float(scale), pte)
                chunks[index].append(low.detach())
    return (
        tuple(torch.cat(values, dim=0) for values in chunks),
        torch.cat(toll_chunks, dim=0).detach(),
    )


def predicted_low_admitted_fraction(
    model,
    props,
    policies,
    predicted_tms,
    capacities,
    topology,
    batch_size=25,
):
    """ESM-only sequential admission score, one value per snapshot.

    Deliberately absent from the signature: actual TMs and oracle flows/MLUs.
    """
    values = []
    with torch.no_grad():
        for start in range(0, len(capacities), batch_size):
            stop = min(start + batch_size, len(capacities))
            batch_policies = [value[start:stop] for value in policies]
            batch_predictions = [value[start:stop] for value in predicted_tms]
            ratios = model.simulate(
                batch_policies,
                batch_predictions,
                capacities[start:stop],
                topology,
                stop - start,
                props,
                rate_cap=props.rate_cap,
            )[:3]
            low_prediction = batch_predictions[2].squeeze(-1)
            admitted = ratios[2].reshape(stop - start, -1) * low_prediction
            # Dataset demand is repeated once per candidate path.
            demand = low_prediction.sum(dim=1) / K
            values.append(admitted.sum(dim=1) / demand.clamp_min(1e-9))
    return torch.cat(values, dim=0)


def select_prediction_only(
    model,
    props,
    policies,
    candidate_lows,
    predicted_tms,
    capacities,
    topology,
):
    """Choose the first (smallest) scale attaining the largest ESM score."""
    columns = []
    for low in candidate_lows:
        columns.append(
            predicted_low_admitted_fraction(
                model,
                props,
                (policies[0], policies[1], low.unsqueeze(-1)),
                predicted_tms,
                capacities,
                topology,
            )
        )
    scores = torch.stack(columns, dim=1)
    # torch.argmax returns the first index on an exact tie. SCALES is ascending,
    # so the predeclared tie-break is minimum scale.
    choice = scores.argmax(dim=1)
    stacked = torch.stack(candidate_lows, dim=1)
    gather_index = choice.reshape(-1, 1, 1).expand(-1, 1, stacked.shape[-1])
    selected = stacked.gather(1, gather_index).squeeze(1)
    return selected, scores, choice


class LowOnlyAdapter:
    def adapt_batch(self, policies, batch):
        policies[2] = batch["path_features"].unsqueeze(-1)
        return policies


def evaluate_low(runtime, model, props, cache, low=None):
    if low is None:
        rows, summary = runtime.evaluate_cache(
            model, props, cache, None, batch_size=25
        )
    else:
        candidate = replace(cache, path_features=low)
        rows, summary = runtime.evaluate_cache(
            model, props, candidate, LowOnlyAdapter(), batch_size=25
        )
    indexed = runtime.summary_index(summary)
    absolute = {
        key: float(indexed["Low"][f"norm_fulfill_{key}"])
        for key in ("mean", "p1", "p10")
    }
    return rows, indexed, absolute


def low_values(rows):
    return np.asarray(
        [float(row["norm_fulfill"]) for row in rows if row["class"] == "Low"],
        dtype=np.float64,
    )


def policy_preservation(cache, low):
    before = cache.policies
    after = (before[0], before[1], low.unsqueeze(-1))
    return {
        name: {
            "torch_equal": bool(torch.equal(before[index], after[index])),
            "same_storage": bool(
                before[index].untyped_storage().data_ptr()
                == after[index].untyped_storage().data_ptr()
            ),
            "max_abs_delta": float((before[index] - after[index]).abs().max().item()),
        }
        for index, name in enumerate(("High", "Medium"))
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("Refusing to write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_ecdf_plot(path: Path, values: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "Core epoch41": "#2f7ed8",
        "Full LowGuard": "#e6550d",
        "ESM-selected": "#31a354",
    }
    styles = {"Core epoch41": "-", "Full LowGuard": "--", "ESM-selected": "-."}
    figure, axis = plt.subplots(figsize=(8.4, 5.4), dpi=160)
    for label, sample in values.items():
        ordered = np.sort(sample)
        y = np.arange(1, len(ordered) + 1, dtype=np.float64) / len(ordered)
        axis.step(
            ordered,
            y,
            where="post",
            label=label,
            color=colors[label],
            linestyle=styles[label],
            linewidth=2.2,
        )
    lo = min(float(sample.min()) for sample in values.values())
    hi = max(float(sample.max()) for sample in values.values())
    margin = max(0.01, 0.08 * (hi - lo))
    axis.set_xlim(lo - margin, hi + margin)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Low NormFulFill")
    axis.set_ylabel("CDF")
    axis.set_title("Prediction-only LowGuard scale audit (snapshots 300–349)")
    axis.grid(True, color="#d9dee7", linewidth=0.7, alpha=0.85)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    if (START, STOP) != (300, 350) or STOP > 350:
        raise RuntimeError("This audit is hard-limited to snapshots 300--349")
    if OUTPUT_DIR.exists():
        raise RuntimeError(f"Refusing to overwrite prior audit output: {OUTPUT_DIR}")

    precommit, frozen_before = precommit_snapshot()
    core_sha = sha256(CORE)
    head_sha = sha256(HEAD)
    if core_sha != EXPECTED_CORE_SHA256 or head_sha != EXPECTED_HEAD_SHA256:
        raise RuntimeError("Frozen core/head SHA256 mismatch")

    model_module = load_module("epoch41_esm_scale_model", MODEL_RUNNER)
    guard = load_module("epoch41_esm_scale_guard", GUARD_LIBRARY)
    edge = load_module("epoch41_esm_scale_edge", EDGE_RUNNER)
    strict = load_module("epoch41_esm_scale_strict", STRICT_EVALUATOR)
    runtime = edge.runtime
    runtime.set_seed(20260824)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)

    core_payload = torch.load(CORE, map_location=device, weights_only=False)
    head_payload = torch.load(HEAD, map_location=device, weights_only=False)
    if int(core_payload.get("epoch", -1)) != 41:
        raise RuntimeError("Core is not the frozen epoch-41 checkpoint")
    if head_payload.get("backbone_sha256") != core_sha:
        raise RuntimeError("LowGuard was not trained against this core")
    if head_payload.get("test_data_read") is not False:
        raise RuntimeError("LowGuard checkpoint is not test-blind")

    model = model_module.PersistentStage2SparseAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    model.load_state_dict(core_payload["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Enforce the permitted split at the dataset constructor boundary.
    dataset_calls = []
    original_dataset_constructor = runtime.DM_Dataset_within_Cluster

    def split_guarded_dataset(current_props, mode, start, end, *args, **kwargs):
        start = int(start)
        end = int(end)
        if start < START or end > STOP or start >= end:
            raise RuntimeError(f"Forbidden dataset construction attempted: [{start}, {end})")
        dataset_calls.append([start, end])
        return original_dataset_constructor(
            current_props, mode, start, end, *args, **kwargs
        )

    runtime.DM_Dataset_within_Cluster = split_guarded_dataset
    cache, strict_audit = strict.build_strict_policy_cache(
        runtime,
        model,
        props,
        START,
        STOP,
        25,
        actual_input_mode="zero",
    )
    if dataset_calls != [[START, STOP]]:
        raise RuntimeError(f"Unexpected dataset calls: {dataset_calls}")
    if int(cache.dataset.max_source_index_read) != 349:
        raise RuntimeError("Reader exceeded or failed to cover the declared window")

    head = guard.LowGuard(
        int(head_payload["edge_count"]),
        int(head_payload["feature_count"]),
        float(head_payload["max_toll"]),
    ).to(device)
    head.load_state_dict(head_payload["state_dict"], strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    # Construct a policy-only view with all actual and oracle tensors erased.
    erased = replace(
        cache,
        tms=tuple(torch.zeros_like(value) for value in cache.tms),
        oracle_flows=tuple(torch.zeros_like(value) for value in cache.oracle_flows),
        oracle_mlus=tuple(torch.zeros_like(value) for value in cache.oracle_mlus),
    )
    features = edge.edge_features(erased)
    pte = erased.dataset.pte.coalesce().to(dtype=torch.float32)
    candidate_lows, tolls = build_scaled_low_policies(
        guard, head, erased.policies, features, pte
    )
    selected_low, predicted_scores, choices = select_prediction_only(
        model,
        props,
        erased.policies,
        candidate_lows,
        erased.predicted_tms,
        erased.capacities,
        pte_info(erased),
    )
    selected_scale = torch.tensor(
        [SCALES[int(index)] for index in choices.detach().cpu().tolist()],
        device=device,
    )

    # Freeze choices before the first actual-TM admission replay.
    frozen_choice = choices.detach().cpu().clone()
    frozen_selected_low = selected_low.detach().clone()
    baseline_rows, baseline_summary, baseline_low = evaluate_low(
        runtime, model, props, cache
    )
    full_rows, full_summary, full_low = evaluate_low(
        runtime, model, props, cache, candidate_lows[-1]
    )
    selected_rows, selected_summary, selected_low_metrics = evaluate_low(
        runtime, model, props, cache, frozen_selected_low
    )
    if not torch.equal(choices.detach().cpu(), frozen_choice):
        raise RuntimeError("Scale choices changed after actual-only scoring")

    full_diag = edge.diagnostics(baseline_rows, full_rows)
    selected_diag = edge.diagnostics(baseline_rows, selected_rows)
    full_preservation = policy_preservation(cache, candidate_lows[-1])
    selected_preservation = policy_preservation(cache, frozen_selected_low)
    if not all(
        item["torch_equal"] and item["same_storage"]
        for audit in (full_preservation, selected_preservation)
        for item in audit.values()
    ):
        raise RuntimeError("Low-only candidate changed High or Medium policy")

    # Scale=1 must exactly match the frozen head's standard implementation.
    standard_cache, standard_tolls = guard.build_candidate_cache(
        head, erased, features, batch_size=25
    )
    scale_one_equivalence = {
        "low_policy_torch_equal": bool(
            torch.equal(candidate_lows[-1], standard_cache.path_features)
        ),
        "low_policy_max_abs_delta": float(
            (candidate_lows[-1] - standard_cache.path_features).abs().max().item()
        ),
        "tolls_torch_equal": bool(torch.equal(tolls, standard_tolls)),
        "tolls_max_abs_delta": float((tolls - standard_tolls).abs().max().item()),
    }
    if not all(
        scale_one_equivalence[key]
        for key in ("low_policy_torch_equal", "tolls_torch_equal")
    ):
        raise RuntimeError("Scale=1 does not reproduce the frozen LowGuard")

    # Offline actual-TM scores for each fixed candidate are diagnostic only and
    # occur after the prediction-only choices were frozen.
    fixed_scale_actual = {}
    for scale, low in zip(SCALES, candidate_lows):
        rows, _, absolute = evaluate_low(runtime, model, props, cache, low)
        fixed_scale_actual[str(scale)] = {
            "absolute": absolute,
            "diagnostics_vs_core": edge.diagnostics(baseline_rows, rows)["Low"],
        }

    base_values = low_values(baseline_rows)
    full_values = low_values(full_rows)
    selected_values = low_values(selected_rows)
    base_by_snapshot = {
        int(row["snapshot"]): float(row["norm_fulfill"])
        for row in baseline_rows
        if row["class"] == "Low"
    }
    full_by_snapshot = {
        int(row["snapshot"]): float(row["norm_fulfill"])
        for row in full_rows
        if row["class"] == "Low"
    }
    selected_by_snapshot = {
        int(row["snapshot"]): float(row["norm_fulfill"])
        for row in selected_rows
        if row["class"] == "Low"
    }

    snapshot_rows = []
    score_cpu = predicted_scores.detach().cpu().numpy()
    choices_cpu = frozen_choice.numpy()
    for local, snapshot in enumerate(range(START, STOP)):
        row = {
            "snapshot": snapshot,
            "selected_scale": SCALES[int(choices_cpu[local])],
            "predicted_selected_low_admitted_fraction": float(
                score_cpu[local, choices_cpu[local]]
            ),
            "core_low_norm_fulfill": base_by_snapshot[snapshot],
            "full_guard_low_norm_fulfill": full_by_snapshot[snapshot],
            "esm_selected_low_norm_fulfill": selected_by_snapshot[snapshot],
            "full_minus_core": full_by_snapshot[snapshot] - base_by_snapshot[snapshot],
            "selected_minus_core": selected_by_snapshot[snapshot]
            - base_by_snapshot[snapshot],
        }
        for column, scale in enumerate(SCALES):
            row[f"predicted_score_scale_{scale:g}"] = float(score_cpu[local, column])
        snapshot_rows.append(row)

    ecdf_grid = np.unique(np.concatenate((base_values, full_values, selected_values)))
    ecdf_rows = []
    for value in ecdf_grid:
        ecdf_rows.append(
            {
                "low_norm_fulfill": float(value),
                "core_cdf": float(np.searchsorted(np.sort(base_values), value, side="right") / len(base_values)),
                "full_guard_cdf": float(np.searchsorted(np.sort(full_values), value, side="right") / len(full_values)),
                "esm_selected_cdf": float(np.searchsorted(np.sort(selected_values), value, side="right") / len(selected_values)),
            }
        )

    frozen_after = {path: sha256(Path(path)) for path in frozen_before}
    frozen_unchanged = frozen_before == frozen_after
    if not frozen_unchanged:
        raise RuntimeError("A frozen precommit file changed during the audit")

    scale_counts = Counter(float(value) for value in selected_scale.detach().cpu().tolist())
    report = {
        "status": "completed",
        "scope": {
            "window": [START, STOP],
            "maximum_source_index_read": int(cache.dataset.max_source_index_read),
            "dataset_constructor_calls": dataset_calls,
            "forbidden_window_constructed_or_read": False,
            "device": str(device),
        },
        "frozen_inputs": {
            "core": {"path": str(CORE.resolve()), "sha256": core_sha, "epoch": 41},
            "low_guard": {
                "path": str(HEAD.resolve()),
                "sha256": head_sha,
                "internal_gate_passed": bool(head_payload["internal_gate_passed"]),
            },
            "precommit_declared_file_count": len(precommit["files"]),
            "all_27_hashes_match_before_and_after": frozen_unchanged,
        },
        "selection_protocol": {
            "candidate_scales": list(SCALES),
            "objective": "maximum predicted-TM sequential-admission Low admitted fraction per snapshot",
            "tie_break": "smallest scale (first ascending argmax)",
            "selector_inputs": [
                "strict-ESM policies",
                "ESM predicted TMs",
                "capacities",
                "path-to-edge topology",
                "frozen LowGuard tolls",
            ],
            "selector_forbidden_inputs": ["actual TMs", "oracle flows", "oracle MLUs"],
            "actual_timing": "used only after scale choices and selected Low tensors were frozen",
            "selected_scale_counts": {str(scale): int(scale_counts.get(scale, 0)) for scale in SCALES},
            "selected_scale_mean": float(selected_scale.float().mean().item()),
        },
        "strict_esm_audit": strict_audit,
        "scale_one_equivalence": scale_one_equivalence,
        "policy_preservation": {
            "full_guard": full_preservation,
            "esm_selected": selected_preservation,
        },
        "results": {
            "core": {"absolute_low": baseline_low},
            "full_guard": {
                "absolute_low": full_low,
                "diagnostics_vs_core": full_diag["Low"],
            },
            "esm_selected": {
                "absolute_low": selected_low_metrics,
                "diagnostics_vs_core": selected_diag["Low"],
            },
        },
        "predicted_proxy": {
            "mean_by_fixed_scale": {
                str(scale): float(predicted_scores[:, index].mean().item())
                for index, scale in enumerate(SCALES)
            },
            "mean_selected": float(
                predicted_scores.gather(1, choices.reshape(-1, 1)).mean().item()
            ),
        },
        "offline_fixed_scale_actual_diagnostic": fixed_scale_actual,
        "artifacts": {
            "per_snapshot_csv": "per_snapshot.csv",
            "ecdf_csv": "low_ecdf.csv",
            "ecdf_png": "low_ecdf.png",
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    write_csv(OUTPUT_DIR / "per_snapshot.csv", snapshot_rows)
    write_csv(OUTPUT_DIR / "low_ecdf.csv", ecdf_rows)
    write_ecdf_plot(
        OUTPUT_DIR / "low_ecdf.png",
        {
            "Core epoch41": base_values,
            "Full LowGuard": full_values,
            "ESM-selected": selected_values,
        },
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
