from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import torch

# The traffic pickles were produced by NumPy 2.x.  NumPy 1.x needs the private
# package alias below solely to unpickle them.
if int(np.__version__.split(".", 1)[0]) < 2:
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "test_diff_path"
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
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> float:
    return float((values * weights).sum().item() / weights.sum().item())


def weighted_quantile(values: torch.Tensor, weights: torch.Tensor, q: float) -> float:
    x = values.reshape(-1).detach().cpu().numpy().astype(np.float64)
    w = weights.reshape(-1).detach().cpu().numpy().astype(np.float64)
    order = np.argsort(x, kind="stable")
    x = x[order]
    w = w[order]
    cumulative = np.cumsum(w)
    target = q * cumulative[-1]
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(x) - 1)
    return float(x[index])


def route_nodes(source, path) -> str:
    nodes = [source]
    nodes.extend(edge[1] for edge in path)
    return "→".join(str(node) for node in nodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=400)
    parser.add_argument("--end", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

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
        "native_wh_refinement_runtime",
        TEST_DIR / "shared2x_medium_adapter" / "run_experiment.py",
    )
    dote_runtime = load_module(
        "native_wh_refinement_dote_runtime",
        TEST_DIR / "run_dotemc_priority_mask_experiment.py",
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    props = runtime.build_props(4, device)
    model, backbone = runtime.load_backbone(4, 490, props, device)
    dote_model, dote_mean, dote_std = dote_runtime.load_checkpoint(
        DOTE_CHECKPOINT, device
    )
    model.eval()
    dote_model.eval()

    dataset = runtime.DM_Dataset_within_Cluster(props, 0, args.start, args.end)
    path_masks = runtime.shared.base.move_dataset_static(dataset, props.device)
    loader = runtime.shared.data_loader(dataset, args.batch_size, False, 0)
    pairs = list(dataset.pij.keys())
    num_pairs = int(dataset.num_pairs)
    paths_per_od = int(props.num_paths_per_pair)

    captures: dict[str, list[torch.Tensor]] = {}
    original_forward_pass = model.forward_pass_mlp
    tracked = ("mlp11", "mlp12", "mlp21", "mlp22", "mlp31", "mlp32")

    def recording_forward_pass(
        self, inputs, mlp, num_hidden_layers, violation_data=None
    ):
        output = original_forward_pass(
            inputs, mlp, num_hidden_layers, violation_data=violation_data
        )
        for name in tracked:
            if mlp is getattr(self, name):
                captures.setdefault(name, []).append(output.detach().clone())
                break
        return output

    model.forward_pass_mlp = types.MethodType(recording_forward_pass, model)

    stage_chunks: dict[str, list[torch.Tensor]] = {}
    dote_chunks: list[torch.Tensor] = []
    demand_chunks: list[torch.Tensor] = []
    stage_order: list[str] | None = None
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_policy = True

    with torch.no_grad():
        for inputs in loader:
            captures.clear()
            values = runtime.shared.unpack_to_device(inputs, props)
            returned = runtime.cached_policy_forward(
                model, props, dataset, values, path_masks
            )
            batch = int(values[2].shape[0])

            gamma = captures["mlp11"][0]
            for delta in captures.get("mlp12", []):
                gamma = gamma + delta

            batch_stages: list[tuple[str, torch.Tensor]] = [("Stage 1", gamma)]

            gamma = gamma + captures["mlp21"][0][:, :, :1]
            batch_stages.append(("Stage 2 init", gamma))
            for index, delta in enumerate(captures.get("mlp22", []), start=1):
                gamma = gamma + delta[:, :, :1]
                batch_stages.append((f"Stage 2 RAU {index}", gamma))

            gamma = gamma + captures["mlp31"][0][:, :, :1]
            batch_stages.append(("Stage 3 init", gamma))
            for index, delta in enumerate(captures.get("mlp32", []), start=1):
                gamma = gamma + delta[:, :, :1]
                batch_stages.append((f"Stage 3 RAU {index}", gamma))

            labels = [label for label, _ in batch_stages]
            if stage_order is None:
                stage_order = labels
            elif labels != stage_order:
                raise RuntimeError("The number of refinement steps changed across batches")

            for label, logits in batch_stages:
                policy = torch.softmax(
                    logits.reshape(batch, num_pairs, paths_per_od), dim=-1
                ).float()
                stage_chunks.setdefault(label, []).append(policy.detach())

            final_policy = (
                returned[0]
                .squeeze(-1)
                .reshape(batch, num_pairs, paths_per_od)
                .float()
            )
            reconstruction_delta = float(
                (stage_chunks[labels[-1]][-1] - final_policy).abs().max().item()
            )
            if reconstruction_delta > 2e-6:
                raise RuntimeError("Reconstructed final High policy does not match forward")

            features = dote_runtime.make_inputs(
                values[3].float(),
                values[5].float(),
                values[7].float(),
                num_pairs,
                dote_mean,
                dote_std,
            )
            dote_chunks.append(dote_model(features, None)[:, 0].detach().float())
            demand_chunks.append(
                values[3]
                .reshape(batch, num_pairs, paths_per_od, 1)[:, :, 0, 0]
                .detach()
                .float()
                .clamp_min(0.0)
            )

    props.research_return_policy = False
    stage_policies = {
        label: torch.cat(chunks, dim=0).cpu()
        for label, chunks in stage_chunks.items()
    }
    dote_policy = torch.cat(dote_chunks, dim=0).cpu()
    demand = torch.cat(demand_chunks, dim=0).cpu().clamp_min(1e-12)
    assert stage_order is not None

    comparisons: list[dict] = []
    comparison_values: dict[str, torch.Tensor] = {}
    for left, right in zip(stage_order, stage_order[1:]):
        label = f"{left} → {right}"
        comparison_values[label] = 0.5 * (
            stage_policies[left] - stage_policies[right]
        ).abs().sum(dim=-1)
    total_label = "Stage 1 → Hattrick final"
    gap_label = "Hattrick final → DOTE-MC"
    comparison_values[total_label] = 0.5 * (
        stage_policies[stage_order[0]] - stage_policies[stage_order[-1]]
    ).abs().sum(dim=-1)
    comparison_values[gap_label] = 0.5 * (
        stage_policies[stage_order[-1]] - dote_policy
    ).abs().sum(dim=-1)

    cdf_thresholds = torch.linspace(0.0, 1.0, 101)
    for label, values in comparison_values.items():
        left_name, right_name = label.split(" → ")
        if left_name == "Hattrick final":
            left_policy, right_policy = stage_policies[stage_order[-1]], dote_policy
        elif label == total_label:
            left_policy = stage_policies[stage_order[0]]
            right_policy = stage_policies[stage_order[-1]]
        else:
            left_policy = stage_policies[left_name]
            right_policy = stage_policies[right_name]
        argmax_changed = (left_policy.argmax(-1) != right_policy.argmax(-1)).float()
        comparisons.append(
            {
                "label": label,
                "weighted_mean_tv": weighted_mean(values, demand),
                "weighted_median_tv": weighted_quantile(values, demand, 0.5),
                "weighted_p90_tv": weighted_quantile(values, demand, 0.9),
                "weighted_argmax_change": weighted_mean(argmax_changed, demand),
                "cdf": [
                    [
                        round(float(threshold), 3),
                        weighted_mean((values <= threshold).float(), demand),
                    ]
                    for threshold in cdf_thresholds
                ],
            }
        )

    total_move = comparison_values[total_label]
    dote_gap = comparison_values[gap_label]
    final_argmax = stage_policies[stage_order[-1]].argmax(-1)
    dote_argmax = dote_policy.argmax(-1)
    candidate = (
        (total_move <= 0.15)
        & (dote_gap >= 0.60)
        & (final_argmax != dote_argmax)
    )
    score = demand * dote_gap * candidate.float()
    selected_flat = torch.argsort(score.reshape(-1), descending=True).tolist()

    chosen: list[tuple[int, int]] = []
    pair_lookup = {tuple(pair): index for index, pair in enumerate(pairs)}
    known_pair = pair_lookup.get((9, 18))
    if known_pair is not None and args.start <= 464 < args.end:
        chosen.append((464 - args.start, known_pair))
    for flat in selected_flat:
        if float(score.reshape(-1)[flat]) <= 0.0:
            break
        snapshot_index, pair_index = divmod(flat, num_pairs)
        key = (snapshot_index, pair_index)
        if key not in chosen:
            chosen.append(key)
        if len(chosen) >= 4:
            break

    examples = []
    for snapshot_index, pair_index in chosen:
        source, destination = pairs[pair_index]
        paths = dataset.pij[(source, destination)]
        distributions = []
        for label in stage_order:
            distributions.append(
                {
                    "stage": label,
                    "probabilities": [
                        round(float(value), 7)
                        for value in stage_policies[label][snapshot_index, pair_index]
                    ],
                }
            )
        distributions.append(
            {
                "stage": "DOTE-MC",
                "probabilities": [
                    round(float(value), 7)
                    for value in dote_policy[snapshot_index, pair_index]
                ],
            }
        )
        examples.append(
            {
                "id": f"{args.start + snapshot_index}:{source}→{destination}",
                "snapshot": args.start + snapshot_index,
                "od_index": pair_index,
                "od": f"{source}→{destination}",
                "esm_high_demand": float(demand[snapshot_index, pair_index]),
                "stage1_to_final_tv": float(total_move[snapshot_index, pair_index]),
                "final_to_dote_tv": float(dote_gap[snapshot_index, pair_index]),
                "hattrick_argmax": int(final_argmax[snapshot_index, pair_index]),
                "dote_argmax": int(dote_argmax[snapshot_index, pair_index]),
                "paths": [route_nodes(source, path) for path in paths],
                "distributions": distributions,
            }
        )

    payload = {
        "title": "Hattrick High path refinement versus DOTE-MC",
        "window": [args.start, args.end - 1],
        "load": "2x",
        "inference": "strict ESM prediction-only",
        "snapshot_count": args.end - args.start,
        "od_count": num_pairs,
        "paths_per_od": paths_per_od,
        "stage_order": stage_order,
        "comparisons": comparisons,
        "examples": examples,
        "artifacts": {
            "hattrick_checkpoint": str(backbone.resolve()),
            "dote_checkpoint": str(DOTE_CHECKPOINT.resolve()),
        },
        "audit": {
            "stage_count": len(stage_order),
            "hattrick_probability_sum_max_error": float(
                max(
                    (policy.sum(-1) - 1.0).abs().max().item()
                    for policy in stage_policies.values()
                )
            ),
            "dote_probability_sum_max_error": float(
                (dote_policy.sum(-1) - 1.0).abs().max().item()
            ),
            "demand_weighting": "ESM predicted High demand",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "stage1_to_final_mean_tv": next(
                    item["weighted_mean_tv"]
                    for item in comparisons
                    if item["label"] == total_label
                ),
                "final_to_dote_mean_tv": next(
                    item["weighted_mean_tv"]
                    for item in comparisons
                    if item["label"] == gap_label
                ),
                "examples": [item["id"] for item in examples],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
