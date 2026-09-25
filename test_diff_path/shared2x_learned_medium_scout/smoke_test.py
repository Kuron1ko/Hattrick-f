from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "shared2x_learned_medium_scout_smoke_runtime",
    THIS_DIR / "run_experiment.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load learned Medium scout runner")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


def compare_policies(expected, actual, label: str, require_equal: bool = True) -> dict:
    result = {}
    for class_name, before, after in zip(
        ("High", "Medium", "Low"), expected, actual
    ):
        difference = (before - after).abs()
        row = {
            "torch_equal": bool(torch.equal(before, after)),
            "different_elements": int(torch.count_nonzero(before != after).item()),
            "max_abs_delta": float(difference.max().item()),
        }
        if require_equal and not row["torch_equal"]:
            raise RuntimeError(f"{label}: {class_name} differs")
        result[class_name] = row
    return result


def permute_actual_traffic(values, num_ods: int, paths_per_od: int):
    changed = list(values)
    total_l1 = 0.0
    for index in (2, 4, 6):
        actual = values[index]
        grouped = actual.reshape(actual.shape[0], num_ods, paths_per_od, -1)
        permuted = torch.flip(grouped, dims=(1,)).reshape_as(actual)
        changed[index] = permuted
        total_l1 += float((actual - permuted).abs().sum().item())
    if total_l1 == 0.0:
        raise RuntimeError("Actual traffic permutation changed nothing")
    return tuple(changed), total_l1


def timed_policy(model, props, dataset, values, path_masks, runs: int) -> list[float]:
    elapsed = []
    with torch.no_grad():
        for _ in range(runs):
            started = time.perf_counter()
            experiment.full.shared.model_forward(
                model, props, dataset, values, path_masks
            )
            elapsed.append(time.perf_counter() - started)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU architecture, strict-input, gradient, and latency smoke"
    )
    parser.add_argument("--snapshot", type=int, default=350)
    parser.add_argument("--timing-runs", type=int, default=5)
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = torch.device("cpu")
    props = experiment.full.shared.build_props(4, device)
    props.checkpoint = 0
    dataset = experiment.full.DM_Dataset_within_Cluster(
        props, 0, args.snapshot, args.snapshot + 1
    )
    path_masks = experiment.full.shared.base.move_dataset_static(dataset, device)
    loader = experiment.full.shared.data_loader(
        dataset, batch_size=1, shuffle=False, seed=490
    )
    values = experiment.full.shared.unpack_to_device(next(iter(loader)), props)
    pte = dataset.pte.coalesce()
    indices = pte.indices()
    pte_info = (pte, indices[0], indices[1], pte.values())
    total_paths, num_edges = int(pte.shape[0]), int(pte.shape[1])
    paths_per_od = int(props.num_paths_per_pair)
    num_ods = total_paths // paths_per_od

    # Copy native weights exactly; only scout modules and zero-output adapters remain new.
    torch.manual_seed(20260824)
    native = experiment.NativeHattrick(props).to(device=device, dtype=props.dtype)
    torch.manual_seed(17)
    candidate = experiment.LearnedMediumScoutHattrick(props).to(
        device=device, dtype=props.dtype
    )
    loaded = candidate.load_state_dict(native.state_dict(), strict=False)
    native_parameters = dict(native.named_parameters())
    candidate_parameters = dict(candidate.named_parameters())
    if not all(
        torch.equal(parameter, candidate_parameters[name])
        for name, parameter in native_parameters.items()
    ):
        raise RuntimeError("Native parameter copy is not exact")
    adapters = (
        candidate.medium_pressure_init_adapter,
        candidate.medium_pressure_rau_adapter,
    )
    if any(
        int(torch.count_nonzero(adapter.body[2].weight).item()) != 0
        or int(torch.count_nonzero(adapter.body[2].bias).item()) != 0
        for adapter in adapters
    ):
        raise RuntimeError("A residual output layer is not initialized to zero")

    props.mode = "test"
    props.research_return_policy = True
    props.research_return_admitted = False
    props.sim_mf_mlu = 0
    native.eval()
    candidate.eval()
    with torch.no_grad():
        native_policy, _ = experiment.full.shared.model_forward(
            native, props, dataset, values, path_masks
        )
        zero_output_policy, _ = experiment.full.shared.model_forward(
            candidate, props, dataset, values, path_masks
        )
    native_equivalence = compare_policies(
        native_policy, zero_output_policy, "zero-output native equivalence"
    )

    # Direct feature gradient audit. The cached topology embedding is constant
    # here, while demand and both scout MLPs remain differentiable.
    candidate.zero_grad(set_to_none=True)
    tm2_probe = values[5].detach().clone().requires_grad_(True)
    features = candidate.compute_medium_pressure_features(
        tm2_probe,
        values[1],
        pte,
        pte_info,
        batch_size=1,
        num_paths_per_pair=paths_per_od,
        props=props,
        path_masks=path_masks,
    )
    if tuple(features.shape) != (1, total_paths, experiment.SCOUT_FEATURE_COUNT):
        raise RuntimeError(f"Unexpected feature shape {tuple(features.shape)}")
    feature_loss = features.square().mean()
    feature_loss.backward()
    if tm2_probe.grad is None or not torch.isfinite(tm2_probe.grad).all().item():
        raise RuntimeError("Missing/non-finite tm2_pred feature gradient")
    scout_gradient_l1 = 0.0
    for module in (candidate.medium_scout_init, candidate.medium_scout_rau):
        for parameter in module.parameters():
            if parameter.grad is not None:
                scout_gradient_l1 += float(parameter.grad.abs().sum().item())
    if scout_gradient_l1 == 0.0:
        raise RuntimeError("Learned scout MLP gradient is zero")

    split = candidate._last_scout_split_ratios.reshape(
        1, num_ods, paths_per_od
    )
    simplex_error = float((split.sum(dim=-1) - 1.0).abs().max().item())
    if simplex_error > 1e-6:
        raise RuntimeError(f"Provisional Medium simplex error {simplex_error}")
    relative = features[..., (1, 3)].reshape(
        1, num_ods, paths_per_od, 2
    )
    relative_center_error = float(relative.sum(dim=2).abs().max().item())
    if relative_center_error > 1e-5:
        raise RuntimeError(
            f"Within-OD relative feature center error {relative_center_error}"
        )

    # Open both output layers to prove the scout can affect policy, then replace every
    # actual TM while retaining predictions to enforce the strict contract.
    with torch.no_grad():
        for adapter in adapters:
            adapter.body[2].weight.fill_(0.01)
        active_policy, _ = experiment.full.shared.model_forward(
            candidate, props, dataset, values, path_masks
        )
        changed_values, actual_permutation_l1 = permute_actual_traffic(
            values, num_ods, paths_per_od
        )
        changed_actual_policy, _ = experiment.full.shared.model_forward(
            candidate, props, dataset, changed_values, path_masks
        )
    active_delta = compare_policies(
        zero_output_policy, active_policy, "nonzero-output activation", require_equal=False
    )
    if max(row["max_abs_delta"] for row in active_delta.values()) == 0.0:
        raise RuntimeError("Opening scout residual outputs did not affect policy")
    strict_actual = compare_policies(
        active_policy,
        changed_actual_policy,
        "strict actual-TM replacement",
    )

    # Fair cached-transformer CPU latency: both models are warmed and timed in
    # alternating order after static topology caches exist.
    with torch.no_grad():
        for adapter in adapters:
            adapter.body[2].weight.zero_()
            adapter.body[2].bias.zero_()
        experiment.full.shared.model_forward(native, props, dataset, values, path_masks)
        experiment.full.shared.model_forward(candidate, props, dataset, values, path_masks)
    native_seconds = timed_policy(
        native, props, dataset, values, path_masks, args.timing_runs
    )
    candidate_seconds = timed_policy(
        candidate, props, dataset, values, path_masks, args.timing_runs
    )
    native_median = statistics.median(native_seconds)
    candidate_median = statistics.median(candidate_seconds)
    latency_ratio = candidate_median / max(native_median, 1e-12)
    if latency_ratio > 2.0:
        raise RuntimeError(f"CPU inference latency ratio {latency_ratio:.3f} > 2")

    added_parameters = {
        name: parameter.numel()
        for name, parameter in candidate_parameters.items()
        if name not in native_parameters
    }
    result = {
        "status": "ok",
        "device": "cpu",
        "snapshot": args.snapshot,
        "strict_input": "ESM tm2_pred; actual TM excluded from scout/policy",
        "actual_permutation_l1": actual_permutation_l1,
        "forward_is_inherited_native": (
            experiment.LearnedMediumScoutHattrick.forward
            is experiment.NativeHattrick.forward
        ),
        "native_parameters": sum(p.numel() for p in native.parameters()),
        "candidate_parameters": sum(p.numel() for p in candidate.parameters()),
        "added_parameters": sum(added_parameters.values()),
        "added_parameter_tensors": added_parameters,
        "missing_state_keys": list(loaded.missing_keys),
        "unexpected_state_keys": list(loaded.unexpected_keys),
        "feature_shape": list(features.shape),
        "feature_min": float(features.detach().min().item()),
        "feature_max": float(features.detach().max().item()),
        "feature_mean": float(features.detach().mean().item()),
        "relative_center_max_error": relative_center_error,
        "provisional_split_simplex_max_error": simplex_error,
        "tm2_pred_gradient_l1": float(tm2_probe.grad.abs().sum().item()),
        "scout_mlp_gradient_l1": scout_gradient_l1,
        "scout_audit": candidate.last_scout_audit,
        "zero_output_native_policy": native_equivalence,
        "nonzero_output_policy_delta": active_delta,
        "actual_tm_replacement_policy": strict_actual,
        "latency": {
            "timing_runs": args.timing_runs,
            "native_seconds": native_seconds,
            "candidate_seconds": candidate_seconds,
            "native_median_seconds": native_median,
            "candidate_median_seconds": candidate_median,
            "candidate_over_native": latency_ratio,
            "passes_2x": latency_ratio <= 2.0,
        },
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
