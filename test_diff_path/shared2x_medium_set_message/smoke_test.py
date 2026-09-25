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
    "shared2x_medium_set_message_smoke_runtime",
    THIS_DIR / "run_experiment.py",
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load Medium set-message runner")
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


def flattened_parameter_slices(model) -> dict[str, slice]:
    result = {}
    offset = 0
    for name, parameter in model.named_parameters():
        result[name] = slice(offset, offset + parameter.numel())
        offset += parameter.numel()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU correctness, strict-input, gradient and projection smoke"
    )
    parser.add_argument("--snapshot", type=int, default=350)
    parser.add_argument("--timing-runs", type=int, default=3)
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

    torch.manual_seed(20260824)
    native = experiment.NativeHattrick(props).to(device=device, dtype=props.dtype)
    torch.manual_seed(17)
    candidate = experiment.MediumSetMessageHattrick(props).to(
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
    for adapter in adapters:
        for branch in (adapter.scalar_branch, adapter.vector_branch):
            if int(torch.count_nonzero(branch[2].weight).item()) != 0:
                raise RuntimeError("Residual output weight is not zero")
            if int(torch.count_nonzero(branch[2].bias).item()) != 0:
                raise RuntimeError("Residual output bias is not zero")
    if int(torch.count_nonzero(candidate.medium_set_gate.weight).item()) != 0:
        raise RuntimeError("Set-message gate is not initialized to zero")

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
        candidate_policy, _ = experiment.full.shared.model_forward(
            candidate, props, dataset, values, path_masks
        )
    native_equivalence = compare_policies(
        native_policy, candidate_policy, "zero-output native equivalence"
    )
    if candidate.medium_set_cache_builds != 1:
        raise RuntimeError("Static path-length cache was not built exactly once")
    if candidate._medium_set_transformer_cache is not None:
        raise RuntimeError("Temporary path-embedding cache retained a completed graph")

    gate = candidate._last_medium_set_gate_split.reshape(
        1, num_ods, paths_per_od
    )
    uniform = candidate._last_medium_set_uniform_split.reshape(
        1, num_ods, paths_per_od
    )
    gate_simplex_error = float((gate.sum(dim=-1) - 1.0).abs().max().item())
    gate_uniform_error = float((gate - uniform).abs().max().item())
    if gate_simplex_error > 1e-6 or gate_uniform_error > 1e-7:
        raise RuntimeError("Zero-logit gate is not a uniform simplex")

    # A synthetic feasible-path mask exercises masked set pooling and softmax.
    base_masks = [
        torch.ones(total_paths, dtype=torch.bool, device=device) for _ in range(3)
    ]
    base_masks[1].reshape(num_ods, paths_per_od)[:, -1] = False
    synthetic_masks = torch.stack(base_masks, dim=0)
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
        path_masks=synthetic_masks,
    )
    expected_shape = (1, total_paths, experiment.TOTAL_FEATURES)
    if tuple(features.shape) != expected_shape:
        raise RuntimeError(f"Unexpected feature shape {tuple(features.shape)}")
    masked_gate = candidate._last_medium_set_gate_split.reshape(
        1, num_ods, paths_per_od
    )
    masked_simplex_error = float(
        (masked_gate.sum(dim=-1) - 1.0).abs().max().item()
    )
    masked_disabled_max = float(masked_gate[..., -1].abs().max().item())
    if masked_simplex_error > 1e-6 or masked_disabled_max != 0.0:
        raise RuntimeError("Masked Medium gate is invalid")

    vector_features = features[..., experiment.ANCHOR_FEATURES :]
    feature_loss = vector_features.square().mean()
    feature_loss.backward()
    message_gradient_l1 = {}
    for module_name in (
        "medium_set_path_token",
        "medium_set_context",
        "medium_set_gate",
    ):
        module = getattr(candidate, module_name)
        value = sum(
            float(parameter.grad.abs().sum().item())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        message_gradient_l1[module_name] = value
        if value == 0.0:
            raise RuntimeError(f"Direct feature gradient is zero for {module_name}")
    if tm2_probe.grad is None or float(tm2_probe.grad.abs().sum().item()) == 0.0:
        raise RuntimeError("Medium set-message has no tm2_pred gradient")

    # Open only the vector branch, then replace all actual TMs while preserving
    # ESM predictions.  The active policy must change, yet remain actual-blind.
    with torch.no_grad():
        for adapter in adapters:
            adapter.vector_branch[2].weight.fill_(0.01)
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
        candidate_policy,
        active_policy,
        "nonzero vector-branch activation",
        require_equal=False,
    )
    if max(row["max_abs_delta"] for row in active_delta.values()) == 0.0:
        raise RuntimeError("Opening the vector branch did not affect policy")
    strict_actual = compare_policies(
        active_policy, changed_actual_policy, "strict actual-TM replacement"
    )

    with torch.no_grad():
        for adapter in adapters:
            adapter.vector_branch[2].weight.zero_()
            adapter.vector_branch[2].bias.zero_()
    cache_builds_after_replays = candidate.medium_set_cache_builds
    if cache_builds_after_replays != 1:
        raise RuntimeError("Static cache was rebuilt across identical forwards")

    # Fair cached-transformer CPU latency.
    with torch.no_grad():
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
    if latency_ratio >= 2.0:
        raise RuntimeError(f"CPU inference latency ratio {latency_ratio:.3f} >= 2")

    # One real six-objective forward/backward verifies parameter coverage by the
    # unchanged ordered projection.  Exact-zero adapters intentionally mean the
    # first step reaches their output layers before the upstream message MLPs.
    candidate.zero_grad(set_to_none=True)
    candidate.train()
    props.mode = "train"
    props.research_return_policy = False
    props.research_return_admitted = True
    props.sim_mf_mlu = 0
    losses, _ = experiment.full.build_objectives(
        candidate, props, dataset, values, path_masks
    )
    projection = experiment.full.ordered_project_gradients(candidate, losses)
    parameter_count = sum(parameter.numel() for parameter in candidate.parameters())
    if int(projection.final_gradient.numel()) != parameter_count:
        raise RuntimeError("Ordered projection omitted model parameters")
    if not torch.isfinite(projection.final_gradient).all().item():
        raise RuntimeError("Ordered projection returned a non-finite gradient")
    slices = flattened_parameter_slices(candidate)
    adapter_output_names = [
        name
        for name in slices
        if "medium_pressure" in name and ".2.weight" in name
    ]
    if len(adapter_output_names) != 4:
        raise RuntimeError("Expected four dual-branch output weights")
    output_gradient_l1 = {
        name: [
            float(raw[slices[name]].abs().sum().item())
            for raw in projection.raw_gradients
        ]
        for name in adapter_output_names
    }
    if not any(any(value > 0.0 for value in row) for row in output_gradient_l1.values()):
        raise RuntimeError("Six objectives do not reach any adapter output weight")

    added_parameters = {
        name: parameter.numel()
        for name, parameter in candidate_parameters.items()
        if name not in native_parameters
    }
    result = {
        "status": "ok",
        "device": "cpu",
        "snapshot": args.snapshot,
        "strict_input": "ESM tm2_pred; actual TM excluded from message/policy",
        "actual_permutation_l1": actual_permutation_l1,
        "forward_is_inherited_native": (
            experiment.MediumSetMessageHattrick.forward
            is experiment.NativeHattrick.forward
        ),
        "native_parameters": sum(p.numel() for p in native.parameters()),
        "candidate_parameters": parameter_count,
        "added_parameters": sum(added_parameters.values()),
        "added_parameter_tensors": added_parameters,
        "missing_state_keys": list(loaded.missing_keys),
        "unexpected_state_keys": list(loaded.unexpected_keys),
        "feature_shape": list(features.shape),
        "anchor_shape": list(features[..., : experiment.ANCHOR_FEATURES].shape),
        "vector_shape": list(vector_features.shape),
        "vector_min": float(vector_features.detach().min().item()),
        "vector_max": float(vector_features.detach().max().item()),
        "vector_mean_abs": float(vector_features.detach().abs().mean().item()),
        "gate_simplex_max_error": gate_simplex_error,
        "zero_gate_uniform_max_error": gate_uniform_error,
        "synthetic_mask_simplex_max_error": masked_simplex_error,
        "synthetic_disabled_path_max": masked_disabled_max,
        "static_cache_builds": cache_builds_after_replays,
        "tm2_pred_gradient_l1": float(tm2_probe.grad.abs().sum().item()),
        "message_gradient_l1": message_gradient_l1,
        "set_message_audit": candidate.last_medium_set_audit,
        "zero_output_native_policy": native_equivalence,
        "nonzero_vector_policy_delta": active_delta,
        "actual_tm_replacement_policy": strict_actual,
        "projection": {
            "objectives": list(experiment.full.OBJECTIVE_NAMES),
            "parameter_count": parameter_count,
            "gradient_width": int(projection.final_gradient.numel()),
            "final_gradient_norm": float(
                torch.linalg.vector_norm(projection.final_gradient).item()
            ),
            "adapter_output_raw_gradient_l1": output_gradient_l1,
        },
        "latency": {
            "timing_runs": args.timing_runs,
            "native_seconds": native_seconds,
            "candidate_seconds": candidate_seconds,
            "native_median_seconds": native_median,
            "candidate_median_seconds": candidate_median,
            "candidate_over_native": latency_ratio,
            "passes_2x": latency_ratio < 2.0,
        },
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
