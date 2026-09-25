from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "test_diff_path"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "benchmarks" / "policy_forward_latency"
NATIVE_CHECKPOINT = (
    TEST_DIR
    / "shared2x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
    / "final_model.pt"
)
LOOKAHEAD_RUNNER = TEST_DIR / "shared2x_medium_pressure_lookahead" / "run_experiment.py"
LOOKAHEAD_CHECKPOINT = (
    TEST_DIR
    / "shared2x_medium_pressure_lookahead"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
    / "epoch_checkpoints"
    / "epoch_052.pt"
)
POLICY_RUNTIME = TEST_DIR / "shared2x_medium_adapter" / "run_experiment.py"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_class: Any
    checkpoint: Optional[Path]
    runner: Optional[Path]
    constructor_source: str


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import {}".format(path))
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


def state_dict_from_checkpoint(payload: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a dictionary")
    for key in ("model_state_dict", "state_dict", "model_state"):
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    if payload and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise KeyError("No model state dictionary found in checkpoint")


def make_props(runtime, device: torch.device):
    props = runtime.build_props(4, device)
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_policy = True
    props.research_return_admitted = False
    return props


def prepare_inputs(runtime, props, batch_sizes: Sequence[int]):
    end = 400 + max(batch_sizes)
    dataset = runtime.DM_Dataset_within_Cluster(props, 0, 400, end)
    if int(dataset.max_source_index_read) != end - 1:
        raise RuntimeError("Strict test-window reader audit failed")
    path_masks = runtime.shared.base.move_dataset_static(dataset, props.device)
    batches = {}
    for batch_size in batch_sizes:
        loader = runtime.shared.data_loader(dataset, batch_size, False, 0)
        inputs = next(iter(loader))
        values = list(runtime.shared.unpack_to_device(inputs, props))
        if int(values[2].shape[0]) != batch_size:
            raise RuntimeError("Unexpected prepared batch size")
        # Strict-prediction benchmark. The native forward still executes its normal
        # post-policy replay, but no real traffic value can affect the emitted policy.
        for index in (2, 4, 6):
            values[index] = torch.zeros_like(values[index])
        batches[int(batch_size)] = tuple(values)
    return dataset, path_masks, batches


def instantiate_model(spec: ModelSpec, props, device: torch.device):
    model = spec.model_class(props).to(device=device, dtype=props.dtype)
    checkpoint_meta: Dict[str, Any] = {"loaded": False}
    if spec.checkpoint is not None:
        checkpoint_path = spec.checkpoint.resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict_from_checkpoint(payload), strict=True)
        checkpoint_meta = {
            "loaded": True,
            "path": str(checkpoint_path),
            "sha256": sha256(checkpoint_path),
            "epoch": payload.get("epoch") if isinstance(payload, dict) else None,
        }
    model.eval()
    return model, checkpoint_meta


def model_parameter_stats(model: torch.nn.Module) -> Dict[str, int]:
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    return {
        "total_parameters": int(sum(value.numel() for value in parameters)),
        "trainable_parameters": int(
            sum(value.numel() for value in parameters if value.requires_grad)
        ),
        "parameter_bytes": int(
            sum(value.numel() * value.element_size() for value in parameters)
        ),
        "buffer_bytes": int(sum(value.numel() * value.element_size() for value in buffers)),
    }


def model_runtime_attributes(model: torch.nn.Module) -> Dict[str, Any]:
    names = (
        "virtual_medium_steps",
        "virtual_medium_temperature",
        "virtual_medium_price_power",
        "medium_pressure_feature_count",
        "research_medium_pressure_lookahead_enabled",
        "research_coverage_build_count",
        "coverage_build_count",
        "medium_coverage_cache_builds",
        "medium_irreplaceability_gamma",
    )
    output = {}
    for name in names:
        if hasattr(model, name):
            value = getattr(model, name)
            if isinstance(value, (bool, int, float, str)):
                output[name] = value
    return output


def coverage_cache_snapshot(model: torch.nn.Module) -> Dict[str, Dict[str, Any]]:
    """Capture discoverable static coverage tensors without copying them."""

    tokens = (
        "coverage",
        "irreplace",
        "od_edge",
        "medium_edge_weight",
        "medium_feasible_mask",
    )
    output: Dict[str, Dict[str, Any]] = {}
    seen = set()
    for name, value in model.named_buffers(recurse=True):
        lowered = name.lower()
        if any(token in lowered for token in tokens):
            output["buffer:" + name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "data_ptr": int(value.data_ptr()),
                "version": int(value._version),
            }
            seen.add(id(value))
    for module_name, module in model.named_modules():
        for name, value in vars(module).items():
            lowered = name.lower()
            if (
                torch.is_tensor(value)
                and id(value) not in seen
                and any(token in lowered for token in tokens)
            ):
                qualified = ".".join(part for part in (module_name, name) if part)
                output["attribute:" + qualified] = {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "data_ptr": int(value.data_ptr()),
                    "version": int(value._version),
                }
                seen.add(id(value))
    return output


def coverage_build_counts(model: torch.nn.Module) -> Dict[str, int]:
    output = {}
    for module_name, module in model.named_modules():
        for name, value in vars(module).items():
            lowered = name.lower()
            if (
                "coverage" in lowered
                and ("count" in lowered or "build" in lowered)
                and isinstance(value, int)
            ):
                qualified = ".".join(part for part in (module_name, name) if part)
                output[qualified] = int(value)
    return output


def compare_coverage_cache(
    before: Dict[str, Dict[str, Any]],
    after: Dict[str, Dict[str, Any]],
    counts_before: Dict[str, int],
    counts_after: Dict[str, int],
) -> Dict[str, Any]:
    common = sorted(set(before).intersection(after))
    pointers_unchanged = all(
        before[name]["data_ptr"] == after[name]["data_ptr"] for name in common
    )
    metadata_unchanged = all(
        all(before[name][field] == after[name][field] for field in ("shape", "dtype", "device"))
        for name in common
    )
    count_keys = sorted(set(counts_before).union(counts_after))
    count_delta = {
        name: counts_after.get(name, 0) - counts_before.get(name, 0)
        for name in count_keys
    }
    return {
        "discoverable_tensor_count": len(before),
        "same_tensor_keys_after_timing": set(before) == set(after),
        "common_data_ptrs_unchanged": pointers_unchanged,
        "common_metadata_unchanged": metadata_unchanged,
        "build_counts_before": counts_before,
        "build_counts_after": counts_after,
        "build_count_delta_during_timed_forwards": count_delta,
        "cache_reuse_confirmed": bool(
            before
            and set(before) == set(after)
            and pointers_unchanged
            and metadata_unchanged
            and all(delta == 0 for delta in count_delta.values())
        ),
        "before": before,
        "after": after,
    }


def clear_topology_cache(model: torch.nn.Module) -> None:
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")


def policy_forward(runtime, model, props, dataset, values, path_masks):
    return runtime.cached_policy_forward(model, props, dataset, values, path_masks)


def policy_audit(policies: Sequence[torch.Tensor], batch_size: int, paths_per_od: int):
    output = []
    for class_index, policy in enumerate(policies):
        flattened = policy.squeeze(-1)
        grouped = flattened.reshape(batch_size, -1, paths_per_od).float()
        output.append(
            {
                "class_index": class_index,
                "shape": list(policy.shape),
                "min": float(grouped.min().item()),
                "max": float(grouped.max().item()),
                "max_probability_sum_error": float(
                    (grouped.sum(dim=-1) - 1.0).abs().max().item()
                ),
                "checksum": float(grouped.sum().item()),
            }
        )
    return output


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def benchmark_one_batch(
    runtime,
    model,
    props,
    dataset,
    path_masks,
    values,
    batch_size: int,
    warmup: int,
    iterations: int,
    paths_per_od: int,
) -> Dict[str, Any]:
    clear_topology_cache(model)
    # `torch.sparse.mm` in the repository's PyTorch build cannot consume an
    # inference-mode sparse tensor because `.to(dtype=...)` touches its version
    # counter. `no_grad` has the same benchmark intent without that incompatibility.
    with torch.no_grad():
        # The first call materializes Hattrick's static topology cache. It is not
        # included in steady-state latency, matching repeated online inference.
        policies = policy_forward(runtime, model, props, dataset, values, path_masks)
        for _ in range(warmup):
            policies = policy_forward(runtime, model, props, dataset, values, path_masks)
    coverage_before = coverage_cache_snapshot(model)
    coverage_counts_before = coverage_build_counts(model)
    if props.device.type == "cuda":
        torch.cuda.synchronize(props.device)
        torch.cuda.reset_peak_memory_stats(props.device)
        allocated_before = int(torch.cuda.memory_allocated(props.device))
        reserved_before = int(torch.cuda.memory_reserved(props.device))
    else:
        allocated_before = 0
        reserved_before = 0

    latencies_ms: List[float] = []
    with torch.no_grad():
        for _ in range(iterations):
            if props.device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(props.device)
                start.record()
                policies = policy_forward(
                    runtime, model, props, dataset, values, path_masks
                )
                stop.record()
                stop.synchronize()
                latencies_ms.append(float(start.elapsed_time(stop)))
            else:
                started = time.perf_counter()
                policies = policy_forward(
                    runtime, model, props, dataset, values, path_masks
                )
                latencies_ms.append((time.perf_counter() - started) * 1000.0)

    if props.device.type == "cuda":
        torch.cuda.synchronize(props.device)
        peak_allocated = int(torch.cuda.max_memory_allocated(props.device))
        peak_reserved = int(torch.cuda.max_memory_reserved(props.device))
    else:
        peak_allocated = 0
        peak_reserved = 0
    coverage_after = coverage_cache_snapshot(model)
    coverage_counts_after = coverage_build_counts(model)
    mean_ms = float(statistics.fmean(latencies_ms))
    return {
        "batch_size": int(batch_size),
        "warmup_iterations": int(warmup),
        "timed_iterations": int(iterations),
        "latency_ms": {
            "mean": mean_ms,
            "median": float(statistics.median(latencies_ms)),
            "p10": percentile(latencies_ms, 10.0),
            "p90": percentile(latencies_ms, 90.0),
            "p95": percentile(latencies_ms, 95.0),
            "min": float(min(latencies_ms)),
            "max": float(max(latencies_ms)),
        },
        "throughput_snapshots_per_second": float(batch_size * 1000.0 / mean_ms),
        "cuda_memory_bytes": {
            "allocated_before_timing": allocated_before,
            "reserved_before_timing": reserved_before,
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
            "incremental_peak_allocated": max(0, peak_allocated - allocated_before),
            "incremental_peak_reserved": max(0, peak_reserved - reserved_before),
        },
        "policy_audit": policy_audit(policies, batch_size, paths_per_od),
        "coverage_cache_audit": compare_coverage_cache(
            coverage_before,
            coverage_after,
            coverage_counts_before,
            coverage_counts_after,
        ),
    }


def external_spec(
    name: str,
    runner: Optional[Path],
    checkpoint: Optional[Path],
    explicit_class: Optional[str],
    module_name: str,
    known_classes: Sequence[str],
) -> Optional[ModelSpec]:
    if runner is None:
        return None
    runner_path = runner.resolve()
    if not runner_path.exists():
        raise FileNotFoundError(runner_path)
    module = load_module(module_name, runner_path)
    if hasattr(module, "benchmark_model_class"):
        model_class = module.benchmark_model_class()
        source = "benchmark_model_class()"
    elif explicit_class and hasattr(module, explicit_class):
        model_class = getattr(module, explicit_class)
        source = explicit_class
    else:
        found = [candidate for candidate in known_classes if hasattr(module, candidate)]
        if len(found) != 1:
            raise RuntimeError(
                "Specify an explicit model class, or expose exactly one known class in {}".format(
                    runner_path
                )
            )
        source = found[0]
        model_class = getattr(module, found[0])
    return ModelSpec(
        name=name,
        model_class=model_class,
        checkpoint=checkpoint.resolve() if checkpoint else None,
        runner=runner_path,
        constructor_source=source,
    )


def virtual_spec(args) -> Optional[ModelSpec]:
    return external_spec(
        "virtual-medium-dual",
        args.virtual_runner,
        args.virtual_checkpoint,
        args.virtual_class,
        "policy_latency_virtual_runner",
        (
            "VirtualDualHattrick",
            "VirtualMediumDualHattrick",
            "VirtualScoutHattrick",
            "MediumScoutHattrick",
        ),
    )


def irreplaceability_spec(args) -> Optional[ModelSpec]:
    return external_spec(
        "medium-irreplaceability",
        args.irreplaceability_runner,
        args.irreplaceability_checkpoint,
        args.irreplaceability_class,
        "policy_latency_irreplaceability_runner",
        (
            "MediumIrreplaceabilityHattrick",
            "IrreplaceabilityHattrick",
            "MediumCoverageHattrick",
            "MediumIrreplaceabilityScoutHattrick",
        ),
    )


def write_csv_summary(path: Path, results: Sequence[Dict[str, Any]]) -> None:
    rows = []
    for result in results:
        for measurement in result["measurements"]:
            latency = measurement["latency_ms"]
            memory = measurement["cuda_memory_bytes"]
            rows.append(
                {
                    "model": result["model"],
                    "batch_size": measurement["batch_size"],
                    "parameter_count": result["parameters"]["total_parameters"],
                    "latency_mean_ms": latency["mean"],
                    "latency_median_ms": latency["median"],
                    "latency_p95_ms": latency["p95"],
                    "throughput_snapshots_per_second": measurement[
                        "throughput_snapshots_per_second"
                    ],
                    "peak_allocated_mib": memory["peak_allocated"] / 2**20,
                    "incremental_peak_allocated_mib": memory[
                        "incremental_peak_allocated"
                    ]
                    / 2**20,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Steady-state strict-ESM Hattrick policy-forward latency benchmark"
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 16, 20))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--virtual-runner", type=Path)
    parser.add_argument("--virtual-checkpoint", type=Path)
    parser.add_argument("--virtual-class")
    parser.add_argument("--irreplaceability-runner", type=Path)
    parser.add_argument("--irreplaceability-checkpoint", type=Path)
    parser.add_argument("--irreplaceability-class")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.batch_sizes or any(value <= 0 for value in args.batch_sizes):
        raise ValueError("Batch sizes must be positive")
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("Warmup and iteration counts must be positive")
    for path in (
        ROOT,
        TEST_DIR,
        TEST_DIR / "shared2x_medium_adapter",
        TEST_DIR / "shared2x_order_regularizer",
        TEST_DIR / "shared2x_full_objectives",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    runtime = load_module("policy_latency_runtime", POLICY_RUNTIME)
    lookahead = load_module("policy_latency_lookahead", LOOKAHEAD_RUNNER)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    common_props = make_props(runtime, device)
    batch_sizes = tuple(sorted(set(int(value) for value in args.batch_sizes)))
    dataset, path_masks, batches = prepare_inputs(
        runtime, common_props, batch_sizes
    )
    specs = [
        ModelSpec(
            name="native-hattrick",
            model_class=runtime.Hattrick,
            checkpoint=NATIVE_CHECKPOINT,
            runner=POLICY_RUNTIME,
            constructor_source="runtime.Hattrick",
        ),
        ModelSpec(
            name="hattrick-la-e52",
            model_class=lookahead.MediumPressureHattrick,
            checkpoint=LOOKAHEAD_CHECKPOINT,
            runner=LOOKAHEAD_RUNNER,
            constructor_source="MediumPressureHattrick",
        ),
    ]
    virtual = virtual_spec(args)
    irreplaceability = irreplaceability_spec(args)
    for optional in (virtual, irreplaceability):
        if optional is not None:
            specs.append(optional)

    results = []
    for spec in specs:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        props = make_props(runtime, device)
        model, checkpoint_meta = instantiate_model(spec, props, device)
        parameter_stats = model_parameter_stats(model)
        measurements = []
        for batch_size in batch_sizes:
            measurements.append(
                benchmark_one_batch(
                    runtime,
                    model,
                    props,
                    dataset,
                    path_masks,
                    batches[batch_size],
                    batch_size,
                    args.warmup,
                    args.iterations,
                    int(props.num_paths_per_pair),
                )
            )
        results.append(
            {
                "model": spec.name,
                "constructor_source": spec.constructor_source,
                "runner": str(spec.runner.resolve()) if spec.runner else None,
                "runner_sha256": sha256(spec.runner) if spec.runner else None,
                "checkpoint": checkpoint_meta,
                "parameters": parameter_stats,
                "runtime_attributes": model_runtime_attributes(model),
                "measurements": measurements,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "benchmark": "strict-ESM steady-state policy forward",
        "window_prefix": [400, 400 + max(batch_sizes)],
        "batch_sizes": list(batch_sizes),
        "actual_traffic_matrices_zeroed": True,
        "topology_cache": "first materialization excluded; steady-state cache reused",
        "timing": "CUDA Events with synchronize before each timed forward",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "results": results,
        "virtual_dual_status": (
            "included" if virtual is not None else "not supplied; pass --virtual-runner"
        ),
        "medium_irreplaceability_status": (
            "included"
            if irreplaceability is not None
            else "not supplied; pass --irreplaceability-runner"
        ),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "policy_forward_latency.json"
    csv_path = output_dir / "policy_forward_latency.csv"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv_summary(csv_path, results)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    print(str(json_path), flush=True)


if __name__ == "__main__":
    main()
