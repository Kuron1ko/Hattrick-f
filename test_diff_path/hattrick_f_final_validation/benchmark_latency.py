from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# PyTorch requires this setting to be present before CUDA is initialized when
# deterministic algorithms are requested.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from frozen_sources import EXPECTED as FROZEN_EXPECTED
from frozen_sources import verify as verify_frozen_sources


HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
ROOT = TEST_DIR.parent
RUNTIME_PATH = TEST_DIR / "shared2x_medium_adapter" / "run_experiment.py"
TOPOLOGY = "geant_priomask500_shared_load2x_train"
DEFAULT_OUTPUT = (
    HERE / "artifacts" / "latency" / "benchmark_latency_current_three_seed.json"
)
FORMAL_MODEL_SEEDS = (490, 491, 492)
PHASE_A_CURRENT_ROOT = HERE / "artifacts" / "phase_a_current"
PHASE_A_SHARED_OUTPUT_ROOT = (
    TEST_DIR / "shared2x_full_objectives" / "artifacts" / "level4_confirmation"
)
HATTRICK_F_CURRENT_ROOT = (
    HERE / "artifacts" / "hattrick_f" / "fh_release_low_budget_0p03"
)
CONTROL_CURRENT_ROOT = (
    HERE
    / "artifacts"
    / "six_loss_continuation_control"
    / "fh_release_low_budget_0p03"
)
SIX_LOSS_OBJECTIVES = ("Fh", "Uh", "Fhm", "Uhm", "Fhml", "Uhml")
HATTRICK_F_OBJECTIVES = ("Fh", "Fhm", "Fhml")

PATHS_PER_OD = 8
ACTUAL_TM_INDICES = (2, 4, 6)
PREDICTED_TM_INDICES = (3, 5, 7)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
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


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def checkpoint_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict", "model_state"):
        if key in payload:
            return payload[key]
    raise KeyError("checkpoint has no model state dictionary")


@dataclass(frozen=True)
class ArtifactSet:
    seed: int
    checkpoints: dict[str, Path]
    configs: dict[str, dict]
    completion: dict[str, dict]
    audit: dict[str, Any]


PHASE_A_SOURCE_FILES = {
    "run_experiment.py": TEST_DIR / "shared2x_full_objectives" / "run_experiment.py",
    "ordered_projection.py": (
        TEST_DIR / "shared2x_full_objectives" / "ordered_projection.py"
    ),
    "frameworks/hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
    "utils/training_utils.py": ROOT / "utils" / "training_utils.py",
    "shared2x_order_regularizer/run_experiment.py": (
        TEST_DIR / "shared2x_order_regularizer" / "run_experiment.py"
    ),
}
CONTINUATION_SOURCE_FILES = {
    "run_level4.py": TEST_DIR / "shared2x_hattrick_f" / "run_level4.py",
    "run_experiment.py": TEST_DIR / "shared2x_hattrick_f" / "run_experiment.py",
    "ordered_projection.py": (
        TEST_DIR / "shared2x_full_objectives" / "ordered_projection.py"
    ),
    "hattrick_system.py": ROOT / "frameworks" / "hattrick_system.py",
}


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def expected_frozen_hash(path: Path) -> str:
    try:
        return FROZEN_EXPECTED[path]
    except KeyError as error:
        raise RuntimeError(f"source is absent from frozen verifier: {path}") from error


def verify_source_hashes(config: dict, role: str) -> None:
    source_hashes = config.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise RuntimeError(f"{role} config has no source_sha256 map")
    expected_files = (
        PHASE_A_SOURCE_FILES if role == "hattrick_phase_a" else CONTINUATION_SOURCE_FILES
    )
    mismatches = {}
    for config_key, path in expected_files.items():
        expected = expected_frozen_hash(path)
        observed = source_hashes.get(config_key)
        if observed != expected:
            mismatches[config_key] = {"expected": expected, "observed": observed}
    if mismatches:
        raise RuntimeError(f"{role} was not built from frozen current sources: {mismatches}")


def validate_run_artifact(
    checkpoint: Path,
    *,
    seed: int,
    role: str,
    phase_a_sha256: str | None = None,
) -> tuple[dict, dict, dict]:
    checkpoint = checkpoint.resolve()
    if checkpoint.name != "best_model.pt" or not checkpoint.is_file():
        raise FileNotFoundError(f"missing selected checkpoint for {role}: {checkpoint}")
    run_dir = checkpoint.parent
    config_path = run_dir / "config.json"
    complete_path = run_dir / "complete.json"
    config = read_json(config_path)
    complete = read_json(complete_path)
    if int(config.get("seed", -1)) != int(seed):
        raise RuntimeError(
            f"{role} seed mismatch: requested={seed}, config={config.get('seed')}"
        )
    # The frozen Phase-A source predates an explicit status field, whereas the
    # continuation writer always emits it.  In either case the selected
    # checkpoint must be bound to complete.json by hash below.
    if role == "hattrick_phase_a":
        if complete.get("status") not in (None, "COMPLETE"):
            raise RuntimeError(
                f"{role} completion status is not COMPLETE: {complete.get('status')}"
            )
    elif complete.get("status") != "COMPLETE":
        raise RuntimeError(
            f"{role} completion status is not COMPLETE: {complete.get('status')}"
        )
    if "seed" in complete and int(complete["seed"]) != int(seed):
        raise RuntimeError(
            f"{role} complete.json seed mismatch: "
            f"requested={seed}, complete={complete.get('seed')}"
        )
    verify_source_hashes(config, role)

    if role == "hattrick_phase_a":
        if int(config.get("epochs", -1)) != 60:
            raise RuntimeError("Phase-A is not the protocol-fixed 60-epoch run")
        if tuple(config.get("objectives", ())) != SIX_LOSS_OBJECTIVES:
            raise RuntimeError("Phase-A does not use the complete six-loss objective order")
        if float(config.get("load_factor", -1.0)) != 2.0:
            raise RuntimeError("Phase-A is not a 2x-load artifact")
        if str(config.get("topology")) != TOPOLOGY:
            raise RuntimeError(f"unexpected Phase-A topology: {config.get('topology')}")
    else:
        if int(config.get("phase_f_epochs", -1)) != 30:
            raise RuntimeError(f"{role} is not a 30-epoch continuation")
        parent = config.get("phase_a")
        if not isinstance(parent, dict) or parent.get("sha256") != phase_a_sha256:
            raise RuntimeError(
                f"{role} parent Phase-A mismatch: "
                f"expected={phase_a_sha256}, observed={parent}"
            )
        phase_f = config.get("phase_f")
        if not isinstance(phase_f, dict):
            raise RuntimeError(f"{role} config has no phase_f block")
        expected_objectives = (
            HATTRICK_F_OBJECTIVES
            if role == "hattrick_f"
            else SIX_LOSS_OBJECTIVES
        )
        if tuple(phase_f.get("objectives", ())) != expected_objectives:
            raise RuntimeError(
                f"{role} continuation objectives differ from protocol: {phase_f}"
            )
        if phase_f.get("optimizer_reset") is not True:
            raise RuntimeError(f"{role} did not reset Adam")
        if phase_f.get("all_parameters_trainable") is not True:
            raise RuntimeError(f"{role} did not keep all parameters trainable")

    checkpoint_digest = sha256(checkpoint)
    declared_digest = complete.get("best_checkpoint_sha256")
    if declared_digest is None:
        artifact_hashes = complete.get("artifact_sha256", {})
        if isinstance(artifact_hashes, dict):
            declared_digest = artifact_hashes.get("best_model.pt")
    if not declared_digest:
        raise RuntimeError(f"{role} complete.json does not bind best_model.pt by SHA-256")
    if declared_digest != checkpoint_digest:
        raise RuntimeError(
            f"{role} selected checkpoint hash disagrees with complete.json: "
            f"{checkpoint_digest}/{declared_digest}"
        )
    audit = {
        "role": role,
        "seed": int(seed),
        "run_directory": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "complete": str(complete_path.resolve()),
        "complete_sha256": sha256(complete_path),
        "parent_phase_a_sha256": (
            None if role == "hattrick_phase_a" else phase_a_sha256
        ),
        "complete_seed_field": complete.get("seed"),
        "complete_seed_bound_via_checkpoint_sha256": True,
    }
    return config, complete, audit


def validate_artifact_set(
    seed: int, phase_a: Path, hattrick_f: Path, control: Path
) -> ArtifactSet:
    phase_config, phase_complete, phase_audit = validate_run_artifact(
        phase_a, seed=seed, role="hattrick_phase_a"
    )
    phase_digest = phase_audit["checkpoint_sha256"]
    h_config, h_complete, h_audit = validate_run_artifact(
        hattrick_f,
        seed=seed,
        role="hattrick_f",
        phase_a_sha256=phase_digest,
    )
    c_config, c_complete, c_audit = validate_run_artifact(
        control,
        seed=seed,
        role="six_loss_control",
        phase_a_sha256=phase_digest,
    )
    return ArtifactSet(
        seed=int(seed),
        checkpoints={
            "hattrick_phase_a": phase_a.resolve(),
            "hattrick_f": hattrick_f.resolve(),
            "six_loss_control": control.resolve(),
        },
        configs={
            "hattrick_phase_a": phase_config,
            "hattrick_f": h_config,
            "six_loss_control": c_config,
        },
        completion={
            "hattrick_phase_a": phase_complete,
            "hattrick_f": h_complete,
            "six_loss_control": c_complete,
        },
        audit={
            "seed": int(seed),
            "phase_a_sha256": phase_digest,
            "runs": [phase_audit, h_audit, c_audit],
            "all_continuations_share_phase_a": True,
            "all_complete": True,
            "all_config_seeds_match": True,
        },
    )


def resolve_automatic_artifact_set(seed: int) -> ArtifactSet:
    hattrick_f = HATTRICK_F_CURRENT_ROOT / f"seed_{seed}" / "best_model.pt"
    control = CONTROL_CURRENT_ROOT / f"seed_{seed}" / "best_model.pt"
    # Some current frozen-source Phase-A runs were emitted by the shared runner
    # into its historical output root.  Directory names are not provenance:
    # validate_run_artifact accepts these only when config/source hashes and the
    # complete/checkpoint binding match the current frozen protocol.  Thus an
    # older checkpoint in the same directory is rejected content-wise.
    phase_candidates = (
        PHASE_A_CURRENT_ROOT / f"seed_{seed}" / "best_model.pt",
        PHASE_A_SHARED_OUTPUT_ROOT / f"seed_{seed}" / "best_model.pt",
    )
    failures = []
    for phase_a in phase_candidates:
        try:
            return validate_artifact_set(seed, phase_a, hattrick_f, control)
        except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
            failures.append(
                {"phase_a": str(phase_a.resolve()), "error": str(error)}
            )
    raise RuntimeError(
        f"no complete current artifact trio exists for seed {seed}; "
        "unverified legacy fallback is disabled: "
        f"{json.dumps(failures, ensure_ascii=False)}"
    )


def resolve_artifact_sets(args: argparse.Namespace) -> tuple[list[ArtifactSet], str]:
    explicit_checkpoints = (
        args.phase_a_checkpoint,
        args.hattrick_f_checkpoint,
        args.control_checkpoint,
    )
    explicit_roots = (
        args.phase_a_root,
        args.hattrick_f_root,
        args.control_root,
    )
    if any(path is not None for path in explicit_checkpoints) and any(
        path is not None for path in explicit_roots
    ):
        raise ValueError(
            "explicit checkpoint mode and explicit artifact-root mode are mutually exclusive"
        )
    if any(path is not None for path in explicit_checkpoints):
        if not all(path is not None for path in explicit_checkpoints):
            raise ValueError(
                "explicit mode requires --phase-a-checkpoint, "
                "--hattrick-f-checkpoint, and --control-checkpoint together"
            )
        if len(args.model_seeds) != 1:
            raise ValueError("explicit checkpoint mode requires exactly one --model-seeds value")
        seed = int(args.model_seeds[0])
        return [
            validate_artifact_set(
                seed,
                explicit_checkpoints[0],
                explicit_checkpoints[1],
                explicit_checkpoints[2],
            )
        ], "explicit_single_seed"
    if any(path is not None for path in explicit_roots):
        if not all(path is not None for path in explicit_roots):
            raise ValueError(
                "explicit root mode requires --phase-a-root, "
                "--hattrick-f-root, and --control-root together"
            )
        return [
            validate_artifact_set(
                int(seed),
                explicit_roots[0] / f"seed_{seed}" / "best_model.pt",
                explicit_roots[1] / f"seed_{seed}" / "best_model.pt",
                explicit_roots[2] / f"seed_{seed}" / "best_model.pt",
            )
            for seed in args.model_seeds
        ], "explicit_artifact_roots"
    return [
        resolve_automatic_artifact_set(int(seed)) for seed in args.model_seeds
    ], "automatic_current_artifacts"


def make_props(runtime, device: torch.device):
    runtime.shared.TOPOLOGY = TOPOLOGY
    props = runtime.build_props(4, device)
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_policy = True
    props.research_return_admitted = False
    return props


def prepare_strict_esm_input(
    runtime,
    props,
    snapshot_start: int,
    batch_size: int,
) -> dict[str, Any]:
    """Load once, then remove actual traffic before the benchmark begins."""
    snapshot_end = snapshot_start + batch_size
    dataset = runtime.DM_Dataset_within_Cluster(
        props, 0, snapshot_start, snapshot_end
    )
    if int(dataset.max_source_index_read) != snapshot_end - 1:
        raise RuntimeError(
            "split-safe input audit failed: "
            f"expected last source {snapshot_end - 1}, "
            f"read {dataset.max_source_index_read}"
        )
    path_masks = runtime.shared.base.move_dataset_static(dataset, props.device)
    loader = runtime.shared.data_loader(dataset, batch_size, False, 0)
    inputs = list(loader)
    if len(inputs) != 1:
        raise RuntimeError(f"expected exactly one fixed batch, got {len(inputs)}")
    values = list(runtime.shared.unpack_to_device(inputs[0], props))
    if int(values[2].shape[0]) != batch_size:
        raise RuntimeError(
            f"expected batch {batch_size}, got {int(values[2].shape[0])}"
        )

    predicted_audit = []
    for class_index, (actual_index, predicted_index) in enumerate(
        zip(ACTUAL_TM_INDICES, PREDICTED_TM_INDICES)
    ):
        prediction = values[predicted_index]
        if not torch.isfinite(prediction).all():
            raise RuntimeError(f"non-finite ESM tensor for class {class_index}")
        predicted_l1 = float(prediction.double().abs().sum().item())
        if predicted_l1 <= 0.0:
            raise RuntimeError(f"empty ESM tensor for class {class_index}")
        predicted_audit.append(
            {
                "class_index": class_index,
                "shape": list(prediction.shape),
                "l1": predicted_l1,
                "sha256": tensor_sha256(prediction),
            }
        )
        values[actual_index] = torch.zeros_like(values[actual_index])

    actual_l1 = [
        float(values[index].double().abs().sum().item())
        for index in ACTUAL_TM_INDICES
    ]
    if any(value != 0.0 for value in actual_l1):
        raise RuntimeError(f"strict-ESM audit failed; actual L1={actual_l1}")

    return {
        "dataset": dataset,
        "path_masks": path_masks,
        "values": tuple(values),
        "audit": {
            "snapshot_indices_zero_based": list(
                range(snapshot_start, snapshot_end)
            ),
            "batch_size": batch_size,
            "actual_traffic_zeroed": True,
            "actual_tm_l1_after_zeroing": actual_l1,
            "predicted_tm": predicted_audit,
        },
    }


def load_model(
    runtime,
    props,
    device: torch.device,
    checkpoint_path: Path,
    *,
    expected_seed: int,
    expected_config: dict,
    expected_complete: dict,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload_config = payload.get("config")
    if payload_config != expected_config:
        raise RuntimeError(
            f"checkpoint payload config disagrees with config.json: {checkpoint_path}"
        )
    if int(payload_config.get("seed", -1)) != int(expected_seed):
        raise RuntimeError(
            f"checkpoint payload seed mismatch for {checkpoint_path}: "
            f"{payload_config.get('seed')}/{expected_seed}"
        )
    if "best_epoch" in expected_complete and int(
        expected_complete["best_epoch"]
    ) != int(payload.get("epoch", -1)):
        raise RuntimeError(
            f"complete/checkpoint best_epoch mismatch for {checkpoint_path}: "
            f"{expected_complete.get('best_epoch')}/{payload.get('epoch')}"
        )
    if "best_rank" in expected_complete and list(
        expected_complete["best_rank"]
    ) != list(payload.get("rank", [])):
        raise RuntimeError(
            f"complete/checkpoint best_rank mismatch for {checkpoint_path}"
        )
    state = checkpoint_state(payload)
    model = runtime.Hattrick(props).to(device=device, dtype=props.dtype)
    load_result = model.load_state_dict(state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"non-strict load for {checkpoint_path}: {load_result}"
        )
    model.eval()
    model.requires_grad_(False)
    metadata = {
        "path": str(checkpoint_path),
        "sha256": sha256(checkpoint_path),
        "epoch": payload.get("epoch"),
        "config_seed": int(payload_config["seed"]),
        "config_sha256_in_memory": hashlib.sha256(
            json.dumps(payload_config, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameter_count": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
        "state_keys_and_shapes": [
            [name, list(value.shape)] for name, value in state.items()
        ],
    }
    return model, payload, metadata


def clear_topology_cache(model: torch.nn.Module) -> None:
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")


def policy_forward(runtime, model, props, fixed_input):
    return runtime.cached_policy_forward(
        model,
        props,
        fixed_input["dataset"],
        fixed_input["values"],
        fixed_input["path_masks"],
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_forward(callable_, device: torch.device):
    # The synchronization before the timer drains unrelated GPU work. The one
    # after the forward makes the measured wall time include completion of all
    # kernels launched by this request, but excludes the drain itself.
    synchronize(device)
    started_ns = time.perf_counter_ns()
    output = callable_()
    synchronize(device)
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    return output, float(elapsed_ms)


def balanced_order(names: Sequence[str], iteration: int) -> list[str]:
    """Deterministic rotations, reversed every cycle, balance thermal order."""
    names = list(names)
    cycle, offset = divmod(iteration, len(names))
    base = names if cycle % 2 == 0 else list(reversed(names))
    return base[offset:] + base[:offset]


def summarize(samples: Sequence[float]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90.0)),
        "p99_ms": float(np.percentile(values, 99.0)),
        "mean_ms": float(statistics.fmean(samples)),
        "std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def policy_audit(
    policies: Sequence[torch.Tensor], batch_size: int
) -> list[dict[str, Any]]:
    result = []
    for class_index, policy in enumerate(policies):
        if not torch.isfinite(policy).all():
            raise RuntimeError(f"non-finite policy for class {class_index}")
        grouped = policy.detach().float().reshape(
            batch_size, -1, PATHS_PER_OD
        )
        result.append(
            {
                "class_index": class_index,
                "shape": list(policy.shape),
                "min": float(grouped.min().item()),
                "max": float(grouped.max().item()),
                "max_path_probability_sum_error": float(
                    (grouped.sum(dim=-1) - 1.0).abs().max().item()
                ),
                "sha256": tensor_sha256(policy),
            }
        )
    return result


def benchmark_batch(
    runtime,
    models: dict[str, torch.nn.Module],
    props,
    fixed_input: dict[str, Any],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    names = list(models)
    calls = {
        name: (
            lambda model=model: policy_forward(
                runtime, model, props, fixed_input
            )
        )
        for name, model in models.items()
    }
    final_policies: dict[str, Sequence[torch.Tensor]] = {}

    with torch.no_grad():
        # The first call builds the static topology cache. Cache construction is
        # deliberately excluded: this is repeated online policy latency.
        for name in names:
            clear_topology_cache(models[name])
            final_policies[name] = calls[name]()
        synchronize(device)

        # Interleave warmups for the same reason as timed calls.
        for iteration in range(warmup):
            for name in balanced_order(names, iteration):
                final_policies[name] = calls[name]()
        synchronize(device)

    samples = {name: [] for name in names}
    execution_order = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        with torch.no_grad():
            for iteration in range(iterations):
                order = balanced_order(names, iteration)
                execution_order.append(order)
                for name in order:
                    policies, elapsed_ms = timed_forward(calls[name], device)
                    final_policies[name] = policies
                    samples[name].append(elapsed_ms)
    finally:
        synchronize(device)
        if gc_was_enabled:
            gc.enable()

    batch_size = int(fixed_input["audit"]["batch_size"])
    results = {}
    for name in names:
        summary = summarize(samples[name])
        summary["throughput_snapshots_per_second_from_mean"] = float(
            batch_size * 1000.0 / summary["mean_ms"]
        )
        results[name] = {
            "summary": summary,
            "raw_latency_ms": samples[name],
            "policy_audit": policy_audit(final_policies[name], batch_size),
        }
    baseline = results["hattrick_phase_a"]["summary"]
    comparisons = {}
    for name in names:
        if name == "hattrick_phase_a":
            continue
        candidate = results[name]["summary"]
        comparisons[f"{name}_vs_hattrick_phase_a"] = {
            "median_delta_ms": float(
                candidate["median_ms"] - baseline["median_ms"]
            ),
            "median_ratio": float(
                candidate["median_ms"] / baseline["median_ms"]
            ),
            "median_percent_change": float(
                100.0
                * (candidate["median_ms"] - baseline["median_ms"])
                / baseline["median_ms"]
            ),
            "mean_delta_ms": float(
                candidate["mean_ms"] - baseline["mean_ms"]
            ),
            "mean_ratio": float(candidate["mean_ms"] / baseline["mean_ms"]),
        }
    return {
        "batch_size": batch_size,
        "input": fixed_input["audit"],
        "topology_cache_build_calls_per_model_excluded": 1,
        "warmup_calls_per_model": warmup,
        "timed_calls_per_model": iterations,
        "execution_order": execution_order,
        "models": results,
        "comparisons": comparisons,
    }


def across_seed_latency(seed_results: list[dict]) -> list[dict]:
    """Summarize per-seed timing without treating calls as independent seeds."""

    grouped: dict[tuple[int, str], list[dict]] = {}
    for seed_result in seed_results:
        for batch in seed_result["batch_results"]:
            for method, result in batch["models"].items():
                grouped.setdefault((int(batch["batch_size"]), method), []).append(
                    {
                        "seed": int(seed_result["seed"]),
                        **result["summary"],
                    }
                )
    output = []
    for (batch_size, method), rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "batch_size": batch_size,
            "method": method,
            "seed_count": len(rows),
            "seeds": [row["seed"] for row in sorted(rows, key=lambda value: value["seed"])],
        }
        for metric in ("median_ms", "mean_ms", "p90_ms", "p99_ms"):
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            item[f"{metric}_across_seed_mean"] = float(values.mean())
            item[f"{metric}_across_seed_min"] = float(values.min())
            item[f"{metric}_across_seed_max"] = float(values.max())
        output.append(item)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproducible strict-ESM steady-state policy-forward latency: "
            "Hattrick Phase-A vs Hattrick-f vs equal-budget six-loss control"
        )
    )
    parser.add_argument("--phase-a-checkpoint", type=Path)
    parser.add_argument("--hattrick-f-checkpoint", type=Path)
    parser.add_argument(
        "--control-checkpoint",
        type=Path,
        help=(
            "explicit same-architecture six-loss continuation checkpoint; all "
            "three checkpoint options must be supplied together for one model seed"
        ),
    )
    parser.add_argument(
        "--phase-a-root",
        type=Path,
        help="explicit root containing seed_<N>/best_model.pt Phase-A artifacts",
    )
    parser.add_argument(
        "--hattrick-f-root",
        type=Path,
        help="explicit root containing seed_<N>/best_model.pt Hattrick-f artifacts",
    )
    parser.add_argument(
        "--control-root",
        type=Path,
        help="explicit root containing seed_<N>/best_model.pt control artifacts",
    )
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=list(FORMAL_MODEL_SEEDS),
        help=(
            "training seeds to resolve from current complete artifacts; formal "
            "comparison requires 490 491 492"
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 16],
        help="fixed online request batches; supported values are 1 and 16",
    )
    parser.add_argument("--snapshot-start", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="explicitly permit fewer than 100 timed calls; never a formal result",
    )
    args = parser.parse_args()
    if args.warmup < 1:
        parser.error("--warmup must be positive")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.iterations < 100 and not args.smoke:
        parser.error("formal runs require --iterations >= 100 (or use --smoke)")
    if not args.batch_sizes or any(value not in (1, 16) for value in args.batch_sizes):
        parser.error("--batch-sizes may contain only 1 and/or 16")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        parser.error("--batch-sizes contains duplicates")
    if not args.model_seeds or len(set(args.model_seeds)) != len(args.model_seeds):
        parser.error("--model-seeds must be a non-empty list without duplicates")
    if args.snapshot_start < 0:
        parser.error("--snapshot-start must be non-negative")
    return args


def main() -> None:
    # Abort before loading data or CUDA if any frozen source/artifact lineage moved.
    frozen_hashes = verify_frozen_sources()
    args = parse_args()
    artifact_sets, artifact_resolution = resolve_artifact_sets(args)
    model_seeds = [artifact.seed for artifact in artifact_sets]
    if len(model_seeds) == 1:
        comparison_scope = "single_seed"
    elif tuple(sorted(model_seeds)) == FORMAL_MODEL_SEEDS and len(model_seeds) == 3:
        comparison_scope = "three_seed"
    else:
        comparison_scope = f"{len(model_seeds)}_seed"
    formal_disqualifications = []
    if args.smoke:
        formal_disqualifications.append("smoke_mode")
    if args.iterations < 100:
        formal_disqualifications.append("fewer_than_100_timed_calls")
    if comparison_scope != "three_seed":
        formal_disqualifications.append(comparison_scope)
    formal_result = not formal_disqualifications

    for path in (ROOT, TEST_DIR, TEST_DIR / "shared2x_order_regularizer"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    runtime = load_module("hattrick_f_final_latency_runtime", RUNTIME_PATH)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False

    props = make_props(runtime, device)
    fixed_inputs = {
        batch_size: prepare_strict_esm_input(
            runtime, props, args.snapshot_start, batch_size
        )
        for batch_size in args.batch_sizes
    }

    seed_results = []
    global_reference_shapes = None
    for artifact in artifact_sets:
        models = {}
        checkpoint_metadata = {}
        for name, checkpoint_path in artifact.checkpoints.items():
            model, _payload, metadata = load_model(
                runtime,
                props,
                device,
                checkpoint_path,
                expected_seed=artifact.seed,
                expected_config=artifact.configs[name],
                expected_complete=artifact.completion[name],
            )
            models[name] = model
            checkpoint_metadata[name] = metadata

        reference_shapes = checkpoint_metadata["hattrick_phase_a"][
            "state_keys_and_shapes"
        ]
        for name, metadata in checkpoint_metadata.items():
            if metadata["state_keys_and_shapes"] != reference_shapes:
                raise RuntimeError(
                    f"seed {artifact.seed}: {name} is not the same "
                    "architecture/state shape as Phase-A"
                )
        if global_reference_shapes is None:
            global_reference_shapes = reference_shapes
        elif reference_shapes != global_reference_shapes:
            raise RuntimeError(
                f"seed {artifact.seed} architecture differs from earlier seeds"
            )

        batch_results = []
        for batch_size in args.batch_sizes:
            batch_results.append(
                benchmark_batch(
                    runtime,
                    models,
                    props,
                    fixed_inputs[batch_size],
                    device,
                    args.warmup,
                    args.iterations,
                )
            )
        seed_results.append(
            {
                "seed": artifact.seed,
                "artifact_lineage": artifact.audit,
                "checkpoints": checkpoint_metadata,
                "batch_results": batch_results,
            }
        )
        del models
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "benchmark": (
            "strict-ESM steady-state Hattrick policy-forward response time"
        ),
        "formal_result": formal_result,
        "formal_disqualification_reasons": formal_disqualifications,
        "smoke": bool(args.smoke),
        "comparison_scope": comparison_scope,
        "artifact_resolution": artifact_resolution,
        "unverified_legacy_artifact_fallback_enabled": False,
        "scope": {
            "included": "policy generation from ESM predictions",
            "excluded": [
                "traffic/dataset I/O",
                "checkpoint loading",
                "static topology-cache construction",
                "sequential admission",
                "oracle evaluation",
                "metric serialization",
            ],
            "strict_esm": (
                "actual traffic tensors are zero; only ESM tensors reach policy forward"
            ),
            "timing": (
                "wall-clock perf_counter_ns; torch.cuda.synchronize immediately "
                "before and after each forward"
            ),
            "drift_control": (
                "deterministic cyclic order, reversed on alternating cycles"
            ),
            "cold_topology_cache_included": False,
            "cold_start_result": False,
            "sequential_admission_included": False,
            "end_to_end_response_time": False,
            "scope_label": "warm-cache steady-state policy-forward only",
        },
        "configuration": {
            "topology": TOPOLOGY,
            "model_seeds": model_seeds,
            "formal_model_seeds": list(FORMAL_MODEL_SEEDS),
            "snapshot_start_zero_based": args.snapshot_start,
            "batch_sizes": args.batch_sizes,
            "warmup_calls_per_model": args.warmup,
            "timed_calls_per_model": args.iterations,
            "benchmark_rng_seed": args.seed,
            "deterministic_algorithms": True,
            "garbage_collection_disabled_only_during_timing": True,
        },
        "environment": {
            "device": str(device),
            "gpu": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
            "cpu": platform.processor(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "provenance": {
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "runtime": {
                "path": str(RUNTIME_PATH.resolve()),
                "sha256": sha256(RUNTIME_PATH),
            },
            "frozen_sources": frozen_hashes,
            "artifact_sets": [artifact.audit for artifact in artifact_sets],
            "all_checkpoint_state_keys_and_shapes_equal": True,
            "all_artifact_sets_complete": True,
            "all_config_seeds_match": True,
            "all_continuations_match_their_phase_a_sha256": True,
            "unverified_legacy_artifact_fallback_enabled": False,
            "model_constructor": "frameworks.hattrick_system.Hattrick",
        },
        "seed_results": seed_results,
        "across_seed_latency": across_seed_latency(seed_results),
        "batch_results": (
            seed_results[0]["batch_results"] if len(seed_results) == 1 else None
        ),
    }

    output = args.output.resolve()
    if args.output.resolve() == DEFAULT_OUTPUT.resolve() and not formal_result:
        seed_label = "_".join(str(seed) for seed in model_seeds)
        run_label = "smoke" if args.smoke else "nonformal"
        output = (
            DEFAULT_OUTPUT.parent
            / f"benchmark_latency_{comparison_scope}_{seed_label}_{run_label}.json"
        ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
