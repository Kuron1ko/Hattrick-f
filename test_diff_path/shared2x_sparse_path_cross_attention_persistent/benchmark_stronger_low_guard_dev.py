from __future__ import annotations

"""Online latency of the development Low-only guard, using snapshots <=349."""

import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
CORE = THIS_DIR / "level4_selection_validation_only" / "selected_checkpoint.pt"
WINNER = (
    THIS_DIR
    / "artifacts"
    / "development_stronger_low_guard_300_349"
    / "development_winner.pt"
)
NATIVE = (
    TEST_DIR
    / "shared2x_full_objectives"
    / "artifacts"
    / "level4_confirmation"
    / "seed_490"
    / "final_model.pt"
)
MODEL_RUNNER = THIS_DIR / "run_experiment.py"
GUARD_RUNNER = THIS_DIR / "train_low_guard.py"
EDGE_RUNNER = TEST_DIR / "shared2x_edge_toll_head" / "run_experiment.py"
STRICT_RUNNER = (
    TEST_DIR / "shared2x_sparse_path_cross_attention" / "evaluate_strict_esm_sequential.py"
)
OUTPUT = (
    THIS_DIR
    / "artifacts"
    / "development_stronger_low_guard_300_349"
    / "latency_strict_esm.json"
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(callable_, warmups=10, repetitions=30):
    for _ in range(warmups):
        callable_()
    sync()
    values = []
    for _ in range(repetitions):
        sync()
        start = time.perf_counter_ns()
        callable_()
        sync()
        values.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "warmups": warmups,
        "repetitions": repetitions,
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": float(np.percentile(np.asarray(values), 95)),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main():
    model_module = load("strong_latency_model", MODEL_RUNNER)
    guard = load("strong_latency_guard", GUARD_RUNNER)
    edge = load("strong_latency_edge", EDGE_RUNNER)
    strict = load("strong_latency_strict", STRICT_RUNNER)
    runtime = edge.runtime
    runtime.set_seed(20260824)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)

    core_payload = torch.load(CORE, map_location=device, weights_only=False)
    core = model_module.PersistentStage2SparseAttentionHattrick(props).to(
        device=device, dtype=props.dtype
    )
    core.load_state_dict(core_payload["model_state_dict"], strict=True)
    core.eval()
    native_payload = torch.load(NATIVE, map_location=device, weights_only=False)
    native = runtime.Hattrick(props).to(device=device, dtype=props.dtype)
    native.load_state_dict(native_payload["model_state_dict"], strict=True)
    native.eval()
    winner = torch.load(WINNER, map_location=device, weights_only=False)
    head = guard.LowGuard(
        int(winner["edge_count"]),
        int(winner["feature_count"]),
        float(winner["max_toll"]),
    ).to(device)
    head.load_state_dict(winner["state_dict"], strict=True)
    head.eval()

    report = {
        "status": "development latency; no snapshot >=350 constructed",
        "scope": "resident-model online strict-ESM forward; dataset I/O excluded",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
        "torch": torch.__version__,
        "batches": {},
        "test_data_read": False,
    }
    for batch_size in (1, 16, 20):
        start = 300
        end = start + batch_size
        dataset = runtime.DM_Dataset_within_Cluster(props, 0, start, end)
        if int(dataset.max_source_index_read) >= 350:
            raise RuntimeError("Latency benchmark crossed the development boundary")
        path_masks = runtime.shared.base.move_dataset_static(dataset, device)
        loader = runtime.shared.data_loader(dataset, batch_size, False, 0)
        original = runtime.shared.unpack_to_device(next(iter(loader)), props)
        values, _ = strict.policy_actual_inputs(
            original, "zero", int(dataset.num_pairs), int(props.num_paths_per_pair)
        )
        predicted_tms = (values[3], values[5], values[7])
        capacities = values[1]
        if capacities.shape[0] == 1 and batch_size > 1:
            capacities_live = capacities.expand(batch_size, -1)
        else:
            capacities_live = capacities
        pte = dataset.pte.coalesce().to(device=device, dtype=torch.float32)
        props.mode = "test"
        props.sim_mf_mlu = 0
        props.research_return_policy = True
        strict.remove_topology_cache(core)
        strict.remove_topology_cache(native)

        def core_forward():
            with torch.no_grad():
                return runtime.cached_policy_forward(
                    core, props, dataset, values, path_masks
                )

        def native_forward():
            with torch.no_grad():
                return runtime.cached_policy_forward(
                    native, props, dataset, values, path_masks
                )

        def refine(policies):
            with torch.no_grad():
                loads = []
                for policy, demand in zip(policies, predicted_tms):
                    flow = policy.squeeze(-1).float() * demand.squeeze(-1).float()
                    loads.append(
                        torch.sparse.mm(pte.t(), flow.t()).t()
                        / capacities_live.float().clamp_min(1e-9)
                    )
                high, medium, low = loads
                features = torch.stack(
                    (
                        high,
                        medium,
                        low,
                        high + medium,
                        high + medium + low,
                        torch.relu(1.0 - high),
                        torch.relu(1.0 - high - medium),
                    ),
                    dim=-1,
                )
                tolls = head(features)
                routed = guard.route_low(policies[2].squeeze(-1), tolls, pte)
                return policies[0], policies[1], routed.unsqueeze(-1)

        resident_core_policy = core_forward()

        def refinement_only():
            return refine(resident_core_policy)

        def composite():
            return refine(core_forward())

        native_metrics = measure(native_forward)
        core_metrics = measure(core_forward)
        refine_metrics = measure(refinement_only)
        composite_metrics = measure(composite)
        report["batches"][str(batch_size)] = {
            "native_hattrick": native_metrics,
            "persistent_core": core_metrics,
            "low_guard_increment": refine_metrics,
            "composite": composite_metrics,
            "composite_over_persistent_core": composite_metrics["mean_ms"]
            / core_metrics["mean_ms"],
            "composite_over_native_hattrick": composite_metrics["mean_ms"]
            / native_metrics["mean_ms"],
            "passes_2x_native": composite_metrics["mean_ms"]
            <= 2.0 * native_metrics["mean_ms"],
        }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
