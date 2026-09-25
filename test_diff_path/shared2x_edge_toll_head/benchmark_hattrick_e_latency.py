from __future__ import annotations

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
ROOT = TEST_DIR.parent
ONEX_PATH = TEST_DIR / "shared1x_esm_sar" / "probe_onex_transfer.py"
ONEX_HEAD_PATH = TEST_DIR / "shared1x_edge_toll_head" / "strict_esm_onex_edge_toll.pt"
TWOX_HEAD_PATH = THIS_DIR / "best_edge_toll_head.pt"
OUTPUT = THIS_DIR / "hattrick_e_edge_toll_latency.json"

for item in (str(ROOT), str(TEST_DIR), str(THIS_DIR), str(ONEX_PATH.parent)):
    if item not in sys.path:
        sys.path.insert(0, item)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


edge = load_module("latency_edge_toll", THIS_DIR / "run_experiment.py")
onex = load_module("latency_onex", ONEX_PATH)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def measure(callable_, warmups: int = 20, repetitions: int = 100) -> dict[str, float | int]:
    for _ in range(warmups):
        callable_()
    synchronize()

    samples = []
    for _ in range(repetitions):
        synchronize()
        started = time.perf_counter_ns()
        callable_()
        synchronize()
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)

    return {
        "warmups": warmups,
        "repetitions": repetitions,
        "mean_ms": float(statistics.fmean(samples)),
        "median_ms": float(statistics.median(samples)),
        "p95_ms": percentile(samples, 95),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
    }


def single_snapshot_inputs(runtime, props, snapshot: int):
    dataset = runtime.DM_Dataset_within_Cluster(props, 0, snapshot, snapshot + 1)
    path_masks = runtime.shared.base.move_dataset_static(dataset, props.device)
    loader = runtime.shared.data_loader(dataset, 1, False, 0)
    values = runtime.shared.unpack_to_device(next(iter(loader)), props)
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_policy = True
    return dataset, path_masks, values


def edge_features_from_live_policy(policies, predicted_tms, capacities, pte):
    capacities = capacities.to(dtype=torch.float32).clamp_min(1e-9)
    loads = []
    for policy, demand in zip(policies, predicted_tms):
        flow = policy.squeeze(-1).to(dtype=torch.float32) * demand.squeeze(-1).to(
            dtype=torch.float32
        )
        loads.append(torch.sparse.mm(pte.t(), flow.t()).t() / capacities)
    high, medium, low = loads
    return torch.stack(
        [
            high,
            medium,
            low,
            high + medium,
            high + medium + low,
            torch.relu(1.0 - high),
            torch.relu(1.0 - high - medium),
        ],
        dim=-1,
    )


def route_from_toll(base, toll, pte):
    batch = int(base.shape[0])
    path_cost = torch.sparse.mm(pte, toll.transpose(0, 1)).transpose(0, 1)
    grouped_base = base.reshape(batch, -1, edge.K)
    grouped_cost = path_cost.reshape(batch, -1, edge.K)
    valid = grouped_base > 0
    logits = torch.log(grouped_base.clamp_min(1e-12)) - grouped_cost
    logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
    routed = torch.softmax(logits, dim=-1) * grouped_base.sum(dim=-1, keepdim=True)
    return torch.where(valid, routed, torch.zeros_like(routed)).reshape_as(base)


def load_case(load: int, device: torch.device, snapshot: int):
    if load == 1:
        model, props = onex.load_onex(device)
        runtime = onex.runtime
        saved = torch.load(ONEX_HEAD_PATH, map_location=device, weights_only=False)
        medium_scale = float(saved["medium_scale"])
        low_scale = float(saved["low_scale"])
        checkpoint = ONEX_HEAD_PATH
    elif load == 2:
        # The 1x loader changes the shared topology name, so restore the strict-2x
        # topology before constructing the 2x properties and dataset.
        edge.runtime.shared.TOPOLOGY = "geant_priomask500_shared_load2x_train"
        runtime = edge.runtime
        props = runtime.build_props(4, device)
        model, backbone = runtime.load_backbone(4, 490, props, device)
        saved = torch.load(TWOX_HEAD_PATH, map_location=device, weights_only=False)
        medium_scale = 1.0
        low_scale = 1.0
        checkpoint = TWOX_HEAD_PATH
    else:
        raise ValueError(load)

    model.eval()
    head = edge.EdgeTollHead(saved["edge_count"], saved["feature_count"]).to(device)
    head.load_state_dict(saved["state_dict"])
    head.eval()

    dataset, path_masks, values = single_snapshot_inputs(runtime, props, snapshot)
    pte = dataset.pte.coalesce().to(device=device, dtype=torch.float32)
    predicted_tms = (values[3], values[5], values[7])
    capacities = values[1]

    def hattrick_forward():
        with torch.no_grad():
            return runtime.cached_policy_forward(
                model, props, dataset, values, path_masks
            )

    def refine(policies):
        with torch.no_grad():
            features = edge_features_from_live_policy(
                policies, predicted_tms, capacities, pte
            )
            base = [policies[1].squeeze(-1), policies[2].squeeze(-1)]
            routed, tolls, _ = head(features, base, pte)
            medium = (
                base[0]
                if abs(medium_scale) <= 1e-12
                else (
                    routed[0]
                    if abs(medium_scale - 1.0) <= 1e-12
                    else route_from_toll(
                        base[0], tolls[:, :, 0] * medium_scale, pte
                    )
                )
            )
            low = (
                base[1]
                if abs(low_scale) <= 1e-12
                else (
                    routed[1]
                    if abs(low_scale - 1.0) <= 1e-12
                    else route_from_toll(
                        base[1], tolls[:, :, 1] * low_scale, pte
                    )
                )
            )
            return policies[0], medium.unsqueeze(-1), low.unsqueeze(-1)

    base_policies = hattrick_forward()

    def refinement_only():
        return refine(base_policies)

    def end_to_end():
        return refine(hattrick_forward())

    return {
        "model": model,
        "head": head,
        "checkpoint": str(checkpoint),
        "medium_scale": medium_scale,
        "low_scale": low_scale,
        "hattrick": hattrick_forward,
        "refinement": refinement_only,
        "end_to_end": end_to_end,
    }


def main() -> None:
    torch.manual_seed(20260823)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    snapshot = 400
    report: dict[str, object] = {
        "method": "current Hattrick-e ESM edge-toll residual head",
        "scope": (
            "single-snapshot online inference after the ESM matrix is available; "
            "models/topology are resident and dataset I/O is excluded"
        ),
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else platform.processor() or "CPU"
        ),
        "torch": torch.__version__,
        "snapshot": snapshot,
        "batch_size": 1,
        "loads": {},
    }

    for load in (1, 2):
        case = load_case(load, device, snapshot)
        hattrick = measure(case["hattrick"])
        refinement = measure(case["refinement"])
        end_to_end = measure(case["end_to_end"])
        report["loads"][f"{load}x"] = {
            "checkpoint": case["checkpoint"],
            "medium_scale": case["medium_scale"],
            "low_scale": case["low_scale"],
            "hattrick": hattrick,
            "hattrick_e_increment": refinement,
            "hattrick_e_end_to_end": end_to_end,
            "mean_added_ms": end_to_end["mean_ms"] - hattrick["mean_ms"],
            "mean_slowdown": end_to_end["mean_ms"] / hattrick["mean_ms"],
        }
        print(
            f"{load}x: Hattrick={hattrick['mean_ms']:.3f} ms, "
            f"edge-head={refinement['mean_ms']:.3f} ms, "
            f"Hattrick-e={end_to_end['mean_ms']:.3f} ms, "
            f"slowdown={end_to_end['mean_ms'] / hattrick['mean_ms']:.2f}x",
            flush=True,
        )
        del case
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
