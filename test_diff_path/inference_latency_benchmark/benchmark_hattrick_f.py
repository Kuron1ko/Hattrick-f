from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

# The traffic snapshots were serialized by NumPy 2.x, while the repository's
# pinned runtime exposes the same modules through the NumPy 1.x public aliases.
if not hasattr(np, "_core"):
    sys.modules.setdefault("numpy._core", np.core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "test_diff_path"
RUNTIME_PATH = TEST_DIR / "shared2x_medium_adapter" / "run_experiment.py"
OUTPUT_DIR = ROOT / "output" / "benchmarks" / "hattrick_f_latency"

RUNS = {
    "1x": {
        "topology": "geant_priomask500_shared",
        "hattrick": TEST_DIR
        / "shared1x_full_objectives/artifacts/level4_confirmation/seed_490/best_model.pt",
        "hattrick_f": TEST_DIR
        / "shared1x_hattrick_f/artifacts/level4_confirmation/fh_release_low_budget_0p03/seed_490/best_model.pt",
    },
    "2x": {
        "topology": "geant_priomask500_shared_load2x_train",
        "hattrick": TEST_DIR
        / "shared2x_full_objectives/artifacts/level4_confirmation/seed_490/best_model.pt",
        "hattrick_f": TEST_DIR
        / "shared2x_hattrick_f/artifacts/level4_confirmation/fh_release_low_budget_0p03/seed_490/best_model.pt",
    },
}


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


def state_dict(payload: dict) -> dict[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict", "model_state"):
        if key in payload:
            return payload[key]
    raise KeyError("checkpoint has no model state dictionary")


def make_props(runtime, device: torch.device):
    props = runtime.build_props(4, device)
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_policy = True
    props.research_return_admitted = False
    return props


def prepare_input(runtime, props):
    dataset = runtime.DM_Dataset_within_Cluster(props, 0, 400, 401)
    if int(dataset.max_source_index_read) != 400:
        raise RuntimeError("test-window reader audit failed")
    path_masks = runtime.shared.base.move_dataset_static(dataset, props.device)
    loader = runtime.shared.data_loader(dataset, 1, False, 0)
    values = list(runtime.shared.unpack_to_device(next(iter(loader)), props))
    predicted_l1 = []
    for actual_index, predicted_index in ((2, 3), (4, 5), (6, 7)):
        predicted_l1.append(float(values[predicted_index].abs().sum().item()))
        values[actual_index] = torch.zeros_like(values[actual_index])
    if not all(value > 0.0 for value in predicted_l1):
        raise RuntimeError("strict ESM input audit failed")
    return dataset, path_masks, tuple(values), predicted_l1


def load_model(runtime, props, device: torch.device, checkpoint_path: Path):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = runtime.Hattrick(props).to(device=device, dtype=props.dtype)
    model.load_state_dict(state_dict(payload), strict=True)
    model.eval()
    return model, payload


def forward(runtime, model, props, dataset, values, path_masks):
    return runtime.cached_policy_forward(model, props, dataset, values, path_masks)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_once(callable_, device: torch.device) -> float:
    if device.type == "cuda":
        sync(device)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        callable_()
        end.record()
        end.synchronize()
        return float(begin.elapsed_time(end))
    started = time.perf_counter()
    callable_()
    return 1000.0 * (time.perf_counter() - started)


def latency_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "p05_ms": float(np.percentile(array, 5.0)),
        "p95_ms": float(np.percentile(array, 95.0)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def policy_audit(policies) -> dict:
    return {
        "shapes": [list(value.shape) for value in policies],
        "checksums": [float(value.float().sum().item()) for value in policies],
    }


def benchmark_load(runtime, label: str, spec: dict, device: torch.device, warmup: int, iterations: int):
    runtime.shared.TOPOLOGY = spec["topology"]
    props = make_props(runtime, device)
    dataset, path_masks, values, predicted_l1 = prepare_input(runtime, props)
    models = {}
    payloads = {}
    for name in ("hattrick", "hattrick_f"):
        models[name], payloads[name] = load_model(
            runtime, props, device, Path(spec[name]).resolve()
        )

    state_shapes = {
        name: [(key, list(value.shape)) for key, value in state_dict(payloads[name]).items()]
        for name in models
    }
    if state_shapes["hattrick"] != state_shapes["hattrick_f"]:
        raise RuntimeError("model structures differ")

    calls = {
        name: (
            lambda model=model: forward(
                runtime, model, props, dataset, values, path_masks
            )
        )
        for name, model in models.items()
    }
    policies = {}
    with torch.no_grad():
        # Materialize the identical topology cache before warmup/timing.
        for name in ("hattrick", "hattrick_f"):
            if hasattr(models[name], "transformer_output"):
                delattr(models[name], "transformer_output")
            policies[name] = calls[name]()
        for index in range(warmup):
            order = ("hattrick", "hattrick_f") if index % 2 == 0 else ("hattrick_f", "hattrick")
            for name in order:
                policies[name] = calls[name]()
    sync(device)

    samples = {"hattrick": [], "hattrick_f": []}
    with torch.no_grad():
        # Alternate order so clock/thermal drift is shared by both checkpoints.
        for index in range(iterations):
            order = ("hattrick", "hattrick_f") if index % 2 == 0 else ("hattrick_f", "hattrick")
            for name in order:
                samples[name].append(time_once(calls[name], device))
    sync(device)

    summaries = {name: latency_summary(samples[name]) for name in samples}
    baseline = summaries["hattrick"]["median_ms"]
    candidate = summaries["hattrick_f"]["median_ms"]
    parameter_count = int(sum(value.numel() for value in models["hattrick"].parameters()))
    result = {
        "load": label,
        "topology": spec["topology"],
        "snapshot": 400,
        "actual_traffic_zeroed": True,
        "predicted_tm_l1": predicted_l1,
        "warmup_per_model": warmup,
        "timed_iterations_per_model": iterations,
        "same_constructor": "frameworks.hattrick_system.Hattrick",
        "same_state_keys_and_shapes": True,
        "parameter_count_each": parameter_count,
        "latency": summaries,
        "hattrick_f_minus_hattrick_median_ms": candidate - baseline,
        "hattrick_f_over_hattrick_median_ratio": candidate / baseline,
        "checkpoints": {
            name: {
                "path": str(Path(spec[name]).resolve()),
                "sha256": sha256(Path(spec[name]).resolve()),
                "epoch": int(payloads[name]["epoch"]),
            }
            for name in models
        },
        "policy_audit": {name: policy_audit(policies[name]) for name in policies},
    }
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Hattrick vs Hattrick-f strict-ESM latency")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    for path in (ROOT, TEST_DIR, TEST_DIR / "shared2x_order_regularizer"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    runtime = load_module("hattrick_f_latency_runtime", RUNTIME_PATH)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(20260825)
    results = [
        benchmark_load(runtime, label, spec, device, args.warmup, args.iterations)
        for label, spec in RUNS.items()
    ]
    payload = {
        "benchmark": "strict-ESM steady-state batch-1 policy forward",
        "timing": "interleaved checkpoints; CUDA Events; synchronized each call",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "results": results,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "hattrick_vs_hattrick_f_latency.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    print(output_path, flush=True)


if __name__ == "__main__":
    main()
