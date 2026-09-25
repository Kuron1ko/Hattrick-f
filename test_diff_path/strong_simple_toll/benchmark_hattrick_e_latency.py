from __future__ import annotations

import importlib.util
import json
import platform
from pathlib import Path
import sys
import time

import torch


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


adaptive = load_module(
    "latency_rqp_slack_weight", HERE / "probe_rqp_slack_weight.py"
)
base = adaptive.base
runtime = base.runtime


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(callable_, warmups: int, repetitions: int) -> dict[str, float | int]:
    for _ in range(warmups):
        callable_()
    synchronize()
    started = time.perf_counter()
    for _ in range(repetitions):
        callable_()
    synchronize()
    milliseconds = 1000.0 * (time.perf_counter() - started) / repetitions
    return {"warmups": warmups, "repetitions": repetitions, "mean_ms": milliseconds}


def single_forward_inputs(model, props, snapshot: int):
    dataset = runtime.DM_Dataset_within_Cluster(props, 0, snapshot, snapshot + 1)
    path_masks = runtime.shared.base.move_dataset_static(dataset, props.device)
    loader = runtime.shared.data_loader(dataset, 1, False, 0)
    values = runtime.shared.unpack_to_device(next(iter(loader)), props)
    props.mode = "test"
    props.sim_mf_mlu = 0
    props.research_return_policy = True

    def forward():
        with torch.no_grad():
            return runtime.cached_policy_forward(
                model, props, dataset, values, path_masks
            )

    return forward


def main() -> None:
    runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict[str, object] = {
        "device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
        ),
        "torch": torch.__version__,
        "snapshot": 400,
        "hattrick_e": {
            "strict_raw_esm": True,
            "kl_anchor_weight": 0.0,
            "adam_steps": adaptive.final.STEPS,
        },
        "loads": {},
    }
    for load in (1, 2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        forward = single_forward_inputs(model, props, 400)
        hattrick = measure(forward, warmups=5, repetitions=30)

        cache = runtime.build_policy_cache(model, props, 400, 401, batch_size=1)
        factors = tuple(torch.ones_like(value) for value in cache.predicted_tms)

        def refine():
            return adaptive.construct(
                model,
                props,
                cache,
                factors,
                {"threshold": 0.90, "low_floor": -0.005, "anchor_weight": 0.0},
            )

        refinement = measure(refine, warmups=1, repetitions=5)
        report["loads"][str(load)] = {
            "hattrick_forward": hattrick,
            "hattrick_e_refinement": refinement,
            "end_to_end_mean_ms": hattrick["mean_ms"] + refinement["mean_ms"],
            "slowdown_vs_hattrick": (
                hattrick["mean_ms"] + refinement["mean_ms"]
            )
            / hattrick["mean_ms"],
        }
        print(
            f"{load}x Hattrick={hattrick['mean_ms']:.3f} ms "
            f"refine={refinement['mean_ms']:.3f} ms "
            f"total={report['loads'][str(load)]['end_to_end_mean_ms']:.3f} ms",
            flush=True,
        )

    output = HERE / "hattrick_e_latency.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
