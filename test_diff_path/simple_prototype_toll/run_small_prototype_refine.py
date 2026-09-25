from __future__ import annotations

import importlib.util
import json
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


screen = load_module("small_prototype_screen", HERE / "run_screen.py")
trainer = screen.trainer
runtime = screen.runtime
COUNTS = (2, 4, 8)
MEDIUM_GAINS = (0.125, 0.25, 0.375, 0.5)
LOW_GAINS = (0.0, 0.0625, 0.125, 0.25, 0.5)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def evaluate(model, props, cache, baseline, library, prototypes, gains):
    started = time.perf_counter()
    tolls = screen.predict(library, prototypes, trainer.edge_features(cache)).clone()
    tolls[:, 0].mul_(gains[0])
    tolls[:, 1].mul_(gains[1])
    routed = trainer.route_with_tolls(cache, tolls, 1.0)
    elapsed = time.perf_counter() - started
    _, summary = trainer.evaluate(model, props, cache, routed)
    return {
        "summary": trainer.probe.compact(summary),
        "delta": trainer.probe.gaps(summary, baseline),
        "inference_ms_per_snapshot": 1000.0 * elapsed / len(cache),
    }


def main() -> None:
    torch.manual_seed(20260828)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    payload = {
        "method": "two-to-eight congestion prototypes with fixed edge tolls",
        "strict_esm_inference": True,
        "actual_tm_used_for_policy": False,
        "screen_only": True,
        "evaluation_400_500_used": False,
        "loads": {},
    }
    for load_factor in (2, 3):
        print(f"[{load_factor}x] preparing caches", flush=True)
        model, props, library = screen.load_backbone(load_factor, device)
        caches = {
            name: runtime.build_policy_cache(model, props, *bounds, batch_size=32)
            for name, bounds in screen.SPLITS.items()
        }
        baselines = {}
        for name, cache in caches.items():
            _, baselines[name] = trainer.evaluate(model, props, cache)
        trials = []
        for count in COUNTS:
            prototypes = screen.fit_prototypes(library, count)
            for medium_gain in MEDIUM_GAINS:
                for low_gain in LOW_GAINS:
                    gains = (medium_gain, low_gain)
                    splits = {
                        name: evaluate(
                            model,
                            props,
                            cache,
                            baselines[name],
                            library,
                            prototypes,
                            gains,
                        )
                        for name, cache in caches.items()
                    }
                    trials.append(
                        {
                            "prototypes": count,
                            "medium_gain": medium_gain,
                            "low_gain": low_gain,
                            "cluster_sizes": prototypes["cluster_sizes"],
                            "splits": splits,
                        }
                    )
            print(f"[{load_factor}x] finished P={count}", flush=True)
        payload["loads"][str(load_factor)] = {"trials": trials}
        del model, props, library, caches, baselines
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload["seconds"] = time.perf_counter() - started
    write_json(HERE / "small_prototype_refine_report.json", payload)
    print(json.dumps({"artifact": str(HERE / 'small_prototype_refine_report.json'), "seconds": payload["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
