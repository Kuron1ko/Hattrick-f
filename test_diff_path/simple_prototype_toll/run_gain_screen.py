from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


HERE = Path(__file__).resolve().parent
SCREEN_PATH = HERE / "run_screen.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


screen = load_module("simple_prototype_screen", SCREEN_PATH)
trainer = screen.trainer
runtime = screen.runtime
MEDIUM_GAINS = (0.25, 0.5, 0.75, 1.0)
LOW_GAINS = (0.0, 0.125, 0.25, 0.5, 0.75)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def evaluate(model, props, cache, baseline, library, prototype, medium_gain, low_gain):
    started = time.perf_counter()
    tolls = screen.predict(library, prototype, trainer.edge_features(cache)).clone()
    tolls[:, 0].mul_(float(medium_gain))
    tolls[:, 1].mul_(float(low_gain))
    routed = trainer.route_with_tolls(cache, tolls, 1.0)
    inference_seconds = time.perf_counter() - started
    _, summary = trainer.evaluate(model, props, cache, routed)
    return {
        "summary": trainer.probe.compact(summary),
        "delta": trainer.probe.gaps(summary, baseline),
        "inference_ms_per_snapshot": 1000.0 * inference_seconds / len(cache),
    }


def main() -> None:
    torch.manual_seed(20260828)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    payload = {
        "method": "one static mean edge-toll table per load with two shared gains",
        "strict_esm_inference": True,
        "actual_tm_used_for_policy": False,
        "screen_only": True,
        "evaluation_400_500_used": False,
        "gains": [],
    }
    per_load = {}
    for load_factor in (2, 3):
        print(f"[{load_factor}x] preparing one-prototype model", flush=True)
        model, props, library = screen.load_backbone(load_factor, device)
        prototype = screen.fit_prototypes(library, 1)
        caches = {
            name: runtime.build_policy_cache(model, props, *bounds, batch_size=32)
            for name, bounds in screen.SPLITS.items()
        }
        baselines = {}
        for name, cache in caches.items():
            _, baselines[name] = trainer.evaluate(model, props, cache)
        load_trials = {}
        for medium_gain in MEDIUM_GAINS:
            for low_gain in LOW_GAINS:
                key = f"m{medium_gain:g}_l{low_gain:g}"
                splits = {
                    name: evaluate(
                        model,
                        props,
                        cache,
                        baselines[name],
                        library,
                        prototype,
                        medium_gain,
                        low_gain,
                    )
                    for name, cache in caches.items()
                }
                load_trials[key] = {
                    "medium_gain": medium_gain,
                    "low_gain": low_gain,
                    "splits": splits,
                }
                s, v = splits["safety"]["delta"], splits["validation"]["delta"]
                print(
                    f"[{load_factor}x] gM={medium_gain:.3f} gL={low_gain:.3f} "
                    f"S({s['Medium.norm_fulfill_mean']:+.5f},"
                    f"{s['Low.norm_fulfill_mean']:+.5f}) "
                    f"V({v['Medium.norm_fulfill_mean']:+.5f},"
                    f"{v['Low.norm_fulfill_mean']:+.5f})",
                    flush=True,
                )
        per_load[str(load_factor)] = {
            "baseline": {
                name: trainer.probe.compact(value) for name, value in baselines.items()
            },
            "trials": load_trials,
        }
        del model, props, library, prototype, caches, baselines
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for medium_gain in MEDIUM_GAINS:
        for low_gain in LOW_GAINS:
            key = f"m{medium_gain:g}_l{low_gain:g}"
            payload["gains"].append(
                {
                    "medium_gain": medium_gain,
                    "low_gain": low_gain,
                    "loads": {
                        load: per_load[load]["trials"][key] for load in ("2", "3")
                    },
                }
            )
    payload["loads"] = per_load
    payload["seconds"] = time.perf_counter() - started
    write_json(HERE / "gain_screen_report.json", payload)
    print(json.dumps({"artifact": str(HERE / 'gain_screen_report.json'), "seconds": payload["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
