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


screen = load_module("load_gate_screen", HERE / "run_screen.py")
trainer = screen.trainer
runtime = screen.runtime
THRESHOLDS = (0.68, 0.70, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.80, 0.82)
MEDIUM_GAIN = 0.20
LOW_GAIN = 0.40
PROTOTYPES = 3


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def evaluate(model, props, cache, baseline, library, prototypes, threshold):
    features = trainer.edge_features(cache)
    total_mean = features[:, :, 4].mean(dim=1)
    active = total_mean >= float(threshold)
    started = time.perf_counter()
    tolls = screen.predict(library, prototypes, features).clone()
    tolls[:, 0].mul_(MEDIUM_GAIN)
    tolls[:, 1].mul_(LOW_GAIN)
    tolls.mul_(active[:, None, None])
    routed = trainer.route_with_tolls(cache, tolls, 1.0)
    elapsed = time.perf_counter() - started
    _, summary = trainer.evaluate(model, props, cache, routed)
    return {
        "summary": trainer.probe.compact(summary),
        "delta": trainer.probe.gaps(summary, baseline),
        "active_count": int(active.sum().item()),
        "sample_count": len(cache),
        "total_mean_range": [float(total_mean.min().item()), float(total_mean.max().item())],
        "overlay_ms_per_snapshot": 1000.0 * elapsed / len(cache),
    }


def main() -> None:
    torch.manual_seed(20260828)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    payload = {
        "method": "three congestion prototypes with one strict-ESM mean-load fallback gate",
        "strict_esm_inference": True,
        "actual_tm_used_for_policy": False,
        "evaluation_400_500_used": False,
        "prototypes": PROTOTYPES,
        "medium_gain": MEDIUM_GAIN,
        "low_gain": LOW_GAIN,
        "threshold_trials": list(THRESHOLDS),
        "loads": {},
    }
    for load_factor in (1, 2, 3):
        model, props, library = screen.load_backbone(load_factor, device)
        prototypes = screen.fit_prototypes(library, PROTOTYPES)
        caches = {
            name: runtime.build_policy_cache(model, props, *bounds, batch_size=32)
            for name, bounds in screen.SPLITS.items()
        }
        baselines = {}
        for name, cache in caches.items():
            _, baselines[name] = trainer.evaluate(model, props, cache)
        trials = []
        for threshold in THRESHOLDS:
            splits = {
                name: evaluate(
                    model,
                    props,
                    cache,
                    baselines[name],
                    library,
                    prototypes,
                    threshold,
                )
                for name, cache in caches.items()
            }
            trials.append({"threshold": threshold, "splits": splits})
            s, v = splits["safety"], splits["validation"]
            print(
                f"[{load_factor}x] t={threshold:.2f} "
                f"active={s['active_count']}/{s['sample_count']},"
                f"{v['active_count']}/{v['sample_count']} "
                f"dM={s['delta']['Medium.norm_fulfill_mean']:+.5f},"
                f"{v['delta']['Medium.norm_fulfill_mean']:+.5f} "
                f"dL={s['delta']['Low.norm_fulfill_mean']:+.5f},"
                f"{v['delta']['Low.norm_fulfill_mean']:+.5f}",
                flush=True,
            )
        payload["loads"][str(load_factor)] = {"trials": trials}
        del model, props, library, prototypes, caches, baselines
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload["seconds"] = time.perf_counter() - started
    write_json(HERE / "load_gate_screen_report.json", payload)
    print(json.dumps({"artifact": str(HERE / 'load_gate_screen_report.json'), "seconds": payload["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
