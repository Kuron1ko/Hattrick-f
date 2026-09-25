from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hybrid = load_module("hybrid_level4_base", HERE / "run_hybrid_screen.py")
base = hybrid.base
CONFIG = (8, 0.08, 0.10, 4.00)
GATE = 0.70


def compact_rows(rows):
    return [
        {
            "snapshot": int(row["snapshot"]),
            "class": str(row["class"]),
            "norm_fulfill": float(row["norm_fulfill"]),
        }
        for row in rows
    ]


def main():
    base.runtime.set_seed(20260823)
    base.GATE = GATE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "method": "Hattrick-SR: short strict-ESM Medium refinement with stable Low toll",
        "config": list(CONFIG),
        "gate": GATE,
        "strict_esm": True,
        "actual_tm_used_for_policy": False,
        "evaluation": [400, 500],
        "device": str(device),
        "loads": {},
    }
    for load in (1, 2, 3):
        model, props, library = base.screen.load_backbone(load, device)
        prototypes = base.screen.fit_prototypes(library, 3)
        cache = base.runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
        baseline_rows, baseline_summary = base.trainer.evaluate(model, props, cache)
        active = base.gate(cache)
        started = time.perf_counter()
        if bool(active.any().item()):
            value = hybrid.run_candidate(
                model,
                props,
                cache,
                baseline_summary,
                library,
                prototypes,
                CONFIG,
                public_only=False,
            )
            candidate_rows = value.pop("rows")
            candidate_summary = value["summary"]
        else:
            candidate_rows = [dict(row) for row in baseline_rows]
            candidate_summary = base.compact(baseline_summary)
            value = {
                "summary": candidate_summary,
                "delta": base.delta(baseline_summary, baseline_summary),
                "active": 0,
                "config": list(CONFIG),
            }
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        overlay_ms = 1000.0 * (time.perf_counter() - started) / len(cache)
        bootstrap = base.screen.method.large.paired_bootstrap(
            baseline_rows, candidate_rows
        )
        report["loads"][str(load)] = {
            "baseline": base.compact(baseline_summary),
            "candidate": candidate_summary,
            "delta": value["delta"],
            "bootstrap": bootstrap,
            "active": int(active.sum().item()),
            "overlay_ms_per_snapshot": overlay_ms,
            "baseline_rows": compact_rows(baseline_rows),
            "candidate_rows": compact_rows(candidate_rows),
        }
        d = value["delta"]
        print(
            f"[{load}x] active={int(active.sum().item())}/100 "
            f"M={d['Medium.norm_fulfill_mean']:+.6f}/"
            f"{d['Medium.norm_fulfill_p1']:+.6f}/"
            f"{d['Medium.norm_fulfill_p10']:+.6f} "
            f"L={d['Low.norm_fulfill_mean']:+.6f}/"
            f"{d['Low.norm_fulfill_p1']:+.6f}/"
            f"{d['Low.norm_fulfill_p10']:+.6f} ms={overlay_ms:.3f}",
            flush=True,
        )
    path = HERE / "hybrid_level4.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
