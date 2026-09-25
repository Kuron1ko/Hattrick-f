from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path
import sys

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


universal = load_module("universal_softmin_screen_base", HERE / "probe_universal_num.py")
base = universal.base


def nonnegative(delta, tolerance=-1e-5):
    return all(value >= tolerance for value in delta.values())


def score(delta):
    high = min(
        delta[f"High.{metric}"]
        for metric in ("norm_fulfill_mean", "norm_fulfill_p1", "norm_fulfill_p10")
    )
    medium = min(
        delta[f"Medium.{metric}"]
        for metric in ("norm_fulfill_mean", "norm_fulfill_p1", "norm_fulfill_p10")
    )
    return high + 0.25 * medium


def evaluate(model, props, cache, baseline, config):
    candidate = universal.correct(model, props, cache, config)
    _, summary = base.runtime.evaluate_cache(
        model, props, candidate, None, batch_size=16
    )
    delta = base.delta(summary, baseline)
    return {
        "config": list(config),
        "summary": base.compact(summary),
        "delta": delta,
        "all_metrics_nonnegative": nonnegative(delta),
        "score": score(delta),
    }


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props, _ = base.screen.load_backbone(3, device)
    cache = base.runtime.build_policy_cache(model, props, 350, 400, batch_size=32)
    _, baseline = base.trainer.evaluate(model, props, cache)
    validation = []
    for tau, steps, learning_rate in itertools.product(
        (0.005, 0.008, 0.010, 0.012),
        (12, 16, 20, 24),
        (0.04, 0.06, 0.08),
    ):
        config = (f"relative_softmin_{tau:g}", steps, learning_rate, 0.02)
        value = evaluate(model, props, cache, baseline, config)
        validation.append(value)
        if value["all_metrics_nonnegative"]:
            d = value["delta"]
            print(
                f"[validation] tau={tau:g} s={steps} lr={learning_rate:g} "
                f"score={value['score']:.5f} "
                f"H={d['High.norm_fulfill_mean']:+.5f}/"
                f"{d['High.norm_fulfill_p1']:+.5f}/"
                f"{d['High.norm_fulfill_p10']:+.5f} "
                f"M={d['Medium.norm_fulfill_mean']:+.5f}/"
                f"{d['Medium.norm_fulfill_p1']:+.5f}/"
                f"{d['Medium.norm_fulfill_p10']:+.5f}",
                flush=True,
            )
    validation.sort(
        key=lambda value: (value["all_metrics_nonnegative"], value["score"]),
        reverse=True,
    )
    frozen = [value["config"] for value in validation[:6]]
    evaluation_cache = base.runtime.build_policy_cache(
        model, props, 400, 500, batch_size=32
    )
    _, evaluation_baseline = base.trainer.evaluate(model, props, evaluation_cache)
    evaluation = []
    for config in frozen:
        value = evaluate(
            model, props, evaluation_cache, evaluation_baseline, tuple(config)
        )
        evaluation.append(value)
        d = value["delta"]
        print(
            f"[evaluation] {config} all={value['all_metrics_nonnegative']} "
            f"H={d['High.norm_fulfill_mean']:+.5f}/"
            f"{d['High.norm_fulfill_p1']:+.5f}/"
            f"{d['High.norm_fulfill_p10']:+.5f} "
            f"M={d['Medium.norm_fulfill_mean']:+.5f}/"
            f"{d['Medium.norm_fulfill_p1']:+.5f}/"
            f"{d['Medium.norm_fulfill_p10']:+.5f} "
            f"L={d['Low.norm_fulfill_mean']:+.5f}/"
            f"{d['Low.norm_fulfill_p1']:+.5f}/"
            f"{d['Low.norm_fulfill_p10']:+.5f}",
            flush=True,
        )
    report = {
        "method": "class-symmetric max-min relative gain",
        "strict_esm": True,
        "selection_bounds": [350, 400],
        "evaluation_bounds": [400, 500],
        "validation": validation,
        "frozen_configs": frozen,
        "evaluation": evaluation,
    }
    path = HERE / "universal_softmin_screen.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
