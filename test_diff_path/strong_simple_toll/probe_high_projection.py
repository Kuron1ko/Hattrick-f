from __future__ import annotations

from dataclasses import replace
import importlib.util
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


base = load_module("high_projection_base", HERE / "probe_high_manifold.py")
LOW_WEIGHTS = (0.10, 0.20, 0.24, 0.30, 0.40)
PENALTIES = (0.0, 3.0, 10.0, 30.0)


def project_to_high_floor(model, props, cache, candidate, tolerance=1e-7):
    original = [value.squeeze(-1).detach() for value in cache.policies]
    proposed = [value.squeeze(-1).detach() for value in candidate.policies]
    with torch.no_grad():
        baseline = base.base.correction.predicted_fulfillment(
            model, props, cache, list(cache.policies)
        )[:, 0]
        best_alpha = torch.zeros_like(baseline)
        # A dense scalar scan is robust to small non-monotonicities of admission.
        for scalar in torch.linspace(0.0, 1.0, 101, device=baseline.device):
            policies = [
                (1.0 - scalar) * old + scalar * new
                for old, new in zip(original, proposed)
            ]
            fulfill = base.base.correction.predicted_fulfillment(
                model,
                props,
                cache,
                [value.unsqueeze(-1) for value in policies],
            )[:, 0]
            feasible = fulfill >= baseline - tolerance
            best_alpha = torch.where(
                feasible & (scalar > best_alpha), scalar, best_alpha
            )
        alpha = best_alpha[:, None]
        projected = [
            (1.0 - alpha) * old + alpha * new
            for old, new in zip(original, proposed)
        ]
    return replace(
        cache,
        policies=tuple(value.unsqueeze(-1) for value in projected),
        path_features=torch.cat([projected[1], projected[2]], dim=1),
    ), best_alpha


def main():
    base.base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "purpose": "actual-TM per-snapshot projection onto Hattrick High floor",
        "deployable": False,
        "loads": {},
    }
    for load in (2, 3):
        model, props, _ = base.base.screen.load_backbone(load, device)
        cache = base.base.runtime.build_policy_cache(
            model, props, 400, 500, batch_size=32
        )
        _, baseline = base.base.trainer.evaluate(model, props, cache)
        oracle_cache = replace(cache, predicted_tms=cache.tms)
        trials = []
        for penalty in PENALTIES:
            for low_weight in LOW_WEIGHTS:
                raw = base.correct_on_manifold(
                    model, props, oracle_cache, low_weight, penalty
                )
                projected, alpha = project_to_high_floor(
                    model, props, oracle_cache, raw
                )
                candidate = replace(
                    cache,
                    policies=projected.policies,
                    path_features=projected.path_features,
                )
                _, summary = base.base.runtime.evaluate_cache(
                    model, props, candidate, None, batch_size=16
                )
                value = {
                    "penalty": penalty,
                    "low_weight": low_weight,
                    "alpha_mean": float(alpha.mean()),
                    "alpha_p10": float(torch.quantile(alpha, 0.10)),
                    "alpha_zero_fraction": float((alpha == 0).float().mean()),
                    "summary": base.base.compact(summary),
                    "delta": base.base.delta(summary, baseline),
                }
                trials.append(value)
                d = value["delta"]
                print(
                    f"[{load}x] rho={penalty:g} l={low_weight:g} "
                    f"a={value['alpha_mean']:.3f}/{value['alpha_p10']:.3f} "
                    f"H={d['High.norm_fulfill_mean']:+.6f}/"
                    f"{d['High.norm_fulfill_p1']:+.6f}/"
                    f"{d['High.norm_fulfill_p10']:+.6f} "
                    f"M={d['Medium.norm_fulfill_mean']:+.6f}/"
                    f"{d['Medium.norm_fulfill_p1']:+.6f}/"
                    f"{d['Medium.norm_fulfill_p10']:+.6f} "
                    f"L={d['Low.norm_fulfill_mean']:+.6f}/"
                    f"{d['Low.norm_fulfill_p1']:+.6f}/"
                    f"{d['Low.norm_fulfill_p10']:+.6f}",
                    flush=True,
                )
        report["loads"][str(load)] = {
            "baseline": base.base.compact(baseline),
            "trials": trials,
        }
    path = HERE / "high_projection.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
