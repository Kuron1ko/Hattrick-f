from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
WITH_KL_REPORT = HERE / "rqp_slack_esm_ablation.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


slack = load_module(
    "rqp_slack_kl_ablation_base", HERE / "run_final_rqp_slack_weight.py"
)
adaptive, final, base = slack.adaptive, slack.final, slack.base

NO_KL_SPEC = {
    "threshold": slack.SPEC["threshold"],
    "low_floor": slack.SPEC["low_floor"],
    "anchor_weight": 0.0,
}


def raw_factors(cache):
    return tuple(torch.ones_like(value[:1]) for value in cache.predicted_tms)


def expand_summary(compact):
    return [
        {"class": class_name, **metrics}
        for class_name, metrics in compact.items()
    ]


def max_policy_diff(left, right):
    return max(
        float((a - b).abs().max())
        for a, b in zip(left.policies, right.policies)
    )


def information_audit(model, props, cache, factors):
    single = final.slice_cache(cache, 1)
    original, original_active, _ = adaptive.construct(
        model, props, single, factors, NO_KL_SPEC
    )
    counterfactual, counterfactual_active, _ = adaptive.construct(
        model,
        props,
        replace(
            single,
            tms=tuple(torch.zeros_like(value) for value in single.tms),
        ),
        factors,
        NO_KL_SPEC,
    )
    return {
        "single_zero_actual_policy_max_abs_diff": max_policy_diff(
            original, counterfactual
        ),
        "single_zero_actual_activation_mismatch": int(
            (original_active != counterfactual_active).sum()
        ),
        "current_actual_tm_used_online": False,
    }


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with_kl_payload = json.loads(WITH_KL_REPORT.read_text(encoding="utf-8"))
    report = {
        "method": "strict one-variable ablation of the RQP-S KL anchor",
        "strict_esm_online": True,
        "current_actual_tm_used_online": False,
        "esm_variant": "raw ESM",
        "fixed_components": [
            "backbone",
            "Level-4 snapshots 400-499",
            "High-slack adaptive weights",
            "40 Adam steps and learning rate 0.08",
            "soft-min and mean-gain terms",
            "High/Medium/Low predicted guards",
        ],
        "ablated_component": "KL(P,P0) anchor",
        "lambda_with_kl": float(adaptive.priority.ANCHOR),
        "lambda_without_kl": 0.0,
        "loads": {},
    }
    for load in (1, 2, 3):
        model, props, _ = base.screen.load_backbone(load, device)
        cache = base.runtime.build_policy_cache(
            model, props, 400, 500, batch_size=32
        )
        baseline_rows, baseline_summary = base.trainer.evaluate(
            model, props, cache
        )
        factors = raw_factors(cache)
        candidate, active, gate = adaptive.construct(
            model, props, cache, factors, NO_KL_SPEC
        )
        without_rows, without_summary = base.runtime.evaluate_cache(
            model, props, candidate, None, batch_size=16
        )
        reference = with_kl_payload["loads"][str(load)]["evaluation"]
        with_summary = reference["raw_esm"]
        with_rows = reference["rows"]["raw_esm"]
        compact_baseline = final.compact_rows(baseline_rows)
        compact_without = final.compact_rows(without_rows)
        report["loads"][str(load)] = {
            "active_without_kl": int(active.sum()),
            "medium_gate_without_kl": {
                "mean": float(gate.mean()),
                "p10": float(torch.quantile(gate, 0.10)),
                "p90": float(torch.quantile(gate, 0.90)),
            },
            "evaluation": {
                "baseline": base.compact(baseline_summary),
                "with_kl": with_summary,
                "without_kl": base.compact(without_summary),
                "delta_with_kl_vs_hattrick": reference[
                    "delta_raw_vs_hattrick"
                ],
                "delta_without_kl_vs_hattrick": base.delta(
                    without_summary, baseline_summary
                ),
                "delta_without_vs_with_kl": base.delta(
                    without_summary, expand_summary(with_summary)
                ),
                "bootstrap_without_vs_with_kl": (
                    base.screen.method.large.paired_bootstrap(
                        with_rows, compact_without
                    )
                ),
                "bootstrap_without_vs_hattrick": (
                    base.screen.method.large.paired_bootstrap(
                        compact_baseline, compact_without
                    )
                ),
                "rows": {
                    "baseline": compact_baseline,
                    "with_kl": with_rows,
                    "without_kl": compact_without,
                },
            },
            "information_audit_without_kl": information_audit(
                model, props, cache, factors
            ),
        }
        d = report["loads"][str(load)]["evaluation"][
            "delta_without_vs_with_kl"
        ]
        print(
            f"[{load}x noKL-withKL] "
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
    path = HERE / "rqp_slack_kl_ablation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
