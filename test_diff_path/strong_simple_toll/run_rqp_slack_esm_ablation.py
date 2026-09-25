from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
CORRECTED_REPORT = HERE / "final_rqp_slack_weight.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


slack = load_module(
    "rqp_slack_esm_ablation_base", HERE / "run_final_rqp_slack_weight.py"
)
adaptive, final, base = slack.adaptive, slack.final, slack.base


def raw_factors(cache):
    return tuple(
        torch.ones_like(value[:1]) for value in cache.predicted_tms
    )


def max_row_diff(left, right):
    left_values = {
        (int(row["snapshot"]), str(row["class"])): float(row["norm_fulfill"])
        for row in left
    }
    right_values = {
        (int(row["snapshot"]), str(row["class"])): float(row["norm_fulfill"])
        for row in right
    }
    if left_values.keys() != right_values.keys():
        raise ValueError("row keys differ")
    return max(abs(left_values[key] - right_values[key]) for key in left_values)


def expand_summary(compact):
    return [
        {"class": class_name, **metrics}
        for class_name, metrics in compact.items()
    ]


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corrected = json.loads(CORRECTED_REPORT.read_text(encoding="utf-8"))
    report = {
        "method": "strict one-variable ablation of RQP-S ESM correction",
        "fixed_components": [
            "backbone",
            "Level-4 snapshots 400-499",
            "40-step adaptive routing objective",
            "High-slack threshold and temperature",
            "High/Medium weights",
            "High/Medium/Low predicted guards",
        ],
        "ablated_component": "55th-percentile residual factor with 0.5 blend",
        "strict_esm_online": True,
        "current_actual_tm_used_online": False,
        "same_configuration_for_all_loads": True,
        "evaluation_bounds": [400, 500],
        "corrected_config": corrected["config"],
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
        raw_candidate, raw_active, raw_gate = adaptive.construct(
            model, props, cache, factors, slack.SPEC
        )
        raw_rows, raw_summary = base.runtime.evaluate_cache(
            model, props, raw_candidate, None, batch_size=16
        )
        corrected_load = corrected["loads"][str(load)]
        corrected_evaluation = corrected_load["evaluation"]
        corrected_summary = corrected_evaluation["candidate"]
        corrected_rows = corrected_evaluation["rows"]["candidate"]
        compact_baseline = final.compact_rows(baseline_rows)
        compact_raw = final.compact_rows(raw_rows)
        result = {
            "active": {
                "raw_esm": int(raw_active.sum()),
                "corrected_esm": int(corrected_load["active"]),
            },
            "medium_gate": {
                "raw_esm": {
                    "mean": float(raw_gate.mean()),
                    "p10": float(torch.quantile(raw_gate, 0.10)),
                    "p90": float(torch.quantile(raw_gate, 0.90)),
                },
                "corrected_esm": corrected_load["medium_gate"],
            },
            "evaluation": {
                "baseline": base.compact(baseline_summary),
                "raw_esm": base.compact(raw_summary),
                "corrected_esm": corrected_summary,
                "delta_raw_vs_hattrick": base.delta(
                    raw_summary, baseline_summary
                ),
                "delta_corrected_vs_hattrick": corrected_evaluation["delta"],
                "delta_corrected_vs_raw": base.delta(
                    expand_summary(corrected_summary), raw_summary
                ),
                "bootstrap_corrected_vs_raw": (
                    base.screen.method.large.paired_bootstrap(
                        compact_raw, corrected_rows
                    )
                ),
                "bootstrap_raw_vs_hattrick": (
                    base.screen.method.large.paired_bootstrap(
                        compact_baseline, compact_raw
                    )
                ),
                "bootstrap_corrected_vs_hattrick": corrected_evaluation[
                    "bootstrap"
                ],
                "rows": {
                    "baseline": compact_baseline,
                    "raw_esm": compact_raw,
                    "corrected_esm": corrected_rows,
                },
            },
            "audit": {
                "baseline_rows_max_abs_diff_vs_corrected_run": max_row_diff(
                    compact_baseline,
                    corrected_evaluation["rows"]["baseline"],
                ),
                "raw_esm_information_audit": slack.information_audit(
                    model, props, cache, factors
                ),
            },
        }
        report["loads"][str(load)] = result
        d = result["evaluation"]["delta_corrected_vs_raw"]
        print(
            f"[{load}x corrected-raw] "
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
    path = HERE / "rqp_slack_esm_ablation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
