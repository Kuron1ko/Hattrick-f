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


final = load_module("p8_cpu_audit_base", HERE / "run_final_p8_sar.py")


def main():
    final.base.runtime.set_seed(20260823)
    final.base.GATE = final.GATE
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "purpose": "deterministic information-boundary audit",
        "actual_tm_used_for_policy": False,
        "loads": {},
    }
    for load in (1, 2, 3):
        model, props, _ = final.base.screen.load_backbone(load, device)
        cache = final.base.runtime.build_policy_cache(
            model, props, 400, 500, batch_size=32
        )
        original, active = final.construct_policy(model, props, cache)
        repeated, repeated_active = final.construct_policy(model, props, cache)
        zero_actual_cache = replace(
            cache, tms=tuple(torch.zeros_like(value) for value in cache.tms)
        )
        zero_actual, zero_active = final.construct_policy(
            model, props, zero_actual_cache
        )
        zero_predicted_cache = replace(
            cache,
            predicted_tms=tuple(
                torch.zeros_like(value) for value in cache.predicted_tms
            ),
        )
        zero_predicted, predicted_active = final.construct_policy(
            model, props, zero_predicted_cache
        )
        report["loads"][str(load)] = {
            "snapshots": [400, 500],
            "active": int(active.sum().item()),
            "repeat_activation_mismatch": int(
                (active != repeated_active).sum().item()
            ),
            "repeat_policy_max_abs_diff": float(
                (original.path_features - repeated.path_features).abs().max().item()
            ),
            "zero_actual_activation_mismatch": int((active != zero_active).sum().item()),
            "zero_actual_policy_max_abs_diff": float(
                (original.path_features - zero_actual.path_features).abs().max().item()
            ),
            "zero_predicted_activation_mismatch": int(
                (active != predicted_active).sum().item()
            ),
            "zero_predicted_policy_max_abs_diff": float(
                (original.path_features - zero_predicted.path_features).abs().max().item()
            ),
        }
        print(load, report["loads"][str(load)], flush=True)
    path = HERE / "cpu_information_audit.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
