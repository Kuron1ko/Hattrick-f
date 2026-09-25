from __future__ import annotations

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


push = load_module("robust_priority_mid_base", HERE / "probe_robust_priority_push.py")
priority, base, cal = push.priority, push.base, push.cal


def main():
    base.runtime.set_seed(20260823)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props, _ = base.screen.load_backbone(3, device)
    train_cache = base.runtime.build_policy_cache(model, props, 0, 318, batch_size=32)
    train_x, train_y, widths = cal.arrays(train_cache)
    specs = [
        {
            "kind": "multiplicative",
            "quantile": 0.55,
            "blend": blend,
            "high_weight": high_weight,
            "steps": 40,
            "secondary_budget": 0.02,
        }
        for blend in (0.40, 0.50)
        for high_weight in (0.82, 0.84, 0.86, 0.88)
    ]
    validation_cache = base.runtime.build_policy_cache(model, props, 334, 400, batch_size=32)
    validation_baseline, validation = push.run_stage(
        model, props, validation_cache, train_x, train_y, widths, specs, "validation"
    )
    # Freeze the four candidates with the strongest High floor on validation.
    validation.sort(key=priority.rank, reverse=True)
    frozen = [priority.spec_only(value) for value in validation[:4]]
    evaluation_cache = base.runtime.build_policy_cache(model, props, 400, 500, batch_size=32)
    evaluation_baseline, evaluation = push.run_stage(
        model, props, evaluation_cache, train_x, train_y, widths, frozen, "evaluation"
    )
    evaluation.sort(key=priority.rank, reverse=True)
    report = {
        "method": "narrow frozen refinement near the robust-priority knee",
        "strict_esm_online": True,
        "current_actual_tm_used_online": False,
        "training_bounds": [0, 318],
        "validation_bounds": [334, 400],
        "evaluation_bounds": [400, 500],
        "validation_baseline": base.compact(validation_baseline),
        "validation": validation,
        "frozen_specs": frozen,
        "evaluation_baseline": base.compact(evaluation_baseline),
        "evaluation": evaluation,
    }
    path = HERE / "robust_priority_mid.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
