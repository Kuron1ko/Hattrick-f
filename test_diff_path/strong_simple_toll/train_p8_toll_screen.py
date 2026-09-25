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


base = load_module("p8_toll_base", HERE / "run_tradeoff_screen.py")
screen = base.screen
trainer = base.trainer
runtime = base.runtime
probe = screen.method.trainer.probe
knn = screen.method.knn
CONFIG = (8, 0.08, 0.241, 0.01)
SPLITS = {"safety": (318, 350), "validation": (350, 400)}
KMS = (1, 2, 4, 8)
KLS = (8, 16, 32, 64)


def p8_tolls(model, props, cache):
    teacher = base.correction.correct_cache(model, props, cache, *CONFIG)
    path_count = int(cache.policies[0].shape[1])
    pte = cache.dataset.pte.to_dense().to(device=cache.capacities.device)
    medium, medium_tolls, medium_stats = probe.fit_edge_tolls(
        cache.policies[1].squeeze(-1),
        teacher.path_features[:, :path_count],
        pte,
        0.01,
        cache.predicted_tms[1].squeeze(-1),
        "target_flow",
    )
    low, low_tolls, low_stats = probe.fit_edge_tolls(
        cache.policies[2].squeeze(-1),
        teacher.path_features[:, path_count:],
        pte,
        0.01,
        cache.predicted_tms[2].squeeze(-1),
        "target_flow",
    )
    return torch.stack([medium_tolls, low_tolls], dim=1), {
        "medium": medium_stats,
        "low": low_stats,
    }


def main():
    runtime.set_seed(20260823)
    base.GATE = 0.70
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "method": "local edge-toll distillation of P8-SAR",
        "strict_esm": True,
        "actual_tm_used_for_training_or_policy": False,
        "teacher": list(CONFIG),
        "loads": {},
    }
    artifact = {"method": report["method"], "teacher": list(CONFIG), "loads": {}}
    for load in (2, 3):
        print(f"[{load}x] teacher and projection", flush=True)
        model, props, _ = screen.load_backbone(load, device)
        train_cache = runtime.build_policy_cache(model, props, 0, 318, batch_size=32)
        features = trainer.edge_features(train_cache)
        targets, projection = p8_tolls(model, props, train_cache)
        library = knn.fit_library(features, targets)
        split_report = {}
        for split, bounds in SPLITS.items():
            cache = runtime.build_policy_cache(model, props, *bounds, batch_size=32)
            _, baseline = trainer.evaluate(model, props, cache)
            rows = []
            for km in KMS:
                for kl in KLS:
                    value = base.public(
                        base.run_knn(
                            model,
                            props,
                            cache,
                            baseline,
                            library,
                            (km, kl, 1.0, 1.0),
                        )
                    )
                    rows.append(value)
                    d = value["delta"]
                    print(
                        f"[{load}x/{split}] k={km}/{kl} "
                        f"M={d['Medium.norm_fulfill_mean']:+.6f}/"
                        f"{d['Medium.norm_fulfill_p1']:+.6f}/"
                        f"{d['Medium.norm_fulfill_p10']:+.6f} "
                        f"L={d['Low.norm_fulfill_mean']:+.6f}/"
                        f"{d['Low.norm_fulfill_p1']:+.6f}/"
                        f"{d['Low.norm_fulfill_p10']:+.6f}",
                        flush=True,
                    )
            split_report[split] = rows
        report["loads"][str(load)] = {
            "projection": projection,
            "splits": split_report,
        }
        artifact["loads"][str(load)] = knn.cpu_library(library)
    (HERE / "p8_toll_screen.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    torch.save(artifact, HERE / "p8_toll_library.pt")
    print(HERE / "p8_toll_screen.json", flush=True)


if __name__ == "__main__":
    main()
