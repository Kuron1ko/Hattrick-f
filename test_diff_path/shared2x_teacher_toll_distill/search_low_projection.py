from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent


def load_module(name: str, filename: str):
    path = HERE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainer = load_module("low_projection_trainer", "train_linear_toll.py")
probe = trainer.probe
runtime = trainer.runtime


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def build_teacher(model, props, cache):
    teacher = probe.correction.correct_cache(
        model, props, cache, *probe.TEACHER_CONFIG
    )
    path_count = int(cache.policies[0].shape[1])
    return teacher, path_count


def project_class(cache, target, class_index: int, weighting: str, ridge: float):
    return probe.fit_edge_tolls(
        cache.policies[class_index].squeeze(-1),
        target,
        probe.dense_path_edge(cache),
        ridge,
        cache.predicted_tms[class_index].squeeze(-1),
        weighting,
    )


def evaluate(model, props, cache, features, baseline_summary):
    adapted = replace(cache, path_features=features)
    _, summary = trainer.evaluate(model, props, cache, adapted)
    return {
        "summary": probe.compact(summary),
        "delta": probe.gaps(summary, baseline_summary),
    }


def run_split(model, props, start: int, stop: int, choices=None):
    cache = runtime.build_policy_cache(model, props, start, stop, batch_size=16)
    _, baseline_summary = trainer.evaluate(model, props, cache)
    teacher, path_count = build_teacher(model, props, cache)
    teacher_medium = teacher.path_features[:, :path_count]
    teacher_low = teacher.path_features[:, path_count:]
    projected_medium, _, medium_stats = project_class(
        cache, teacher_medium, 1, "target_flow", 0.01
    )
    grid = choices or [
        (weighting, ridge)
        for weighting in ("uniform", "target_route", "target_flow")
        for ridge in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    ]
    results = []
    for weighting, ridge in grid:
        projected_low, _, low_stats = project_class(
            cache, teacher_low, 2, weighting, ridge
        )
        result = evaluate(
            model,
            props,
            cache,
            torch.cat([projected_medium, projected_low], dim=1),
            baseline_summary,
        )
        result.update(
            {
                "weighting": weighting,
                "ridge": ridge,
                "low_projection": low_stats,
                "medium_projection": medium_stats,
            }
        )
        results.append(result)
        print(
            f"[{start}:{stop}] {weighting} ridge={ridge:g} "
            f"M={result['delta']['Medium.norm_fulfill_mean']:+.6f} "
            f"L={result['delta']['Low.norm_fulfill_mean']:+.6f} "
            f"L10={result['delta']['Low.norm_fulfill_p10']:+.6f}",
            flush=True,
        )
    return results


def eligible(row: dict) -> bool:
    delta = row["delta"]
    return (
        delta["Low.norm_fulfill_mean"] >= -0.003
        and delta["Low.norm_fulfill_p10"] >= -0.01
    )


def main() -> None:
    torch.manual_seed(20260822)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(4, device)
    model, checkpoint = runtime.load_backbone(4, 490, props, device)

    safety = run_split(model, props, 318, 350)
    candidates = [row for row in safety if eligible(row)]
    if candidates:
        selected = max(
            candidates,
            key=lambda row: (
                row["delta"]["Low.norm_fulfill_mean"],
                row["delta"]["Low.norm_fulfill_p10"],
            ),
        )
        choice = [(selected["weighting"], selected["ridge"])]
        validation = run_split(model, props, 350, 400, choice)[0]
    else:
        selected = None
        validation = None

    payload = {
        "hypothesis": "Low loss is introduced by an unsuitable edge-toll projection metric",
        "checkpoint": str(checkpoint),
        "strict_esm": True,
        "medium_projection_fixed": {"weighting": "target_flow", "ridge": 0.01},
        "safety_range": [318, 350],
        "safety_grid": safety,
        "selected": selected,
        "validation_range": [350, 400],
        "validation": validation,
        "hypothesis_pass": bool(validation and eligible(validation)),
    }
    write_json(
        HERE / "artifacts" / "level4_linear_toll" / "low_projection_search.json",
        payload,
    )
    print(json.dumps({
        "selected": selected,
        "validation": validation,
        "hypothesis_pass": payload["hypothesis_pass"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
