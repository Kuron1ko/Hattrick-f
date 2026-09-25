from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


HERE = Path(__file__).resolve().parent
TEST_DIR = HERE.parent
ROOT = TEST_DIR.parent
METHOD_PATH = (
    TEST_DIR / "shared2x_teacher_toll_distill" / "train_asymmetric_knn_toll.py"
)
LOADS = {
    1: {
        "topology": "geant_priomask500_shared",
        "checkpoint": ROOT / "hattrick_geant_priomask500_shared_8sp.pkl",
        "library": TEST_DIR / "shared1x_priority_fallback_lcr" / "model.pt",
        "active": "low",
    },
    2: {
        "topology": "geant_priomask500_shared_load2x_train",
        "checkpoint": TEST_DIR
        / "shared2x_full_objectives"
        / "artifacts"
        / "level4_confirmation"
        / "seed_490"
        / "best_model.pt",
        "runtime_checkpoint": True,
        "library": TEST_DIR
        / "shared2x_teacher_toll_distill"
        / "artifacts"
        / "asymmetric_knn_level4"
        / "model.pt",
        "active": "both",
    },
    3: {
        "topology": "geant_priomask500_shared_load3x_train",
        "checkpoint": ROOT
        / "hattrick_geant_priomask500_shared_load3x_train_8sp.pkl",
        "library": TEST_DIR
        / "shared_multi_asymmetric_knn_toll"
        / "artifacts"
        / "3x_level4"
        / "model.pt",
        "active": "both",
    },
}
SPLITS = {"safety": (318, 350), "validation": (350, 400)}
PROTOTYPE_COUNTS = (1, 2, 4, 8, 16, 32)
SCALES = (0.5, 1.0)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


method = load_module("simple_prototype_toll_method", METHOD_PATH)
trainer = method.trainer
runtime = method.runtime


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_backbone(load_factor: int, device: torch.device):
    config = LOADS[load_factor]
    runtime.shared.TOPOLOGY = config["topology"]
    props = runtime.build_props(4, device)
    props.mode = "test"
    props.sim_mf_mlu = 0
    if config.get("runtime_checkpoint", False):
        model, resolved_checkpoint = runtime.load_backbone(4, 490, props, device)
        if resolved_checkpoint.resolve() != config["checkpoint"].resolve():
            raise RuntimeError(
                f"Unexpected 2x checkpoint: {resolved_checkpoint}"
            )
    else:
        model = torch.load(
            config["checkpoint"], map_location=device, weights_only=False
        ).to(device=device, dtype=props.dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    trainer.runtime = runtime
    trainer.probe.runtime = runtime
    method.runtime = runtime
    library_artifact = torch.load(
        config["library"], map_location=device, weights_only=False
    )
    return model, props, library_artifact["library"]


def squared_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left[:, None] - right[None]).square().mean(dim=2)


def fit_prototypes(library: dict, count: int, iterations: int = 30) -> dict:
    features = library["features"].reshape(library["features"].shape[0], -1)
    targets = library["targets"]
    count = min(int(count), features.shape[0])
    # Deterministic farthest-first initialization avoids random-screen noise.
    first = torch.argmin(features.square().mean(dim=1))
    selected = [int(first.item())]
    nearest = squared_distance(features, features[first : first + 1]).squeeze(1)
    for _ in range(1, count):
        index = int(torch.argmax(nearest).item())
        selected.append(index)
        distance = squared_distance(features, features[index : index + 1]).squeeze(1)
        nearest = torch.minimum(nearest, distance)
    centroids = features[selected].clone()
    assignment = torch.zeros(features.shape[0], device=features.device, dtype=torch.long)
    for _ in range(iterations):
        new_assignment = squared_distance(features, centroids).argmin(dim=1)
        if torch.equal(new_assignment, assignment) and _ > 0:
            break
        assignment = new_assignment
        updated = []
        for cluster in range(count):
            members = features[assignment == cluster]
            updated.append(members.mean(dim=0) if len(members) else centroids[cluster])
        centroids = torch.stack(updated)
    prototype_targets = []
    cluster_sizes = []
    for cluster in range(count):
        mask = assignment == cluster
        cluster_sizes.append(int(mask.sum().item()))
        prototype_targets.append(
            targets[mask].mean(dim=0) if mask.any() else targets[selected[cluster]]
        )
    return {
        "centroids": centroids,
        "targets": torch.stack(prototype_targets),
        "cluster_sizes": cluster_sizes,
    }


def predict(library: dict, prototypes: dict, query: torch.Tensor) -> torch.Tensor:
    normalized = (query - library["mean"]) / library["std"]
    flattened = normalized.reshape(normalized.shape[0], -1)
    indices = squared_distance(flattened, prototypes["centroids"]).argmin(dim=1)
    return prototypes["targets"][indices]


def selective_tolls(tolls: torch.Tensor, active: str) -> torch.Tensor:
    if active == "both":
        return tolls
    selected = tolls.clone()
    if active == "low":
        selected[:, 0].zero_()
    elif active == "medium":
        selected[:, 1].zero_()
    else:
        raise ValueError(active)
    return selected


def compact(summary: list[dict]) -> dict:
    return trainer.probe.compact(summary)


def evaluate_candidate(
    model,
    props,
    cache,
    baseline_summary,
    library: dict,
    prototypes: dict,
    active: str,
    scale: float,
) -> dict:
    started = time.perf_counter()
    query = trainer.edge_features(cache)
    tolls = selective_tolls(predict(library, prototypes, query), active)
    routed = trainer.route_with_tolls(cache, tolls, scale)
    inference_seconds = time.perf_counter() - started
    rows, summary = trainer.evaluate(model, props, cache, routed)
    return {
        "summary": compact(summary),
        "delta": trainer.probe.gaps(summary, baseline_summary),
        "inference_seconds": inference_seconds,
        "inference_ms_per_snapshot": 1000.0 * inference_seconds / len(cache),
        "rows": rows,
    }


def public(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "rows"}


def main() -> None:
    torch.manual_seed(20260828)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    payload = {
        "method": "nearest congestion prototype with one fixed edge-toll table",
        "strict_esm_inference": True,
        "actual_tm_used_for_policy": False,
        "screen_only": True,
        "evaluation_400_500_used": False,
        "splits": {key: list(value) for key, value in SPLITS.items()},
        "loads": {},
    }
    for load_factor in (1, 2, 3):
        print(f"[{load_factor}x] loading backbone and development caches", flush=True)
        model, props, library = load_backbone(load_factor, device)
        prototypes_by_count = {
            count: fit_prototypes(library, count) for count in PROTOTYPE_COUNTS
        }
        split_caches = {
            name: runtime.build_policy_cache(
                model, props, start, end, batch_size=32
            )
            for name, (start, end) in SPLITS.items()
        }
        baselines = {}
        for name, cache in split_caches.items():
            _, summary = trainer.evaluate(model, props, cache)
            baselines[name] = summary
        trials = []
        for count in PROTOTYPE_COUNTS:
            for scale in SCALES:
                split_results = {}
                for name, cache in split_caches.items():
                    split_results[name] = public(
                        evaluate_candidate(
                            model,
                            props,
                            cache,
                            baselines[name],
                            library,
                            prototypes_by_count[count],
                            LOADS[load_factor]["active"],
                            scale,
                        )
                    )
                trial = {
                    "prototypes": count,
                    "scale": scale,
                    "cluster_sizes": prototypes_by_count[count]["cluster_sizes"],
                    "splits": split_results,
                }
                trials.append(trial)
                safety = split_results["safety"]["delta"]
                validation = split_results["validation"]["delta"]
                print(
                    f"[{load_factor}x] P={count:2d} s={scale:.1f} "
                    f"safety dM={safety['Medium.norm_fulfill_mean']:+.5f} "
                    f"dL={safety['Low.norm_fulfill_mean']:+.5f}; "
                    f"validation dM={validation['Medium.norm_fulfill_mean']:+.5f} "
                    f"dL={validation['Low.norm_fulfill_mean']:+.5f}",
                    flush=True,
                )
        payload["loads"][str(load_factor)] = {
            "topology": LOADS[load_factor]["topology"],
            "active": LOADS[load_factor]["active"],
            "baseline": {name: compact(value) for name, value in baselines.items()},
            "trials": trials,
        }
        del model, props, library, prototypes_by_count, split_caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    payload["seconds"] = time.perf_counter() - started
    write_json(HERE / "screen_report.json", payload)
    print(json.dumps({"artifact": str(HERE / 'screen_report.json'), "seconds": payload["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
