from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
TEST_DIR = THIS_DIR.parent
ROOT = TEST_DIR.parent
METHOD_PATH = TEST_DIR / "shared2x_active_set_router" / "probe_esm_self_correction.py"
spec = importlib.util.spec_from_file_location("shared1x_esm_sar_method", METHOD_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load ESM-SAR implementation")
method = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = method
spec.loader.exec_module(method)
runtime = method.runtime


TOPOLOGY = "geant_priomask500_shared"
MODEL_PATH = ROOT / "hattrick_geant_priomask500_shared_8sp.pkl"
TRANSFER_CONFIG = (24, 0.06, 0.25, 0.01)


def load_onex(device: torch.device):
    runtime.shared.TOPOLOGY = TOPOLOGY
    props = runtime.build_props(4, device)
    props.mode = "test"
    props.sim_mf_mlu = 0
    model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = model.to(device=device, dtype=props.dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if hasattr(model, "transformer_output"):
        delattr(model, "transformer_output")
    return model, props


def slice_cache(cache, start: int, stop: int):
    def sliced(values):
        return tuple(value[start:stop].clone() for value in values)

    return replace(
        cache,
        source_start=cache.source_start + start,
        policies=sliced(cache.policies),
        tms=sliced(cache.tms),
        predicted_tms=sliced(cache.predicted_tms),
        capacities=cache.capacities[start:stop].clone(),
        oracle_flows=sliced(cache.oracle_flows),
        oracle_mlus=sliced(cache.oracle_mlus),
        path_features=None,
    )


def correct_strict(model, props, cache, config=TRANSFER_CONFIG):
    values = []
    for index in range(len(cache)):
        one = method.correct_cache(
            model, props, slice_cache(cache, index, index + 1), *config
        )
        values.append(one.path_features)
    return replace(cache, path_features=torch.cat(values, dim=0))


def evaluate_range(model, props, start: int, stop: int) -> dict:
    cache = runtime.build_policy_cache(model, props, start, stop, batch_size=32)
    _, baseline = runtime.evaluate_cache(model, props, cache, None, batch_size=32)
    corrected = correct_strict(model, props, cache)
    adapter = method.CorrectedPolicyAdapter(int(cache.policies[0].shape[1]))
    _, candidate = runtime.evaluate_cache(model, props, corrected, adapter, batch_size=32)
    delta = method.gaps(candidate, baseline)
    return {
        "range": [start, stop],
        "baseline": method.compact(baseline),
        "candidate": method.compact(candidate),
        "delta": delta,
    }


def main() -> None:
    torch.manual_seed(20260821)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, props = load_onex(device)
    payload = {
        "method": "ESM-SAR 2x configuration transferred unchanged to 1x",
        "topology": TOPOLOGY,
        "model": str(MODEL_PATH),
        "model_sha256": runtime.sha256(MODEL_PATH),
        "config": {
            "steps": TRANSFER_CONFIG[0],
            "learning_rate": TRANSFER_CONFIG[1],
            "low_weight": TRANSFER_CONFIG[2],
            "anchor_weight": TRANSFER_CONFIG[3],
        },
        "input_contract": "ESM predictions only for correction; actual TMs evaluation only",
        "small": evaluate_range(model, props, 350, 358),
        "holdout": evaluate_range(model, props, 358, 400),
    }
    THIS_DIR.mkdir(parents=True, exist_ok=True)
    output = THIS_DIR / "onex_transfer_validation.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
