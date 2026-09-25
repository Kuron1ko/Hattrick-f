from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
MODULE_PATH = THIS_DIR / "probe_esm_self_correction.py"
spec = importlib.util.spec_from_file_location("test_esm_self_correction_runtime", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load self-correction module")
method = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = method
spec.loader.exec_module(method)
runtime = method.runtime


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = runtime.build_props(3, device)
    model, _ = runtime.load_backbone(3, 490, props, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cache = runtime.build_policy_cache(model, props, 350, 351, batch_size=1)
    config = (4, 0.03, 0.25, 0.01)
    first = method.correct_cache(model, props, cache, *config)

    # Current actual traffic matrices must be irrelevant to policy correction.
    changed_actual = replace(
        cache,
        tms=tuple(torch.randn_like(value) * 1000 for value in cache.tms),
    )
    second = method.correct_cache(model, props, changed_actual, *config)
    assert torch.equal(first.path_features, second.path_features)

    path_count = int(cache.policies[0].shape[1])
    for class_index, offset in ((1, 0), (2, path_count)):
        base = cache.policies[class_index].squeeze(-1).reshape(1, -1, 8)
        corrected = first.path_features[:, offset : offset + path_count].reshape(1, -1, 8)
        assert torch.isfinite(corrected).all()
        assert float((base.sum(-1) - corrected.sum(-1)).abs().max().item()) < 2e-6
        disabled = base == 0
        if disabled.any():
            assert float(corrected[disabled].abs().max().item()) == 0.0

    adapter = method.CorrectedPolicyAdapter(path_count)
    policies = [value.clone() for value in cache.policies]
    adapted = adapter.adapt_batch(policies, {"path_features": first.path_features})
    assert torch.equal(adapted[0], cache.policies[0])
    print("all self-correction invariants passed")


if __name__ == "__main__":
    main()
