from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.training_utils import loss_mlu, loss_mlu_overflow_area  # noqa: E402


def main() -> None:
    torch.manual_seed(490)
    utilization = (0.7 + 1.4 * torch.rand(8, 72)).requires_grad_(True)
    optimum = 0.8 + 0.3 * torch.rand(8)
    native, native_value = loss_mlu(utilization, optimum)
    candidate, candidate_value = loss_mlu_overflow_area(
        utilization, optimum, 0.0
    )
    native_gradient = torch.autograd.grad(native, utilization, retain_graph=True)[0]
    candidate_gradient = torch.autograd.grad(candidate, utilization)[0]
    result = {
        "lambda": 0.0,
        "loss_abs_difference": float(abs(native.item() - candidate.item())),
        "reported_value_abs_difference": float(abs(native_value - candidate_value)),
        "gradient_max_abs_difference": float(
            (native_gradient - candidate_gradient).abs().max().item()
        ),
    }
    result["identity_pass"] = all(value == 0.0 for value in result.values())
    output = HERE / "artifacts" / "overflow_area_identity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["identity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
