from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
PIMA_DIR = THIS_DIR.parent / "shared2x_medium_adapter"
for item in (str(THIS_DIR), str(PIMA_DIR)):
    if item not in sys.path:
        sys.path.insert(0, item)

from adapter import EndpointSharedMediumAdapter

spec = importlib.util.spec_from_file_location(
    "shared2x_cpima_gain_test_runtime", THIS_DIR / "run_experiment.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Unable to load local gain runtime")
gain_runtime = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gain_runtime
spec.loader.exec_module(gain_runtime)
exponential_tilt = gain_runtime.exponential_tilt


def test_exponential_tilt_is_nonnegative_and_preserves_od_mass() -> None:
    head = EndpointSharedMediumAdapter(
        torch.tensor([0, 0, 1]), torch.tensor([1, 2, 2]), 8
    )
    with torch.no_grad():
        head.source_bias.normal_()
        head.destination_bias.normal_()
        head.path_rank_bias.normal_()
    base = torch.rand(4, 24, 1)
    before = base.reshape(4, 3, 8).sum(dim=-1)
    after = exponential_tilt(base, head, 1.5)
    assert float(after.min()) >= 0.0
    assert torch.allclose(after.reshape(4, 3, 8).sum(dim=-1), before, atol=1e-6)


def test_zero_temperature_is_identity() -> None:
    head = EndpointSharedMediumAdapter(
        torch.tensor([0, 0]), torch.tensor([1, 2]), 8
    )
    with torch.no_grad():
        head.source_bias.normal_()
        head.destination_bias.normal_()
        head.path_rank_bias.normal_()
    base = torch.rand(3, 16, 1)
    assert torch.allclose(exponential_tilt(base, head, 0.0), base, atol=1e-7)
