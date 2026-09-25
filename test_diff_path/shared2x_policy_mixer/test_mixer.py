import torch

from run_experiment import mix_cache, runtime


def test_convex_medium_only() -> None:
    base = tuple(torch.rand(2, 8, 1) for _ in range(3))
    dote = tuple(torch.rand(2, 8, 1) for _ in range(3))
    cache = runtime.PolicyCache(
        dataset=None,
        path_masks=None,
        source_start=0,
        policies=base,
        tms=base,
        predicted_tms=base,
        capacities=torch.ones(2, 1),
        oracle_flows=tuple(torch.ones(2) for _ in range(3)),
        oracle_mlus=tuple(torch.ones(2) for _ in range(3)),
    )
    mixed = mix_cache(cache, dote, 0.25, "medium")
    assert torch.equal(mixed.policies[0], base[0])
    assert torch.equal(mixed.policies[2], base[2])
    assert torch.allclose(mixed.policies[1], 0.75 * base[1] + 0.25 * dote[1])
    paired = mix_cache(cache, dote, 0.25, "high_medium")
    assert torch.allclose(paired.policies[0], 0.75 * base[0] + 0.25 * dote[0])
    assert torch.equal(paired.policies[2], base[2])
    lower = mix_cache(cache, dote, 0.25, "medium_low")
    assert torch.equal(lower.policies[0], base[0])
    assert torch.allclose(lower.policies[1], 0.75 * base[1] + 0.25 * dote[1])
    assert torch.allclose(lower.policies[2], 0.75 * base[2] + 0.25 * dote[2])


if __name__ == "__main__":
    test_convex_medium_only()
