from __future__ import annotations

import torch

from adapter import (
    CausalMediumLowAdapter,
    DynamicCausalAdapter,
    EndpointSharedMediumAdapter,
    PriorityIsolatedMediumAdapter,
    SlackAwareCausalAdapter,
)


def test_zero_initialization_is_exact_identity() -> None:
    torch.manual_seed(1)
    policy = torch.softmax(torch.randn(3, 5, 8), dim=-1).reshape(3, 40, 1)
    adapter = PriorityIsolatedMediumAdapter(5, 8)
    actual = adapter(policy)
    assert torch.equal(actual, policy)


def test_od_mass_is_preserved_and_output_is_positive() -> None:
    torch.manual_seed(2)
    policy = torch.softmax(torch.randn(4, 7, 8), dim=-1).reshape(4, 56, 1)
    adapter = PriorityIsolatedMediumAdapter(7, 8)
    with torch.no_grad():
        adapter.bias.normal_(mean=0.0, std=2.0)
    actual = adapter(policy).reshape(4, 7, 8)
    expected = policy.reshape(4, 7, 8)
    torch.testing.assert_close(actual.sum(-1), expected.sum(-1), rtol=2e-6, atol=2e-7)
    assert torch.all(actual >= 0)


def test_gradient_reaches_adapter_but_not_frozen_policy() -> None:
    torch.manual_seed(3)
    policy = torch.softmax(torch.randn(2, 3, 8), dim=-1).reshape(2, 24, 1).detach()
    adapter = PriorityIsolatedMediumAdapter(3, 8)
    weights = torch.linspace(-1.0, 1.0, 24).reshape(1, 24, 1)
    loss = (adapter(policy) * weights).sum()
    loss.backward()
    assert adapter.bias.grad is not None
    assert torch.isfinite(adapter.bias.grad).all()
    assert float(adapter.bias.grad.abs().sum()) > 0.0
    assert policy.grad is None


def test_endpoint_shared_adapter_identity_mass_and_parameter_count() -> None:
    sources = torch.tensor([0, 0, 1, 1, 2, 2])
    destinations = torch.tensor([1, 2, 0, 2, 0, 1])
    policy = torch.softmax(torch.randn(3, 6, 8), dim=-1).reshape(3, 48, 1)
    adapter = EndpointSharedMediumAdapter(sources, destinations, 8)
    assert torch.equal(adapter(policy), policy)
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 56
    with torch.no_grad():
        adapter.source_bias.normal_()
        adapter.destination_bias.normal_()
        adapter.path_rank_bias.normal_()
    actual = adapter(policy).reshape(3, 6, 8)
    expected = policy.reshape(3, 6, 8)
    torch.testing.assert_close(actual.sum(-1), expected.sum(-1), rtol=2e-6, atol=2e-7)


def test_causal_adapter_zero_identity_and_class_isolation() -> None:
    sources = torch.tensor([0, 0, 1, 1, 2, 2])
    destinations = torch.tensor([1, 2, 0, 2, 0, 1])
    policies = [
        torch.softmax(torch.randn(2, 6, 8), dim=-1).reshape(2, 48, 1)
        for _ in range(3)
    ]
    adapter = CausalMediumLowAdapter(sources, destinations, 8)
    zero = adapter.adapt_policies(policies)
    assert all(torch.equal(before, after) for before, after in zip(policies, zero))
    with torch.no_grad():
        adapter.low_head.path_rank_bias[0] = 1.0
    changed = adapter.adapt_policies(policies)
    assert torch.equal(changed[0], policies[0])
    assert torch.equal(changed[1], policies[1])
    assert not torch.equal(changed[2], policies[2])


def test_dynamic_adapter_is_identity_at_zero_and_responds_to_features() -> None:
    policies = [
        torch.softmax(torch.randn(2, 4, 8), dim=-1).reshape(2, 32, 1)
        for _ in range(3)
    ]
    features = torch.randn(2, 32, 9)
    adapter = DynamicCausalAdapter(9, 12, 8)
    batch = {"path_features": features}
    zero = adapter.adapt_batch(policies, batch)
    assert all(torch.equal(before, after) for before, after in zip(policies, zero))
    with torch.no_grad():
        adapter.medium_head.output_layer.weight[0, 0] = 0.5
    changed = adapter.adapt_batch(policies, batch)
    assert torch.equal(changed[0], policies[0])
    assert not torch.equal(changed[1], policies[1])
    assert torch.equal(changed[2], policies[2])


def test_slack_adapter_can_change_high_without_parameter_sharing() -> None:
    sources = torch.tensor([0, 0, 1, 1, 2, 2])
    destinations = torch.tensor([1, 2, 0, 2, 0, 1])
    policies = [
        torch.softmax(torch.randn(2, 6, 8), dim=-1).reshape(2, 48, 1)
        for _ in range(3)
    ]
    adapter = SlackAwareCausalAdapter(sources, destinations, 8)
    zero = adapter.adapt_policies(policies)
    assert all(torch.equal(before, after) for before, after in zip(policies, zero))
    with torch.no_grad():
        adapter.high_head.path_rank_bias[0] = 1.0
    changed = adapter.adapt_policies(policies)
    assert not torch.equal(changed[0], policies[0])
    assert torch.equal(changed[1], policies[1])
    assert torch.equal(changed[2], policies[2])
