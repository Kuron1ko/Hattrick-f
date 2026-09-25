from __future__ import annotations

import torch
from torch import nn


class PriorityIsolatedMediumAdapter(nn.Module):
    """A topology-sized residual head that can only change Medium paths.

    The residual is centred under the backbone distribution for every OD.  This
    preserves the OD total without touching either the High or Low policy.  At
    the all-zero initialization the output is bit-for-bit equal to the input.
    """

    def __init__(self, num_pairs: int, paths_per_pair: int, amplitude: float = 0.49):
        super().__init__()
        if num_pairs <= 0 or paths_per_pair <= 1:
            raise ValueError("num_pairs must be positive and paths_per_pair must exceed one")
        if not 0.0 < amplitude < 0.5:
            raise ValueError("amplitude must be in (0, 0.5) so every multiplier stays positive")
        self.num_pairs = int(num_pairs)
        self.paths_per_pair = int(paths_per_pair)
        self.amplitude = float(amplitude)
        self.bias = nn.Parameter(torch.zeros(self.num_pairs, self.paths_per_pair))

    def forward(self, base_medium_policy: torch.Tensor) -> torch.Tensor:
        original_shape = base_medium_policy.shape
        if base_medium_policy.ndim not in (2, 3):
            raise ValueError(f"Expected [B,P] or [B,P,1], got {tuple(original_shape)}")
        flat = base_medium_policy.squeeze(-1) if base_medium_policy.ndim == 3 else base_medium_policy
        expected_paths = self.num_pairs * self.paths_per_pair
        if flat.shape[-1] != expected_paths:
            raise ValueError(f"Expected {expected_paths} paths, got {flat.shape[-1]}")

        base = flat.reshape(flat.shape[0], self.num_pairs, self.paths_per_pair)
        residual = torch.tanh(self.bias).unsqueeze(0)
        base_mass = base.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(base.dtype).tiny)
        weighted_mean = (base * residual).sum(dim=-1, keepdim=True) / base_mass
        multiplier = 1.0 + self.amplitude * (residual - weighted_mean)
        adapted = (base * multiplier).reshape_as(flat)
        return adapted.unsqueeze(-1) if base_medium_policy.ndim == 3 else adapted

    @torch.no_grad()
    def project_parameter_box(self, limit: float = 6.0) -> None:
        """Avoid needlessly saturated tanh parameters after many manual steps."""
        self.bias.clamp_(min=-float(limit), max=float(limit))


class EndpointSharedMediumAdapter(nn.Module):
    """A lower-capacity PIMA shared by source, destination, and KSP rank."""

    def __init__(
        self,
        pair_sources: torch.Tensor,
        pair_destinations: torch.Tensor,
        paths_per_pair: int,
        amplitude: float = 0.49,
    ):
        super().__init__()
        sources = torch.as_tensor(pair_sources, dtype=torch.long).reshape(-1)
        destinations = torch.as_tensor(pair_destinations, dtype=torch.long).reshape(-1)
        if sources.shape != destinations.shape or sources.numel() == 0:
            raise ValueError("pair source/destination arrays must be non-empty and aligned")
        if not 0.0 < amplitude < 0.5:
            raise ValueError("amplitude must be in (0, 0.5)")
        self.num_pairs = int(sources.numel())
        self.paths_per_pair = int(paths_per_pair)
        self.amplitude = float(amplitude)
        self.num_nodes = int(torch.maximum(sources.max(), destinations.max()).item()) + 1
        self.register_buffer("pair_sources", sources)
        self.register_buffer("pair_destinations", destinations)
        self.source_bias = nn.Parameter(torch.zeros(self.num_nodes, self.paths_per_pair))
        self.destination_bias = nn.Parameter(torch.zeros(self.num_nodes, self.paths_per_pair))
        self.path_rank_bias = nn.Parameter(torch.zeros(self.paths_per_pair))

    def forward(self, base_medium_policy: torch.Tensor) -> torch.Tensor:
        original_shape = base_medium_policy.shape
        if base_medium_policy.ndim not in (2, 3):
            raise ValueError(f"Expected [B,P] or [B,P,1], got {tuple(original_shape)}")
        flat = base_medium_policy.squeeze(-1) if base_medium_policy.ndim == 3 else base_medium_policy
        expected_paths = self.num_pairs * self.paths_per_pair
        if flat.shape[-1] != expected_paths:
            raise ValueError(f"Expected {expected_paths} paths, got {flat.shape[-1]}")
        base = flat.reshape(flat.shape[0], self.num_pairs, self.paths_per_pair)
        logits = (
            self.source_bias[self.pair_sources]
            + self.destination_bias[self.pair_destinations]
            + self.path_rank_bias.unsqueeze(0)
        )
        residual = torch.tanh(logits).unsqueeze(0)
        base_mass = base.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(base.dtype).tiny)
        weighted_mean = (base * residual).sum(dim=-1, keepdim=True) / base_mass
        multiplier = 1.0 + self.amplitude * (residual - weighted_mean)
        adapted = (base * multiplier).reshape_as(flat)
        return adapted.unsqueeze(-1) if base_medium_policy.ndim == 3 else adapted

    @torch.no_grad()
    def project_parameter_box(self, limit: float = 6.0) -> None:
        for parameter in self.parameters():
            parameter.clamp_(min=-float(limit), max=float(limit))


class CausalMediumLowAdapter(nn.Module):
    """Two small heads following the causal order High -> Medium -> Low.

    The Medium head is optimized first.  The Low shield then reroutes only the
    final-priority class, which cannot alter already admitted High or Medium.
    """

    def __init__(
        self,
        pair_sources: torch.Tensor,
        pair_destinations: torch.Tensor,
        paths_per_pair: int,
        amplitude: float = 0.49,
    ):
        super().__init__()
        self.medium_head = EndpointSharedMediumAdapter(
            pair_sources, pair_destinations, paths_per_pair, amplitude
        )
        self.low_head = EndpointSharedMediumAdapter(
            pair_sources, pair_destinations, paths_per_pair, amplitude
        )

    def forward(self, base_medium_policy: torch.Tensor) -> torch.Tensor:
        return self.medium_head(base_medium_policy)

    def adapt_policies(self, policies: list[torch.Tensor]) -> list[torch.Tensor]:
        return [policies[0], self.medium_head(policies[1]), self.low_head(policies[2])]

    def medium_parameters(self) -> list[nn.Parameter]:
        return list(self.medium_head.parameters())

    def low_parameters(self) -> list[nn.Parameter]:
        return list(self.low_head.parameters())

    @torch.no_grad()
    def project_parameter_box(self, limit: float = 6.0) -> None:
        self.medium_head.project_parameter_box(limit)
        self.low_head.project_parameter_box(limit)


class DynamicResidualHead(nn.Module):
    """Shared per-path residual conditioned on predicted congestion state."""

    def __init__(self, feature_dim: int, hidden_dim: int, paths_per_pair: int, amplitude: float = 0.49):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.paths_per_pair = int(paths_per_pair)
        self.amplitude = float(amplitude)
        self.input_layer = nn.Linear(self.feature_dim, int(hidden_dim))
        self.output_layer = nn.Linear(int(hidden_dim), 1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, base_policy: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        original_shape = base_policy.shape
        flat = base_policy.squeeze(-1) if base_policy.ndim == 3 else base_policy
        if features.shape[:2] != flat.shape or features.shape[-1] != self.feature_dim:
            raise ValueError("Dynamic feature shape does not match policy")
        if flat.shape[1] % self.paths_per_pair:
            raise ValueError("Path count is not divisible by paths_per_pair")
        batch = flat.shape[0]
        num_pairs = flat.shape[1] // self.paths_per_pair
        hidden = torch.nn.functional.silu(self.input_layer(features))
        residual = torch.tanh(self.output_layer(hidden)).reshape(
            batch, num_pairs, self.paths_per_pair
        )
        base = flat.reshape(batch, num_pairs, self.paths_per_pair)
        base_mass = base.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(base.dtype).tiny)
        weighted_mean = (base * residual).sum(dim=-1, keepdim=True) / base_mass
        multiplier = 1.0 + self.amplitude * (residual - weighted_mean)
        adapted = (base * multiplier).reshape_as(flat)
        return adapted.unsqueeze(-1) if base_policy.ndim == 3 else adapted


class DynamicCausalAdapter(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, paths_per_pair: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.medium_head = DynamicResidualHead(feature_dim, hidden_dim, paths_per_pair)
        self.low_head = DynamicResidualHead(feature_dim, hidden_dim, paths_per_pair)

    def forward(
        self, base_medium_policy: torch.Tensor, features: torch.Tensor | None = None
    ) -> torch.Tensor:
        if features is None:
            features = torch.zeros(
                (*base_medium_policy.shape[:2], self.feature_dim),
                device=base_medium_policy.device,
                dtype=base_medium_policy.dtype,
            )
        return self.medium_head(base_medium_policy, features)

    def adapt_batch(self, policies: list[torch.Tensor], batch: dict) -> list[torch.Tensor]:
        features = batch.get("path_features")
        if features is None:
            raise ValueError("Dynamic adapter requires path_features")
        return [
            policies[0],
            self.medium_head(policies[1], features),
            self.low_head(policies[2], features),
        ]

    def medium_parameters(self) -> list[nn.Parameter]:
        return list(self.medium_head.parameters())

    def low_parameters(self) -> list[nn.Parameter]:
        return list(self.low_head.parameters())

    @torch.no_grad()
    def project_parameter_box(self, limit: float = 6.0) -> None:
        for parameter in self.parameters():
            parameter.clamp_(min=-float(limit), max=float(limit))


class SlackAwareCausalAdapter(nn.Module):
    """Three isolated heads that may spend High's explicit fulfillment slack."""

    def __init__(
        self,
        pair_sources: torch.Tensor,
        pair_destinations: torch.Tensor,
        paths_per_pair: int,
        amplitude: float = 0.49,
    ):
        super().__init__()
        self.high_head = EndpointSharedMediumAdapter(
            pair_sources, pair_destinations, paths_per_pair, amplitude
        )
        self.medium_head = EndpointSharedMediumAdapter(
            pair_sources, pair_destinations, paths_per_pair, amplitude
        )
        self.low_head = EndpointSharedMediumAdapter(
            pair_sources, pair_destinations, paths_per_pair, amplitude
        )

    def forward(self, base_medium_policy: torch.Tensor) -> torch.Tensor:
        return self.medium_head(base_medium_policy)

    def adapt_policies(self, policies: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            self.high_head(policies[0]),
            self.medium_head(policies[1]),
            self.low_head(policies[2]),
        ]

    def medium_parameters(self) -> list[nn.Parameter]:
        # The objective is Medium, but its gradient may deliberately reroute High.
        return list(self.high_head.parameters()) + list(self.medium_head.parameters())

    def high_parameters(self) -> list[nn.Parameter]:
        """Parameters whose update explicitly spends the High fulfillment slack."""
        return list(self.high_head.parameters())

    def low_parameters(self) -> list[nn.Parameter]:
        return list(self.low_head.parameters())

    @torch.no_grad()
    def project_parameter_box(self, limit: float = 6.0) -> None:
        self.high_head.project_parameter_box(limit)
        self.medium_head.project_parameter_box(limit)
        self.low_head.project_parameter_box(limit)
