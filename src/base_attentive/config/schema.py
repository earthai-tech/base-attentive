"""Schema objects for the V2 architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BaseAttentiveComponentSpec:
    """Logical component selections for ``BaseAttentiveV2``."""

    static_projection: str = "projection.static"
    dynamic_projection: str = "projection.dynamic"
    future_projection: str = "projection.future"
    dynamic_encoder: str = "encoder.temporal_self_attention"
    future_encoder: str = "encoder.temporal_self_attention"
    sequence_pooling: str = "pool.mean"
    fusion: str = "fusion.concat"
    hidden_projection: str = "projection.hidden"
    point_head: str = "head.point_forecast"
    quantile_head: str = "head.quantile_forecast"


@dataclass(frozen=True)
class BaseAttentiveSpec:
    """Backend-neutral configuration for the experimental V2 model."""

    static_input_dim: int
    dynamic_input_dim: int
    future_input_dim: int
    output_dim: int = 1
    forecast_horizon: int = 1
    embed_dim: int = 32
    hidden_units: int = 64
    attention_heads: int = 4
    layer_norm_epsilon: float = 1e-6
    dropout_rate: float = 0.0
    activation: str = "relu"
    backend_name: str = "tensorflow"
    head_type: str = "point"
    quantiles: tuple[float, ...] = ()
    components: BaseAttentiveComponentSpec = field(
        default_factory=BaseAttentiveComponentSpec
    )
    extras: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "BaseAttentiveComponentSpec",
    "BaseAttentiveSpec",
]
