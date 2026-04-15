"""Schema objects for BaseAttentive configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BaseAttentiveArchitectureSpec:
    """Logical architecture choices for BaseAttentive models."""

    encoder_type: str = "transformer"
    decoder_attention_stack: tuple[str, ...] = ("cross",)
    feature_processing: str = "dense"


@dataclass(frozen=True)
class BaseAttentiveRuntimeSpec:
    """Runtime behavior shared across legacy and V2 paths."""

    mode: str | None = None
    num_encoder_layers: int = 2
    max_window_size: int = 10
    memory_size: int = 100
    scales: tuple[int, ...] | str = ()
    multi_scale_agg: str = "last"
    final_agg: str = "last"
    use_residuals: bool = True
    use_batch_norm: bool = False
    apply_dtw: bool = True
    verbose: int = 0


@dataclass(frozen=True)
class BaseAttentiveComponentSpec:
    """Logical component selections for resolver-driven models."""

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

    # Planned legacy-to-resolver keys.
    static_processor: str = "feature.static_processor"
    dynamic_processor: str = "feature.dynamic_processor"
    future_processor: str = "feature.future_processor"
    positional_encoder: str = "embedding.positional"
    hybrid_encoder: str = "encoder.hybrid_multiscale"
    dynamic_window: str = "encoder.dynamic_window"
    decoder_cross_attention: str = "decoder.cross_attention"
    decoder_hierarchical_attention: str = (
        "decoder.hierarchical_attention"
    )
    decoder_memory_attention: str = "decoder.memory_attention"
    decoder_fusion: str = "fusion.multi_resolution_attention"
    multi_horizon_head: str = "head.multi_horizon"
    quantile_distribution_head: str = (
        "head.quantile_distribution"
    )
    final_pool_last: str = "pool.final_last"
    final_pool_mean: str = "pool.final_mean"
    final_pool_flatten: str = "pool.final_flatten"


@dataclass(frozen=True)
class BaseAttentiveSpec:
    """Backend-neutral configuration for BaseAttentive models."""

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

    # Legacy-compatible scalar fields.
    lstm_units: int | tuple[int, ...] = 64
    attention_units: int = 32
    vsn_units: int | None = None

    architecture: BaseAttentiveArchitectureSpec = field(
        default_factory=BaseAttentiveArchitectureSpec
    )
    runtime: BaseAttentiveRuntimeSpec = field(
        default_factory=BaseAttentiveRuntimeSpec
    )
    components: BaseAttentiveComponentSpec = field(
        default_factory=BaseAttentiveComponentSpec
    )
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def num_heads(self) -> int:
        """Legacy alias for ``attention_heads``."""
        return self.attention_heads

    @property
    def num_encoder_layers(self) -> int:
        return self.runtime.num_encoder_layers

    @property
    def mode(self) -> str | None:
        return self.runtime.mode

    @property
    def max_window_size(self) -> int:
        return self.runtime.max_window_size

    @property
    def memory_size(self) -> int:
        return self.runtime.memory_size

    @property
    def scales(self) -> tuple[int, ...] | str:
        return self.runtime.scales

    @property
    def multi_scale_agg(self) -> str:
        return self.runtime.multi_scale_agg

    @property
    def final_agg(self) -> str:
        return self.runtime.final_agg

    @property
    def use_residuals(self) -> bool:
        return self.runtime.use_residuals

    @property
    def use_batch_norm(self) -> bool:
        return self.runtime.use_batch_norm

    @property
    def apply_dtw(self) -> bool:
        return self.runtime.apply_dtw

    @property
    def verbose(self) -> int:
        return self.runtime.verbose

    @property
    def objective(self) -> str:
        return self.architecture.encoder_type

    @property
    def use_vsn(self) -> bool:
        return self.architecture.feature_processing == "vsn"

    @property
    def attention_levels(self) -> tuple[str, ...]:
        return self.architecture.decoder_attention_stack


__all__ = [
    "BaseAttentiveArchitectureSpec",
    "BaseAttentiveComponentSpec",
    "BaseAttentiveRuntimeSpec",
    "BaseAttentiveSpec",
]
