"""Default configuration values for the V2 architecture."""

from __future__ import annotations

DEFAULT_BASE_ATTENTIVE_COMPONENTS = {
    "static_projection": "projection.static",
    "dynamic_projection": "projection.dynamic",
    "future_projection": "projection.future",
    "dynamic_encoder": "encoder.temporal_self_attention",
    "future_encoder": "encoder.temporal_self_attention",
    "sequence_pooling": "pool.mean",
    "fusion": "fusion.concat",
    "hidden_projection": "projection.hidden",
    "point_head": "head.point_forecast",
    "quantile_head": "head.quantile_forecast",
}

DEFAULT_BASE_ATTENTIVE_V2_CONFIG = {
    "output_dim": 1,
    "forecast_horizon": 1,
    "embed_dim": 32,
    "hidden_units": 64,
    "attention_heads": 4,
    "layer_norm_epsilon": 1e-6,
    "dropout_rate": 0.0,
    "activation": "relu",
    "backend_name": "tensorflow",
    "head_type": "point",
    "quantiles": (),
}

__all__ = [
    "DEFAULT_BASE_ATTENTIVE_COMPONENTS",
    "DEFAULT_BASE_ATTENTIVE_V2_CONFIG",
]
