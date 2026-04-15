"""Default configuration values for BaseAttentive models."""

from __future__ import annotations

DEFAULT_BASE_ATTENTIVE_ARCHITECTURE = {
    "encoder_type": "transformer",
    "decoder_attention_stack": ("cross",),
    "feature_processing": "dense",
}

DEFAULT_LEGACY_ARCHITECTURE = {
    "encoder_type": "hybrid",
    "decoder_attention_stack": (
        "cross",
        "hierarchical",
        "memory",
    ),
    "feature_processing": "vsn",
}

DEFAULT_BASE_ATTENTIVE_RUNTIME = {
    "mode": None,
    "num_encoder_layers": 2,
    "max_window_size": 10,
    "memory_size": 100,
    "scales": (),
    "multi_scale_agg": "last",
    "final_agg": "last",
    "use_residuals": True,
    "use_batch_norm": False,
    "apply_dtw": True,
    "verbose": 0,
}

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
    "static_processor": "feature.static_processor",
    "dynamic_processor": "feature.dynamic_processor",
    "future_processor": "feature.future_processor",
    "positional_encoder": "embedding.positional",
    "hybrid_encoder": "encoder.hybrid_multiscale",
    "dynamic_window": "encoder.dynamic_window",
    "decoder_cross_attention": "decoder.cross_attention",
    "decoder_hierarchical_attention": (
        "decoder.hierarchical_attention"
    ),
    "decoder_memory_attention": "decoder.memory_attention",
    "decoder_fusion": "fusion.multi_resolution_attention",
    "multi_horizon_head": "head.multi_horizon",
    "quantile_distribution_head": (
        "head.quantile_distribution"
    ),
    "final_pool_last": "pool.final_last",
    "final_pool_mean": "pool.final_mean",
    "final_pool_flatten": "pool.final_flatten",
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
    "lstm_units": 64,
    "attention_units": 32,
    "vsn_units": None,
}

__all__ = [
    "DEFAULT_BASE_ATTENTIVE_ARCHITECTURE",
    "DEFAULT_BASE_ATTENTIVE_COMPONENTS",
    "DEFAULT_BASE_ATTENTIVE_RUNTIME",
    "DEFAULT_BASE_ATTENTIVE_V2_CONFIG",
    "DEFAULT_LEGACY_ARCHITECTURE",
]
