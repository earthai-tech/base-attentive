"""JAX-specific implementations of BaseAttentiveV2 components.

This package provides high-performance JAX implementations of the V2 components.
Components are registered when the resolver detects JAX as the active backend.

Key advantages over generic implementations:
- Pure functional implementations (JAX-friendly)
- XLA compilation for maximum performance
- Automatic differentiation optimized
- GPU/TPU acceleration ready
- Pytree-compatible data structures
"""

from __future__ import annotations

from .base_attentive_v2 import (
    _JaxTemporalSelfAttentionEncoder,
    _build_jax_concat_fusion,
    _build_jax_dense_projection,
    _build_jax_last_pool,
    _build_jax_mean_pool,
    _build_jax_point_forecast_head,
    _build_jax_quantile_head,
    _build_jax_temporal_self_attention_encoder,
    ensure_jax_v2_registered,
)

__all__ = [
    "ensure_jax_v2_registered",
    "_JaxTemporalSelfAttentionEncoder",
    "_build_jax_dense_projection",
    "_build_jax_temporal_self_attention_encoder",
    "_build_jax_mean_pool",
    "_build_jax_last_pool",
    "_build_jax_concat_fusion",
    "_build_jax_point_forecast_head",
    "_build_jax_quantile_head",
]

__all__ = [
    "ensure_jax_v2_registered",
    "_JaxTemporalSelfAttentionEncoder",
    "_build_jax_dense_projection",
    "_build_jax_temporal_self_attention_encoder",
    "_build_jax_mean_pool",
    "_build_jax_last_pool",
    "_build_jax_concat_fusion",
    "_build_jax_point_forecast_head",
    "_build_jax_quantile_head",
]
