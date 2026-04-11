"""TensorFlow-specific implementations of BaseAttentiveV2 components.

This package provides high-performance TensorFlow implementations of the V2 components.
Components are registered when the resolver detects TensorFlow as the active backend.

Key advantages over generic implementations:
- Uses native TensorFlow Keras layers (no abstraction overhead)
- Optimized TensorFlow attention kernels
- tf.function compilation for performance
- Native mixed precision support
- Efficient memory management
- Better gradient computation
"""

from __future__ import annotations

from .base_attentive_v2 import (
    _TFTemporalSelfAttentionEncoder,
    _build_tf_concat_fusion,
    _build_tf_dense_projection,
    _build_tf_last_pool,
    _build_tf_mean_pool,
    _build_tf_point_forecast_head,
    _build_tf_quantile_head,
    _build_tf_temporal_self_attention_encoder,
    ensure_tensorflow_v2_registered,
)

__all__ = [
    "ensure_tensorflow_v2_registered",
    "_TFTemporalSelfAttentionEncoder",
    "_build_tf_dense_projection",
    "_build_tf_temporal_self_attention_encoder",
    "_build_tf_mean_pool",
    "_build_tf_last_pool",
    "_build_tf_concat_fusion",
    "_build_tf_point_forecast_head",
    "_build_tf_quantile_head",
]

__all__ = [
    "ensure_tensorflow_v2_registered",
    "_TFTemporalSelfAttentionEncoder",
    "_build_tf_dense_projection",
    "_build_tf_temporal_self_attention_encoder",
    "_build_tf_mean_pool",
    "_build_tf_last_pool",
    "_build_tf_concat_fusion",
    "_build_tf_point_forecast_head",
    "_build_tf_quantile_head",
]
