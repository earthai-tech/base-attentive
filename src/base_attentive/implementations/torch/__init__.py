"""PyTorch-specific implementations of BaseAttentiveV2 components.

This package provides high-performance PyTorch implementations of the V2 components.
Components are registered when the resolver detects PyTorch as the active backend.

Key advantages over generic implementations:
- Uses native torch.nn.Module layers
- Optimized PyTorch attention (C++ backend acceleration)
- Automatic mixed precision (AMP) support
- CUDA/device-agnostic implementations
- Better gradient computation
- Native support for torchscript
"""

from __future__ import annotations

from .base_attentive_v2 import (
    _build_torch_concat_fusion,
    _build_torch_dense_projection,
    _build_torch_last_pool,
    _build_torch_mean_pool,
    _build_torch_point_forecast_head,
    _build_torch_quantile_head,
    _build_torch_temporal_self_attention_encoder,
    _TorchTemporalSelfAttentionEncoder,
    ensure_torch_v2_registered,
)

__all__ = [
    "ensure_torch_v2_registered",
    "_TorchTemporalSelfAttentionEncoder",
    "_build_torch_dense_projection",
    "_build_torch_temporal_self_attention_encoder",
    "_build_torch_mean_pool",
    "_build_torch_last_pool",
    "_build_torch_concat_fusion",
    "_build_torch_point_forecast_head",
    "_build_torch_quantile_head",
]

__all__ = [
    "ensure_torch_v2_registered",
    "_TorchTemporalSelfAttentionEncoder",
    "_build_torch_dense_projection",
    "_build_torch_temporal_self_attention_encoder",
    "_build_torch_mean_pool",
    "_build_torch_last_pool",
    "_build_torch_concat_fusion",
    "_build_torch_point_forecast_head",
    "_build_torch_quantile_head",
]
