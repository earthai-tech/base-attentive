"""TensorFlow-specific implementations of BaseAttentiveV2 components."""

from __future__ import annotations

import importlib
from typing import Any

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


def __getattr__(name: str) -> Any:
    if name in __all__ or name == "base_attentive_v2":
        module = importlib.import_module(
            ".base_attentive_v2", __name__
        )
        return (
            module
            if name == "base_attentive_v2"
            else getattr(module, name)
        )
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )
