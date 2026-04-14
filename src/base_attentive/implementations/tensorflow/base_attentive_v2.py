"""TensorFlow-optimized V2 component builders.

This module provides TensorFlow-specific implementations of BaseAttentiveV2
components with optimizations including:
- Native TensorFlow Keras layers (no abstraction overhead)
- tf.function compilation for performance
- Mixed precision support
- TensorFlow's optimized attention mechanisms
- Efficient tensor operations
"""

from __future__ import annotations

from ...resolver.builder_contract import resolve_head_units

from typing import Any

try:
    import tensorflow as tf
    from tensorflow.keras import layers
except ImportError:
    tf = None
    layers = None

_BUILDER_META_KEYS = {
    "component_key",
    "in_features",
    "forecast_horizon",
}


def _clean_builder_kwargs(kwargs):
    return {
        key: value
        for key, value in kwargs.items()
        if key not in _BUILDER_META_KEYS
    }


def _ensure_tensorflow():
    """Ensure TensorFlow is available."""
    if tf is None:
        raise ImportError(
            "TensorFlow is required for tensorflow backend implementations. "
            "Install it with: pip install tensorflow"
        )


class _TFTemporalSelfAttentionEncoder(
    layers.Layer if layers else object
):
    """TensorFlow-optimized temporal encoder with multi-head attention.

    Advantages over generic version:
    - Uses native TensorFlow MultiHeadAttention (highly optimized)
    - Supports tf.function compilation
    - Better memory efficiency with TensorFlow's attention kernels
    - Native support for mixed precision
    """

    def __init__(
        self,
        *,
        units: int,
        hidden_units: int,
        num_heads: int,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        layer_norm_epsilon: float = 1e-6,
        name: str | None = None,
        **kwargs,
    ):
        _ensure_tensorflow()
        super().__init__(name=name, **_clean_builder_kwargs(kwargs))

        key_dim = max(1, units // max(1, num_heads))

        # Use TensorFlow's highly optimized MultiHeadAttention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate,
            attention_axes=None,  # Full attention
            name=f"{name}_mha" if name else "mha",
            dtype=None,  # Respects global precision policy
        )

        self.norm1 = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name=f"{name}_ln1" if name else "ln1",
        )

        # FFN: Dense -> Activation -> Dropout -> Dense
        self.ffn_hidden = layers.Dense(
            hidden_units,
            activation=activation,
            name=f"{name}_ffn_hidden"
            if name
            else "ffn_hidden",
        )

        self.ffn_output = layers.Dense(
            units,
            name=f"{name}_ffn_output"
            if name
            else "ffn_output",
        )

        self.dropout = None
        if dropout_rate > 0:
            self.dropout = layers.Dropout(dropout_rate)

        self.norm2 = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name=f"{name}_ln2" if name else "ln2",
        )

    def call(
        self, inputs: Any, training: bool = False
    ) -> Any:  # noqa: FBT002
        """Forward pass with residual connections and layer normalization."""
        # Self-attention with residual
        attn_output = self.attention(
            inputs,
            inputs,
            attention_mask=None,
            training=training,
        )
        x = self.norm1(inputs + attn_output)

        # FFN with residual
        ffn_output = self.ffn_hidden(x)
        if self.dropout is not None:
            ffn_output = self.dropout(
                ffn_output, training=training
            )
        ffn_output = self.ffn_output(ffn_output)

        return self.norm2(x + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.ffn_output.units,
                "hidden_units": self.ffn_hidden.units,
                "num_heads": self.attention.num_heads,
                "key_dim": self.attention.key_dim,
                "activation": self.ffn_hidden.activation.__name__
                if self.ffn_hidden.activation
                else None,
                "dropout_rate": self.dropout.rate
                if self.dropout
                else 0.0,
                "layer_norm_epsilon": self.norm1.epsilon,
            }
        )
        return config


def _build_tf_dense_projection(
    *,
    context=None,
    spec=None,
    units: int,
    activation: str | None = None,
    name: str | None = None,
    **kwargs,
) -> layers.Dense:
    """Build a TensorFlow Dense projection layer.

    Arguments:
        units: Output dimension
        activation: Activation function name (e.g., 'relu', 'sigmoid')
        name: Layer name
        **kwargs: Additional kwargs passed to Dense

    Returns:
        layers.Dense with optional activation
    """
    del context, spec
    _ensure_tensorflow()
    return layers.Dense(
        units, activation=activation, name=name, **kwargs
    )


def _build_tf_temporal_self_attention_encoder(
    *,
    context=None,
    spec=None,
    units: int,
    hidden_units: int,
    num_heads: int,
    activation: str = "relu",
    dropout_rate: float = 0.0,
    layer_norm_epsilon: float = 1e-6,
    name: str | None = None,
    **kwargs,
) -> _TFTemporalSelfAttentionEncoder:
    """Build a TensorFlow-optimized temporal self-attention encoder.

    Advantages:
    - Native TensorFlow implementation (no abstraction overhead)
    - Optimized attention kernels via tf.function compilation
    - Better memory management and gradient computation

    Arguments:
        units: Model dimension
        hidden_units: FFN hidden dimension
        num_heads: Number of attention heads
        activation: FFN activation function
        dropout_rate: Dropout rate for attention and FFN
        layer_norm_epsilon: Layer normalization epsilon
        name: Layer name
        **kwargs: Additional kwargs passed to the layer

    Returns:
        _TFTemporalSelfAttentionEncoder instance
    """
    del context, spec
    _ensure_tensorflow()
    return _TFTemporalSelfAttentionEncoder(
        units=units,
        hidden_units=hidden_units,
        num_heads=num_heads,
        activation=activation,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        name=name,
        **_clean_builder_kwargs(kwargs),
    )


def _build_tf_mean_pool(
    *,
    context=None,
    spec=None,
    axis: int | None = None,
    keepdims: bool = False,
    name: str | None = None,
    **kwargs,
):
    """Build a pooling layer using TensorFlow's Lambda layer.

    For simple operations like mean pooling, a Lambda layer is efficient.
    In TensorFlow, ops are highly optimized and graph-compiled.

    Arguments:
        axis: Axis along which to compute the mean
        keepdims: Whether to keep reduced dimensions
        name: Layer name
        **kwargs: Additional kwargs

    Returns:
        layers.Lambda that computes mean along axis
    """
    del context, spec, kwargs
    _ensure_tensorflow()

    def mean_pool_fn(x):
        return tf.reduce_mean(x, axis=axis, keepdims=keepdims)

    return layers.Lambda(
        mean_pool_fn, name=name or "mean_pool"
    )


def _build_tf_last_pool(
    *,
    context=None,
    spec=None,
    axis: int = 1,
    name: str | None = None,
    **kwargs,
):
    """Build a layer that extracts the last timestep.

    In TensorFlow, direct indexing is efficient and can be graph-compiled.

    Arguments:
        name: Layer name
        **kwargs: Additional kwargs

    Returns:
        layers.Lambda that extracts the last timestep
    """
    del context, spec, kwargs
    _ensure_tensorflow()
    if axis != 1:
        raise ValueError(
            "tensorflow last-pool currently supports axis=1 only."
        )

    def last_pool_fn(x):
        return x[:, -1, :]

    return layers.Lambda(
        last_pool_fn, name=name or "last_pool"
    )


def _build_tf_concat_fusion(
    *,
    context=None,
    spec=None,
    axis: int = -1,
    name: str | None = None,
    **kwargs,
):
    """Build a concatenation layer using TensorFlow's optimized ops.

    TensorFlow has highly optimized concatenation kernels.

    Arguments:
        axis: Axis along which to concatenate
        name: Layer name
        **kwargs: Additional kwargs

    Returns:
        layers.Lambda that concatenates inputs along axis
    """
    del context, spec, kwargs
    _ensure_tensorflow()

    def concat_fn(inputs):
        return tf.concat(inputs, axis=axis)

    return layers.Lambda(
        concat_fn, name=name or "concat_fusion"
    )


def _build_tf_point_forecast_head(
    *,
    context=None,
    spec=None,
    units: int | None = None,
    output_dim: int = 1,
    forecast_horizon: int = 1,
    quantiles: tuple[float, ...] | None = None,
    activation: str | None = None,
    name: str | None = None,
    **kwargs,
) -> layers.Dense:
    """Build a TensorFlow-optimized point forecast output head."""
    del context, spec, quantiles
    _ensure_tensorflow()
    total_output = resolve_head_units(
        units=units,
        output_dim=output_dim,
        forecast_horizon=forecast_horizon,
        is_quantile=False,
    )
    return layers.Dense(
        total_output,
        activation=activation,
        name=name or "head",
        **kwargs,
    )


def _build_tf_quantile_head(
    *,
    context=None,
    spec=None,
    units: int | None = None,
    output_dim: int = 1,
    forecast_horizon: int = 1,
    quantiles: tuple[float, ...] | None = None,
    activation: str | None = None,
    name: str | None = None,
    **kwargs,
) -> layers.Dense:
    """Build a TensorFlow-optimized quantile forecast output head."""
    del context, spec, activation
    _ensure_tensorflow()
    total_output = resolve_head_units(
        units=units,
        output_dim=output_dim,
        forecast_horizon=forecast_horizon,
        quantiles=quantiles,
        is_quantile=True,
    )
    return layers.Dense(
        total_output,
        activation=None,
        name=name or "quantile_head",
        **kwargs,
    )


def ensure_tensorflow_v2_registered(
    registry=None,
) -> None:
    """Register all TensorFlow-optimized V2 components.

    This registers TensorFlow-specific implementations that will be preferred
    over generic implementations when the TensorFlow backend is active.

    Arguments:
        registry: ComponentRegistry instance. If None, uses DEFAULT_COMPONENT_REGISTRY.
    """
    _ensure_tensorflow()

    if registry is None:
        from ...registry import DEFAULT_COMPONENT_REGISTRY

        registry = DEFAULT_COMPONENT_REGISTRY

    # Register TensorFlow-optimized projections
    registry.register(
        "projection.dense",
        _build_tf_dense_projection,
        backend="tensorflow",
        replace=True,
    )
    registry.register(
        "projection.static",
        _build_tf_dense_projection,
        backend="tensorflow",
        replace=True,
    )
    registry.register(
        "projection.dynamic",
        _build_tf_dense_projection,
        backend="tensorflow",
        replace=True,
    )
    registry.register(
        "projection.future",
        _build_tf_dense_projection,
        backend="tensorflow",
        replace=True,
    )
    registry.register(
        "projection.hidden",
        _build_tf_dense_projection,
        backend="tensorflow",
        replace=True,
    )

    # Register TensorFlow-optimized encoders
    registry.register(
        "encoder.temporal_self_attention",
        _build_tf_temporal_self_attention_encoder,
        backend="tensorflow",
        replace=True,
    )

    # Register TensorFlow-optimized pooling
    registry.register(
        "pool.mean",
        _build_tf_mean_pool,
        backend="tensorflow",
        replace=True,
    )
    registry.register(
        "pool.last",
        _build_tf_last_pool,
        backend="tensorflow",
        replace=True,
    )

    # Register TensorFlow-optimized fusion
    registry.register(
        "fusion.concat",
        _build_tf_concat_fusion,
        backend="tensorflow",
        replace=True,
    )

    # Register TensorFlow-optimized heads
    registry.register(
        "head.point_forecast",
        _build_tf_point_forecast_head,
        backend="tensorflow",
        replace=True,
    )
    registry.register(
        "head.quantile_forecast",
        _build_tf_quantile_head,
        backend="tensorflow",
        replace=True,
    )
    registry.register(
        "head.quantile",
        _build_tf_quantile_head,
        backend="tensorflow",
        replace=True,
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
