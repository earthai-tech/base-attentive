"""Torch-backend V2 builders implemented with Keras 3 primitives.

For Keras 3 multi-backend support, the Torch backend should return Keras
layers and callables that execute on the active Torch runtime, rather
than raw ``torch.nn.Module`` objects that Keras cannot fully track.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from ... import KERAS_DEPS
from ...keras_runtime import get_layer_class
from ...resolver.builder_contract import resolve_head_units
from ..generic.base_attentive_v2 import (
    _build_dynamic_window as _build_generic_dynamic_window,
    _build_feature_processor as _build_generic_feature_processor,
    _build_flatten_pool as _build_generic_flatten_pool,
    _build_hybrid_multiscale_encoder as _build_generic_hybrid_multiscale_encoder,
    _build_positional_encoding as _build_generic_positional_encoding,
)

Dense = KERAS_DEPS.Dense
Dropout = KERAS_DEPS.Dropout
Layer = get_layer_class()
LayerNormalization = KERAS_DEPS.LayerNormalization
MultiHeadAttention = KERAS_DEPS.MultiHeadAttention
concat_op = KERAS_DEPS.concat
mean_op = KERAS_DEPS.reduce_mean
register_keras_serializable = (
    KERAS_DEPS.register_keras_serializable
)
SERIALIZATION_PACKAGE = __name__

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


def _ensure_torch():
    if importlib.util.find_spec("torch") is None:
        raise ImportError(
            "PyTorch is required for torch backend implementations. "
            "Install it with: pip install torch"
        )


@register_keras_serializable(
    SERIALIZATION_PACKAGE,
    name="TorchTemporalSelfAttentionEncoder",
)
class _TorchTemporalSelfAttentionEncoder(Layer):
    """Keras multi-backend temporal encoder for the Torch backend."""

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
        _ensure_torch()
        super().__init__(
            name=name, **_clean_builder_kwargs(kwargs)
        )
        self.units = int(units)
        self.hidden_units = int(hidden_units)
        self.num_heads = int(num_heads)
        self.activation = activation
        self.dropout_rate = float(dropout_rate)
        self.layer_norm_epsilon = float(layer_norm_epsilon)
        key_dim = max(1, self.units // max(1, self.num_heads))
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate,
            name=f"{name}_mha" if name else "mha",
        )
        self.norm1 = LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name=f"{name}_ln1" if name else "ln1",
        )
        self.ffn_hidden = Dense(
            self.hidden_units,
            activation=self.activation,
            name=f"{name}_ffn_hidden"
            if name
            else "ffn_hidden",
        )
        self.ffn_output = Dense(
            self.units,
            name=f"{name}_ffn_output"
            if name
            else "ffn_output",
        )
        self.dropout = (
            Dropout(
                self.dropout_rate,
                name=f"{name}_dropout" if name else None,
            )
            if self.dropout_rate > 0
            else None
        )
        self.norm2 = LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name=f"{name}_ln2" if name else "ln2",
        )

    def forward(
        self, inputs: Any, training: bool = False
    ) -> Any:  # noqa: FBT002
        return self.call(inputs, training=training)

    def call(
        self, inputs: Any, training: bool = False
    ) -> Any:  # noqa: FBT002
        attn_output = self.attention(
            inputs, inputs, training=training
        )
        x = self.norm1(inputs + attn_output)
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
                "units": self.units,
                "hidden_units": self.hidden_units,
                "num_heads": self.num_heads,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
                "layer_norm_epsilon": self.layer_norm_epsilon,
            }
        )
        return config


@register_keras_serializable(
    SERIALIZATION_PACKAGE, name="TorchMeanPool"
)
class _MeanPool(Layer):
    def __init__(
        self,
        *,
        axis=None,
        keepdims=False,
        name=None,
        **kwargs,
    ):
        super().__init__(
            name=name, **_clean_builder_kwargs(kwargs)
        )
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs, training: bool = False):  # noqa: ARG002, FBT002
        return mean_op(
            inputs, axis=self.axis, keepdims=self.keepdims
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {"axis": self.axis, "keepdims": self.keepdims}
        )
        return config


@register_keras_serializable(
    SERIALIZATION_PACKAGE, name="TorchLastPool"
)
class _LastPool(Layer):
    def __init__(self, *, axis=1, name=None, **kwargs):
        if axis != 1:
            raise ValueError(
                "torch last-pool currently supports axis=1 only."
            )
        super().__init__(
            name=name, **_clean_builder_kwargs(kwargs)
        )
        self.axis = axis

    def call(self, inputs, training: bool = False):  # noqa: ARG002, FBT002
        return inputs[:, -1, :]

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@register_keras_serializable(
    SERIALIZATION_PACKAGE, name="TorchConcatFusion"
)
class _ConcatFusion(Layer):
    def __init__(self, *, axis=-1, name=None, **kwargs):
        super().__init__(
            name=name, **_clean_builder_kwargs(kwargs)
        )
        self.axis = axis

    def call(self, inputs, training: bool = False):  # noqa: ARG002, FBT002
        active = [
            value for value in inputs if value is not None
        ]
        if len(active) == 1:
            return active[0]
        return concat_op(active, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


expand_dims_op = KERAS_DEPS.expand_dims
tile_op = KERAS_DEPS.tile
shape_op = KERAS_DEPS.shape


class _SelfAttentionBlockMixin:
    @staticmethod
    def _key_dim(units: int, num_heads: int) -> int:
        return max(1, int(units) // max(1, int(num_heads)))


@register_keras_serializable(
    SERIALIZATION_PACKAGE,
    name="TorchCrossAttention",
)
class _TorchCrossAttention(Layer, _SelfAttentionBlockMixin):
    def __init__(
        self,
        *,
        units: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(
            name=name, **_clean_builder_kwargs(kwargs)
        )
        self.units = int(units)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout_rate)
        key_dim = self._key_dim(self.units, self.num_heads)
        self.query_dense = Dense(
            self.units,
            name=f"{name}_query" if name else None,
        )
        self.key_value_dense = Dense(
            self.units,
            name=f"{name}_key_value" if name else None,
        )
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate,
            name=f"{name}_mha" if name else None,
        )

    def call(self, inputs, training: bool = False):  # noqa: FBT002
        query, context = inputs
        query = self.query_dense(query)
        context = self.key_value_dense(context)
        return self.attention(
            query=query,
            key=context,
            value=context,
            training=training,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@register_keras_serializable(
    SERIALIZATION_PACKAGE,
    name="TorchHierarchicalAttention",
)
class _TorchHierarchicalAttention(
    Layer, _SelfAttentionBlockMixin
):
    def __init__(
        self,
        *,
        units: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(
            name=name, **_clean_builder_kwargs(kwargs)
        )
        self.units = int(units)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout_rate)
        key_dim = self._key_dim(self.units, self.num_heads)
        self.short_dense = Dense(
            self.units,
            name=f"{name}_short" if name else None,
        )
        self.long_dense = Dense(
            self.units,
            name=f"{name}_long" if name else None,
        )
        self.short_attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate,
            name=f"{name}_short_mha" if name else None,
        )
        self.long_attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate,
            name=f"{name}_long_mha" if name else None,
        )

    def call(self, inputs, training: bool = False):  # noqa: FBT002
        if (
            isinstance(inputs, (list, tuple))
            and len(inputs) == 2
        ):
            short_term, long_term = inputs
        else:
            short_term = long_term = inputs
        short_term = self.short_dense(short_term)
        long_term = self.long_dense(long_term)
        short_attended = self.short_attention(
            query=short_term,
            key=short_term,
            value=short_term,
            training=training,
        )
        long_attended = self.long_attention(
            query=long_term,
            key=long_term,
            value=long_term,
            training=training,
        )
        return short_attended + long_attended

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@register_keras_serializable(
    SERIALIZATION_PACKAGE,
    name="TorchMemoryAttention",
)
class _TorchMemoryAttention(Layer, _SelfAttentionBlockMixin):
    def __init__(
        self,
        *,
        units: int,
        memory_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(
            name=name, **_clean_builder_kwargs(kwargs)
        )
        self.units = int(units)
        self.memory_size = int(memory_size)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout_rate)
        key_dim = self._key_dim(self.units, self.num_heads)
        self.input_dense = Dense(
            self.units,
            name=f"{name}_input" if name else None,
        )
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate,
            name=f"{name}_mha" if name else None,
        )

    def build(self, input_shape):
        self.memory = self.add_weight(
            name="memory",
            shape=(self.memory_size, self.units),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, training: bool = False):  # noqa: FBT002
        encoded = self.input_dense(inputs)
        batch_size = shape_op(encoded)[0]
        memory = expand_dims_op(self.memory, axis=0)
        memory = tile_op(memory, [batch_size, 1, 1])
        attended = self.attention(
            query=encoded,
            key=memory,
            value=memory,
            training=training,
        )
        return attended + encoded

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "memory_size": self.memory_size,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@register_keras_serializable(
    SERIALIZATION_PACKAGE,
    name="TorchMultiResolutionAttentionFusion",
)
class _TorchMultiResolutionAttentionFusion(
    Layer, _SelfAttentionBlockMixin
):
    def __init__(
        self,
        *,
        units: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(
            name=name, **_clean_builder_kwargs(kwargs)
        )
        self.units = int(units)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout_rate)
        key_dim = self._key_dim(self.units, self.num_heads)
        self.input_dense = Dense(
            self.units,
            name=f"{name}_input" if name else None,
        )
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate,
            name=f"{name}_mha" if name else None,
        )

    def call(self, inputs, training: bool = False):  # noqa: FBT002
        encoded = self.input_dense(inputs)
        return self.attention(
            query=encoded,
            key=encoded,
            value=encoded,
            training=training,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@register_keras_serializable(
    SERIALIZATION_PACKAGE,
    name="TorchMultiHorizonHead",
)
class _TorchMultiHorizonHead(Layer):
    def __init__(
        self,
        *,
        output_dim: int,
        forecast_horizon: int,
        activation: str | None = None,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(
            name=name, **_clean_builder_kwargs(kwargs)
        )
        self.output_dim = int(output_dim)
        self.forecast_horizon = int(forecast_horizon)
        self.activation = activation
        total_units = self.output_dim * self.forecast_horizon
        self.projection = Dense(
            total_units,
            activation=self.activation,
            name=f"{name}_projection" if name else None,
        )

    def call(self, inputs, training: bool = False):  # noqa: ARG002, FBT002
        outputs = self.projection(inputs)
        batch_size = shape_op(outputs)[0]
        return KERAS_DEPS.reshape(
            outputs,
            [
                batch_size,
                self.forecast_horizon,
                self.output_dim,
            ],
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "forecast_horizon": self.forecast_horizon,
                "activation": self.activation,
            }
        )
        return config


@register_keras_serializable(
    SERIALIZATION_PACKAGE,
    name="TorchQuantileDistributionHead",
)
class _TorchQuantileDistributionHead(Layer):
    def __init__(
        self,
        *,
        quantiles: tuple[float, ...] | list[float],
        output_dim: int,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(
            name=name, **_clean_builder_kwargs(kwargs)
        )
        self.quantiles = tuple(float(q) for q in quantiles)
        self.output_dim = int(output_dim)
        self.projection = Dense(
            self.output_dim * len(self.quantiles),
            name=f"{name}_projection" if name else None,
        )

    def call(self, inputs, training: bool = False):  # noqa: ARG002, FBT002
        outputs = self.projection(inputs)
        shape = shape_op(outputs)
        batch_size = shape[0]
        horizon = shape[1]
        return KERAS_DEPS.reshape(
            outputs,
            [
                batch_size,
                horizon,
                len(self.quantiles),
                self.output_dim,
            ],
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "quantiles": list(self.quantiles),
                "output_dim": self.output_dim,
            }
        )
        return config


def _delegate_generic(builder, **kwargs):
    return builder(**kwargs)


def _build_torch_dense_projection(
    *,
    context=None,
    spec=None,
    units: int,
    activation: str | None = None,
    name: str | None = None,
    **kwargs,
):
    del context, spec
    _ensure_torch()
    return Dense(
        units,
        activation=activation,
        name=name,
        **_clean_builder_kwargs(kwargs),
    )


def _build_torch_temporal_self_attention_encoder(
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
):
    del context, spec
    _ensure_torch()
    return _TorchTemporalSelfAttentionEncoder(
        units=units,
        hidden_units=hidden_units,
        num_heads=num_heads,
        activation=activation,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        name=name,
        **_clean_builder_kwargs(kwargs),
    )


def _build_torch_mean_pool(
    *,
    context=None,
    spec=None,
    axis: int | None = None,
    keepdims: bool = False,
    name: str | None = None,
    **kwargs,
):
    del context, spec, kwargs
    _ensure_torch()
    return _MeanPool(axis=axis, keepdims=keepdims, name=name)


def _build_torch_last_pool(
    *,
    context=None,
    spec=None,
    axis: int = 1,
    name: str | None = None,
    **kwargs,
):
    del context, spec, kwargs
    _ensure_torch()
    return _LastPool(axis=axis, name=name)


def _build_torch_concat_fusion(
    *,
    context=None,
    spec=None,
    axis: int = -1,
    name: str | None = None,
    **kwargs,
):
    del context, spec, kwargs
    _ensure_torch()
    return _ConcatFusion(axis=axis, name=name)


def _build_torch_point_forecast_head(
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
):
    del context, spec, quantiles
    _ensure_torch()
    total_units = resolve_head_units(
        units=units,
        output_dim=output_dim,
        forecast_horizon=forecast_horizon,
        is_quantile=False,
    )
    return Dense(
        total_units,
        activation=activation,
        name=name,
        **_clean_builder_kwargs(kwargs),
    )


def _build_torch_quantile_head(
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
):
    del context, spec
    _ensure_torch()
    total_units = resolve_head_units(
        units=units,
        output_dim=output_dim,
        forecast_horizon=forecast_horizon,
        quantiles=quantiles,
        is_quantile=True,
    )
    return Dense(
        total_units,
        activation=activation,
        name=name,
        **_clean_builder_kwargs(kwargs),
    )


def _build_torch_static_processor(**kwargs):
    _ensure_torch()
    return _delegate_generic(
        _build_generic_feature_processor, **kwargs
    )


def _build_torch_dynamic_processor(**kwargs):
    _ensure_torch()
    return _delegate_generic(
        _build_generic_feature_processor, **kwargs
    )


def _build_torch_future_processor(**kwargs):
    _ensure_torch()
    return _delegate_generic(
        _build_generic_feature_processor, **kwargs
    )


def _build_torch_positional_encoding(**kwargs):
    _ensure_torch()
    return _delegate_generic(
        _build_generic_positional_encoding, **kwargs
    )


def _build_torch_hybrid_multiscale_encoder(**kwargs):
    _ensure_torch()
    return _delegate_generic(
        _build_generic_hybrid_multiscale_encoder, **kwargs
    )


def _build_torch_dynamic_window(**kwargs):
    _ensure_torch()
    return _delegate_generic(
        _build_generic_dynamic_window, **kwargs
    )


def _build_torch_cross_attention(
    *,
    context=None,
    spec=None,
    units: int | None = None,
    num_heads: int | None = None,
    dropout_rate: float = 0.0,
    name: str | None = None,
    **kwargs,
):
    del context
    _ensure_torch()
    active_units = int(
        units or getattr(spec, "attention_units", 32)
    )
    active_heads = int(
        num_heads or getattr(spec, "num_heads", 1)
    )
    return _TorchCrossAttention(
        units=active_units,
        num_heads=active_heads,
        dropout_rate=dropout_rate,
        name=name,
        **_clean_builder_kwargs(kwargs),
    )


def _build_torch_hierarchical_attention(
    *,
    context=None,
    spec=None,
    units: int | None = None,
    num_heads: int | None = None,
    dropout_rate: float = 0.0,
    name: str | None = None,
    **kwargs,
):
    del context
    _ensure_torch()
    active_units = int(
        units or getattr(spec, "attention_units", 32)
    )
    active_heads = int(
        num_heads or getattr(spec, "num_heads", 1)
    )
    return _TorchHierarchicalAttention(
        units=active_units,
        num_heads=active_heads,
        dropout_rate=dropout_rate,
        name=name,
        **_clean_builder_kwargs(kwargs),
    )


def _build_torch_memory_attention(
    *,
    context=None,
    spec=None,
    units: int | None = None,
    memory_size: int | None = None,
    num_heads: int | None = None,
    dropout_rate: float = 0.0,
    name: str | None = None,
    **kwargs,
):
    del context
    _ensure_torch()
    active_units = int(
        units or getattr(spec, "attention_units", 32)
    )
    active_memory = int(
        memory_size or getattr(spec, "memory_size", 1)
    )
    active_heads = int(
        num_heads or getattr(spec, "num_heads", 1)
    )
    return _TorchMemoryAttention(
        units=active_units,
        memory_size=active_memory,
        num_heads=active_heads,
        dropout_rate=dropout_rate,
        name=name,
        **_clean_builder_kwargs(kwargs),
    )


def _build_torch_multi_resolution_attention_fusion(
    *,
    context=None,
    spec=None,
    units: int | None = None,
    num_heads: int | None = None,
    dropout_rate: float = 0.0,
    name: str | None = None,
    **kwargs,
):
    del context
    _ensure_torch()
    active_units = int(
        units or getattr(spec, "attention_units", 32)
    )
    active_heads = int(
        num_heads or getattr(spec, "num_heads", 1)
    )
    return _TorchMultiResolutionAttentionFusion(
        units=active_units,
        num_heads=active_heads,
        dropout_rate=dropout_rate,
        name=name,
        **_clean_builder_kwargs(kwargs),
    )


def _build_torch_flatten_pool(**kwargs):
    _ensure_torch()
    return _delegate_generic(
        _build_generic_flatten_pool, **kwargs
    )


def _build_torch_multi_horizon_head(
    *,
    context=None,
    spec=None,
    output_dim: int | None = None,
    forecast_horizon: int | None = None,
    activation: str | None = None,
    name: str | None = None,
    **kwargs,
):
    del context
    _ensure_torch()
    active_output_dim = int(
        output_dim
        if output_dim is not None
        else spec.output_dim
    )
    active_horizon = int(
        forecast_horizon
        if forecast_horizon is not None
        else spec.forecast_horizon
    )
    return _TorchMultiHorizonHead(
        output_dim=active_output_dim,
        forecast_horizon=active_horizon,
        activation=activation,
        name=name,
        **_clean_builder_kwargs(kwargs),
    )


def _build_torch_quantile_distribution_head(
    *,
    context=None,
    spec=None,
    output_dim: int | None = None,
    quantiles: tuple[float, ...] | list[float] | None = None,
    name: str | None = None,
    **kwargs,
):
    del context
    _ensure_torch()
    active_output_dim = int(
        output_dim
        if output_dim is not None
        else spec.output_dim
    )
    active_quantiles = tuple(
        quantiles if quantiles is not None else spec.quantiles
    )
    return _TorchQuantileDistributionHead(
        quantiles=active_quantiles,
        output_dim=active_output_dim,
        name=name,
        **_clean_builder_kwargs(kwargs),
    )


def ensure_torch_v2_registered(registry=None) -> None:
    _ensure_torch()
    if registry is None:
        from ...registry import DEFAULT_COMPONENT_REGISTRY

        registry = DEFAULT_COMPONENT_REGISTRY
    registrations = [
        ("projection.dense", _build_torch_dense_projection),
        ("projection.static", _build_torch_dense_projection),
        ("projection.dynamic", _build_torch_dense_projection),
        ("projection.future", _build_torch_dense_projection),
        ("projection.hidden", _build_torch_dense_projection),
        (
            "feature.static_processor",
            _build_torch_static_processor,
        ),
        (
            "feature.dynamic_processor",
            _build_torch_dynamic_processor,
        ),
        (
            "feature.future_processor",
            _build_torch_future_processor,
        ),
        (
            "embedding.positional",
            _build_torch_positional_encoding,
        ),
        (
            "encoder.hybrid_multiscale",
            _build_torch_hybrid_multiscale_encoder,
        ),
        (
            "encoder.temporal_self_attention",
            _build_torch_temporal_self_attention_encoder,
        ),
        (
            "encoder.dynamic_window",
            _build_torch_dynamic_window,
        ),
        (
            "decoder.cross_attention",
            _build_torch_cross_attention,
        ),
        (
            "decoder.hierarchical_attention",
            _build_torch_hierarchical_attention,
        ),
        (
            "decoder.memory_attention",
            _build_torch_memory_attention,
        ),
        (
            "fusion.multi_resolution_attention",
            _build_torch_multi_resolution_attention_fusion,
        ),
        ("pool.mean", _build_torch_mean_pool),
        ("pool.last", _build_torch_last_pool),
        ("pool.final_last", _build_torch_last_pool),
        ("pool.final_mean", _build_torch_mean_pool),
        ("pool.final_flatten", _build_torch_flatten_pool),
        ("fusion.concat", _build_torch_concat_fusion),
        (
            "head.multi_horizon",
            _build_torch_multi_horizon_head,
        ),
        (
            "head.quantile_distribution",
            _build_torch_quantile_distribution_head,
        ),
        (
            "head.point_forecast",
            _build_torch_point_forecast_head,
        ),
        (
            "head.quantile_forecast",
            _build_torch_quantile_head,
        ),
    ]
    for key, builder in registrations:
        registry.register(key, builder, backend="torch")


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
