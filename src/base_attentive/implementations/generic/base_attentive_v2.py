"""Generic V2 component builders and model assembler."""

from __future__ import annotations

from ... import KERAS_DEPS
from ...keras_runtime import get_layer_class
from ...components._temporal_utils import (
    aggregate_multiscale_on_3d,
)
from ...components.attention import (
    CrossAttention,
    HierarchicalAttention,
    MemoryAugmentedAttention,
    MultiResolutionAttentionFusion,
)
from ...components.encoder_decoder import MultiDecoder
from ...components.gating_norm import (
    GatedResidualNetwork,
    VariableSelectionNetwork,
)
from ...components.heads import QuantileDistributionModeling
from ...components.misc import PositionalEncoding
from ...components.temporal import (
    DynamicTimeWindow,
    MultiScaleLSTM,
)
from ...registry import (
    DEFAULT_COMPONENT_REGISTRY,
    DEFAULT_MODEL_REGISTRY,
    ComponentRegistry,
    ModelRegistry,
)
from ...resolver.assembly import BaseAttentiveV2Assembly
from ...resolver.builder_contract import (
    build_head_kwargs,
    resolve_head_units,
)
from ...resolver.component_resolver import build_component

register_keras_serializable = (
    KERAS_DEPS.register_keras_serializable
)
Layer = get_layer_class()
SERIALIZATION_PACKAGE = __name__


def _invoke(component, inputs, *, training: bool = False):
    """Call a component, passing ``training`` when accepted."""
    if component is None:
        return inputs
    try:
        return component(inputs, training=training)
    except TypeError:
        return component(inputs)


@register_keras_serializable(
    SERIALIZATION_PACKAGE,
    name="GenericFeatureProcessor",
)
class _GenericFeatureProcessor(Layer):
    """Generic feature processor for static, dynamic, and future inputs."""

    def __init__(
        self,
        *,
        role: str,
        input_dim: int,
        output_units: int,
        vsn_units: int | None,
        feature_processing: str,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.role = role
        self.input_dim = int(input_dim)
        self.output_units = int(output_units)
        self.vsn_units = int(vsn_units or output_units)
        self.feature_processing = str(feature_processing)
        self.activation = activation
        self.dropout_rate = float(dropout_rate)
        self.use_batch_norm = bool(use_batch_norm)

        self._vsn = None
        self._post_grn = None
        self._dense = None

        if self.input_dim <= 0:
            return

        if self.feature_processing == "vsn":
            use_time_distributed = role in {
                "dynamic",
                "future",
            }
            self._vsn = VariableSelectionNetwork(
                num_inputs=self.input_dim,
                units=self.vsn_units,
                dropout_rate=self.dropout_rate,
                use_time_distributed=use_time_distributed,
                use_batch_norm=self.use_batch_norm,
                activation="elu",
                name=name,
            )
            self._post_grn = GatedResidualNetwork(
                units=self.output_units,
                dropout_rate=self.dropout_rate,
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
                name=(f"{name}_grn" if name else None),
            )
        else:
            dense_activation = (
                self.activation if role == "static" else None
            )
            dense_cls = getattr(KERAS_DEPS, "Dense", None)
            if dense_cls is None:
                raise ImportError(
                    "Dense is required for generic feature processing."
                )
            self._dense = dense_cls(
                self.output_units,
                activation=dense_activation,
                name=(f"{name}_dense" if name else None),
            )
            if role == "static":
                self._post_grn = GatedResidualNetwork(
                    units=self.output_units,
                    dropout_rate=self.dropout_rate,
                    activation=self.activation,
                    use_batch_norm=self.use_batch_norm,
                    name=(f"{name}_grn" if name else None),
                )

    def call(self, inputs, training: bool = False):  # noqa: FBT002
        if inputs is None or self.input_dim <= 0:
            return None

        if self._vsn is not None:
            encoded = _invoke(
                self._vsn, inputs, training=training
            )
            return _invoke(
                self._post_grn,
                encoded,
                training=training,
            )

        encoded = _invoke(
            self._dense, inputs, training=training
        )
        if self._post_grn is not None:
            encoded = _invoke(
                self._post_grn,
                encoded,
                training=training,
            )
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "role": self.role,
                "input_dim": self.input_dim,
                "output_units": self.output_units,
                "vsn_units": self.vsn_units,
                "feature_processing": self.feature_processing,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
                "use_batch_norm": self.use_batch_norm,
            }
        )
        return config


@register_keras_serializable(
    SERIALIZATION_PACKAGE,
    name="GenericMeanPool",
)
class _GenericMeanPool(Layer):
    """Serializable mean-pooling layer for resolver assemblies."""

    def __init__(
        self,
        *,
        axis: int | None = 1,
        keepdims: bool = False,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs, training: bool = False):  # noqa: ARG002, FBT002
        return KERAS_DEPS.reduce_mean(
            inputs,
            axis=self.axis,
            keepdims=self.keepdims,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {"axis": self.axis, "keepdims": self.keepdims}
        )
        return config


@register_keras_serializable(
    SERIALIZATION_PACKAGE,
    name="GenericLastPool",
)
class _GenericLastPool(Layer):
    """Serializable last-step pooling layer for resolver assemblies."""

    def __init__(
        self,
        *,
        axis: int = 1,
        name: str | None = None,
        **kwargs,
    ):
        if axis != 1:
            raise ValueError(
                "generic last-pool currently supports axis=1 only."
            )
        super().__init__(name=name, **kwargs)
        self.axis = axis

    def call(self, inputs, training: bool = False):  # noqa: ARG002, FBT002
        return inputs[:, -1, :]

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@register_keras_serializable(
    SERIALIZATION_PACKAGE,
    name="GenericFlattenPool",
)
class _GenericFlattenPool(Layer):
    """Serializable flatten-pooling layer for resolver assemblies."""

    def __init__(
        self,
        *,
        axis: int = 1,
        name: str | None = None,
        **kwargs,
    ):
        if axis != 1:
            raise ValueError(
                "generic flatten-pool currently supports axis=1 only."
            )
        super().__init__(name=name, **kwargs)
        self.axis = axis

    def call(self, inputs, training: bool = False):  # noqa: ARG002, FBT002
        shape = KERAS_DEPS.shape(inputs)
        return KERAS_DEPS.reshape(
            inputs, [shape[0], shape[1] * shape[2]]
        )

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@register_keras_serializable(
    SERIALIZATION_PACKAGE,
    name="GenericConcatFusion",
)
class _GenericConcatFusion(Layer):
    """Serializable concatenation fusion layer for resolver assemblies."""

    def __init__(
        self,
        *,
        axis: int = -1,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.axis = axis

    def __call__(self, inputs, *args, **kwargs):
        if isinstance(inputs, (list, tuple)) and not any(
            feature is not None for feature in inputs
        ):
            raise ValueError(
                "fusion received no active feature tensors."
            )
        return super().__call__(inputs, *args, **kwargs)

    def call(self, inputs, training: bool = False):  # noqa: ARG002, FBT002
        active = [
            feature
            for feature in inputs
            if feature is not None
        ]
        if not active:
            raise ValueError(
                "fusion received no active feature tensors."
            )
        if len(active) == 1:
            return active[0]
        return KERAS_DEPS.concat(active, axis=self.axis)

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


@register_keras_serializable(
    SERIALIZATION_PACKAGE,
    name="GenericTemporalSelfAttentionEncoder",
)
class _TemporalSelfAttentionEncoder(Layer):
    """Small backend-neutral temporal encoder built from Keras primitives."""

    def __init__(
        self,
        *,
        units: int,
        hidden_units: int,
        num_heads: int,
        activation: str,
        dropout_rate: float,
        layer_norm_epsilon: float,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        layers = KERAS_DEPS
        if (
            getattr(layers, "MultiHeadAttention", None)
            is None
        ):
            raise ImportError(
                "MultiHeadAttention is required for the V2 temporal encoder."
            )
        if (
            getattr(layers, "LayerNormalization", None)
            is None
        ):
            raise ImportError(
                "LayerNormalization is required for the V2 temporal encoder."
            )

        self.units = int(units)
        self.hidden_units = int(hidden_units)
        self.num_heads = int(num_heads)
        self.activation = activation
        self.dropout_rate = float(dropout_rate)
        self.layer_norm_epsilon = float(layer_norm_epsilon)

        key_dim = max(1, self.units // max(1, self.num_heads))
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout_rate,
            name=f"{name}_self_attention" if name else None,
        )
        self.norm1 = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name=f"{name}_norm_1" if name else None,
        )
        self.ffn_hidden = layers.Dense(
            self.hidden_units,
            activation=self.activation,
            name=f"{name}_ffn_hidden" if name else None,
        )
        self.ffn_output = layers.Dense(
            self.units,
            name=f"{name}_ffn_output" if name else None,
        )
        self.dropout = None
        if (
            self.dropout_rate > 0
            and getattr(layers, "Dropout", None) is not None
        ):
            self.dropout = layers.Dropout(
                self.dropout_rate,
                name=f"{name}_dropout" if name else None,
            )
        self.norm2 = layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            name=f"{name}_norm_2" if name else None,
        )

    def call(self, inputs, training: bool = False):  # noqa: FBT002
        attention_output = self.attention(
            inputs,
            inputs,
            training=training,
        )
        encoded = self.norm1(inputs + attention_output)
        ffn_output = self.ffn_hidden(encoded)
        ffn_output = self.ffn_output(ffn_output)
        if self.dropout is not None:
            ffn_output = self.dropout(
                ffn_output,
                training=training,
            )
        return self.norm2(encoded + ffn_output)

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
    SERIALIZATION_PACKAGE,
    name="GenericHybridMultiScaleEncoder",
)
class _HybridMultiScaleEncoder(Layer):
    """Hybrid multi-scale encoder for the migrated legacy path."""

    def __init__(
        self,
        *,
        lstm_units: int,
        scales: tuple[int, ...] | list[int] | str,
        sequence_mode: str = "concat",
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.lstm_units = int(lstm_units)
        if isinstance(scales, str) or not scales:
            active_scales = [1]
        else:
            active_scales = [int(value) for value in scales]
        self.scales = tuple(active_scales)
        self.sequence_mode = sequence_mode or "concat"
        self.multi_scale_lstm = MultiScaleLSTM(
            lstm_units=self.lstm_units,
            scales=list(self.scales),
            return_sequences=True,
            name=name,
        )

    def call(self, inputs, training: bool = False):  # noqa: FBT002
        outputs = self.multi_scale_lstm(
            inputs,
            training=training,
        )
        mode = self.sequence_mode
        if mode != "concat":
            mode = "concat"
        return aggregate_multiscale_on_3d(outputs, mode=mode)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "lstm_units": self.lstm_units,
                "scales": list(self.scales),
                "sequence_mode": self.sequence_mode,
            }
        )
        return config


def _resolve_lstm_units(
    value: int | tuple[int, ...] | list[int],
) -> int:
    if isinstance(value, int):
        return value
    values = list(value)
    if not values:
        raise ValueError("lstm_units cannot be empty.")
    return int(values[0])


def _build_dense_projection(
    *,
    context,
    spec=None,
    units: int,
    activation: str | None = None,
    name: str | None = None,
    **kwargs,
):
    del spec, kwargs
    return context.layers.Dense(
        units,
        activation=activation,
        name=name,
    )


def _build_feature_processor(
    *,
    context,
    spec=None,
    role: str,
    input_dim: int,
    output_units: int,
    vsn_units: int | None = None,
    activation: str = "relu",
    dropout_rate: float = 0.0,
    use_batch_norm: bool = False,
    name: str | None = None,
    **kwargs,
):
    del kwargs
    if spec is None:
        raise ValueError("feature processors require a spec.")
    return _GenericFeatureProcessor(
        role=role,
        input_dim=input_dim,
        output_units=output_units,
        vsn_units=vsn_units,
        feature_processing=spec.architecture.feature_processing,
        activation=activation,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        name=name,
    )


def _build_positional_encoding(
    *,
    context,
    spec=None,
    name: str | None = None,
    **kwargs,
):
    del context, spec, kwargs
    return PositionalEncoding(name=name)


def _build_hybrid_multiscale_encoder(
    *,
    context,
    spec=None,
    lstm_units: int
    | tuple[int, ...]
    | list[int]
    | None = None,
    scales: tuple[int, ...] | list[int] | str | None = None,
    aggregation: str | None = None,
    name: str | None = None,
    **kwargs,
):
    del context, kwargs
    if spec is None and lstm_units is None:
        raise ValueError(
            "hybrid encoder requires a spec or lstm_units."
        )

    active_lstm_units = _resolve_lstm_units(
        lstm_units
        if lstm_units is not None
        else spec.lstm_units
    )
    active_scales = (
        scales if scales is not None else spec.scales
    )
    sequence_mode = aggregation
    if sequence_mode is None and spec is not None:
        sequence_mode = spec.runtime.multi_scale_agg
    return _HybridMultiScaleEncoder(
        lstm_units=active_lstm_units,
        scales=active_scales,
        sequence_mode=sequence_mode or "concat",
    )


def _build_temporal_self_attention_encoder(
    *,
    context,
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
    del spec, kwargs
    return _TemporalSelfAttentionEncoder(
        units=units,
        hidden_units=hidden_units,
        num_heads=num_heads,
        activation=activation,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=layer_norm_epsilon,
    )


def _build_dynamic_window(
    *,
    context,
    spec=None,
    max_window_size: int | None = None,
    name: str | None = None,
    **kwargs,
):
    del context, kwargs
    active_window = max_window_size
    if active_window is None and spec is not None:
        active_window = spec.max_window_size
    return DynamicTimeWindow(
        max_window_size=int(active_window or 1),
    )


def _build_cross_attention(
    *,
    context,
    spec=None,
    units: int | None = None,
    num_heads: int | None = None,
    name: str | None = None,
    **kwargs,
):
    del context, kwargs
    active_units = int(
        units or getattr(spec, "attention_units", 32)
    )
    active_heads = int(
        num_heads or getattr(spec, "num_heads", 1)
    )
    return CrossAttention(
        units=active_units,
        num_heads=active_heads,
    )


def _build_hierarchical_attention(
    *,
    context,
    spec=None,
    units: int | None = None,
    num_heads: int | None = None,
    name: str | None = None,
    **kwargs,
):
    del context, kwargs
    active_units = int(
        units or getattr(spec, "attention_units", 32)
    )
    active_heads = int(
        num_heads or getattr(spec, "num_heads", 1)
    )
    return HierarchicalAttention(
        units=active_units,
        num_heads=active_heads,
    )


def _build_memory_attention(
    *,
    context,
    spec=None,
    units: int | None = None,
    memory_size: int | None = None,
    num_heads: int | None = None,
    name: str | None = None,
    **kwargs,
):
    del context, kwargs
    active_units = int(
        units or getattr(spec, "attention_units", 32)
    )
    active_memory = int(
        memory_size or getattr(spec, "memory_size", 1)
    )
    active_heads = int(
        num_heads or getattr(spec, "num_heads", 1)
    )
    return MemoryAugmentedAttention(
        units=active_units,
        memory_size=active_memory,
        num_heads=active_heads,
    )


def _build_multi_resolution_attention_fusion(
    *,
    context,
    spec=None,
    units: int | None = None,
    num_heads: int | None = None,
    name: str | None = None,
    **kwargs,
):
    del context, kwargs
    active_units = int(
        units or getattr(spec, "attention_units", 32)
    )
    active_heads = int(
        num_heads or getattr(spec, "num_heads", 1)
    )
    return MultiResolutionAttentionFusion(
        units=active_units,
        num_heads=active_heads,
    )


def _build_mean_pool(
    *,
    context,
    spec=None,
    axis: int = 1,
    name: str | None = None,
    keepdims: bool = False,
    **kwargs,
):
    del context, spec, kwargs
    return _GenericMeanPool(
        axis=axis, keepdims=keepdims, name=name
    )


def _build_last_pool(
    *,
    context,
    spec=None,
    axis: int = 1,
    name: str | None = None,
    **kwargs,
):
    del context, spec, kwargs
    return _GenericLastPool(axis=axis, name=name)


def _build_flatten_pool(
    *,
    context,
    spec=None,
    axis: int = 1,
    name: str | None = None,
    **kwargs,
):
    del context, spec, kwargs
    return _GenericFlattenPool(axis=axis, name=name)


def _build_concat_fusion(
    *,
    context,
    spec=None,
    axis: int = -1,
    name: str | None = None,
    **kwargs,
):
    del context, spec, kwargs
    return _GenericConcatFusion(axis=axis, name=name)


def _build_point_forecast_head(
    *,
    context,
    spec=None,
    units: int | None = None,
    output_dim: int | None = None,
    forecast_horizon: int | None = None,
    quantiles: tuple[float, ...] | list[float] | None = None,
    activation: str | None = None,
    name: str | None = None,
    **kwargs,
):
    del spec, kwargs
    total_units = resolve_head_units(
        units=units,
        output_dim=output_dim,
        forecast_horizon=forecast_horizon,
        quantiles=quantiles,
        is_quantile=False,
    )
    return context.layers.Dense(
        total_units,
        activation=activation,
        name=name,
    )


def _build_multi_horizon_head(
    *,
    context,
    spec=None,
    output_dim: int | None = None,
    forecast_horizon: int | None = None,
    name: str | None = None,
    **kwargs,
):
    del context, kwargs, name
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
    return MultiDecoder(
        output_dim=active_output_dim,
        num_horizons=active_horizon,
    )


def _build_quantile_forecast_head(
    *,
    context,
    spec=None,
    units: int | None = None,
    output_dim: int | None = None,
    forecast_horizon: int | None = None,
    quantiles: tuple[float, ...] | list[float] | None = None,
    activation: str | None = None,
    name: str | None = None,
    **kwargs,
):
    del spec, kwargs
    total_units = resolve_head_units(
        units=units,
        output_dim=output_dim,
        forecast_horizon=forecast_horizon,
        quantiles=quantiles,
        is_quantile=True,
    )
    return context.layers.Dense(
        total_units,
        activation=activation,
        name=name,
    )


def _build_quantile_distribution_head(
    *,
    context,
    spec=None,
    output_dim: int | None = None,
    forecast_horizon: int | None = None,
    quantiles: tuple[float, ...] | list[float] | None = None,
    name: str | None = None,
    **kwargs,
):
    del context, kwargs, forecast_horizon
    active_output_dim = int(
        output_dim
        if output_dim is not None
        else spec.output_dim
    )
    active_quantiles = tuple(
        quantiles if quantiles is not None else spec.quantiles
    )
    return QuantileDistributionModeling(
        quantiles=list(active_quantiles),
        output_dim=active_output_dim,
        name=name,
    )


def _resolve_final_pool_key(spec) -> str:
    mode = (spec.final_agg or "flatten").replace("-", "_")
    if mode == "average":
        return spec.components.final_pool_mean
    if mode == "flatten":
        return spec.components.final_pool_flatten
    return spec.components.final_pool_last


def _assemble_base_attentive_v2(
    *,
    spec,
    backend_context,
    component_registry,
):
    static_processor = None
    if spec.static_input_dim > 0:
        static_processor = build_component(
            spec.components.static_processor,
            backend_context=backend_context,
            registry=component_registry,
            spec=spec,
            role="static",
            input_dim=spec.static_input_dim,
            output_units=spec.hidden_units,
            vsn_units=spec.vsn_units,
            activation=spec.activation,
            dropout_rate=spec.dropout_rate,
            use_batch_norm=spec.use_batch_norm,
            name="v2_static_processor",
        )

    dynamic_processor = build_component(
        spec.components.dynamic_processor,
        backend_context=backend_context,
        registry=component_registry,
        spec=spec,
        role="dynamic",
        input_dim=spec.dynamic_input_dim,
        output_units=spec.embed_dim,
        vsn_units=spec.vsn_units,
        activation=spec.activation,
        dropout_rate=spec.dropout_rate,
        use_batch_norm=spec.use_batch_norm,
        name="v2_dynamic_processor",
    )

    future_processor = None
    if spec.future_input_dim > 0:
        future_processor = build_component(
            spec.components.future_processor,
            backend_context=backend_context,
            registry=component_registry,
            spec=spec,
            role="future",
            input_dim=spec.future_input_dim,
            output_units=spec.embed_dim,
            vsn_units=spec.vsn_units,
            activation=spec.activation,
            dropout_rate=spec.dropout_rate,
            use_batch_norm=spec.use_batch_norm,
            name="v2_future_processor",
        )

    encoder_positional_encoding = build_component(
        spec.components.positional_encoder,
        backend_context=backend_context,
        registry=component_registry,
        spec=spec,
        name="v2_encoder_positional_encoding",
    )

    future_positional_encoding = None
    if future_processor is not None:
        future_positional_encoding = build_component(
            spec.components.positional_encoder,
            backend_context=backend_context,
            registry=component_registry,
            spec=spec,
            name="v2_future_positional_encoding",
        )

    if spec.architecture.encoder_type == "hybrid":
        dynamic_encoder = build_component(
            spec.components.hybrid_encoder,
            backend_context=backend_context,
            registry=component_registry,
            spec=spec,
            lstm_units=spec.lstm_units,
            scales=spec.scales,
            aggregation="concat",
            name="v2_dynamic_hybrid_encoder",
        )
        future_encoder = None
    else:
        dynamic_encoder = build_component(
            spec.components.dynamic_encoder,
            backend_context=backend_context,
            registry=component_registry,
            spec=spec,
            units=spec.embed_dim,
            hidden_units=spec.hidden_units,
            num_heads=spec.attention_heads,
            activation=spec.activation,
            dropout_rate=spec.dropout_rate,
            layer_norm_epsilon=spec.layer_norm_epsilon,
            name="v2_dynamic_encoder",
        )
        future_encoder = None
        if future_processor is not None:
            future_encoder = build_component(
                spec.components.future_encoder,
                backend_context=backend_context,
                registry=component_registry,
                spec=spec,
                units=spec.embed_dim,
                hidden_units=spec.hidden_units,
                num_heads=spec.attention_heads,
                activation=spec.activation,
                dropout_rate=spec.dropout_rate,
                layer_norm_epsilon=spec.layer_norm_epsilon,
                name="v2_future_encoder",
            )

    dynamic_window = None
    if spec.apply_dtw:
        dynamic_window = build_component(
            spec.components.dynamic_window,
            backend_context=backend_context,
            registry=component_registry,
            spec=spec,
            max_window_size=spec.max_window_size,
            name="v2_dynamic_window",
        )

    sequence_pool = build_component(
        spec.components.sequence_pooling,
        backend_context=backend_context,
        registry=component_registry,
        spec=spec,
        axis=1,
        keepdims=False,
        name="v2_sequence_pool",
    )
    fusion = build_component(
        spec.components.fusion,
        backend_context=backend_context,
        registry=component_registry,
        spec=spec,
        axis=-1,
        name="v2_feature_fusion",
    )
    hidden_projection = build_component(
        spec.components.hidden_projection,
        backend_context=backend_context,
        registry=component_registry,
        spec=spec,
        units=spec.hidden_units,
        activation=spec.activation,
        name="v2_hidden_projection",
    )

    decoder_input_projection = build_component(
        spec.components.hidden_projection,
        backend_context=backend_context,
        registry=component_registry,
        spec=spec,
        units=spec.attention_units,
        activation=spec.activation,
        name="v2_decoder_input_projection",
    )
    decoder_cross_attention = build_component(
        spec.components.decoder_cross_attention,
        backend_context=backend_context,
        registry=component_registry,
        spec=spec,
        units=spec.attention_units,
        num_heads=spec.attention_heads,
        name="v2_decoder_cross_attention",
    )
    decoder_cross_postprocess = GatedResidualNetwork(
        units=spec.attention_units,
        dropout_rate=spec.dropout_rate,
        activation=spec.activation,
        use_batch_norm=spec.use_batch_norm,
        name="v2_decoder_cross_postprocess",
    )
    decoder_hierarchical_attention = build_component(
        spec.components.decoder_hierarchical_attention,
        backend_context=backend_context,
        registry=component_registry,
        spec=spec,
        units=spec.attention_units,
        num_heads=spec.attention_heads,
        name="v2_decoder_hierarchical_attention",
    )
    decoder_memory_attention = build_component(
        spec.components.decoder_memory_attention,
        backend_context=backend_context,
        registry=component_registry,
        spec=spec,
        units=spec.attention_units,
        memory_size=spec.memory_size,
        num_heads=spec.attention_heads,
        name="v2_decoder_memory_attention",
    )
    decoder_fusion = build_component(
        spec.components.decoder_fusion,
        backend_context=backend_context,
        registry=component_registry,
        spec=spec,
        units=spec.attention_units,
        num_heads=spec.attention_heads,
        name="v2_decoder_fusion",
    )
    residual_projection = None
    decoder_residual_add = None
    decoder_residual_norm = None
    final_residual_add = None
    final_residual_norm = None
    if spec.use_residuals:
        residual_projection = build_component(
            spec.components.hidden_projection,
            backend_context=backend_context,
            registry=component_registry,
            spec=spec,
            units=spec.attention_units,
            activation=None,
            name="v2_residual_projection",
        )
        if backend_context.layers.Add is not None:
            decoder_residual_add = backend_context.layers.Add(
                name="v2_decoder_residual_add",
            )
            final_residual_add = backend_context.layers.Add(
                name="v2_final_residual_add",
            )
        if (
            backend_context.layers.LayerNormalization
            is not None
        ):
            decoder_residual_norm = (
                backend_context.layers.LayerNormalization(
                    epsilon=spec.layer_norm_epsilon,
                    name="v2_decoder_residual_norm",
                )
            )
            final_residual_norm = (
                backend_context.layers.LayerNormalization(
                    epsilon=spec.layer_norm_epsilon,
                    name="v2_final_residual_norm",
                )
            )

    final_pool = build_component(
        _resolve_final_pool_key(spec),
        backend_context=backend_context,
        registry=component_registry,
        spec=spec,
        axis=1,
        name="v2_final_pool",
    )

    multi_horizon_head = build_component(
        spec.components.multi_horizon_head,
        backend_context=backend_context,
        registry=component_registry,
        spec=spec,
        output_dim=spec.output_dim,
        forecast_horizon=spec.forecast_horizon,
        name="v2_multi_horizon_head",
    )

    quantile_distribution_head = None
    if spec.head_type == "quantile":
        quantile_distribution_head = build_component(
            spec.components.quantile_distribution_head,
            backend_context=backend_context,
            registry=component_registry,
            spec=spec,
            output_dim=spec.output_dim,
            forecast_horizon=spec.forecast_horizon,
            quantiles=spec.quantiles,
            name="v2_quantile_distribution_head",
        )

    output_head = multi_horizon_head

    dropout = None
    if (
        spec.dropout_rate > 0
        and backend_context.layers.Dropout is not None
    ):
        dropout = backend_context.layers.Dropout(
            spec.dropout_rate,
            name="v2_dropout",
        )

    return BaseAttentiveV2Assembly(
        backend_context=backend_context,
        static_projection=static_processor,
        dynamic_projection=dynamic_processor,
        future_projection=future_processor,
        dynamic_encoder=dynamic_encoder,
        future_encoder=future_encoder,
        sequence_pool=sequence_pool,
        fusion=fusion,
        hidden_projection=hidden_projection,
        output_head=output_head,
        dropout=dropout,
        static_processor=static_processor,
        dynamic_processor=dynamic_processor,
        future_processor=future_processor,
        encoder_positional_encoding=encoder_positional_encoding,
        future_positional_encoding=future_positional_encoding,
        dynamic_window=dynamic_window,
        decoder_input_projection=decoder_input_projection,
        decoder_cross_attention=decoder_cross_attention,
        decoder_cross_postprocess=decoder_cross_postprocess,
        decoder_hierarchical_attention=decoder_hierarchical_attention,
        decoder_memory_attention=decoder_memory_attention,
        decoder_fusion=decoder_fusion,
        residual_projection=residual_projection,
        decoder_residual_add=decoder_residual_add,
        decoder_residual_norm=decoder_residual_norm,
        final_residual_add=final_residual_add,
        final_residual_norm=final_residual_norm,
        final_pool=final_pool,
        multi_horizon_head=multi_horizon_head,
        quantile_distribution_head=quantile_distribution_head,
    )


def ensure_generic_v2_registered(
    *,
    component_registry: ComponentRegistry | None = None,
    model_registry: ModelRegistry | None = None,
):
    """Register the generic V2 builders once."""
    active_component_registry = (
        component_registry or DEFAULT_COMPONENT_REGISTRY
    )
    active_model_registry = (
        model_registry or DEFAULT_MODEL_REGISTRY
    )

    component_defs = [
        (
            "projection.dense",
            _build_dense_projection,
            "Generic dense projection layer (fallback for all backends).",
        ),
        (
            "projection.static",
            _build_dense_projection,
            "Static feature projection layer.",
        ),
        (
            "projection.dynamic",
            _build_dense_projection,
            "Dynamic sequence projection layer.",
        ),
        (
            "projection.future",
            _build_dense_projection,
            "Future covariate projection layer.",
        ),
        (
            "projection.hidden",
            _build_dense_projection,
            "Post-fusion hidden projection layer.",
        ),
        (
            "feature.static_processor",
            _build_feature_processor,
            "Migrated static feature processor.",
        ),
        (
            "feature.dynamic_processor",
            _build_feature_processor,
            "Migrated dynamic feature processor.",
        ),
        (
            "feature.future_processor",
            _build_feature_processor,
            "Migrated future feature processor.",
        ),
        (
            "embedding.positional",
            _build_positional_encoding,
            "Sinusoidal positional encoding layer.",
        ),
        (
            "encoder.hybrid_multiscale",
            _build_hybrid_multiscale_encoder,
            "Migrated hybrid multi-scale encoder.",
        ),
        (
            "encoder.temporal_self_attention",
            _build_temporal_self_attention_encoder,
            "Temporal self-attention encoder block.",
        ),
        (
            "encoder.dynamic_window",
            _build_dynamic_window,
            "Dynamic time-window slicing layer.",
        ),
        (
            "decoder.cross_attention",
            _build_cross_attention,
            "Migrated decoder cross-attention block.",
        ),
        (
            "decoder.hierarchical_attention",
            _build_hierarchical_attention,
            "Migrated decoder hierarchical attention block.",
        ),
        (
            "decoder.memory_attention",
            _build_memory_attention,
            "Migrated decoder memory attention block.",
        ),
        (
            "fusion.multi_resolution_attention",
            _build_multi_resolution_attention_fusion,
            "Migrated decoder attention fusion block.",
        ),
        (
            "pool.mean",
            _build_mean_pool,
            "Sequence mean pooling.",
        ),
        (
            "pool.last",
            _build_last_pool,
            "Last-step sequence pooling.",
        ),
        (
            "pool.final_last",
            _build_last_pool,
            "Final temporal last-step aggregation.",
        ),
        (
            "pool.final_mean",
            _build_mean_pool,
            "Final temporal average aggregation.",
        ),
        (
            "pool.final_flatten",
            _build_flatten_pool,
            "Final temporal flatten aggregation.",
        ),
        (
            "fusion.concat",
            _build_concat_fusion,
            "Feature concatenation fusion.",
        ),
        (
            "head.multi_horizon",
            _build_multi_horizon_head,
            "Migrated multi-horizon decoder head.",
        ),
        (
            "head.quantile_distribution",
            _build_quantile_distribution_head,
            "Migrated quantile distribution head.",
        ),
        (
            "head.point_forecast",
            _build_point_forecast_head,
            "Point forecast head.",
        ),
        (
            "head.quantile_forecast",
            _build_quantile_forecast_head,
            "Quantile forecast head.",
        ),
    ]

    for key, builder, description in component_defs:
        if not active_component_registry.has(
            key,
            backend="generic",
        ):
            active_component_registry.register(
                key,
                builder,
                backend="generic",
                description=description,
                experimental=True,
            )

    if not active_model_registry.has(
        "base_attentive.v2",
        backend="generic",
    ):
        active_model_registry.register(
            "base_attentive.v2",
            _assemble_base_attentive_v2,
            backend="generic",
            description=(
                "Experimental generic BaseAttentiveV2 assembler."
            ),
            experimental=True,
        )


__all__ = ["ensure_generic_v2_registered"]
