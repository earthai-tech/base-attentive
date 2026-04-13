"""Generic V2 component builders and model assembler."""

from __future__ import annotations

from ...registry import (
    DEFAULT_COMPONENT_REGISTRY,
    DEFAULT_MODEL_REGISTRY,
    ComponentRegistry,
    ModelRegistry,
)
from ...resolver.assembly import BaseAttentiveV2Assembly
from ...resolver.component_resolver import build_component


def _build_dense_projection(
    *,
    context,
    units: int,
    activation: str | None = None,
    name: str | None = None,
):
    return context.layers.Dense(
        units,
        activation=activation,
        name=name,
    )


class _TemporalSelfAttentionEncoder:
    """Small backend-neutral temporal encoder built from Keras primitives."""

    def __init__(
        self,
        *,
        context,
        units: int,
        hidden_units: int,
        num_heads: int,
        activation: str,
        dropout_rate: float,
        layer_norm_epsilon: float,
        name: str | None = None,
    ):
        layers = context.layers
        if layers.MultiHeadAttention is None:
            raise ImportError(
                "MultiHeadAttention is required for the V2 temporal encoder."
            )
        if layers.LayerNormalization is None:
            raise ImportError(
                "LayerNormalization is required for the V2 temporal encoder."
            )

        key_dim = max(1, units // max(1, num_heads))
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout_rate,
            name=f"{name}_self_attention" if name else None,
        )
        self.norm1 = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name=f"{name}_norm_1" if name else None,
        )
        self.ffn_hidden = layers.Dense(
            hidden_units,
            activation=activation,
            name=f"{name}_ffn_hidden" if name else None,
        )
        self.ffn_output = layers.Dense(
            units,
            name=f"{name}_ffn_output" if name else None,
        )
        self.dropout = None
        if dropout_rate > 0 and layers.Dropout is not None:
            self.dropout = layers.Dropout(
                dropout_rate,
                name=f"{name}_dropout" if name else None,
            )
        self.norm2 = layers.LayerNormalization(
            epsilon=layer_norm_epsilon,
            name=f"{name}_norm_2" if name else None,
        )

    def __call__(self, inputs, training: bool = False):
        attention_output = self.attention(
            inputs, inputs, training=training
        )
        encoded = self.norm1(inputs + attention_output)
        ffn_output = self.ffn_hidden(encoded)
        ffn_output = self.ffn_output(ffn_output)
        if self.dropout is not None:
            ffn_output = self.dropout(ffn_output, training=training)
        return self.norm2(encoded + ffn_output)


def _build_temporal_self_attention_encoder(
    *,
    context,
    units: int,
    hidden_units: int,
    num_heads: int,
    activation: str = "relu",
    dropout_rate: float = 0.0,
    layer_norm_epsilon: float = 1e-6,
    name: str | None = None,
):
    return _TemporalSelfAttentionEncoder(
        context=context,
        units=units,
        hidden_units=hidden_units,
        num_heads=num_heads,
        activation=activation,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        name=name,
    )


def _build_mean_pool(
    *, context, axis: int = 1, name: str | None = None
):
    del name

    def pool(inputs):
        return context.ops.mean(inputs, axis=axis)

    return pool


def _build_last_pool(
    *, context, axis: int = 1, name: str | None = None
):
    del context, name

    if axis != 1:
        raise ValueError(
            "generic last-pool currently supports axis=1 only."
        )

    def pool(inputs):
        return inputs[:, -1, :]

    return pool


def _build_concat_fusion(
    *, context, axis: int = -1, name: str | None = None
):
    del name

    def fuse(features):
        active = [
            feature for feature in features if feature is not None
        ]
        if not active:
            raise ValueError(
                "fusion received no active feature tensors."
            )
        if len(active) == 1:
            return active[0]
        return context.ops.concatenate(active, axis=axis)

    return fuse


def _build_point_forecast_head(
    *, context, units: int, name: str | None = None
):
    return context.layers.Dense(units, name=name)


def _build_quantile_forecast_head(
    *,
    context,
    units: int,
    name: str | None = None,
):
    return context.layers.Dense(units, name=name)


def _assemble_base_attentive_v2(
    *,
    spec,
    backend_context,
    component_registry,
):
    static_projection = None
    if spec.static_input_dim > 0:
        static_projection = build_component(
            spec.components.static_projection,
            backend_context=backend_context,
            registry=component_registry,
            units=spec.embed_dim,
            activation=spec.activation,
            name="v2_static_projection",
        )

    dynamic_projection = build_component(
        spec.components.dynamic_projection,
        backend_context=backend_context,
        registry=component_registry,
        units=spec.embed_dim,
        activation=spec.activation,
        name="v2_dynamic_projection",
    )

    future_projection = None
    if spec.future_input_dim > 0:
        future_projection = build_component(
            spec.components.future_projection,
            backend_context=backend_context,
            registry=component_registry,
            units=spec.embed_dim,
            activation=spec.activation,
            name="v2_future_projection",
        )

    dynamic_encoder = build_component(
        spec.components.dynamic_encoder,
        backend_context=backend_context,
        registry=component_registry,
        units=spec.embed_dim,
        hidden_units=spec.hidden_units,
        num_heads=spec.attention_heads,
        activation=spec.activation,
        dropout_rate=spec.dropout_rate,
        layer_norm_epsilon=spec.layer_norm_epsilon,
        name="v2_dynamic_encoder",
    )

    future_encoder = None
    if future_projection is not None:
        future_encoder = build_component(
            spec.components.future_encoder,
            backend_context=backend_context,
            registry=component_registry,
            units=spec.embed_dim,
            hidden_units=spec.hidden_units,
            num_heads=spec.attention_heads,
            activation=spec.activation,
            dropout_rate=spec.dropout_rate,
            layer_norm_epsilon=spec.layer_norm_epsilon,
            name="v2_future_encoder",
        )

    sequence_pool = build_component(
        spec.components.sequence_pooling,
        backend_context=backend_context,
        registry=component_registry,
        axis=1,
        name="v2_sequence_pool",
    )
    fusion = build_component(
        spec.components.fusion,
        backend_context=backend_context,
        registry=component_registry,
        axis=-1,
        name="v2_feature_fusion",
    )
    hidden_projection = build_component(
        spec.components.hidden_projection,
        backend_context=backend_context,
        registry=component_registry,
        units=spec.hidden_units,
        activation=spec.activation,
        name="v2_hidden_projection",
    )
    output_head_key = spec.components.point_head
    output_units = spec.forecast_horizon * spec.output_dim
    if spec.head_type == "quantile":
        output_head_key = spec.components.quantile_head
        output_units *= len(spec.quantiles)

    output_head = build_component(
        output_head_key,
        backend_context=backend_context,
        registry=component_registry,
        units=output_units,
        name="v2_point_head",
    )

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
        static_projection=static_projection,
        dynamic_projection=dynamic_projection,
        future_projection=future_projection,
        dynamic_encoder=dynamic_encoder,
        future_encoder=future_encoder,
        sequence_pool=sequence_pool,
        fusion=fusion,
        hidden_projection=hidden_projection,
        output_head=output_head,
        dropout=dropout,
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
    active_model_registry = model_registry or DEFAULT_MODEL_REGISTRY

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
            "encoder.temporal_self_attention",
            _build_temporal_self_attention_encoder,
            "Temporal self-attention encoder block.",
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
            "fusion.concat",
            _build_concat_fusion,
            "Feature concatenation fusion.",
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
        if not active_component_registry.has(key, backend="generic"):
            active_component_registry.register(
                key,
                builder,
                backend="generic",
                description=description,
                experimental=True,
            )

    if not active_model_registry.has(
        "base_attentive.v2", backend="generic"
    ):
        active_model_registry.register(
            "base_attentive.v2",
            _assemble_base_attentive_v2,
            backend="generic",
            description="Experimental generic BaseAttentiveV2 assembler.",
            experimental=True,
        )


__all__ = ["ensure_generic_v2_registered"]
