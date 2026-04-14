"""Experimental V2 model assembled through the resolver path."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from .. import KERAS_DEPS, dependency_message
from ..config import (
    BaseAttentiveSpec,
    normalize_base_attentive_spec,
    serialize_base_attentive_spec,
)
from ..resolver import BackendContext, assemble_model
from ..utils.deps_utils import ensure_pkg

Model = KERAS_DEPS.Model
reshape = KERAS_DEPS.reshape
register_keras_serializable = (
    KERAS_DEPS.register_keras_serializable
)

DEP_MSG = dependency_message("experimental models")


def _extract_spec_payload(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Extract nested or flattened spec fields from a config mapping."""
    payload = dict(config.pop("spec", None) or {})

    nested_keys = (
        "architecture",
        "runtime",
        "components",
        "extras",
    )
    for key in nested_keys:
        if key in config and key not in payload:
            payload[key] = config.pop(key)

    spec_field_names = tuple(
        BaseAttentiveSpec.__dataclass_fields__.keys()
    )
    for key in spec_field_names:
        if key in config and key not in payload:
            payload[key] = config.pop(key)

    return payload


def _invoke(component, inputs, *, training: bool = False):
    """Call a component, passing ``training`` when accepted."""
    if component is None:
        return inputs
    try:
        return component(inputs, training=training)
    except TypeError:
        return component(inputs)


@register_keras_serializable(__name__, name="BaseAttentiveV2")
class BaseAttentiveV2(Model):
    """Experimental V2 model scaffold using config and resolver."""

    @ensure_pkg("keras", extra=DEP_MSG)
    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        output_dim: int = 1,
        forecast_horizon: int = 1,
        *,
        quantiles: tuple[float, ...]
        | list[float]
        | None = None,
        embed_dim: int = 32,
        hidden_units: int = 64,
        attention_heads: int = 4,
        layer_norm_epsilon: float = 1e-6,
        dropout_rate: float = 0.0,
        activation: str = "relu",
        backend_name: str | None = None,
        head_type: str = "point",
        spec: BaseAttentiveSpec
        | dict[str, Any]
        | None = None,
        name: str = "BaseAttentiveV2",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.spec = normalize_base_attentive_spec(
            spec,
            static_input_dim=static_input_dim,
            dynamic_input_dim=dynamic_input_dim,
            future_input_dim=future_input_dim,
            output_dim=output_dim,
            forecast_horizon=forecast_horizon,
            quantiles=quantiles or (),
            embed_dim=embed_dim,
            hidden_units=hidden_units,
            attention_heads=attention_heads,
            layer_norm_epsilon=layer_norm_epsilon,
            dropout_rate=dropout_rate,
            activation=activation,
            backend_name=backend_name,
            head_type=head_type,
        )
        self.backend_context = BackendContext.current(
            self.spec.backend_name
        )
        self._assembly = assemble_model(
            "base_attentive.v2",
            spec=self.spec,
            backend_context=self.backend_context,
        )
        self._track_assembly_components()

    def _normalize_inputs(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise TypeError(
                "BaseAttentiveV2 expects inputs as a list or tuple of tensors."
            )

        if len(inputs) == 3:
            return inputs[0], inputs[1], inputs[2]
        if len(inputs) == 2:
            return None, inputs[0], inputs[1]
        if len(inputs) == 1:
            return None, inputs[0], None

        raise ValueError(
            "BaseAttentiveV2 expects one, two, or three input tensors."
        )

    def _resolve_processor(
        self, modern_name: str, legacy_name: str
    ):
        component = getattr(self._assembly, modern_name, None)
        if component is not None:
            return component
        return getattr(self._assembly, legacy_name, None)

    def _track_assembly_components(self) -> None:
        """Attach resolved components as model attributes for tracking.

        Keras tracks layers and models assigned as attributes on the
        parent model. Keeping the full assembly dataclass is useful for
        readability, but explicit attributes make serialization and layer
        discovery more reliable across the multi-backend runtime.
        """
        tracked_names: list[str] = []
        for field in fields(self._assembly):
            name = field.name
            component = getattr(self._assembly, name)
            setattr(self, name, component)
            if component is not None:
                tracked_names.append(name)
        self._tracked_component_names = tuple(tracked_names)

    def _normalized_mode(self) -> str:
        return (self.spec.mode or "pihal_like").replace(
            "-", "_"
        )

    def _tensor_width(self, value):
        """Return the static last-dimension width when available."""
        shape = getattr(value, "shape", None)
        if shape is None:
            return None
        try:
            width = shape[-1]
        except Exception:
            return None
        if width is None:
            return None
        try:
            return int(width)
        except Exception:
            return width

    def _align_residual_base(
        self,
        value,
        base,
        residual_projection=None,
        *,
        training: bool = False,
    ):
        """Project the residual branch when its width differs."""
        value_width = self._tensor_width(value)
        base_width = self._tensor_width(base)
        if (
            value_width is None
            or base_width is None
            or value_width == base_width
        ):
            return base

        if residual_projection is not None:
            projected = _invoke(
                residual_projection,
                base,
                training=training,
            )
            projected_width = self._tensor_width(projected)
            if (
                projected_width is None
                or projected_width == value_width
            ):
                return projected

        raise ValueError(
            "Residual width mismatch: "
            f"main path width={value_width}, residual width={base_width}."
        )

    def _apply_residual(
        self,
        value,
        base,
        add_layer,
        norm_layer,
        residual_projection=None,
        *,
        training: bool = False,
    ):
        base = self._align_residual_base(
            value,
            base,
            residual_projection,
            training=training,
        )
        if add_layer is not None:
            value = add_layer([value, base])
        else:
            value = value + base
        if norm_layer is not None:
            value = norm_layer(value)
        return value

    def _apply_decoder_stack(
        self,
        decoder_input,
        encoder_sequences,
        *,
        training: bool = False,
    ):
        stack = set(self.spec.attention_levels)
        context_att = decoder_input

        if "cross" in stack:
            cross_out = _invoke(
                self._assembly.decoder_cross_attention,
                [decoder_input, encoder_sequences],
                training=training,
            )
            att_proc = _invoke(
                self._assembly.decoder_cross_postprocess,
                cross_out,
                training=training,
            )
            if self.spec.use_residuals:
                context_att = self._apply_residual(
                    att_proc,
                    decoder_input,
                    self._assembly.decoder_residual_add,
                    self._assembly.decoder_residual_norm,
                    self._assembly.residual_projection,
                    training=training,
                )
            else:
                context_att = att_proc

        if "hierarchical" in stack:
            hierarchical_out = _invoke(
                self._assembly.decoder_hierarchical_attention,
                [context_att, context_att],
                training=training,
            )
        else:
            hierarchical_out = context_att

        if "memory" in stack:
            memory_out = _invoke(
                self._assembly.decoder_memory_attention,
                hierarchical_out,
                training=training,
            )
        else:
            memory_out = hierarchical_out

        final_sequence = _invoke(
            self._assembly.decoder_fusion,
            memory_out,
            training=training,
        )

        if self.spec.use_residuals:
            final_sequence = self._apply_residual(
                final_sequence,
                context_att,
                self._assembly.final_residual_add,
                self._assembly.final_residual_norm,
                self._assembly.residual_projection,
                training=training,
            )

        return final_sequence

    def call(self, inputs, training: bool = False):
        static_x, dynamic_x, future_x = (
            self._normalize_inputs(inputs)
        )
        if dynamic_x is None:
            raise ValueError(
                "dynamic input is required for BaseAttentiveV2."
            )

        static_processor = self._resolve_processor(
            "static_processor",
            "static_projection",
        )
        dynamic_processor = self._resolve_processor(
            "dynamic_processor",
            "dynamic_projection",
        )
        future_processor = self._resolve_processor(
            "future_processor",
            "future_projection",
        )

        static_context = None
        if (
            static_processor is not None
            and static_x is not None
        ):
            static_context = _invoke(
                static_processor,
                static_x,
                training=training,
            )

        dynamic_processed = _invoke(
            dynamic_processor,
            dynamic_x,
            training=training,
        )
        future_processed = None
        if (
            future_processor is not None
            and future_x is not None
        ):
            future_processed = _invoke(
                future_processor,
                future_x,
                training=training,
            )

        time_steps = self.backend_context.shape(dynamic_x)[1]
        mode = self._normalized_mode()

        encoder_parts = [dynamic_processed]
        if (
            mode == "tft_like"
            and future_processed is not None
        ):
            encoder_parts.append(
                future_processed[:, :time_steps, :]
            )
        encoder_input = self.backend_context.concat(
            encoder_parts,
            axis=-1,
        )
        if (
            self._assembly.encoder_positional_encoding
            is not None
        ):
            encoder_input = _invoke(
                self._assembly.encoder_positional_encoding,
                encoder_input,
                training=training,
            )

        encoder_sequences = encoder_input
        if self._assembly.dynamic_encoder is not None:
            encoder_sequences = _invoke(
                self._assembly.dynamic_encoder,
                encoder_sequences,
                training=training,
            )
        if self._assembly.dynamic_window is not None:
            encoder_sequences = _invoke(
                self._assembly.dynamic_window,
                encoder_sequences,
                training=training,
            )

        decoder_future = None
        if future_processed is not None:
            if mode == "tft_like":
                decoder_future = future_processed[
                    :, time_steps:, :
                ]
            else:
                decoder_future = future_processed

        decoder_parts = []
        if static_context is not None:
            static_expanded = (
                self.backend_context.expand_dims(
                    static_context,
                    axis=1,
                )
            )
            static_expanded = self.backend_context.tile(
                static_expanded,
                [1, self.spec.forecast_horizon, 1],
            )
            decoder_parts.append(static_expanded)

        if decoder_future is not None:
            if (
                self._assembly.future_positional_encoding
                is not None
            ):
                decoder_future = _invoke(
                    self._assembly.future_positional_encoding,
                    decoder_future,
                    training=training,
                )
            decoder_parts.append(decoder_future)

        if not decoder_parts:
            batch_size = self.backend_context.shape(
                dynamic_x
            )[0]
            raw_decoder_input = self.backend_context.zeros(
                (
                    batch_size,
                    self.spec.forecast_horizon,
                    self.spec.attention_units,
                )
            )
        else:
            raw_decoder_input = self.backend_context.concat(
                decoder_parts,
                axis=-1,
            )

        projected_decoder_input = raw_decoder_input
        if (
            self._assembly.decoder_input_projection
            is not None
        ):
            projected_decoder_input = _invoke(
                self._assembly.decoder_input_projection,
                projected_decoder_input,
                training=training,
            )

        final_sequence = self._apply_decoder_stack(
            projected_decoder_input,
            encoder_sequences,
            training=training,
        )
        final_features = _invoke(
            self._assembly.final_pool,
            final_sequence,
            training=training,
        )

        hidden = self._assembly.hidden_projection(
            final_features
        )
        if self._assembly.dropout is not None:
            hidden = self._assembly.dropout(
                hidden, training=training
            )

        migrated_multi_horizon = getattr(
            self._assembly,
            "multi_horizon_head",
            None,
        )
        migrated_quantile_head = getattr(
            self._assembly,
            "quantile_distribution_head",
            None,
        )

        if migrated_multi_horizon is not None:
            point_outputs = _invoke(
                migrated_multi_horizon,
                hidden,
                training=training,
            )
            if self.spec.head_type == "quantile":
                if migrated_quantile_head is None:
                    raise RuntimeError(
                        "Quantile head type requires a quantile distribution "
                        "component in the resolved assembly."
                    )
                return _invoke(
                    migrated_quantile_head,
                    point_outputs,
                    training=training,
                )
            return point_outputs

        outputs = self._assembly.output_head(hidden)
        if self.spec.head_type == "quantile":
            return reshape(
                outputs,
                (
                    -1,
                    self.spec.forecast_horizon,
                    self.spec.output_dim,
                    len(self.spec.quantiles),
                ),
            )

        return reshape(
            outputs,
            (
                -1,
                self.spec.forecast_horizon,
                self.spec.output_dim,
            ),
        )

    def get_config(self):
        base_get_config = getattr(super(), "get_config", None)
        config = (
            base_get_config()
            if callable(base_get_config)
            else {}
        )
        config.update(
            {
                "spec": serialize_base_attentive_spec(
                    self.spec
                ),
                "static_input_dim": self.spec.static_input_dim,
                "dynamic_input_dim": self.spec.dynamic_input_dim,
                "future_input_dim": self.spec.future_input_dim,
                "output_dim": self.spec.output_dim,
                "forecast_horizon": self.spec.forecast_horizon,
                "quantiles": list(self.spec.quantiles),
                "embed_dim": self.spec.embed_dim,
                "hidden_units": self.spec.hidden_units,
                "attention_heads": self.spec.attention_heads,
                "layer_norm_epsilon": self.spec.layer_norm_epsilon,
                "dropout_rate": self.spec.dropout_rate,
                "activation": self.spec.activation,
                "backend_name": self.spec.backend_name,
                "head_type": self.spec.head_type,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        payload = dict(config)
        spec_payload = _extract_spec_payload(payload)

        init_kwargs: dict[str, Any] = {}
        passthrough_kwargs = ("name", "trainable", "dtype")
        for key in passthrough_kwargs:
            if key in payload:
                init_kwargs[key] = payload[key]

        if spec_payload:
            normalized = normalize_base_attentive_spec(
                spec_payload
            )
            init_kwargs.update(
                {
                    "static_input_dim": normalized.static_input_dim,
                    "dynamic_input_dim": normalized.dynamic_input_dim,
                    "future_input_dim": normalized.future_input_dim,
                    "output_dim": normalized.output_dim,
                    "forecast_horizon": normalized.forecast_horizon,
                    "quantiles": tuple(normalized.quantiles),
                    "embed_dim": normalized.embed_dim,
                    "hidden_units": normalized.hidden_units,
                    "attention_heads": normalized.attention_heads,
                    "layer_norm_epsilon": normalized.layer_norm_epsilon,
                    "dropout_rate": normalized.dropout_rate,
                    "activation": normalized.activation,
                    "backend_name": normalized.backend_name,
                    "head_type": normalized.head_type,
                    "spec": serialize_base_attentive_spec(
                        normalized
                    ),
                }
            )
            return cls(**init_kwargs)

        return cls(**payload)


__all__ = ["BaseAttentiveV2"]
