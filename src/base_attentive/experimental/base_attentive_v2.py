"""Experimental V2 model assembled through the new resolver path."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .. import KERAS_DEPS, dependency_message
from ..config import BaseAttentiveSpec, normalize_base_attentive_spec
from ..resolver import BackendContext, assemble_model
from ..utils.deps_utils import ensure_pkg

Model = KERAS_DEPS.Model
reshape = KERAS_DEPS.reshape
register_keras_serializable = KERAS_DEPS.register_keras_serializable

DEP_MSG = dependency_message("experimental models")


@register_keras_serializable(__name__, name="BaseAttentiveV2")
class BaseAttentiveV2(Model):
    """Experimental V2 model scaffold using config, registry, and resolver."""

    @ensure_pkg("keras", extra=DEP_MSG)
    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        output_dim: int = 1,
        forecast_horizon: int = 1,
        *,
        quantiles: tuple[float, ...] | list[float] | None = None,
        embed_dim: int = 32,
        hidden_units: int = 64,
        attention_heads: int = 4,
        layer_norm_epsilon: float = 1e-6,
        dropout_rate: float = 0.0,
        activation: str = "relu",
        backend_name: str | None = None,
        head_type: str = "point",
        spec: BaseAttentiveSpec | dict[str, Any] | None = None,
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
        self.backend_context = BackendContext.current(self.spec.backend_name)
        self._assembly = assemble_model(
            "base_attentive.v2",
            spec=self.spec,
            backend_context=self.backend_context,
        )

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

    def call(self, inputs, training: bool = False):
        static_x, dynamic_x, future_x = self._normalize_inputs(inputs)
        if dynamic_x is None:
            raise ValueError("dynamic input is required for BaseAttentiveV2.")

        features = []
        if self._assembly.static_projection is not None and static_x is not None:
            features.append(self._assembly.static_projection(static_x))

        dynamic_encoded = self._assembly.dynamic_projection(dynamic_x)
        if self._assembly.dynamic_encoder is not None:
            dynamic_encoded = self._assembly.dynamic_encoder(
                dynamic_encoded,
                training=training,
            )
        features.append(self._assembly.sequence_pool(dynamic_encoded))

        if self._assembly.future_projection is not None and future_x is not None:
            future_encoded = self._assembly.future_projection(future_x)
            if self._assembly.future_encoder is not None:
                future_encoded = self._assembly.future_encoder(
                    future_encoded,
                    training=training,
                )
            features.append(self._assembly.sequence_pool(future_encoded))

        fused = self._assembly.fusion(features)
        hidden = self._assembly.hidden_projection(fused)

        if self._assembly.dropout is not None:
            hidden = self._assembly.dropout(hidden, training=training)

        outputs = self._assembly.output_head(hidden)
        if self.spec.head_type == "quantile":
            return reshape(
                outputs,
                (
                    -1,
                    self.spec.forecast_horizon,
                    len(self.spec.quantiles),
                    self.spec.output_dim,
                ),
            )

        return reshape(
            outputs,
            (-1, self.spec.forecast_horizon, self.spec.output_dim),
        )

    def get_config(self):
        config = super().get_config()
        config.update(asdict(self.spec))
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


__all__ = ["BaseAttentiveV2"]
