# SPDX-License-Identifier: Apache-2.0
"""Legacy-compatible public ``BaseAttentive`` facade.

This module preserves the public constructor surface of the original
``BaseAttentive`` class while routing execution through the resolver-
driven V2 architecture.
"""

from __future__ import annotations

from typing import Any

from .._bootstrap import KERAS_DEPS, dependency_message
from ..api.property import NNLearner
from ..config import legacy_base_attentive_to_spec
from ..experimental.base_attentive_v2 import BaseAttentiveV2
from ..utils.deps_utils import ensure_pkg

register_keras_serializable = (
    KERAS_DEPS.register_keras_serializable
)
DEP_MSG = dependency_message("models")
SERIALIZATION_PACKAGE = __name__


def _copy_sequence(value):
    if value is None:
        return None
    return list(value)


@register_keras_serializable(
    SERIALIZATION_PACKAGE,
    name="BaseAttentive",
)
class BaseAttentive(BaseAttentiveV2, NNLearner):
    """Compatibility wrapper over the resolver-driven V2 model.

    The facade keeps the legacy constructor signature intact, converts
    the payload into :class:`BaseAttentiveSpec`, and delegates the
    actual model assembly and execution to ``BaseAttentiveV2``.
    """

    @ensure_pkg("keras", extra=DEP_MSG)
    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        output_dim: int = 1,
        forecast_horizon: int = 1,
        mode: str | None = None,
        num_encoder_layers: int = 2,
        quantiles: list[float]
        | tuple[float, ...]
        | None = None,
        embed_dim: int = 32,
        hidden_units: int = 64,
        lstm_units: int | tuple[int, ...] = 64,
        attention_units: int = 32,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        max_window_size: int = 10,
        memory_size: int = 100,
        scales: list[int]
        | tuple[int, ...]
        | str
        | None = None,
        multi_scale_agg: str = "last",
        final_agg: str = "last",
        activation: str = "relu",
        use_residuals: bool = True,
        use_vsn: bool = True,
        vsn_units: int | None = None,
        use_batch_norm: bool = False,
        apply_dtw: bool = True,
        attention_levels: str
        | list[str]
        | tuple[str, ...]
        | None = None,
        objective: str = "hybrid",
        architecture_config: dict[str, Any] | None = None,
        backend_name: str | None = None,
        component_overrides: dict[str, Any] | None = None,
        verbose: int = 0,
        name: str = "BaseAttentive",
        **kwargs,
    ):
        spec = legacy_base_attentive_to_spec(
            static_input_dim=static_input_dim,
            dynamic_input_dim=dynamic_input_dim,
            future_input_dim=future_input_dim,
            output_dim=output_dim,
            forecast_horizon=forecast_horizon,
            mode=mode,
            num_encoder_layers=num_encoder_layers,
            quantiles=quantiles,
            embed_dim=embed_dim,
            hidden_units=hidden_units,
            lstm_units=lstm_units,
            attention_units=attention_units,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            max_window_size=max_window_size,
            memory_size=memory_size,
            scales=scales,
            multi_scale_agg=multi_scale_agg,
            final_agg=final_agg,
            activation=activation,
            use_residuals=use_residuals,
            use_vsn=use_vsn,
            vsn_units=vsn_units,
            use_batch_norm=use_batch_norm,
            apply_dtw=apply_dtw,
            attention_levels=attention_levels,
            objective=objective,
            architecture_config=architecture_config,
            backend_name=backend_name,
            component_overrides=component_overrides,
            verbose=verbose,
        )
        self._legacy_config = {
            "static_input_dim": static_input_dim,
            "dynamic_input_dim": dynamic_input_dim,
            "future_input_dim": future_input_dim,
            "output_dim": output_dim,
            "forecast_horizon": forecast_horizon,
            "mode": mode,
            "num_encoder_layers": num_encoder_layers,
            "quantiles": _copy_sequence(quantiles),
            "embed_dim": embed_dim,
            "hidden_units": hidden_units,
            "lstm_units": lstm_units,
            "attention_units": attention_units,
            "num_heads": num_heads,
            "dropout_rate": dropout_rate,
            "max_window_size": max_window_size,
            "memory_size": memory_size,
            "scales": scales,
            "multi_scale_agg": multi_scale_agg,
            "final_agg": final_agg,
            "activation": activation,
            "use_residuals": use_residuals,
            "use_vsn": use_vsn,
            "vsn_units": vsn_units,
            "use_batch_norm": use_batch_norm,
            "apply_dtw": apply_dtw,
            "attention_levels": _copy_sequence(
                attention_levels
            ),
            "objective": objective,
            "architecture_config": dict(
                architecture_config or {}
            ),
            "backend_name": backend_name,
            "component_overrides": dict(
                component_overrides or {}
            ),
            "verbose": verbose,
            "name": name,
        }
        super().__init__(
            static_input_dim=static_input_dim,
            dynamic_input_dim=dynamic_input_dim,
            future_input_dim=future_input_dim,
            output_dim=output_dim,
            forecast_horizon=forecast_horizon,
            quantiles=tuple(quantiles or ()),
            embed_dim=embed_dim,
            hidden_units=hidden_units,
            attention_heads=num_heads,
            dropout_rate=dropout_rate,
            activation=activation,
            backend_name=spec.backend_name,
            head_type=spec.head_type,
            spec=spec,
            name=name,
            **kwargs,
        )

        # Legacy-compatible public attributes.
        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.future_input_dim = future_input_dim
        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon
        self.mode = mode
        self.num_encoder_layers = num_encoder_layers
        self.quantiles = _copy_sequence(quantiles)
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.lstm_units = lstm_units
        self.attention_units = attention_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.max_window_size = max_window_size
        self.memory_size = memory_size
        self.scales = scales
        self.multi_scale_agg = multi_scale_agg
        self.final_agg = final_agg
        self.activation = activation
        self.use_residuals = use_residuals
        self.use_vsn = use_vsn
        self.vsn_units = vsn_units
        self.use_batch_norm = use_batch_norm
        self.apply_dtw = apply_dtw
        self.attention_levels = tuple(spec.attention_levels)
        self.objective = objective
        self.architecture_config = dict(
            architecture_config or {}
        )
        self.backend_name = spec.backend_name
        self.verbose = verbose

    def get_config(self) -> dict[str, Any]:
        return dict(self._legacy_config)

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        return cls(**config)


__all__ = ["BaseAttentive"]
