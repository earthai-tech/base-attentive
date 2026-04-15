# SPDX-License-Identifier: Apache-2.0
"""Legacy-compatible public ``BaseAttentive`` facade.

This module preserves the public constructor surface of the original
``BaseAttentive`` class while routing execution through the resolver-
driven V2 architecture.
"""

from __future__ import annotations

import warnings
from typing import Any

from .._bootstrap import KERAS_DEPS, dependency_message
from ..api.property import NNLearner
from ..compat.versioning import (
    BASE_ATTENTIVE_PARAMETER_RULES,
    UnsupportedCompatibilityWarning,
    n_quantiles_to_quantiles,
    resolve_deprecated_kwargs,
)
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
        static_dim: int | None = None,
        dynamic_dim: int | None = None,
        future_dim: int | None = None,
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
        lookback_window: int = 10,
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
        attention_stack: str
        | list[str]
        | tuple[str, ...]
        | None = None,
        objective: str = "hybrid",
        architecture_config: dict[str, Any] | None = None,
        backend_name: str | None = None,
        component_overrides: dict[str, Any] | None = None,
        verbose: int = 0,
        output_mode: str | None = None,
        n_quantiles: int | None = None,
        name: str = "BaseAttentive",
        *,
        static_input_dim: int | None = None,
        dynamic_input_dim: int | None = None,
        future_input_dim: int | None = None,
        max_window_size: int | None = None,
        attention_levels: str
        | list[str]
        | tuple[str, ...]
        | None = None,
        **kwargs,
    ):
        incoming = {
            "static_dim": static_dim,
            "dynamic_dim": dynamic_dim,
            "future_dim": future_dim,
            "output_dim": output_dim,
            "forecast_horizon": forecast_horizon,
            "mode": mode,
            "num_encoder_layers": num_encoder_layers,
            "quantiles": quantiles,
            "embed_dim": embed_dim,
            "hidden_units": hidden_units,
            "lstm_units": lstm_units,
            "attention_units": attention_units,
            "num_heads": num_heads,
            "dropout_rate": dropout_rate,
            "lookback_window": lookback_window,
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
            "attention_stack": attention_stack,
            "objective": objective,
            "architecture_config": architecture_config,
            "backend_name": backend_name,
            "component_overrides": component_overrides,
            "verbose": verbose,
            "output_mode": output_mode,
            "n_quantiles": n_quantiles,
            "name": name,
            "static_input_dim": static_input_dim,
            "dynamic_input_dim": dynamic_input_dim,
            "future_input_dim": future_input_dim,
            "max_window_size": max_window_size,
            "attention_levels": attention_levels,
        }
        resolved = resolve_deprecated_kwargs(
            incoming,
            BASE_ATTENTIVE_PARAMETER_RULES,
            component_name="BaseAttentive",
        )

        if (
            resolved.get("quantiles") is None
            and resolved.get("n_quantiles") is not None
        ):
            resolved["quantiles"] = n_quantiles_to_quantiles(
                resolved["n_quantiles"]
            )
            warnings.warn(
                "BaseAttentive: 'n_quantiles' is a compatibility helper "
                "that expands to evenly spaced 'quantiles'. Prefer "
                "passing explicit quantiles for long-term stability.",
                category=UserWarning,
                stacklevel=3,
            )

        output_mode = resolved.get("output_mode")
        if output_mode is not None:
            normalized_output_mode = (
                str(output_mode).strip().lower()
            )
            resolved["output_mode"] = normalized_output_mode
            if normalized_output_mode in {
                "gaussian",
                "mixture",
            }:
                warnings.warn(
                    "BaseAttentive: 'output_mode=%s' is accepted for a "
                    "smooth transition, but the current BaseAttentive "
                    "facade still builds the point/quantile kernel path. "
                    "The value is recorded for compatibility but does not "
                    "yet activate a dedicated '%s' output head."
                    % (
                        normalized_output_mode,
                        normalized_output_mode,
                    ),
                    UnsupportedCompatibilityWarning,
                    stacklevel=3,
                )
            elif normalized_output_mode not in {
                "point",
                "quantile",
            }:
                warnings.warn(
                    "BaseAttentive: 'output_mode=%s' is not implemented "
                    "in the current facade and will be ignored."
                    % normalized_output_mode,
                    UnsupportedCompatibilityWarning,
                    stacklevel=3,
                )
        else:
            normalized_output_mode = None

        static_dim = resolved.get("static_dim")
        dynamic_dim = resolved.get("dynamic_dim")
        future_dim = resolved.get("future_dim")
        if (
            static_dim is None
            or dynamic_dim is None
            or future_dim is None
        ):
            raise TypeError(
                "BaseAttentive requires 'static_dim', 'dynamic_dim', and "
                "'future_dim'. Legacy aliases 'static_input_dim', "
                "'dynamic_input_dim', and 'future_input_dim' remain "
                "supported during the transition."
            )

        quantiles = resolved.get("quantiles")
        lookback_window = resolved.get("lookback_window")
        attention_stack = resolved.get("attention_stack")
        architecture_config = dict(
            resolved.get("architecture_config") or {}
        )
        component_overrides = dict(
            resolved.get("component_overrides") or {}
        )

        spec = legacy_base_attentive_to_spec(
            static_input_dim=static_dim,
            dynamic_input_dim=dynamic_dim,
            future_input_dim=future_dim,
            output_dim=resolved["output_dim"],
            forecast_horizon=resolved["forecast_horizon"],
            mode=resolved["mode"],
            num_encoder_layers=resolved["num_encoder_layers"],
            quantiles=quantiles,
            embed_dim=resolved["embed_dim"],
            hidden_units=resolved["hidden_units"],
            lstm_units=resolved["lstm_units"],
            attention_units=resolved["attention_units"],
            num_heads=resolved["num_heads"],
            dropout_rate=resolved["dropout_rate"],
            max_window_size=lookback_window,
            memory_size=resolved["memory_size"],
            scales=resolved["scales"],
            multi_scale_agg=resolved["multi_scale_agg"],
            final_agg=resolved["final_agg"],
            activation=resolved["activation"],
            use_residuals=resolved["use_residuals"],
            use_vsn=resolved["use_vsn"],
            vsn_units=resolved["vsn_units"],
            use_batch_norm=resolved["use_batch_norm"],
            apply_dtw=resolved["apply_dtw"],
            attention_levels=attention_stack,
            objective=resolved["objective"],
            architecture_config=architecture_config,
            backend_name=resolved["backend_name"],
            component_overrides=component_overrides,
            verbose=resolved["verbose"],
            extras={
                key: value
                for key, value in {
                    "output_mode": normalized_output_mode,
                    "n_quantiles": resolved.get(
                        "n_quantiles"
                    ),
                }.items()
                if value is not None
            },
        )
        self._legacy_config = {
            "static_dim": static_dim,
            "dynamic_dim": dynamic_dim,
            "future_dim": future_dim,
            "output_dim": resolved["output_dim"],
            "forecast_horizon": resolved["forecast_horizon"],
            "mode": resolved["mode"],
            "num_encoder_layers": resolved[
                "num_encoder_layers"
            ],
            "quantiles": _copy_sequence(quantiles),
            "embed_dim": resolved["embed_dim"],
            "hidden_units": resolved["hidden_units"],
            "lstm_units": resolved["lstm_units"],
            "attention_units": resolved["attention_units"],
            "num_heads": resolved["num_heads"],
            "dropout_rate": resolved["dropout_rate"],
            "lookback_window": lookback_window,
            "memory_size": resolved["memory_size"],
            "scales": resolved["scales"],
            "multi_scale_agg": resolved["multi_scale_agg"],
            "final_agg": resolved["final_agg"],
            "activation": resolved["activation"],
            "use_residuals": resolved["use_residuals"],
            "use_vsn": resolved["use_vsn"],
            "vsn_units": resolved["vsn_units"],
            "use_batch_norm": resolved["use_batch_norm"],
            "apply_dtw": resolved["apply_dtw"],
            "attention_stack": _copy_sequence(
                attention_stack
            ),
            "objective": resolved["objective"],
            "architecture_config": architecture_config,
            "backend_name": resolved["backend_name"],
            "component_overrides": component_overrides,
            "verbose": resolved["verbose"],
            "output_mode": normalized_output_mode,
            "n_quantiles": resolved.get("n_quantiles"),
            "name": name,
        }
        super().__init__(
            static_input_dim=static_dim,
            dynamic_input_dim=dynamic_dim,
            future_input_dim=future_dim,
            output_dim=resolved["output_dim"],
            forecast_horizon=resolved["forecast_horizon"],
            quantiles=tuple(quantiles or ()),
            embed_dim=resolved["embed_dim"],
            hidden_units=resolved["hidden_units"],
            attention_heads=resolved["num_heads"],
            dropout_rate=resolved["dropout_rate"],
            activation=resolved["activation"],
            backend_name=spec.backend_name,
            head_type=spec.head_type,
            spec=spec,
            name=name,
            **kwargs,
        )

        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.future_dim = future_dim
        self.lookback_window = lookback_window
        self.attention_stack = _copy_sequence(attention_stack)
        self.output_mode = normalized_output_mode
        self.n_quantiles = resolved.get("n_quantiles")

        # Backward-compatible aliases.
        self.static_input_dim = static_dim
        self.dynamic_input_dim = dynamic_dim
        self.future_input_dim = future_dim
        self.max_window_size = lookback_window
        self.attention_levels = tuple(spec.attention_levels)

        self.output_dim = resolved["output_dim"]
        self.forecast_horizon = resolved["forecast_horizon"]
        self.mode = resolved["mode"]
        self.num_encoder_layers = resolved[
            "num_encoder_layers"
        ]
        self.quantiles = _copy_sequence(quantiles)
        self.embed_dim = resolved["embed_dim"]
        self.hidden_units = resolved["hidden_units"]
        self.lstm_units = resolved["lstm_units"]
        self.attention_units = resolved["attention_units"]
        self.num_heads = resolved["num_heads"]
        self.dropout_rate = resolved["dropout_rate"]
        self.memory_size = resolved["memory_size"]
        self.scales = resolved["scales"]
        self.multi_scale_agg = resolved["multi_scale_agg"]
        self.final_agg = resolved["final_agg"]
        self.activation = resolved["activation"]
        self.use_residuals = resolved["use_residuals"]
        self.use_vsn = resolved["use_vsn"]
        self.vsn_units = resolved["vsn_units"]
        self.use_batch_norm = resolved["use_batch_norm"]
        self.apply_dtw = resolved["apply_dtw"]
        self.objective = resolved["objective"]
        self.architecture_config = architecture_config
        self.backend_name = spec.backend_name
        self.verbose = resolved["verbose"]

    def get_config(self) -> dict[str, Any]:
        return dict(self._legacy_config)

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        return cls(**config)


__all__ = ["BaseAttentive"]
