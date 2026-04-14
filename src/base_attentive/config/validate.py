"""Validation helpers for BaseAttentive specs."""

from __future__ import annotations

from .schema import (
    BaseAttentiveArchitectureSpec,
    BaseAttentiveRuntimeSpec,
    BaseAttentiveSpec,
)

_ALLOWED_ENCODERS = {"hybrid", "transformer"}
_ALLOWED_FEATURE_PROCESSING = {"vsn", "dense"}
_ALLOWED_ATTENTION_LEVELS = {
    "cross",
    "hierarchical",
    "memory",
}
_ALLOWED_HEAD_TYPES = {"point", "quantile"}
_ALLOWED_MULTI_SCALE_AGG = {
    "last",
    "average",
    "flatten",
    "auto",
    "sum",
    "concat",
}
_ALLOWED_FINAL_AGG = {"last", "average", "flatten"}
_ALLOWED_MODES = {
    None,
    "tft",
    "pihal",
    "tft_like",
    "pihal_like",
    "tft-like",
    "pihal-like",
}


def _ensure_positive_int(
    name: str,
    value: int,
    *,
    allow_zero: bool = False,
) -> None:
    if not isinstance(value, int):
        raise TypeError(
            f"{name} must be an integer, got {type(value).__name__}."
        )
    if allow_zero:
        if value < 0:
            raise ValueError(
                f"{name} must be >= 0, got {value}."
            )
        return
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def _validate_architecture(
    spec: BaseAttentiveArchitectureSpec,
) -> None:
    if spec.encoder_type not in _ALLOWED_ENCODERS:
        raise ValueError(
            "architecture.encoder_type must be one of "
            f"{sorted(_ALLOWED_ENCODERS)!r}."
        )
    if (
        spec.feature_processing
        not in _ALLOWED_FEATURE_PROCESSING
    ):
        raise ValueError(
            "architecture.feature_processing must be 'vsn' or "
            f"'dense', got {spec.feature_processing!r}."
        )

    invalid_levels = [
        item
        for item in spec.decoder_attention_stack
        if item not in _ALLOWED_ATTENTION_LEVELS
    ]
    if invalid_levels:
        raise ValueError(
            "architecture.decoder_attention_stack contains "
            f"unsupported entries: {invalid_levels!r}."
        )


def _validate_runtime(spec: BaseAttentiveRuntimeSpec) -> None:
    _ensure_positive_int(
        "runtime.num_encoder_layers",
        spec.num_encoder_layers,
    )
    _ensure_positive_int(
        "runtime.max_window_size",
        spec.max_window_size,
    )
    _ensure_positive_int(
        "runtime.memory_size",
        spec.memory_size,
    )
    _ensure_positive_int(
        "runtime.verbose",
        spec.verbose,
        allow_zero=True,
    )

    if spec.multi_scale_agg not in _ALLOWED_MULTI_SCALE_AGG:
        raise ValueError(
            "runtime.multi_scale_agg must be one of "
            f"{sorted(_ALLOWED_MULTI_SCALE_AGG)!r}."
        )
    if spec.final_agg not in _ALLOWED_FINAL_AGG:
        raise ValueError(
            "runtime.final_agg must be one of "
            f"{sorted(_ALLOWED_FINAL_AGG)!r}."
        )
    if spec.mode not in _ALLOWED_MODES:
        raise ValueError(
            f"runtime.mode is unsupported: {spec.mode!r}."
        )
    if spec.scales != "auto":
        for index, value in enumerate(spec.scales):
            _ensure_positive_int(
                f"runtime.scales[{index}]",
                value,
            )


def validate_base_attentive_spec(
    spec: BaseAttentiveSpec,
) -> BaseAttentiveSpec:
    """Validate a normalized ``BaseAttentiveSpec``."""
    if not isinstance(spec, BaseAttentiveSpec):
        raise TypeError(
            "spec must be a BaseAttentiveSpec instance."
        )

    _ensure_positive_int(
        "static_input_dim",
        spec.static_input_dim,
        allow_zero=True,
    )
    _ensure_positive_int(
        "dynamic_input_dim", spec.dynamic_input_dim
    )
    _ensure_positive_int(
        "future_input_dim",
        spec.future_input_dim,
        allow_zero=True,
    )
    _ensure_positive_int("output_dim", spec.output_dim)
    _ensure_positive_int(
        "forecast_horizon",
        spec.forecast_horizon,
    )
    _ensure_positive_int("embed_dim", spec.embed_dim)
    _ensure_positive_int("hidden_units", spec.hidden_units)
    _ensure_positive_int(
        "attention_heads",
        spec.attention_heads,
    )
    _ensure_positive_int(
        "attention_units",
        spec.attention_units,
    )

    if isinstance(spec.lstm_units, tuple):
        for index, value in enumerate(spec.lstm_units):
            _ensure_positive_int(
                f"lstm_units[{index}]",
                value,
            )
    else:
        _ensure_positive_int("lstm_units", spec.lstm_units)

    if spec.vsn_units is not None:
        _ensure_positive_int("vsn_units", spec.vsn_units)

    if not 0.0 <= spec.dropout_rate <= 1.0:
        raise ValueError(
            "dropout_rate must be between 0 and 1 inclusive."
        )
    if spec.layer_norm_epsilon <= 0:
        raise ValueError("layer_norm_epsilon must be > 0.")
    if spec.head_type not in _ALLOWED_HEAD_TYPES:
        raise ValueError(
            "head_type must be either 'point' or 'quantile', "
            f"got {spec.head_type!r}."
        )
    if spec.head_type == "quantile" and not spec.quantiles:
        raise ValueError(
            "quantile head_type requires at least one quantile value."
        )

    _validate_architecture(spec.architecture)
    _validate_runtime(spec.runtime)
    return spec


__all__ = ["validate_base_attentive_spec"]
