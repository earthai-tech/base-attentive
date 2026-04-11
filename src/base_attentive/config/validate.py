"""Validation helpers for V2 specs."""

from __future__ import annotations

from .schema import BaseAttentiveSpec


def _ensure_positive_int(name: str, value: int, *, allow_zero: bool = False):
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if allow_zero:
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}.")
    elif value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}.")


def validate_base_attentive_spec(spec: BaseAttentiveSpec) -> BaseAttentiveSpec:
    """Validate a normalized ``BaseAttentiveSpec``."""
    if not isinstance(spec, BaseAttentiveSpec):
        raise TypeError("spec must be a BaseAttentiveSpec instance.")

    _ensure_positive_int("static_input_dim", spec.static_input_dim, allow_zero=True)
    _ensure_positive_int("dynamic_input_dim", spec.dynamic_input_dim)
    _ensure_positive_int("future_input_dim", spec.future_input_dim, allow_zero=True)
    _ensure_positive_int("output_dim", spec.output_dim)
    _ensure_positive_int("forecast_horizon", spec.forecast_horizon)
    _ensure_positive_int("embed_dim", spec.embed_dim)
    _ensure_positive_int("hidden_units", spec.hidden_units)
    _ensure_positive_int("attention_heads", spec.attention_heads)

    if not 0.0 <= spec.dropout_rate <= 1.0:
        raise ValueError("dropout_rate must be between 0 and 1 inclusive.")
    if spec.layer_norm_epsilon <= 0:
        raise ValueError("layer_norm_epsilon must be > 0.")

    if spec.head_type not in {"point", "quantile"}:
        raise ValueError(
            "head_type must be either 'point' or 'quantile', "
            f"got {spec.head_type!r}."
        )

    if spec.head_type == "quantile" and not spec.quantiles:
        raise ValueError(
            "quantile head_type requires at least one quantile value."
        )

    return spec


__all__ = ["validate_base_attentive_spec"]
