"""Model-specific utility functions."""

from __future__ import annotations

from typing import Any

__all__ = ["set_default_params"]


def set_default_params(
    model_params: Any,
    *args: Any,
    **defaults: Any,
) -> Any:
    """
    Set default parameters for model configuration.

    Parameters
    ----------
    model_params : Any
        Either a parameter dictionary or the ``quantiles`` value expected by
        ``BaseAttentive``.
    *args : Any
        Additional positional arguments. When provided, this function supports
        the ``BaseAttentive`` calling convention:
        ``set_default_params(quantiles, scales, multi_scale_agg)``.
    **defaults
        Default parameter values to apply when ``model_params`` is a dict.

    Returns
    -------
    Any
        Either a merged parameter dict or a normalized
        ``(quantiles, scales, lstm_return_sequences)`` tuple.
    """
    if isinstance(model_params, dict):
        result = dict(defaults)
        result.update(model_params)
        return result

    scales = args[0] if len(args) > 0 else None
    multi_scale_agg = args[1] if len(args) > 1 else "last"

    normalized_quantiles = (
        list(model_params) if isinstance(model_params, (list, tuple)) else model_params
    )
    normalized_scales = (
        list(scales)
        if isinstance(scales, (list, tuple))
        else [1]
        if scales in (None, "auto")
        else scales
    )
    lstm_return_sequences = multi_scale_agg != "last"

    return (
        normalized_quantiles,
        normalized_scales,
        lstm_return_sequences,
    )
