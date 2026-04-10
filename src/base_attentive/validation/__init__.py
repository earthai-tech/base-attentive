"""Utilities for backend-agnostic tensor validation."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np

from ..compat.types import TensorLike
from ..logging import get_logger

try:
    from .. import KERAS_BACKEND, KERAS_DEPS
except Exception:
    KERAS_BACKEND = ""
    KERAS_DEPS = None

__all__ = [
    "validate_model_inputs",
    "maybe_reduce_quantiles_bh",
    "ensure_bh1",
]

_logger = get_logger(__name__)


def _has_runtime() -> bool:
    return bool(KERAS_BACKEND and KERAS_DEPS is not None)


def validate_model_inputs(
    inputs: Union[Any, np.ndarray, list],
    static_input_dim: Optional[int] = None,
    dynamic_input_dim: Optional[int] = None,
    future_covariate_dim: Optional[int] = None,
    forecast_horizon: Optional[int] = None,
    error: str = "raise",
    mode: str = "strict",
    deep_check: Optional[bool] = None,
    model_name: Optional[str] = None,
    verbose: int = 0,
    **kwargs,
) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    Validate and homogenize input tensors for model workflows.

    This entrypoint intentionally stays lightweight: it normalizes the input
    container shape and converts values into the active Keras runtime tensor
    type when a runtime is available. When no Keras runtime is configured, the
    raw values are returned unchanged.
    """
    if not _has_runtime():
        if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
            return inputs[0], inputs[1], inputs[2]
        return inputs, None, None

    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    while len(inputs) < 3:
        inputs.append(None)
    inputs = inputs[:3]

    tensors = []
    for inp in inputs:
        if inp is None:
            tensors.append(None)
            continue
        try:
            tensors.append(KERAS_DEPS.convert_to_tensor(inp))
        except Exception as exc:
            if error == "raise":
                raise ValueError(
                    f"Could not convert input to tensor: {exc}"
                ) from exc
            tensors.append(inp)

    static, dynamic, future = tensors

    if verbose > 0:
        _logger.info("Validating input tensors...")
        if static is not None:
            _logger.info("  Static shape: %s", getattr(static, "shape", None))
        if dynamic is not None:
            _logger.info(
                "  Dynamic shape: %s",
                getattr(dynamic, "shape", None),
            )
        if future is not None:
            _logger.info("  Future shape: %s", getattr(future, "shape", None))

    return static, dynamic, future


def maybe_reduce_quantiles_bh(
    x: Any,
    *,
    name: str = "tensor",
    axis: int = 2,
    reduction: Union[str, callable] = "mean",
) -> Any:
    """Reduce a quantile axis when a backend tensor carries one."""
    if not _has_runtime():
        return x

    x = KERAS_DEPS.convert_to_tensor(x)
    rank = len(getattr(x, "shape", ()))

    if rank >= 4:
        if callable(reduction):
            return reduction(x, axis=axis)
        if reduction == "mean":
            return KERAS_DEPS.reduce_mean(x, axis=axis)
        if reduction == "sum":
            return KERAS_DEPS.reduce_sum(x, axis=axis)
    elif rank == 3:
        last_dim = x.shape[-1]
        if last_dim is not None and last_dim > 1:
            if callable(reduction):
                return reduction(x, axis=axis)
            if reduction == "mean":
                return KERAS_DEPS.reduce_mean(x, axis=axis)
            if reduction == "sum":
                return KERAS_DEPS.reduce_sum(x, axis=axis)

    return x


def ensure_bh1(
    x: Any,
    *,
    name: str = "tensor",
    dtype: Optional[Any] = None,
    reduce_axis: Optional[int] = None,
    reduction: Union[str, callable] = "mean",
    allow_rank1: bool = False,
) -> Any:
    """Ensure a tensor-like value has shape ``(B, H, 1)``."""
    if not _has_runtime():
        return x

    x = KERAS_DEPS.convert_to_tensor(x)

    while len(getattr(x, "shape", ())) < 3:
        x = KERAS_DEPS.expand_dims(x, axis=-1)

    if reduce_axis is not None:
        if callable(reduction):
            x = reduction(x, axis=reduce_axis)
        elif reduction == "mean":
            x = KERAS_DEPS.reduce_mean(x, axis=reduce_axis)
        elif reduction == "sum":
            x = KERAS_DEPS.reduce_sum(x, axis=reduce_axis)

    if dtype is not None:
        x = KERAS_DEPS.cast(x, dtype)

    return x
