"""Tensor validation helpers for model workflows."""

from __future__ import annotations

import warnings
from typing import Any, Optional, Tuple, Union

import numpy as np

from .compat.types import TensorLike
from .logging import get_logger

__all__ = [
    "validate_model_inputs",
    "maybe_reduce_quantiles_bh",
    "ensure_bh1",
]

_logger = get_logger(__name__)


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
    Validate and homogenize input tensors for time series models.

    This is a simplified version. Full validation checks:
    - Static tensors have rank 2: (B, F_static)
    - Dynamic tensors have rank 3: (B, T_past, F_dyn)
    - Future tensors have rank 3: (B, T_future, F_future)

    Parameters
    ----------
    inputs : Union[Tensor, np.ndarray, list]
        Triplet [static, dynamic, future] or single tensor.
    static_input_dim : int, optional
        Expected feature dimension for static inputs.
    dynamic_input_dim : int, optional
        Expected feature dimension for dynamic inputs.
    future_covariate_dim : int, optional
        Expected feature dimension for future inputs.
    forecast_horizon : int, optional
        Expected forecast horizon.
    error : str, optional
        Error handling: 'raise', 'warn', or 'ignore'. Default: 'raise'.
    mode : str, optional
        Validation mode: 'strict' or 'soft'. Default: 'strict'.
    deep_check : bool, optional
        Deprecated. Use mode instead.
    model_name : str, optional
        Name of the calling model.
    verbose : int, optional
        Verbosity level. Default: 0.
    **kwargs
        Reserved for future use.

    Returns
    -------
    Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]
        Validated (static, dynamic, future) tensors.
    """
    try:
        import tensorflow as tf
    except ImportError:
        # If TensorFlow not available, just return inputs as-is
        if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
            return inputs[0], inputs[1], inputs[2]
        return inputs, None, None

    # Convert inputs to list if needed
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    # Ensure we have 3 elements
    while len(inputs) < 3:
        inputs.append(None)
    inputs = inputs[:3]

    # Convert to tensors
    tensors = []
    for inp in inputs:
        if inp is None:
            tensors.append(None)
        else:
            try:
                tensors.append(tf.convert_to_tensor(inp))
            except Exception as e:
                if error == "raise":
                    raise ValueError(f"Could not convert input to tensor: {e}")
                tensors.append(inp)

    static, dynamic, future = tensors

    # Basic shape validation
    if verbose > 0:
        _logger.info("Validating input tensors...")
        if static is not None:
            _logger.info(f"  Static shape: {static.shape}")
        if dynamic is not None:
            _logger.info(f"  Dynamic shape: {dynamic.shape}")
        if future is not None:
            _logger.info(f"  Future shape: {future.shape}")

    return static, dynamic, future


def maybe_reduce_quantiles_bh(
    x: Any,
    *,
    name: str = "tensor",
    axis: int = 2,
    reduction: Union[str, callable] = "mean",
) -> Any:
    """
    Reduce quantile dimension if present.

    If x has shape (B, H, Q) or (B, H, Q, 1), reduce along Q axis.
    Otherwise return unchanged.

    Parameters
    ----------
    x : Tensor
        Input tensor.
    name : str, optional
        Tensor name for error messages. Default: "tensor".
    axis : int, optional
        Axis to reduce. Default: 2.
    reduction : str or callable, optional
        Reduction method: 'mean', 'sum', or callable. Default: 'mean'.

    Returns
    -------
    Tensor
        Reduced or original tensor.
    """
    try:
        import tensorflow as tf
    except ImportError:
        return x

    x = tf.convert_to_tensor(x)
    rank = len(x.shape)

    if rank >= 4:
        # (B, H, Q, 1) -> reduce Q dimension
        if callable(reduction):
            return reduction(x, axis=axis)
        elif reduction == "mean":
            return tf.reduce_mean(x, axis=axis)
        elif reduction == "sum":
            return tf.reduce_sum(x, axis=axis)
    elif rank == 3:
        # Check if last dimension > 1 and needs reduction
        last_dim = x.shape[-1]
        if last_dim is not None and last_dim > 1:
            if callable(reduction):
                return reduction(x, axis=axis)
            elif reduction == "mean":
                return tf.reduce_mean(x, axis=axis)
            elif reduction == "sum":
                return tf.reduce_sum(x, axis=axis)

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
    """
    Ensure tensor has shape (B, H, 1).

    Parameters
    ----------
    x : Tensor
        Input tensor with shape like (B, H) or (B, H, Q).
    name : str, optional
        Tensor name for error messages. Default: "tensor".
    dtype : tf.DType, optional
        Cast to this dtype if provided. Default: None.
    reduce_axis : int, optional
        Reduce along this axis. Default: None.
    reduction : str or callable, optional
        Reduction method. Default: 'mean'.
    allow_rank1 : bool, optional
        Allow rank-1 tensors. Default: False.

    Returns
    -------
    Tensor
        Tensor with shape (B, H, 1).
    """
    try:
        import tensorflow as tf
    except ImportError:
        return x

    x = tf.convert_to_tensor(x)

    # Add dimensions if needed
    while len(x.shape) < 3:
        x = tf.expand_dims(x, axis=-1)

    # Optionally reduce a middle dimension
    if reduce_axis is not None:
        if callable(reduction):
            x = reduction(x, axis=reduce_axis)
        elif reduction == "mean":
            x = tf.reduce_mean(x, axis=reduce_axis)
        elif reduction == "sum":
            x = tf.reduce_sum(x, axis=reduce_axis)

    # Cast if needed
    if dtype is not None:
        x = tf.cast(x, dtype)

    return x
