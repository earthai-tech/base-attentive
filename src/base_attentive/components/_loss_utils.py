# SPDX-License-Identifier: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>
# Adapted from: earthai-tech/fusionlab-learn https://github.com/earthai-tech/gofast
# Modified for GeoPrior-v3 API.

"""
Loss utilities for different use-cases, including quantile loss,
mean squared error, and more. These utilities can be used with
different models and can be easily extended for future requirements.
"""

from __future__ import annotations

from ._config import (
    Loss,
    Tensor,
    register_keras_serializable,
    tf_abs,
    tf_cast,
    tf_constant,
    tf_float32,
    tf_maximum,
    tf_reduce_mean,
    tf_reduce_sum,
    tf_square,
    tf_where,
)

__all__ = [
    "MeanSquaredErrorLoss",
    "QuantileLoss",
    "HuberLoss",
    "WeightedLoss",
    "compute_loss_with_reduction",
    "compute_quantile_loss",
]
SERIALIZATION_PACKAGE = __name__


def _normalize_loss_reduction(reduction: str | None) -> str | None:
    if reduction == "auto":
        return "sum_over_batch_size"
    return reduction


@register_keras_serializable(
    SERIALIZATION_PACKAGE, name="MeanSquaredErrorLoss"
)
class MeanSquaredErrorLoss(Loss):
    """
    Mean Squared Error (MSE) loss function.

    Args:
    - reduction (str): Defines the reduction method for the loss:
      'auto', 'sum', 'mean', or 'none'.
    """

    def __init__(
        self, reduction: str = "auto", name: str = "MSELoss"
    ):
        super().__init__(
            reduction=_normalize_loss_reduction(reduction), name=name
        )

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute the Mean Squared Error loss.

        Args:
        - y_true: Ground truth values
        - y_pred: Predicted values

        Returns:
        - Tensor: Computed loss value
        """
        return tf_reduce_mean(tf_square(y_true - y_pred))


@register_keras_serializable(
    SERIALIZATION_PACKAGE, name="QuantileLoss"
)
class QuantileLoss(Loss):
    """
    Adaptive Quantile Loss layer that computes quantile loss for given
    quantiles.

    Args:
    - quantiles (List[float]): List of quantiles to compute the loss.
    """

    def __init__(
        self,
        quantiles: list[float],
        reduction: str = "auto",
        name: str = "AdaptiveQuantileLoss",
    ):
        super().__init__(
            reduction=_normalize_loss_reduction(reduction), name=name
        )
        self.quantiles = quantiles
        self.q = len(quantiles)
        self._qs = tf_constant(quantiles, dtype=tf_float32)

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """Compute quantile loss across common rank-2 and rank-4 layouts."""
        return compute_quantile_loss(
            y_true, y_pred, quantiles=self.quantiles
        )


@register_keras_serializable(SERIALIZATION_PACKAGE, name="HuberLoss")
class HuberLoss(Loss):
    """
    Huber loss function which is less sensitive to outliers in data.

    Args:
    - delta (float): Threshold for switching between MSE and MAE.
    """

    def __init__(
        self,
        delta: float = 1.0,
        reduction: str = "auto",
        name: str = "HuberLoss",
    ):
        super().__init__(
            reduction=_normalize_loss_reduction(reduction), name=name
        )
        self.delta = delta

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute the Huber loss.

        Args:
        - y_true: Ground truth values
        - y_pred: Predicted values

        Returns:
        - Tensor: Computed loss value
        """
        error = y_true - y_pred
        abs_error = tf_abs(error)

        # Huber loss formula
        condition = abs_error <= self.delta
        squared_loss = tf_square(error) / 2
        linear_loss = self.delta * (abs_error - (self.delta / 2))

        return tf_reduce_mean(
            tf_where(condition, squared_loss, linear_loss)
        )


@register_keras_serializable(
    SERIALIZATION_PACKAGE, name="WeightedLoss"
)
class WeightedLoss(Loss):
    """
    Weighted loss function to apply different weights for each sample.

    Args:
    - weight (float or Tensor): Weighting factor for the loss calculation.
    """

    def __init__(
        self,
        base_loss=None,
        weight: float = 1.0,
        reduction: str = "auto",
        name: str = "WeightedLoss",
    ):
        super().__init__(
            reduction=_normalize_loss_reduction(reduction), name=name
        )
        self.base_loss = base_loss
        self.weight = weight

    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Compute the weighted loss.

        Args:
        - y_true: Ground truth values
        - y_pred: Predicted values

        Returns:
        - Tensor: Computed weighted loss value
        """
        if self.base_loss is not None:
            return self.weight * self.base_loss(y_true, y_pred)
        return self.weight * tf_reduce_mean(
            tf_square(y_true - y_pred)
        )


### Utility Functions for Losses ###


def compute_loss_with_reduction(
    loss_or_values,
    y_true: Tensor | None = None,
    y_pred: Tensor | None = None,
    reduction: str = "mean",
) -> Tensor:
    """
    Compute the loss using the specified reduction method.

    Args:
    - loss_or_values: Loss function to compute the loss or already computed values
    - y_true: Ground truth values
    - y_pred: Predicted values
    - reduction: String indicating reduction method ('sum', 'mean', etc.)

    Returns:
    - Tensor: Loss value
    """
    if callable(loss_or_values):
        loss_value = loss_or_values(y_true, y_pred)
    else:
        loss_value = loss_or_values

    if reduction == "mean":
        return tf_reduce_mean(loss_value)
    elif reduction == "sum":
        return tf_reduce_sum(loss_value)
    elif reduction == "none":
        return loss_value
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")


def compute_quantile_loss(
    y_true: Tensor,
    y_pred: Tensor,
    quantiles: list[float] | None = None,
    *,
    quantile: float | None = None,
) -> Tensor:
    """
    Compute the quantile loss for a given set of quantiles.

    Args:
    - y_true: Ground truth values
    - y_pred: Predicted values
    - quantiles: List of quantiles to compute the loss
    - quantile: Convenience alias for a single quantile

    Returns:
    - Tensor: Quantile loss value
    """
    if quantiles is None:
        if quantile is None:
            raise ValueError("Provide `quantiles` or `quantile`.")
        quantiles = [quantile]

    y_true = tf_cast(y_true, tf_float32)
    y_pred = tf_cast(y_pred, tf_float32)
    qs = tf_constant(quantiles, dtype=tf_float32)
    # Align qs to the same device as y_pred (MPS/CUDA compatibility)
    _dev = getattr(y_pred, "device", None)
    _to = getattr(qs, "to", None)
    if _dev is not None and callable(_to):
        qs = _to(_dev)
    error = y_true - y_pred
    a = qs * error
    b = (qs - 1.0) * error
    # Use torch.maximum directly when torch tensors are present to avoid
    # the numpy-based fallback that cannot handle MPS tensors.
    try:
        import torch as _torch

        if isinstance(a, _torch.Tensor):
            if not isinstance(b, _torch.Tensor):
                b = _torch.as_tensor(b, device=a.device)
            elif a.device != b.device:
                b = b.to(a.device)
            pinball_loss = _torch.maximum(a, b)
        else:
            pinball_loss = tf_maximum(a, b)
    except ImportError:
        pinball_loss = tf_maximum(a, b)
    try:
        return tf_reduce_mean(pinball_loss, axis=2)
    except Exception:
        return tf_reduce_mean(pinball_loss)
