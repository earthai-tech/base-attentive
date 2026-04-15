# SPDX-License-Identifier: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>
# Adapted from: earthai-tech/fusionlab-learn https://github.com/earthai-tech/gofast
# Modified for GeoPrior-v3 API.

"""
Mask helpers for attention / sequence ops.

- pad_mask_from_lengths: build (B, T) masks from per-seq lengths
- sequence_mask_3d: broadcast a 2D mask to 3D for (B, T, F) tensors
"""

from __future__ import annotations

from ..compat.types import TensorLike
from ._config import (
    Tensor,
    bool_dtype,
    cast,
    expand_dims,
    float32,
    int32,
    logical_not,
    range,
    reduce_max,
    shape,
)

__all__ = ["pad_mask_from_lengths", "sequence_mask_3d"]


def pad_mask_from_lengths(
    lengths: Tensor,
    max_len: int | None = None,
    *,
    dtype: Tensor = bool_dtype,
    invert: bool = False,
) -> Tensor:
    """
    Build a mask of shape (B, T) from integer lengths.

    Parameters
    ----------
    lengths : Tensor[int32/int64], shape (B,)
        Valid lengths per sequence.
    max_len : int, optional
        If None, uses max(lengths).
    dtype : tf.bool or tf.float32
        Output dtype.
    invert : bool
        If True, returns mask==False positions as True.

    Returns
    -------
    Tensor
        Mask of shape (B, T). True/1.0 = keep (unless invert=True).
    """
    lengths = cast(lengths, int32)
    if max_len is None:
        max_len = reduce_max(lengths)
    max_len = cast(max_len, int32)

    rng = range(max_len)  # (T,)
    rng = expand_dims(rng, 0)  # (1, T)
    len_col = expand_dims(lengths, 1)  # (B, 1)

    mask = cast(rng < len_col, bool_dtype)  # (B, T) bool
    if invert:
        mask = logical_not(mask)

    if dtype is float32:
        return cast(mask, float32)
    return mask


def sequence_mask_3d(
    data: Tensor,
    *,
    lengths: TensorLike | None = None,
    mask_2d: TensorLike | None = None,
    time_axis: int = 1,
    invert: bool = False,
    dtype: Tensor = bool_dtype,
) -> Tensor:
    """
    Produce a broadcastable mask for a (B, T, F) tensor.

    One of `lengths` or `mask_2d` must be given.

    Parameters
    ----------
    data : Tensor, shape (B, T, F) or compatible
        Reference tensor to infer batch/time dims.
    lengths : Tensor[int], optional shape (B,)
        Build mask from valid lengths.
    mask_2d : Tensor[bool/0-1], optional shape (B, T)
        Precomputed 2D mask.
    time_axis : int
        Axis of time dim in `data` (default 1).
    invert : bool
        Return the inverse mask.
    dtype : tf.bool or tf.float32
        Output dtype.

    Returns
    -------
    Tensor
        Mask shaped (B, T, 1) to multiply with data or feed to MHA.
    """
    if lengths is None and mask_2d is None:
        raise ValueError("Provide `lengths` or `mask_2d`.")

    # Infer batch/time from data
    b = shape(data)[0]  # noqa
    t = shape(data)[time_axis]

    if mask_2d is None:
        mask_2d = pad_mask_from_lengths(
            lengths, max_len=t, dtype=bool_dtype
        )
    else:
        mask_2d = cast(mask_2d, bool_dtype)

    if invert:
        mask_2d = logical_not(mask_2d)

    # Expand to (B, T, 1) so it broadcasts over feature dim
    mask_3d = expand_dims(mask_2d, -1)

    if dtype is float32:
        return cast(mask_3d, float32)
    return mask_3d
