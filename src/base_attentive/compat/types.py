"""Type hints and type aliases for the base-attentive package."""

from __future__ import annotations

import os
from typing import Any, Union

# Type aliases
TensorLike = Any  # Can be numpy array, tf.Tensor, list, etc.
DatasetLike = Any  # Can be various dataset formats
PathLike = Union[str, os.PathLike]

__all__ = [
    "TensorLike",
    "DatasetLike",
    "PathLike",
]
