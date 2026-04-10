"""
base_attentive: A foundational blueprint for sequence-to-sequence
time series forecasting models with attention mechanisms.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "LKouadio"
__email__ = "etanoyau@gmail.com"
__license__ = "Apache-2.0"

# Initialize Keras backend and dependencies
import os
from typing import Any

# Resolve Keras backend
KERAS_BACKEND = os.environ.get("KERAS_BACKEND", "tensorflow")


class _KerasDeps:
    """Placeholder for Keras dependencies."""

    def __getattr__(self, name: str) -> Any:
        """Lazy load Keras modules as needed."""
        try:
            if KERAS_BACKEND == "tensorflow":
                import tensorflow.keras as keras
            else:
                import keras
            return getattr(keras, name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Cannot import {name} from Keras. Ensure TensorFlow or Keras is installed."
            ) from e


KERAS_DEPS = _KerasDeps()


def dependency_message(module_name: str) -> str:
    """Return a dependency message for missing packages."""
    return (
        f"TensorFlow/Keras is required for {module_name}. "
        f"Install it with: pip install tensorflow"
    )


# Import backend utilities for framework flexibility
from .backend import (
    get_backend,
    set_backend,
    get_available_backends,
)

try:
    from base_attentive.core import BaseAttentive
    __all__ = [
        "BaseAttentive",
        "KERAS_BACKEND",
        "KERAS_DEPS",
        "dependency_message",
        "get_backend",
        "set_backend",
        "get_available_backends",
    ]
except Exception as e:
    __all__ = [
        "KERAS_BACKEND",
        "KERAS_DEPS",
        "dependency_message",
        "get_backend",
        "set_backend",
        "get_available_backends",
    ]
    import warnings
    warnings.warn(
        f"Failed to import BaseAttentive: {e}. Ensure TensorFlow/Keras is installed.",
        RuntimeWarning
    )
