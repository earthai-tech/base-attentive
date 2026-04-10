"""Backend abstraction for framework support (TensorFlow, PyTorch, etc.)."""

from __future__ import annotations

import os
from typing import Any, Optional, Type

__all__ = [
    "get_backend",
    "set_backend",
    "get_available_backends",
    "Backend",
]

# Global backend instance
_CURRENT_BACKEND: Optional[Backend] = None


class Backend:
    """Abstract base class for backend implementations."""

    name: str = "base"
    framework: str = "unknown"

    # Framework-specific imports (to be set by subclasses)
    Tensor: Any = None
    Layer: Any = None
    Model: Any = None
    Sequential: Any = None
    Dense: Any = None
    LSTM: Any = None
    MultiHeadAttention: Any = None
    LayerNormalization: Any = None
    Dropout: Any = None
    BatchNormalization: Any = None

    def __init__(self):
        """Initialize the backend."""
        self._verify_installation()

    def _verify_installation(self):
        """Verify that the required framework is installed."""
        raise NotImplementedError(
            f"Backend {self.name} must implement _verify_installation()"
        )

    def is_available(self) -> bool:
        """Check if the backend is available."""
        try:
            self._verify_installation()
            return True
        except ImportError:
            return False


class TensorFlowBackend(Backend):
    """TensorFlow backend implementation."""

    name = "tensorflow"
    framework = "tensorflow"

    def __init__(self):
        """Initialize TensorFlow backend."""
        super().__init__()
        self._initialize_imports()

    def _verify_installation(self):
        """Verify TensorFlow is installed."""
        try:
            import tensorflow as tf
            return True
        except ImportError as e:
            raise ImportError(
                "TensorFlow is not installed. "
                "Install it with: pip install tensorflow"
            ) from e

    def _initialize_imports(self):
        """Load all required TensorFlow/Keras modules."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers

            self.tf = tf
            self.keras = keras
            self.layers = layers

            # Set common layer references
            self.Tensor = tf.Tensor
            self.Layer = layers.Layer
            self.Model = keras.Model
            self.Sequential = keras.Sequential
            self.Dense = layers.Dense
            self.LSTM = layers.LSTM
            self.MultiHeadAttention = layers.MultiHeadAttention
            self.LayerNormalization = layers.LayerNormalization
            self.Dropout = layers.Dropout
            self.BatchNormalization = layers.BatchNormalization

        except ImportError as e:
            raise ImportError(
                f"Failed to import TensorFlow components: {e}"
            ) from e


class PyTorchBackend(Backend):
    """PyTorch backend implementation (stub for future use)."""

    name = "pytorch"
    framework = "pytorch"

    def __init__(self):
        """Initialize PyTorch backend."""
        super().__init__()
        self._initialize_imports()

    def _verify_installation(self):
        """Verify PyTorch is installed."""
        try:
            import torch
            return True
        except ImportError as e:
            raise ImportError(
                "PyTorch is not installed. "
                "Install it with: pip install torch"
            ) from e

    def _initialize_imports(self):
        """Load all required PyTorch modules."""
        try:
            import torch
            import torch.nn as nn

            self.torch = torch
            self.nn = nn

            # Set common layer references (will differ from TF)
            self.Tensor = torch.Tensor
            self.Layer = nn.Module
            self.Model = nn.Module
            self.Dense = nn.Linear
            self.LSTM = nn.LSTM
            # PyTorch doesn't have direct MultiHeadAttention in layers,
            # but has it in nn.MultiheadAttention

        except ImportError as e:
            raise ImportError(
                f"Failed to import PyTorch components: {e}"
            ) from e


# Registry of available backends
_BACKENDS: dict[str, Type[Backend]] = {
    "tensorflow": TensorFlowBackend,
    "pytorch": PyTorchBackend,
}


def get_available_backends() -> list[str]:
    """Get list of available backends."""
    available = []
    for name, backend_cls in _BACKENDS.items():
        try:
            backend = backend_cls()
            if backend.is_available():
                available.append(name)
        except ImportError:
            pass
    return available


def get_backend(name: Optional[str] = None) -> Backend:
    """
    Get the current or specified backend.

    Parameters
    ----------
    name : str, optional
        Backend name. If None, uses current backend or environment variable.

    Returns
    -------
    Backend
        The requested backend instance.

    Raises
    ------
    ValueError
        If backend is not available.
    """
    global _CURRENT_BACKEND

    if name is None:
        # Use environment variable or current backend
        name = os.environ.get("BASE_ATTENTIVE_BACKEND")
        if name is None and _CURRENT_BACKEND is not None:
            return _CURRENT_BACKEND
        if name is None:
            # Default to tensorflow
            name = "tensorflow"

    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available: {list(_BACKENDS.keys())}"
        )

    backend_cls = _BACKENDS[name]
    try:
        return backend_cls()
    except ImportError as e:
        available = get_available_backends()
        raise ValueError(
            f"Backend '{name}' is not available. "
            f"Available backends: {available}. "
            f"Install with: pip install {name}"
        ) from e


def set_backend(name: str) -> Backend:
    """
    Set the default backend.

    Parameters
    ----------
    name : str
        Backend name ('tensorflow' or 'pytorch').

    Returns
    -------
    Backend
        The set backend instance.
    """
    global _CURRENT_BACKEND
    _CURRENT_BACKEND = get_backend(name)
    os.environ["BASE_ATTENTIVE_BACKEND"] = name
    return _CURRENT_BACKEND
