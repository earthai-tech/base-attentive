"""Backend runtime abstraction for TensorFlow, JAX, and Torch."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import warnings
from typing import Any, Optional, Type

__all__ = [
    "Backend",
    "TensorFlowBackend",
    "JaxBackend",
    "TorchBackend",
    "PyTorchBackend",
    "get_backend",
    "set_backend",
    "get_available_backends",
    "get_backend_capabilities",
    "normalize_backend_name",
]


_BACKEND_ALIASES = {
    "tf": "tensorflow",
    "tensorflow": "tensorflow",
    "jax": "jax",
    "torch": "torch",
    "pytorch": "torch",
}

_MULTI_BACKEND_BLOCKERS = (
    "BaseAttentive still contains TensorFlow-oriented compatibility paths.",
    "The compat.tf helpers are still TensorFlow-specific.",
    "Some runtime shape/assert checks still assume TensorFlow graph semantics.",
)

# Global backend instance
_CURRENT_BACKEND: Optional["Backend"] = None


def normalize_backend_name(name: Optional[str]) -> str:
    """Normalize user-facing backend aliases to canonical names."""
    if name is None:
        return "tensorflow"

    normalized = str(name).strip().lower()
    if not normalized:
        return "tensorflow"
    if normalized == "keras":
        return normalize_backend_name(
            os.environ.get("KERAS_BACKEND", "tensorflow")
        )
    return _BACKEND_ALIASES.get(normalized, normalized)


def _import_module(module_name: str):
    """Import a module by name."""
    return importlib.import_module(module_name)


def _has_module(module_name: str) -> bool:
    """Return whether a module appears importable without importing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _read_loaded_keras_backend() -> Optional[str]:
    """Return the already-loaded Keras runtime backend, if available."""
    if "keras" not in sys.modules:
        return None

    try:
        keras = sys.modules["keras"]
        backend_ns = getattr(keras, "backend", None)
        backend_fn = getattr(backend_ns, "backend", None)
        if callable(backend_fn):
            return normalize_backend_name(backend_fn())
    except Exception:
        return None
    return None


class Backend:
    """Base class for runtime backend descriptors."""

    name: str = "base"
    framework: str = "unknown"
    required_modules: tuple[str, ...] = ()
    uses_keras_runtime: bool = False
    experimental: bool = False
    supports_base_attentive: bool = False
    blockers: tuple[str, ...] = ()

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

    def __init__(self, load_runtime: bool = True):
        self._verify_installation()
        if load_runtime:
            self._initialize_imports()

    def _verify_installation(self):
        """Verify that the required framework is installed."""
        for module_name in self.required_modules:
            if not _has_module(module_name):
                raise ImportError(
                    f"Backend '{self.name}' requires '{module_name}'."
                )
        return True

    def _initialize_imports(self):
        """Load framework-specific handles."""

    def is_available(self) -> bool:
        """Check whether the backend can be imported."""
        try:
            self._verify_installation()
            return True
        except ImportError:
            return False

    def get_capabilities(self) -> dict[str, Any]:
        """Return a capability summary for the backend."""
        return {
            "name": self.name,
            "framework": self.framework,
            "available": self.is_available(),
            "uses_keras_runtime": self.uses_keras_runtime,
            "experimental": self.experimental,
            "supports_base_attentive": self.supports_base_attentive,
            "blockers": list(self.blockers),
            "loaded_keras_backend": _read_loaded_keras_backend(),
        }


class TensorFlowBackend(Backend):
    """TensorFlow-backed runtime."""

    name = "tensorflow"
    framework = "tensorflow"
    required_modules = ("tensorflow",)
    uses_keras_runtime = True
    supports_base_attentive = True

    def _initialize_imports(self):
        tf = _import_module("tensorflow")

        try:
            keras = _import_module("keras")
        except ImportError:
            keras = tf.keras

        layers = keras.layers

        self.tf = tf
        self.keras = keras
        self.layers = layers
        self.Tensor = getattr(tf, "Tensor", object)
        self.Layer = getattr(layers, "Layer", None)
        self.Model = getattr(keras, "Model", None)
        self.Sequential = getattr(keras, "Sequential", None)
        self.Dense = getattr(layers, "Dense", None)
        self.LSTM = getattr(layers, "LSTM", None)
        self.MultiHeadAttention = getattr(
            layers, "MultiHeadAttention", None
        )
        self.LayerNormalization = getattr(
            layers, "LayerNormalization", None
        )
        self.Dropout = getattr(layers, "Dropout", None)
        self.BatchNormalization = getattr(
            layers, "BatchNormalization", None
        )


class JaxBackend(Backend):
    """Keras-on-JAX runtime descriptor."""

    name = "jax"
    framework = "jax"
    required_modules = ("keras", "jax")
    uses_keras_runtime = True
    experimental = True
    supports_base_attentive = False
    blockers = _MULTI_BACKEND_BLOCKERS

    def _initialize_imports(self):
        keras = _import_module("keras")
        jax = _import_module("jax")

        self.keras = keras
        self.jax = jax
        self.layers = getattr(keras, "layers", None)
        self.Tensor = object
        self.Layer = getattr(self.layers, "Layer", None)
        self.Model = getattr(keras, "Model", None)
        self.Sequential = getattr(keras, "Sequential", None)
        self.Dense = getattr(self.layers, "Dense", None)
        self.LSTM = getattr(self.layers, "LSTM", None)
        self.MultiHeadAttention = getattr(
            self.layers, "MultiHeadAttention", None
        )
        self.LayerNormalization = getattr(
            self.layers, "LayerNormalization", None
        )
        self.Dropout = getattr(self.layers, "Dropout", None)
        self.BatchNormalization = getattr(
            self.layers, "BatchNormalization", None
        )


class TorchBackend(Backend):
    """Keras-on-Torch runtime descriptor."""

    name = "torch"
    framework = "torch"
    required_modules = ("keras", "torch")
    uses_keras_runtime = True
    experimental = True
    supports_base_attentive = False
    blockers = _MULTI_BACKEND_BLOCKERS

    def _initialize_imports(self):
        keras = _import_module("keras")
        torch = _import_module("torch")

        self.keras = keras
        self.torch = torch
        self.layers = getattr(keras, "layers", None)
        self.Tensor = getattr(torch, "Tensor", object)
        self.Layer = getattr(self.layers, "Layer", None)
        self.Model = getattr(keras, "Model", None)
        self.Sequential = getattr(keras, "Sequential", None)
        self.Dense = getattr(self.layers, "Dense", None)
        self.LSTM = getattr(self.layers, "LSTM", None)
        self.MultiHeadAttention = getattr(
            self.layers, "MultiHeadAttention", None
        )
        self.LayerNormalization = getattr(
            self.layers, "LayerNormalization", None
        )
        self.Dropout = getattr(self.layers, "Dropout", None)
        self.BatchNormalization = getattr(
            self.layers, "BatchNormalization", None
        )


class PyTorchBackend(TorchBackend):
    """Backward-compatible alias for the Torch runtime."""


# Registry of available backends
_BACKENDS: dict[str, Type[Backend]] = {
    "tensorflow": TensorFlowBackend,
    "jax": JaxBackend,
    "torch": TorchBackend,
}


def get_available_backends() -> list[str]:
    """Get the installed backends that can be imported."""
    available = []
    for name, backend_cls in _BACKENDS.items():
        try:
            backend = backend_cls(load_runtime=False)
            if backend.is_available():
                available.append(name)
        except ImportError:
            pass
    return available


def get_backend(name: Optional[str] = None) -> Backend:
    """
    Get the current or requested backend runtime.

    When ``name`` is omitted, the lookup order is:

    1. ``BASE_ATTENTIVE_BACKEND``
    2. ``KERAS_BACKEND``
    3. the previously set in-process backend
    4. ``tensorflow``
    """
    global _CURRENT_BACKEND

    if name is None:
        env_name = os.environ.get("BASE_ATTENTIVE_BACKEND")
        if env_name is None:
            env_name = os.environ.get("KERAS_BACKEND")
        if env_name is None and _CURRENT_BACKEND is not None:
            return _CURRENT_BACKEND
        name = env_name or "tensorflow"

    normalized = normalize_backend_name(name)
    if normalized not in _BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available: {list(_BACKENDS.keys())}"
        )

    backend_cls = _BACKENDS[normalized]
    try:
        return backend_cls()
    except ImportError as exc:
        available = get_available_backends()
        raise ValueError(
            f"Backend '{normalized}' is not available. "
            f"Available backends: {available}."
        ) from exc


def get_backend_capabilities(
    name: Optional[str] = None,
) -> dict[str, Any]:
    """Return a capability report for the current or requested backend."""
    if name is None:
        name = (
            os.environ.get("BASE_ATTENTIVE_BACKEND")
            or os.environ.get("KERAS_BACKEND")
            or "tensorflow"
        )
    normalized = normalize_backend_name(name)
    if normalized not in _BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available: {list(_BACKENDS.keys())}"
        )
    return _BACKENDS[normalized](load_runtime=False).get_capabilities()


def set_backend(name: str) -> Backend:
    """
    Set the default backend.

    Notes
    -----
    For Keras 3 multi-backend runtimes, the backend should ideally be set
    before importing ``keras``. If Keras is already loaded with a different
    runtime, a restart is recommended.
    """
    global _CURRENT_BACKEND

    normalized = normalize_backend_name(name)
    loaded_backend = _read_loaded_keras_backend()
    if loaded_backend and loaded_backend != normalized:
        warnings.warn(
            "Keras is already loaded with backend "
            f"'{loaded_backend}'. Restart Python after switching to "
            f"'{normalized}' for the change to take full effect.",
            RuntimeWarning,
            stacklevel=2,
        )

    _CURRENT_BACKEND = get_backend(normalized)
    os.environ["BASE_ATTENTIVE_BACKEND"] = normalized
    os.environ["KERAS_BACKEND"] = normalized
    return _CURRENT_BACKEND
