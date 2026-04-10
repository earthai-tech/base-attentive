"""
base_attentive: A foundational blueprint for sequence-to-sequence
time series forecasting models with attention mechanisms.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "LKouadio"
__email__ = "etanoyau@gmail.com"
__license__ = "Apache-2.0"

# Initialize Keras backend and dependencies
import importlib
import os
from types import SimpleNamespace
from typing import Any

import numpy as np

# Resolve Keras backend


def _normalize_configured_backend(name: str | None) -> str:
    normalized = (name or "tensorflow").strip().lower()
    if normalized == "pytorch":
        return "torch"
    return normalized or "tensorflow"


KERAS_BACKEND = _normalize_configured_backend(
    os.environ.get("KERAS_BACKEND")
    or os.environ.get("BASE_ATTENTIVE_BACKEND")
)


def _safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


def _resolve_scalar(value: Any) -> Any:
    if isinstance(value, (int, float, bool, str)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return getattr(value, "value", None)


def _get_static_value(value: Any) -> Any:
    scalar = _resolve_scalar(value)
    if scalar is not None:
        return scalar

    tensor = _safe_import("tensorflow")
    if tensor is not None:
        get_static_value = getattr(
            tensor, "get_static_value", None
        )
        if callable(get_static_value):
            try:
                return get_static_value(value)
            except Exception:
                return None
    return None


class _KerasAutographExperimental:
    @staticmethod
    def do_not_convert(func=None, **kwargs):
        if func is None:
            def decorator(inner):
                return inner

            return decorator
        return func


class _KerasAutographNamespace:
    experimental = _KerasAutographExperimental()


class _KerasDebuggingNamespace:
    @staticmethod
    def assert_equal(actual, expected, message="", name=None):
        actual_value = _get_static_value(actual)
        expected_value = _get_static_value(expected)
        if (
            actual_value is not None
            and expected_value is not None
            and actual_value != expected_value
        ):
            raise AssertionError(message or f"{actual_value} != {expected_value}")
        return None


class _KerasLinalgNamespace:
    @staticmethod
    def band_part(x, num_lower, num_upper):
        tf = _safe_import("tensorflow")
        if tf is not None:
            return tf.linalg.band_part(x, num_lower, num_upper)
        raise ImportError(
            "linalg.band_part is only available with TensorFlow installed."
        )


class _KerasDeps:
    """Resolve Keras symbols across Keras 3 and TensorFlow namespaces."""

    _OP_ALIASES = {
        "concat": "concatenate",
        "floordiv": "floor_divide",
        "reduce_mean": "mean",
        "reduce_sum": "sum",
        "reduce_max": "max",
        "range": "arange",
    }

    _SEARCH_PATHS = (
        None,
        "layers",
        "losses",
        "activations",
        "initializers",
        "models",
        "ops",
        "random",
        "saving",
        "utils",
    )

    def __init__(self):
        self._cache: dict[str, Any] = {}

    def _load_keras_root(self):
        keras = _safe_import("keras")
        if keras is not None:
            return keras
        if KERAS_BACKEND == "tensorflow":
            return _safe_import("tensorflow.keras")
        return None

    def _load_namespace(self, root: Any, name: str | None):
        if root is None:
            return None
        if name is None:
            return root

        namespace = getattr(root, name, None)
        if namespace is not None:
            return namespace

        module_name = getattr(root, "__name__", None)
        if module_name:
            return _safe_import(f"{module_name}.{name}")
        return None

    def _load_tensorflow(self):
        if KERAS_BACKEND != "tensorflow":
            return None
        return _safe_import("tensorflow")

    def _resolve_special(self, name: str) -> Any:
        if name == "autograph":
            return _KerasAutographNamespace()
        if name == "debugging":
            tf = self._load_tensorflow()
            if tf is not None and hasattr(tf, "debugging"):
                return tf.debugging
            return _KerasDebuggingNamespace()
        if name == "newaxis":
            return None
        if name == "bool":
            return np.bool_
        if name == "float32":
            return np.float32
        if name == "int32":
            return np.int32
        if name == "Assert":
            tf = self._load_tensorflow()
            if tf is not None and hasattr(tf, "Assert"):
                return tf.Assert
            return lambda condition, data=None, summarize=None, name=None: condition
        if name == "Tensor":
            tf = self._load_tensorflow()
            return getattr(tf, "Tensor", object) if tf else object
        if name == "TensorShape":
            tf = self._load_tensorflow()
            return getattr(tf, "TensorShape", tuple) if tf else tuple
        if name == "Reduction":
            keras = self._load_keras_root()
            losses = self._load_namespace(keras, "losses")
            reduction = getattr(losses, "Reduction", None)
            if reduction is not None:
                return reduction
            return SimpleNamespace(AUTO="auto", SUM="sum", NONE="none")
        if name == "get_static_value":
            return _get_static_value
        if name == "linalg":
            tf = self._load_tensorflow()
            if tf is not None and hasattr(tf, "linalg"):
                return tf.linalg
            return _KerasLinalgNamespace()
        return None

    def _resolve_from_keras(self, name: str) -> Any:
        keras = self._load_keras_root()
        if keras is None:
            return None

        if name == "register_keras_serializable":
            for namespace_name in ("saving", "utils"):
                namespace = self._load_namespace(keras, namespace_name)
                value = getattr(
                    namespace, "register_keras_serializable", None
                )
                if value is not None:
                    return value

        if name == "get":
            losses = self._load_namespace(keras, "losses")
            value = getattr(losses, "get", None)
            if value is not None:
                return value

        if name == "activations":
            return self._load_namespace(keras, "activations")
        if name == "random":
            return self._load_namespace(keras, "random")

        target_name = self._OP_ALIASES.get(name, name)
        for namespace_name in self._SEARCH_PATHS:
            namespace = self._load_namespace(keras, namespace_name)
            if namespace is None:
                continue
            value = getattr(namespace, target_name, None)
            if value is not None:
                return value
        return None

    def _resolve_from_tensorflow(self, name: str) -> Any:
        tf = self._load_tensorflow()
        if tf is None:
            return None

        target_name = self._OP_ALIASES.get(name, name)
        if hasattr(tf, target_name):
            return getattr(tf, target_name)

        keras = getattr(tf, "keras", None)
        if keras is not None:
            for namespace_name in ("layers", "losses", "initializers", "utils"):
                namespace = getattr(keras, namespace_name, None)
                if namespace is None:
                    continue
                value = getattr(namespace, target_name, None)
                if value is not None:
                    return value
            if hasattr(keras, target_name):
                return getattr(keras, target_name)
        return None

    def __getattr__(self, name: str) -> Any:
        """Lazy load Keras/TensorFlow symbols as needed."""
        if name in self._cache:
            return self._cache[name]
        if name == "newaxis":
            self._cache[name] = None
            return None

        value = self._resolve_special(name)
        if value is None:
            value = self._resolve_from_keras(name)
        if value is None:
            value = self._resolve_from_tensorflow(name)
        if value is None:
            raise ImportError(
                f"Cannot import {name} from Keras runtime '{KERAS_BACKEND}'."
            )

        self._cache[name] = value
        return value


KERAS_DEPS = _KerasDeps()


def dependency_message(module_name: str) -> str:
    """Return a dependency message for missing packages."""
    return (
        f"Keras is required for {module_name}. "
        f"Install a runtime such as `tensorflow`, `keras jax jaxlib`, "
        f"or `keras torch`."
    )


# Import backend utilities for framework flexibility
from .backend import (
    get_backend,
    set_backend,
    get_available_backends,
    get_backend_capabilities,
)

__all__ = [
    "BaseAttentive",
    "KERAS_BACKEND",
    "KERAS_DEPS",
    "dependency_message",
    "get_backend",
    "set_backend",
    "get_available_backends",
    "get_backend_capabilities",
]


def __getattr__(name: str):
    """Lazily expose heavy package exports."""
    if name != "BaseAttentive":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from base_attentive.core import BaseAttentive as _BaseAttentive
    except Exception as exc:
        raise AttributeError(name) from exc

    globals()["BaseAttentive"] = _BaseAttentive
    return _BaseAttentive
