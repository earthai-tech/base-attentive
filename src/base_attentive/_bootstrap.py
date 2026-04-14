"""
Internal runtime bootstrap for base_attentive.

This module centralizes runtime setup, backend resolution,
and lightweight compatibility helpers used across the package.

It is intentionally private and should not be treated as part of the
stable public API.
"""

from __future__ import annotations

import importlib
import os
from types import SimpleNamespace
from typing import Any

import numpy as np

__all__ = [
    "KERAS_BACKEND",
    "KERAS_DEPS",
    "_ORIGINAL_KERAS_DEPS_GETATTR",
    "dependency_message",
    "get_backend",
    "set_backend",
    "get_available_backends",
    "get_backend_capabilities",
]


_BACKEND_ALIASES = {
    "tf": "tensorflow",
    "tensorflow": "tensorflow",
    "jax": "jax",
    "torch": "torch",
    "pytorch": "torch",
}


def _normalize_configured_backend(name: str | None) -> str:
    if name is None:
        return "tensorflow"
    normalized = str(name).strip().lower()
    if not normalized:
        return "tensorflow"
    if normalized == "keras":
        configured = os.environ.get(
            "KERAS_BACKEND", "tensorflow"
        )
        return _normalize_configured_backend(configured)
    return _BACKEND_ALIASES.get(normalized, normalized)


def _resolve_runtime_backend() -> str:
    configured = os.environ.get("BASE_ATTENTIVE_BACKEND")
    if configured:
        return _normalize_configured_backend(configured)

    configured = os.environ.get("KERAS_BACKEND")
    if configured:
        return _normalize_configured_backend(configured)

    return "tensorflow"


KERAS_BACKEND = _resolve_runtime_backend()
os.environ.setdefault("BASE_ATTENTIVE_BACKEND", KERAS_BACKEND)
os.environ.setdefault("KERAS_BACKEND", KERAS_BACKEND)


def get_backend(name: str | None = None):
    backend = importlib.import_module(
        "base_attentive.backend"
    )
    return backend.get_backend(name)


def set_backend(name: str):
    backend = importlib.import_module(
        "base_attentive.backend"
    )
    return backend.set_backend(name)


def get_available_backends():
    backend = importlib.import_module(
        "base_attentive.backend"
    )
    return backend.get_available_backends()


def get_backend_capabilities(name: str | None = None):
    backend = importlib.import_module(
        "base_attentive.backend"
    )
    return backend.get_backend_capabilities(name)


def _safe_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
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
                result = get_static_value(value)
                if result is not value:
                    return result
            except Exception:
                pass
    return None


def _normalize_dtype(dtype: Any) -> Any:
    """Convert runtime dtypes into a Keras-friendly representation."""
    if dtype is None:
        return None
    if isinstance(dtype, str):
        return dtype

    name = getattr(dtype, "name", None)
    if isinstance(name, str):
        return name

    as_numpy_dtype = getattr(dtype, "as_numpy_dtype", None)
    if as_numpy_dtype is not None:
        try:
            return np.dtype(as_numpy_dtype).name
        except Exception:
            pass

    try:
        return np.dtype(dtype).name
    except Exception:
        return dtype


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
            raise AssertionError(
                message
                or f"{actual_value} != {expected_value}"
            )
        return None


class _KerasLinalgNamespace:
    @staticmethod
    def band_part(x, num_lower, num_upper):
        tf = _safe_import("tensorflow")
        if tf is not None:
            return tf.linalg.band_part(
                x, num_lower, num_upper
            )
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
        self._fallback_runtime = None

    def _load_fallback_runtime(self):
        if self._fallback_runtime is None:
            self._fallback_runtime = _safe_import(
                "base_attentive._keras_fallback"
            )
        return self._fallback_runtime

    def _load_keras_root(self):
        """Prefer standalone Keras and avoid importing ``tf.keras`` directly.

        On TensorFlow-backed Keras 3 installations, traversing ``tf.keras`` can
        recurse through TensorFlow's lazy-loader bridge on some Windows setups.
        The standalone ``keras`` package is the primary API surface for this
        project, so we deliberately avoid using ``tensorflow.keras`` as a
        secondary namespace here.
        """
        return _safe_import("keras")

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
        fallback = self._load_fallback_runtime()
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
            return getattr(
                fallback,
                "Assert",
                lambda condition, data=None, summarize=None, name=None: (
                    condition
                ),
            )
        if name == "Tensor":
            tf = self._load_tensorflow()
            if tf:
                return getattr(tf, "Tensor", object)
            return getattr(fallback, "Tensor", object)
        if name == "TensorShape":
            tf = self._load_tensorflow()
            if tf:
                return getattr(tf, "TensorShape", tuple)
            return getattr(fallback, "TensorShape", tuple)
        if name == "Reduction":
            keras = self._load_keras_root()
            losses = self._load_namespace(keras, "losses")
            reduction = getattr(losses, "Reduction", None)
            if reduction is not None:
                return reduction
            return getattr(
                fallback,
                "Reduction",
                SimpleNamespace(
                    AUTO="auto", SUM="sum", NONE="none"
                ),
            )
        if name == "get_static_value":
            return _get_static_value
        if name == "linalg":
            tf = self._load_tensorflow()
            if tf is not None and hasattr(tf, "linalg"):
                return tf.linalg
            return getattr(
                fallback,
                "linalg",
                _KerasLinalgNamespace(),
            )
        return None

    def _resolve_from_keras(self, name: str) -> Any:
        keras = self._load_keras_root()
        if keras is None:
            return None

        if name == "register_keras_serializable":
            for namespace_name in ("saving", "utils"):
                namespace = self._load_namespace(
                    keras,
                    namespace_name,
                )
                value = getattr(
                    namespace,
                    "register_keras_serializable",
                    None,
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
        ops = self._load_namespace(keras, "ops")

        if name == "constant":
            convert_to_tensor = getattr(
                ops,
                "convert_to_tensor",
                None,
            )
            if callable(convert_to_tensor):

                def _constant(value, dtype=None):
                    normalized = _normalize_dtype(dtype)
                    if normalized is None:
                        return convert_to_tensor(value)
                    return convert_to_tensor(
                        value, dtype=normalized
                    )

                return _constant

        if name == "cast":
            cast = getattr(ops, "cast", None)
            if callable(cast):
                return lambda value, dtype, **kwargs: cast(
                    value,
                    _normalize_dtype(dtype),
                )

        if ops is not None:
            value = getattr(ops, target_name, None)
            if value is not None:
                return value

        for namespace_name in self._SEARCH_PATHS:
            namespace = self._load_namespace(
                keras, namespace_name
            )
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

        # Avoid walking through ``tf.keras`` here. When TensorFlow exposes Keras
        # via its internal lazy loader, attribute access can recurse while it is
        # deciding whether to bridge to Keras 3. For model/layer namespaces we
        # rely on standalone ``keras`` above; TensorFlow is only used here for
        # root-level TF ops and dtypes.
        return None

    def _resolve_from_fallback(self, name: str) -> Any:
        fallback = self._load_fallback_runtime()
        if fallback is None:
            return None

        namespace_map = {
            "activations": "activations",
            "random": "random",
            "register_keras_serializable": "register_keras_serializable",
            "get": "get",
        }
        target_name = namespace_map.get(
            name,
            self._OP_ALIASES.get(name, name),
        )
        return getattr(fallback, target_name, None)

    def __getattr__(self, name: str) -> Any:
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
            value = self._resolve_from_fallback(name)
        if value is None:
            raise ImportError(
                f"Cannot import {name} from Keras runtime "
                f"{KERAS_BACKEND!r}."
            )

        self._cache[name] = value
        return value


KERAS_DEPS = _KerasDeps()
_ORIGINAL_KERAS_DEPS_GETATTR = _KerasDeps.__getattr__


def dependency_message(module_name: str) -> str:
    """Return a dependency hint for missing runtime packages."""
    return (
        f"Keras is required for {module_name}. "
        f"Install a runtime such as `tensorflow`, "
        f"`keras jax jaxlib`, or `keras torch`."
    )
