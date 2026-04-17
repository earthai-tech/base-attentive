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
import sys
from types import SimpleNamespace
from typing import Any

from ._runtime_requirements import (
    backend_install_command,
    backend_packages,
)

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
    "enable_eager_runtime_imports",
]


_ALLOW_EAGER_RUNTIME_IMPORTS = (
    os.environ.get("BASE_ATTENTIVE_EAGER_RUNTIME", "0") == "1"
)


def enable_eager_runtime_imports(
    enabled: bool = True,
) -> None:
    """Toggle eager imports of standalone Keras / TensorFlow runtime modules."""
    global _ALLOW_EAGER_RUNTIME_IMPORTS
    _ALLOW_EAGER_RUNTIME_IMPORTS = bool(enabled)
    os.environ["BASE_ATTENTIVE_EAGER_RUNTIME"] = (
        "1" if enabled else "0"
    )


def _runtime_imports_permitted() -> bool:
    """Return whether importing real runtime modules is allowed now.

    During pytest collection we prefer loaded-module lookups and fallback
    symbols instead of importing TensorFlow or Keras eagerly.
    """
    if "pytest" in sys.modules:
        return False
    return _ALLOW_EAGER_RUNTIME_IMPORTS


_BACKEND_ALIASES = {
    "tf": "tensorflow",
    "tensorflow": "tensorflow",
    "jax": "jax",
    "torch": "torch",
    "pytorch": "torch",
}


def _normalize_configured_backend(name: str | None) -> str | None:
    if name is None:
        return None
    normalized = str(name).strip().lower()
    if not normalized:
        return None
    if normalized == "auto":
        return "auto"
    if normalized == "keras":
        configured = os.environ.get("KERAS_BACKEND")
        return _normalize_configured_backend(configured)
    return _BACKEND_ALIASES.get(normalized, normalized)


def _resolve_runtime_backend() -> str | None:
    configured = os.environ.get("BASE_ATTENTIVE_BACKEND")
    normalized = _normalize_configured_backend(configured)
    if normalized is not None:
        return normalized

    configured = os.environ.get("KERAS_BACKEND")
    normalized = _normalize_configured_backend(configured)
    if normalized is not None:
        return normalized

    return None


def _auto_install_enabled() -> bool:
    return os.environ.get("BASE_ATTENTIVE_AUTO_INSTALL", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _configured_backend_display() -> str:
    configured = _resolve_runtime_backend()
    return configured or "<unset>"


def _backend_not_configured_message(module_name: str) -> str:
    return (
        f"BaseAttentive backend is not configured for {module_name}. "
        "Set BASE_ATTENTIVE_BACKEND to one of: tensorflow, torch, jax, or auto. "
        "Example: BASE_ATTENTIVE_BACKEND=torch. "
        "If you want deferred installation when a runtime is missing, set "
        "BASE_ATTENTIVE_AUTO_INSTALL=1."
    )


def _backend_missing_message(module_name: str, backend_name: str) -> str:
    packages = ", ".join(backend_packages(backend_name)) or backend_name
    install_cmd = backend_install_command(backend_name)
    auto_install_note = (
        " Automatic installation is enabled (BASE_ATTENTIVE_AUTO_INSTALL=1), "
        "so BaseAttentive will try to install it when runtime resolution is attempted."
        if _auto_install_enabled()
        else " Set BASE_ATTENTIVE_AUTO_INSTALL=1 to allow deferred installation when needed."
    )
    return (
        f"BaseAttentive backend '{backend_name}' is configured for {module_name}, "
        f"but its runtime is not installed ({packages}). Install it with: `{install_cmd}`."
        f"{auto_install_note}"
    )


KERAS_BACKEND = _resolve_runtime_backend() or ""


def _set_runtime_backend(name: str | None) -> str | None:
    global KERAS_BACKEND
    normalized = _normalize_configured_backend(name)
    KERAS_BACKEND = normalized or ""
    if normalized:
        os.environ["BASE_ATTENTIVE_BACKEND"] = normalized
        os.environ["KERAS_BACKEND"] = normalized
    else:
        os.environ.pop("BASE_ATTENTIVE_BACKEND", None)
        os.environ.pop("KERAS_BACKEND", None)
    return normalized


def ensure_runtime_backend(module_name: str) -> str:
    """Ensure a configured backend is available when runtime resolution is needed."""
    configured_backend = _resolve_runtime_backend()
    if configured_backend is None:
        raise ImportError(_backend_not_configured_message(module_name))

    detector = importlib.import_module("base_attentive.backend.detector")
    auto_install = _auto_install_enabled()

    if configured_backend == "auto":
        try:
            chosen = detector.ensure_default_backend(
                auto_install=auto_install,
                install_tensorflow=True,
            )
        except RuntimeError as exc:
            raise ImportError(
                dependency_message(module_name)
            ) from exc
        _set_runtime_backend(chosen)
        return chosen

    available = detector.get_available_backends()
    if configured_backend not in available:
        if auto_install:
            try:
                detector.install_backend_runtime(configured_backend)
            except RuntimeError as exc:
                raise ImportError(
                    _backend_missing_message(module_name, configured_backend)
                ) from exc
        else:
            raise ImportError(
                _backend_missing_message(module_name, configured_backend)
            )
    _set_runtime_backend(configured_backend)
    return configured_backend


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

    tensor = sys.modules.get("tensorflow")
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
        self._cache_state: tuple[Any, ...] | None = None

    def _current_state(self) -> tuple[Any, ...]:
        return (
            KERAS_BACKEND,
            id(sys.modules.get("keras")),
            id(sys.modules.get("tensorflow")),
        )

    def _maybe_reset_cache(self) -> None:
        state = self._current_state()
        if self._cache_state != state:
            self._cache.clear()
            self._cache_state = state

    def _load_fallback_runtime(self):
        if self._fallback_runtime is None:
            self._fallback_runtime = _safe_import(
                "base_attentive._keras_fallback"
            )
        return self._fallback_runtime

    def _load_keras_root(self):
        """Return standalone Keras only when already loaded or explicitly enabled."""
        loaded = sys.modules.get("keras")
        if loaded is not None:
            return loaded
        if _runtime_imports_permitted():
            return _safe_import("keras")
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
        loaded_root = (
            sys.modules.get(module_name)
            if module_name
            else None
        )
        # Only walk into submodules when ``root`` is the actual loaded module
        # object, not an arbitrary stand-in that merely advertises a ``__name__``.
        if (
            module_name
            and loaded_root is root
            and isinstance(root, type(sys))
        ):
            return _safe_import(f"{module_name}.{name}")
        return None

    def _load_tensorflow(self):
        if KERAS_BACKEND != "tensorflow":
            return None
        loaded = sys.modules.get("tensorflow")
        if loaded is not None:
            return loaded
        if _runtime_imports_permitted():
            return _safe_import("tensorflow")
        return None

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
                lambda condition,
                data=None,
                summarize=None,
                name=None: (condition),
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
        self._maybe_reset_cache()
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
            configured_backend = _resolve_runtime_backend()
            if configured_backend is None:
                raise ImportError(
                    _backend_not_configured_message("base_attentive runtime")
                )
            if configured_backend == "auto":
                raise ImportError(
                    "BaseAttentive backend is set to 'auto', but no suitable "
                    "runtime has been resolved yet. Install one of: "
                    "tensorflow keras, torch keras, or jax jaxlib keras; "
                    "or set BASE_ATTENTIVE_AUTO_INSTALL=1."
                )
            raise ImportError(
                _backend_missing_message(
                    "base_attentive runtime", configured_backend
                )
            )

        self._cache[name] = value
        return value


KERAS_DEPS = _KerasDeps()
_ORIGINAL_KERAS_DEPS_GETATTR = _KerasDeps.__getattr__


def dependency_message(module_name: str) -> str:
    """Return a dependency hint for missing runtime packages."""
    configured_backend = _resolve_runtime_backend()
    if configured_backend is None:
        return _backend_not_configured_message(module_name)
    if configured_backend == "auto":
        return (
            f"BaseAttentive backend is set to 'auto' for {module_name}. "
            "A runtime will be chosen on first use from the installed backends. "
            "If none is installed, install one of: `tensorflow keras`, `torch keras`, "
            "or `jax jaxlib keras`; or set BASE_ATTENTIVE_AUTO_INSTALL=1."
        )
    return _backend_missing_message(module_name, configured_backend)
