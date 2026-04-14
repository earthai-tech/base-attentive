"""Public package surface for :mod:`base_attentive`."""

from __future__ import annotations

import importlib

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _meta_version

    __version__ = _meta_version("base_attentive")
except PackageNotFoundError:
    __version__ = "unknown"

__author__ = "Laurent Kouadio"
__email__ = "etanoyau@gmail.com"
__license__ = "Apache-2.0"

__all__ = [
    "BaseAttentive",
    "KERAS_BACKEND",
    "KERAS_DEPS",
    "dependency_message",
    "get_backend",
    "set_backend",
    "get_available_backends",
    "get_backend_capabilities",
    "get_layer_class",
    "get_model_class",
    "register_keras_serializable",
    "resolve_keras_dep",
    "make_fast_predict_fn",
    "keras_runtime",
    "_KerasDeps",
    "_ORIGINAL_KERAS_DEPS_GETATTR",
]


for _cached_name in (
    "BaseAttentive",
    "make_fast_predict_fn",
    "keras_runtime",
    "get_backend",
    "set_backend",
    "get_available_backends",
    "get_backend_capabilities",
    "KERAS_BACKEND",
    "KERAS_DEPS",
    "dependency_message",
    "get_layer_class",
    "get_model_class",
    "register_keras_serializable",
    "resolve_keras_dep",
    "_KerasDeps",
    "_ORIGINAL_KERAS_DEPS_GETATTR",
):
    globals().pop(_cached_name, None)

_LAZY_EXPORTS = {
    "BaseAttentive": ("base_attentive.core.base_attentive", "BaseAttentive"),
    "make_fast_predict_fn": (
        "base_attentive.runtime",
        "make_fast_predict_fn",
    ),
    "keras_runtime": (
        "base_attentive.keras_runtime",
        None,
    ),
    "get_backend": (
        "base_attentive.backend",
        "get_backend",
    ),
    "set_backend": (
        "base_attentive.backend",
        "set_backend",
    ),
    "get_available_backends": (
        "base_attentive.backend",
        "get_available_backends",
    ),
    "get_backend_capabilities": (
        "base_attentive.backend",
        "get_backend_capabilities",
    ),
    "KERAS_BACKEND": (
        "base_attentive.keras_runtime",
        "KERAS_BACKEND",
    ),
    "KERAS_DEPS": (
        "base_attentive.keras_runtime",
        "KERAS_DEPS",
    ),
    "dependency_message": (
        "base_attentive.keras_runtime",
        "dependency_message",
    ),
    "get_layer_class": (
        "base_attentive.keras_runtime",
        "get_layer_class",
    ),
    "get_model_class": (
        "base_attentive.keras_runtime",
        "get_model_class",
    ),
    "register_keras_serializable": (
        "base_attentive.keras_runtime",
        "register_keras_serializable",
    ),
    "resolve_keras_dep": (
        "base_attentive.keras_runtime",
        "resolve_keras_dep",
    ),
    "_KerasDeps": (
        "base_attentive._bootstrap",
        "_KerasDeps",
    ),
    "_ORIGINAL_KERAS_DEPS_GETATTR": (
        "base_attentive._bootstrap",
        "_ORIGINAL_KERAS_DEPS_GETATTR",
    ),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module_name, export_name = target
    try:
        module = importlib.import_module(module_name)
        value = module if export_name is None else getattr(module, export_name)
    except Exception as exc:
        raise ImportError(
            f"Failed to import {name} from {module_name}: {exc}"
        ) from exc

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
