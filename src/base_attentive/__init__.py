"""
base_attentive: A foundational blueprint for sequence-to-
sequence time series forecasting models with attention
mechanisms.
"""

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

_LAZY_EXPORTS = {
    "BaseAttentive": (
        "base_attentive.core",
        "BaseAttentive",
    ),
    "KERAS_BACKEND": (
        "base_attentive._bootstrap",
        "KERAS_BACKEND",
    ),
    "KERAS_DEPS": (
        "base_attentive._bootstrap",
        "KERAS_DEPS",
    ),
    "_KerasDeps": (
        "base_attentive._bootstrap",
        "_KerasDeps",
    ),
    "_ORIGINAL_KERAS_DEPS_GETATTR": (
        "base_attentive._bootstrap",
        "_ORIGINAL_KERAS_DEPS_GETATTR",
    ),
    "dependency_message": (
        "base_attentive._bootstrap",
        "dependency_message",
    ),
    "get_available_backends": (
        "base_attentive.backend",
        "get_available_backends",
    ),
    "get_backend": (
        "base_attentive.backend",
        "get_backend",
    ),
    "get_backend_capabilities": (
        "base_attentive.backend",
        "get_backend_capabilities",
    ),
    "set_backend": (
        "base_attentive.backend",
        "set_backend",
    ),
    "make_fast_predict_fn": (
        "base_attentive.runtime",
        "make_fast_predict_fn",
    ),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, export_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc

    try:
        module = importlib.import_module(module_name)
        value = getattr(module, export_name)
    except Exception as exc:
        raise ImportError(
            f"Failed to import {name} from {module_name}: {exc}"
        ) from exc

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
