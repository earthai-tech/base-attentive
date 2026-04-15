"""Public Keras runtime helpers for Base-Attentive.

This module is the public entry point for backend-aware Keras runtime
access. Internal bootstrap details remain private in ``_bootstrap``.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "get_layer_class",
    "get_model_class",
    "register_keras_serializable",
    "resolve_keras_dep",
]


for _cached_name in (
    "KERAS_BACKEND",
    "KERAS_DEPS",
    "dependency_message",
):
    globals().pop(_cached_name, None)


def _bootstrap_module():
    return importlib.import_module(
        "base_attentive._bootstrap"
    )


def _internal_runtime_module():
    return importlib.import_module(
        "base_attentive._keras_runtime"
    )


def __getattr__(name: str) -> Any:
    if name in {
        "KERAS_BACKEND",
        "KERAS_DEPS",
        "dependency_message",
    }:
        value = getattr(_bootstrap_module(), name)
        globals()[name] = value
        return value
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}"
    )


def get_layer_class():
    return _internal_runtime_module().get_layer_class()


def get_model_class():
    return _internal_runtime_module().get_model_class()


def register_keras_serializable(
    package="Custom",
    name=None,
):
    return _internal_runtime_module().register_keras_serializable(
        package=package,
        name=name,
    )


def resolve_keras_dep(name: str, fallback: Any = None) -> Any:
    runtime = _internal_runtime_module()
    if fallback is None:
        return runtime.resolve_keras_dep(name)
    return runtime.resolve_keras_dep(name, fallback=fallback)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
