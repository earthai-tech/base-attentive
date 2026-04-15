"""Lazy runtime accessors for Keras-backed symbols.

This module provides a narrow layer above ``KERAS_DEPS`` so package
modules can avoid eagerly freezing backend symbols at import time.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from ._keras_fallback import (
    register_keras_serializable as _fallback_register,
)

_MISSING = object()


def _bootstrap_module():
    return importlib.import_module(
        "base_attentive._bootstrap"
    )


def _keras_deps():
    return _bootstrap_module().KERAS_DEPS


class _CompatLayer:
    """Lightweight fallback used by tests and fallback runtimes."""

    def __init__(self, *args, name=None, **kwargs):
        del args, kwargs
        self.name = name
        self.built = False

    def build(self, input_shape=None):
        del input_shape
        self.built = True

    def add_weight(
        self,
        name=None,
        shape=None,
        dtype=None,
        initializer=None,
        trainable=True,
    ):
        del trainable
        if callable(initializer):
            try:
                value = initializer(shape, dtype=dtype)
            except TypeError:
                value = initializer(shape)
        else:
            value = np.zeros(shape, dtype=dtype)
        weight = np.asarray(value, dtype=dtype)
        if name:
            setattr(self, name, weight)
        return weight

    def get_config(self):
        return {}

    def __call__(self, *args, **kwargs):
        if not getattr(self, "built", False):
            input_shape = None
            if args:
                input_shape = getattr(args[0], "shape", None)
            try:
                self.build(input_shape)
            except TypeError:
                self.build()
        call = getattr(self, "call", None)
        if callable(call):
            return call(*args, **kwargs)
        if args:
            return args[0]
        return None


def resolve_keras_dep(
    name: str,
    fallback: Any = _MISSING,
) -> Any:
    """Resolve a Keras runtime symbol lazily."""
    try:
        return getattr(_keras_deps(), name)
    except ImportError:
        if fallback is not _MISSING:
            return fallback
        raise


def get_layer_class():
    """Return the active Keras ``Layer`` base class."""
    layer = resolve_keras_dep("Layer", fallback=_CompatLayer)
    if layer is object:
        return _CompatLayer
    return layer


def get_model_class():
    """Return the active Keras ``Model`` base class."""
    return resolve_keras_dep("Model")


def register_keras_serializable(
    package="Custom",
    name=None,
):
    """Return a runtime-aware serializable decorator."""
    register = resolve_keras_dep(
        "register_keras_serializable",
        fallback=_fallback_register,
    )
    return register(package, name=name)


__all__ = [
    "get_layer_class",
    "get_model_class",
    "register_keras_serializable",
    "resolve_keras_dep",
]
