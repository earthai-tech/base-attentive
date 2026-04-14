"""TensorFlow compatibility layer.

This module now keeps only TensorFlow-specific shims. Backend-neutral
Keras imports live in :mod:`base_attentive.compat.keras`.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import warnings

from .keras import import_keras_attr

HAS_TF = importlib.util.find_spec("tensorflow") is not None
tf = None
_tf = None


def _import_tensorflow():
    """Import TensorFlow lazily to avoid heavy import side effects."""
    global tf, _tf
    if tf is not None:
        return tf
    try:
        tf = importlib.import_module("tensorflow")
    except Exception:
        tf = None
    _tf = tf
    return tf


class TFConfig:
    """TensorFlow configuration helper."""

    def __init__(self):
        self.compat_ndim_enabled = False


def standalone_keras(module_name: str):
    """Backward-compatible alias for backend-neutral Keras imports."""
    return import_keras_attr(module_name)


def suppress_tf_warnings():
    """Suppress TensorFlow warnings when TensorFlow is active."""
    tensorflow = _import_tensorflow()
    if HAS_TF and tensorflow is not None:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        tensorflow.get_logger().setLevel("ERROR")
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning
        )


def optional_tf_function(*args, **kwargs):
    """Optional ``tf.function`` decorator that degrades gracefully."""

    if (
        args
        and callable(args[0])
        and len(args) == 1
        and not kwargs
    ):
        func = args[0]
        tensorflow = _import_tensorflow()
        if HAS_TF and tensorflow is not None:
            return tensorflow.function(func)
        return func

    def decorator(func):
        tensorflow = _import_tensorflow()
        if HAS_TF and tensorflow is not None:
            return tensorflow.function(*args, **kwargs)(func)
        return func

    return decorator


def tf_debugging_assert_equal(
    x, y, message="", name="assert_equal"
):
    """TensorFlow ``assert_equal`` wrapper."""
    tensorflow = _import_tensorflow()
    if HAS_TF and tensorflow is not None:
        return tensorflow.debugging.assert_equal(
            x, y, message=message, name=name
        )
    return None


__all__ = [
    "HAS_TF",
    "tf",
    "TFConfig",
    "standalone_keras",
    "suppress_tf_warnings",
    "optional_tf_function",
    "tf_debugging_assert_equal",
]
