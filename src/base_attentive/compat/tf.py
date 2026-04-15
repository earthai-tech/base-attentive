"""TensorFlow compatibility layer.

This module keeps only TensorFlow-specific shims. All helpers are lazy and
will not import TensorFlow unless it has already been imported elsewhere.
That keeps lightweight paths safe on platforms where TensorFlow import is
slow or unstable.
"""

from __future__ import annotations

import os
import sys
import warnings

from .keras import import_keras_attr

HAS_TF = "tensorflow" in sys.modules
tf = sys.modules.get("tensorflow")
_tf = tf


def _import_tensorflow():
    """Return the active TensorFlow module without importing it.

    The compatibility helpers are governed by the module-level ``HAS_TF`` flag.
    When callers set ``HAS_TF = False`` we must *not* reach into a real
    TensorFlow module that may already be present in ``sys.modules``.
    """
    global tf, _tf
    if not HAS_TF:
        return None
    if tf is not None:
        _tf = tf
        return tf

    module = sys.modules.get("tensorflow")
    if module is not None:
        tf = module
        _tf = module
        return module
    return None


class TFConfig:
    """TensorFlow configuration helper."""

    def __init__(self):
        self.compat_ndim_enabled = False


def standalone_keras(module_name: str):
    """Backward-compatible alias for backend-neutral Keras imports."""
    return import_keras_attr(module_name)


def suppress_tf_warnings():
    """Suppress TensorFlow warnings when TensorFlow is already active."""
    if not HAS_TF and "tensorflow" not in sys.modules:
        return None
    tensorflow = _import_tensorflow()
    if tensorflow is not None:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        tensorflow.get_logger().setLevel("ERROR")
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning
        )
    return None


def optional_tf_function(*args, **kwargs):
    """Optional ``tf.function`` decorator that degrades gracefully."""

    def _decorate(func):
        tensorflow = _import_tensorflow()
        if tensorflow is not None:
            return tensorflow.function(*args, **kwargs)(func)
        return func

    if (
        args
        and callable(args[0])
        and len(args) == 1
        and not kwargs
    ):
        func = args[0]
        tensorflow = _import_tensorflow()
        if tensorflow is not None:
            return tensorflow.function(func)
        return func

    return _decorate


def tf_debugging_assert_equal(
    x, y, message="", name="assert_equal"
):
    """TensorFlow ``assert_equal`` wrapper."""
    if not HAS_TF and "tensorflow" not in sys.modules:
        return None
    tensorflow = _import_tensorflow()
    if tensorflow is not None:
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
