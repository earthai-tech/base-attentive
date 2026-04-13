"""TensorFlow compatibility layer."""

from __future__ import annotations

import importlib
import importlib.util
import os
import warnings

HAS_TF = importlib.util.find_spec("tensorflow") is not None
tf = None


def _import_tensorflow():
    """Import TensorFlow lazily to avoid heavy import side effects at module load."""
    global tf
    if tf is not None:
        return tf
    try:
        tf = importlib.import_module("tensorflow")
    except Exception:
        tf = None
    return tf


class TFConfig:
    """TensorFlow configuration helper."""

    def __init__(self):
        self.compat_ndim_enabled = False


def standalone_keras(module_name: str):
    """
    Import module from tensorflow.keras or standalone keras.

    Parameters
    ----------
    module_name : str
        The name of the module to import (e.g., 'activations', 'layers', etc.).

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    ImportError
        If neither tensorflow.keras nor standalone keras is installed.
    """
    try:
        # Try importing from tensorflow.keras
        tf_keras = importlib.import_module("tensorflow.keras")

        return getattr(tf_keras, module_name)
    except (ImportError, AttributeError, ModuleNotFoundError):
        try:
            # Fallback to standalone keras
            keras = importlib.import_module("keras")

            return getattr(keras, module_name)
        except (ImportError, AttributeError, ModuleNotFoundError):
            raise ImportError(
                f"Module '{module_name}' could not be imported from either "
                f"tensorflow.keras or standalone keras. Ensure that TensorFlow "
                f"or standalone Keras is installed and the module exists."
            ) from None


def suppress_tf_warnings():
    """Suppress TensorFlow warnings."""
    tensorflow = _import_tensorflow()
    if HAS_TF and tensorflow is not None:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        tensorflow.get_logger().setLevel("ERROR")
        warnings.filterwarnings("ignore", category=DeprecationWarning)


def optional_tf_function(*args, **kwargs):
    """Optional tf.function decorator that works even if TF is not available."""

    def decorator(func):
        tensorflow = _import_tensorflow()
        if HAS_TF and tensorflow is not None:
            return tensorflow.function(*args, **kwargs)(func)
        return func

    return (
        decorator
        if callable(args[0].__class__ if args else None) or not args
        else decorator
    )


def tf_debugging_assert_equal(x, y, message="", name="assert_equal"):
    """TensorFlow assert_equal wrapper."""
    tensorflow = _import_tensorflow()
    if HAS_TF and tensorflow is not None:
        return tensorflow.debugging.assert_equal(
            x, y, message=message, name=name
        )
    return None


__all__ = [
    "HAS_TF",
    "TFConfig",
    "standalone_keras",
    "suppress_tf_warnings",
    "optional_tf_function",
    "tf_debugging_assert_equal",
]
