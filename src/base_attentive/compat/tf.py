"""TensorFlow compatibility layer."""

from __future__ import annotations

import os
import warnings

# Check if TensorFlow is available
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    tf = None


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
        import tensorflow.keras as tf_keras
        return getattr(tf_keras, module_name)
    except (ImportError, AttributeError):
        try:
            # Fallback to standalone keras
            import keras
            return getattr(keras, module_name)
        except (ImportError, AttributeError):
            raise ImportError(
                f"Module '{module_name}' could not be imported from either "
                f"tensorflow.keras or standalone keras. Ensure that TensorFlow "
                f"or standalone Keras is installed and the module exists."
            )


def suppress_tf_warnings():
    """Suppress TensorFlow warnings."""
    if HAS_TF:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        tf.get_logger().setLevel("ERROR")
        warnings.filterwarnings("ignore", category=DeprecationWarning)


def optional_tf_function(*args, **kwargs):
    """Optional tf.function decorator that works even if TF is not available."""
    def decorator(func):
        if HAS_TF:
            return tf.function(*args, **kwargs)(func)
        return func
    return decorator if callable(args[0].__class__ if args else None) or not args else decorator


def tf_debugging_assert_equal(x, y, message="", name="assert_equal"):
    """TensorFlow assert_equal wrapper."""
    if HAS_TF:
        return tf.debugging.assert_equal(x, y, message=message, name=name)
    return None


__all__ = [
    "HAS_TF",
    "TFConfig",
    "standalone_keras",
    "suppress_tf_warnings",
    "optional_tf_function",
    "tf_debugging_assert_equal",
]
