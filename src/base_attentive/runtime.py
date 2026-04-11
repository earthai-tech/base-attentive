"""Runtime helpers for accelerated inference."""

from __future__ import annotations

import importlib
from typing import Any

from . import KERAS_BACKEND


def _load_tensorflow():
    try:
        return importlib.import_module("tensorflow")
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to build a fast prediction function. "
            "Install TensorFlow and use the TensorFlow backend."
        ) from exc


def make_fast_predict_fn(
    model: Any,
    *,
    jit_compile: bool = True,
    reduce_retracing: bool = True,
    warmup_inputs: Any | None = None,
):
    """
    Create a TensorFlow-traced prediction function for a Keras model.

    The returned callable accepts the same input structure as ``model`` and
    always executes with ``training=False``. This is useful when you want
    a reusable inference function with ``tf.function`` tracing and optional
    XLA compilation.

    Parameters
    ----------
    model : Any
        A Keras-compatible model or layer that can be called as
        ``model(inputs, training=False)``.
    jit_compile : bool, default=True
        Whether to request XLA JIT compilation for the traced prediction
        function.
    reduce_retracing : bool, default=True
        Whether TensorFlow should reduce retracing when input structures
        are reused.
    warmup_inputs : Any, optional
        Example inputs used to trigger tracing before the callable is
        returned.

    Returns
    -------
    callable
        A TensorFlow ``tf.function``-wrapped prediction callable.

    Raises
    ------
    RuntimeError
        If the active package backend is not TensorFlow.
    ImportError
        If TensorFlow cannot be imported.
    """
    if KERAS_BACKEND != "tensorflow":
        raise RuntimeError(
            "make_fast_predict_fn requires the TensorFlow backend. "
            f"Current backend is {KERAS_BACKEND!r}. Set "
            "`KERAS_BACKEND=tensorflow` before importing base_attentive."
        )

    tf = _load_tensorflow()

    @tf.function(
        jit_compile=jit_compile,
        reduce_retracing=reduce_retracing,
    )
    def predict_fn(inputs):
        return model(inputs, training=False)

    if warmup_inputs is not None:
        predict_fn(warmup_inputs)

    return predict_fn


__all__ = ["make_fast_predict_fn"]
