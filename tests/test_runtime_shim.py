"""Tests for the lightweight Keras runtime shim."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np


def test_keras_deps_resolves_symbols_from_standalone_keras(
    monkeypatch,
):
    """The top-level shim should resolve layers, ops, and saving helpers."""
    fake_dense = object()
    fake_layer = object()
    fake_model = object()
    fake_sequential = object()

    def register_keras_serializable(*args, **kwargs):
        def decorator(obj):
            return obj

        return decorator

    fake_ops = types.SimpleNamespace(
        concatenate="concat-op",
        mean="mean-op",
        sum="sum-op",
        max="max-op",
        arange="range-op",
        expand_dims="expand-dims-op",
        cast="cast-op",
        convert_to_tensor="convert-op",
    )
    fake_random = types.SimpleNamespace(
        uniform="uniform-rng",
        normal="normal-rng",
    )
    fake_keras = types.ModuleType("keras")
    fake_keras.backend = types.SimpleNamespace(
        backend=lambda: "jax"
    )
    fake_keras.layers = types.SimpleNamespace(
        Dense=fake_dense,
        Layer=fake_layer,
        Add=object(),
    )
    fake_keras.losses = types.SimpleNamespace(
        Loss=object(),
        get=lambda config: ("loss", config),
        Reduction=types.SimpleNamespace(
            AUTO="auto",
            SUM="sum",
            NONE="none",
        ),
    )
    fake_keras.initializers = types.SimpleNamespace(
        Constant=object()
    )
    fake_keras.saving = types.SimpleNamespace(
        register_keras_serializable=register_keras_serializable
    )
    fake_keras.activations = object()
    fake_keras.ops = fake_ops
    fake_keras.random = fake_random
    fake_keras.Model = fake_model
    fake_keras.Sequential = fake_sequential

    monkeypatch.setenv("KERAS_BACKEND", "jax")
    monkeypatch.delenv(
        "BASE_ATTENTIVE_BACKEND", raising=False
    )
    monkeypatch.setitem(sys.modules, "keras", fake_keras)
    monkeypatch.delitem(
        sys.modules, "tensorflow", raising=False
    )
    monkeypatch.delitem(
        sys.modules, "base_attentive", raising=False
    )

    import base_attentive

    importlib.reload(base_attentive)

    assert base_attentive.KERAS_DEPS.Dense is fake_dense
    assert base_attentive.KERAS_DEPS.Layer is fake_layer
    assert base_attentive.KERAS_DEPS.Model is fake_model
    assert (
        base_attentive.KERAS_DEPS.Sequential
        is fake_sequential
    )
    assert base_attentive.KERAS_DEPS.concat == "concat-op"
    assert base_attentive.KERAS_DEPS.reduce_mean == "mean-op"
    assert base_attentive.KERAS_DEPS.reduce_sum == "sum-op"
    assert base_attentive.KERAS_DEPS.reduce_max == "max-op"
    assert base_attentive.KERAS_DEPS.range == "range-op"
    assert base_attentive.KERAS_DEPS.random is fake_random
    assert (
        base_attentive.KERAS_DEPS.register_keras_serializable
        is register_keras_serializable
    )
    assert base_attentive.KERAS_DEPS.float32 is np.float32
    assert base_attentive.KERAS_DEPS.int32 is np.int32
    assert base_attentive.KERAS_DEPS.newaxis is None
    assert (
        base_attentive.dependency_message("models")
        == "Keras is required for models. Install a runtime such as "
        "`tensorflow`, `keras jax jaxlib`, or `keras torch`."
    )

    base_attentive.KERAS_DEPS.debugging.assert_equal(3, 3)
