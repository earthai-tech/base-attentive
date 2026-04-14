from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest


class _DummyDense:
    def __init__(
        self, units, activation=None, name=None, **kwargs
    ):
        self.units = units
        self.activation = activation
        self.name = name
        self.kwargs = kwargs


class _DummyTFEncoder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyLayer:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.kwargs = kwargs


class _DummyMultiHeadAttention(_DummyLayer):
    def __init__(
        self, num_heads, key_dim, dropout=0.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout


class _DummyLayerNorm(_DummyLayer):
    def __init__(self, epsilon=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon


class _DummyDropout(_DummyLayer):
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate


class _DummyLambda(_DummyLayer):
    def __init__(self, function, **kwargs):
        super().__init__(**kwargs)
        self.function = function


@pytest.fixture
def mod(monkeypatch):
    fake_layers = SimpleNamespace(
        Layer=_DummyLayer,
        Dense=_DummyDense,
        MultiHeadAttention=_DummyMultiHeadAttention,
        LayerNormalization=_DummyLayerNorm,
        Dropout=_DummyDropout,
        Lambda=_DummyLambda,
    )
    fake_tf = types.ModuleType("tensorflow")
    fake_tf_keras = types.ModuleType("tensorflow.keras")
    fake_tf_keras.layers = fake_layers
    fake_tf.keras = fake_tf_keras

    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setitem(
        sys.modules, "tensorflow.keras", fake_tf_keras
    )

    import importlib

    module = importlib.import_module(
        "base_attentive.implementations.tensorflow.base_attentive_v2"
    )
    return importlib.reload(module)


def test_tf_dense_projection_keeps_kwargs(mod):
    layer = mod._build_tf_dense_projection(
        units=16,
        activation="relu",
        name="proj",
        use_bias=False,
        dtype="float32",
    )

    assert layer.units == 16
    assert layer.activation == "relu"
    assert layer.name == "proj"
    assert layer.kwargs["use_bias"] is False
    assert layer.kwargs["dtype"] == "float32"


def test_tf_temporal_encoder_keeps_kwargs(mod, monkeypatch):
    monkeypatch.setattr(
        mod,
        "_TFTemporalSelfAttentionEncoder",
        _DummyTFEncoder,
    )

    layer = mod._build_tf_temporal_self_attention_encoder(
        units=32,
        hidden_units=64,
        num_heads=4,
        activation="gelu",
        dropout_rate=0.1,
        layer_norm_epsilon=1e-5,
        name="encoder",
        trainable=False,
        dtype="float32",
    )

    assert layer.kwargs["units"] == 32
    assert layer.kwargs["hidden_units"] == 64
    assert layer.kwargs["num_heads"] == 4
    assert layer.kwargs["activation"] == "gelu"
    assert layer.kwargs["dropout_rate"] == 0.1
    assert layer.kwargs["layer_norm_epsilon"] == 1e-5
    assert layer.kwargs["name"] == "encoder"
    assert layer.kwargs["trainable"] is False
    assert layer.kwargs["dtype"] == "float32"
