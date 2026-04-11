"""Focused tests for the V2 scaffold."""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np

from base_attentive.config import normalize_base_attentive_spec
from base_attentive.registry import ComponentRegistry, get_backend_capability_report


def test_normalize_base_attentive_spec_applies_defaults_and_aliases():
    """Spec normalization should merge defaults and normalize aliases."""
    spec = normalize_base_attentive_spec(
        {
            "static_input_dim": 0,
            "dynamic_input_dim": 8,
            "future_input_dim": 4,
            "backend_name": "pytorch",
            "architecture": {
                "sequence_pooling": "pool.mean",
                "fusion": "fusion.concat",
            },
            "custom_flag": True,
        }
    )

    assert spec.backend_name == "torch"
    assert spec.output_dim == 1
    assert spec.components.sequence_pooling == "pool.mean"
    assert spec.components.fusion == "fusion.concat"
    assert spec.extras["custom_flag"] is True


def test_normalize_base_attentive_spec_quantiles_are_tupled():
    """Quantiles should normalize to a tuple for stable config handling."""
    spec = normalize_base_attentive_spec(
        {
            "static_input_dim": 1,
            "dynamic_input_dim": 8,
            "future_input_dim": 2,
            "head_type": "quantile",
            "quantiles": [0.1, 0.5, 0.9],
        }
    )

    assert spec.head_type == "quantile"
    assert spec.quantiles == (0.1, 0.5, 0.9)


def test_normalize_base_attentive_spec_adds_temporal_encoder_defaults():
    """V2 specs should include encoder defaults for the sequence path."""
    spec = normalize_base_attentive_spec(
        {
            "static_input_dim": 1,
            "dynamic_input_dim": 8,
            "future_input_dim": 2,
        }
    )

    assert spec.attention_heads == 4
    assert spec.components.dynamic_encoder == "encoder.temporal_self_attention"
    assert spec.components.future_encoder == "encoder.temporal_self_attention"


def test_backend_capability_report_exposes_v2_support():
    """Capability reporting should distinguish legacy and V2 support."""
    report = get_backend_capability_report("torch")

    assert report.name == "torch"
    assert report.supports_base_attentive is False
    assert report.supports_base_attentive_v2 is True
    assert report.v2_blockers


def test_component_registry_prefers_backend_specific_registration():
    """Registry resolution should prefer exact backend matches."""
    registry = ComponentRegistry()

    registry.register("head.point_forecast", lambda **_: "generic", backend="generic")
    registry.register("head.point_forecast", lambda **_: "torch", backend="torch")

    assert (
        registry.resolve("head.point_forecast", backend="torch").backend == "torch"
    )
    assert (
        registry.resolve("head.point_forecast", backend="tensorflow").backend
        == "generic"
    )


def test_import_and_run_base_attentive_v2_with_fake_keras_runtime(monkeypatch):
    """The experimental V2 model should run on a lightweight fake Keras runtime."""

    class FakeModel:
        def __init__(self, name=None, **kwargs):
            self.name = name
            self.kwargs = kwargs

        def get_config(self):
            return {"name": self.name}

    class FakeDense:
        def __init__(self, units, activation=None, name=None):
            self.units = units
            self.activation = activation
            self.name = name

        def __call__(self, inputs):
            array = np.asarray(inputs)
            output_shape = array.shape[:-1] + (self.units,)
            return np.ones(output_shape, dtype=np.float32)

    class FakeDropout:
        def __init__(self, rate, name=None):
            self.rate = rate
            self.name = name

        def __call__(self, inputs, training=False):
            del training
            return inputs

    class FakeLayerNormalization:
        def __init__(self, epsilon=1e-6, name=None):
            self.epsilon = epsilon
            self.name = name

        def __call__(self, inputs):
            return np.asarray(inputs, dtype=np.float32)

    class FakeMultiHeadAttention:
        def __init__(self, num_heads, key_dim, dropout=0.0, name=None):
            self.num_heads = num_heads
            self.key_dim = key_dim
            self.dropout = dropout
            self.name = name

        def __call__(self, query, value, training=False):
            del value, training
            return np.asarray(query, dtype=np.float32)

    def register_keras_serializable(*args, **kwargs):
        del args, kwargs

        def decorator(obj):
            return obj

        return decorator

    fake_keras = types.ModuleType("keras")
    fake_keras.backend = types.SimpleNamespace(backend=lambda: "jax")
    fake_keras.layers = types.SimpleNamespace(
        Dense=FakeDense,
        Dropout=FakeDropout,
        LayerNormalization=FakeLayerNormalization,
        MultiHeadAttention=FakeMultiHeadAttention,
        Layer=object,
    )
    fake_keras.losses = types.SimpleNamespace(
        Reduction=types.SimpleNamespace(AUTO="auto", SUM="sum", NONE="none"),
        get=lambda config: config,
    )
    fake_keras.saving = types.SimpleNamespace(
        register_keras_serializable=register_keras_serializable
    )
    fake_keras.ops = types.SimpleNamespace(
        concatenate=np.concatenate,
        mean=np.mean,
        reshape=np.reshape,
    )
    fake_keras.Model = FakeModel
    fake_keras.Sequential = object

    monkeypatch.setenv("KERAS_BACKEND", "jax")
    monkeypatch.delenv("BASE_ATTENTIVE_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "keras", fake_keras)
    monkeypatch.delitem(sys.modules, "tensorflow", raising=False)
    original_modules = {
        module_name: module
        for module_name, module in list(sys.modules.items())
        if module_name.startswith("base_attentive")
    }

    try:
        for module_name in list(original_modules):
            del sys.modules[module_name]

        import base_attentive

        importlib.reload(base_attentive)

        from base_attentive.experimental import BaseAttentiveV2

        model = BaseAttentiveV2(
            static_input_dim=2,
            dynamic_input_dim=3,
            future_input_dim=2,
            output_dim=1,
            forecast_horizon=4,
            backend_name="jax",
            attention_heads=2,
        )

        outputs = model.call(
            [
                np.ones((2, 2), dtype=np.float32),
                np.ones((2, 5, 3), dtype=np.float32),
                np.ones((2, 4, 2), dtype=np.float32),
            ]
        )

        assert outputs.shape == (2, 4, 1)

        quantile_model = BaseAttentiveV2(
            static_input_dim=2,
            dynamic_input_dim=3,
            future_input_dim=2,
            output_dim=1,
            forecast_horizon=4,
            backend_name="jax",
            head_type="quantile",
            quantiles=[0.1, 0.5, 0.9],
        )

        quantile_outputs = quantile_model.call(
            [
                np.ones((2, 2), dtype=np.float32),
                np.ones((2, 5, 3), dtype=np.float32),
                np.ones((2, 4, 2), dtype=np.float32),
            ]
        )

        assert quantile_outputs.shape == (2, 4, 3, 1)

        restored = BaseAttentiveV2.from_config(quantile_model.get_config())
        assert restored.spec.head_type == "quantile"
        assert restored.spec.quantiles == (0.1, 0.5, 0.9)
    finally:
        for module_name in list(sys.modules):
            if module_name.startswith("base_attentive"):
                del sys.modules[module_name]
        sys.modules.update(original_modules)
