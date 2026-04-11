"""Robust tests for the main BaseAttentive class."""

from __future__ import annotations

import copy
import importlib
import sys
import types
import warnings

import pytest


class FakeScalar:
    """Small scalar wrapper that mimics a TensorFlow shape scalar."""

    def __init__(self, value: int):
        self.value = value
        self.dtype = int

    def __repr__(self) -> str:
        return str(self.value)


class FakeTensor:
    """Simple tensor-like object for shape-driven tests."""

    def __init__(self, shape, name: str = "tensor"):
        self.shape = shape
        self.name = name

    def __repr__(self) -> str:
        return f"FakeTensor(name={self.name!r}, shape={self.shape!r})"


class FakeDebugging:
    """Small debugging namespace compatible with the module's expectations."""

    @staticmethod
    def assert_equal(actual, expected, message=""):
        actual_value = getattr(actual, "value", actual)
        if actual_value != expected:
            raise AssertionError(message)


class FakeModel:
    """Minimal stand-in for a Keras Model base class."""

    def __init__(self, name=None, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def get_config(self):
        return {"name": self.name}


class FakeKerasDeps:
    """Minimal subset of KERAS_DEPS needed to import the class."""

    Add = object
    Dense = object
    Tensor = object
    MultiHeadAttention = object
    Layer = object
    LayerNormalization = object
    LSTM = object
    Model = FakeModel
    debugging = FakeDebugging()

    @staticmethod
    def shape(tensor):
        return [None, FakeScalar(tensor.shape[1])]

    @staticmethod
    def concat(values, axis=0):
        return values

    @staticmethod
    def zeros(shape):
        return shape

    @staticmethod
    def expand_dims(value, axis=-1):
        return value

    @staticmethod
    def tile(value, multiples):
        return value

    @staticmethod
    def convert_to_tensor(value, dtype=None):
        return value

    @staticmethod
    def register_keras_serializable(*args, **kwargs):
        def decorator(obj):
            return obj

        return decorator


@pytest.fixture
def base_attentive_module(monkeypatch):
    """Import the BaseAttentive module with lightweight stub dependencies."""
    monkeypatch.setenv("KERAS_BACKEND", "")
    monkeypatch.setitem(sys.modules, "keras", types.ModuleType("keras"))

    for module_name in (
        "base_attentive",
        "base_attentive.core",
        "base_attentive.core.base_attentive",
    ):
        sys.modules.pop(module_name, None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        import base_attentive

    monkeypatch.setattr(base_attentive, "KERAS_BACKEND", "")
    monkeypatch.setattr(base_attentive, "KERAS_DEPS", FakeKerasDeps())
    monkeypatch.setattr(
        base_attentive,
        "dependency_message",
        lambda name: f"{name} dependencies",
    )

    sys.modules.pop("base_attentive.core.base_attentive", None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        module = importlib.import_module("base_attentive.core.base_attentive")

    monkeypatch.setattr(
        module,
        "Activation",
        lambda activation: types.SimpleNamespace(activation_str=activation),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "set_default_params",
        lambda quantiles, scales, multi_scale_agg: (
            quantiles,
            [1] if scales is None else list(scales),
            multi_scale_agg != "last",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        module.BaseAttentive,
        "_build_attentive_layers",
        lambda self: None,
    )
    return module


@pytest.fixture
def make_base_attentive(base_attentive_module):
    """Factory for creating lightweight BaseAttentive instances."""

    def factory(**kwargs):
        params = {
            "static_input_dim": 2,
            "dynamic_input_dim": 3,
            "future_input_dim": 4,
            "forecast_horizon": 5,
            "mode": "pihal_like",
        }
        params.update(kwargs)
        return base_attentive_module.BaseAttentive(**params)

    return factory


def test_base_attentive_initializes_with_reconciled_architecture(
    make_base_attentive,
):
    """Explicit config inputs should be normalized into a stable architecture."""
    with pytest.warns(FutureWarning, match="deprecated"):
        model = make_base_attentive(
            mode="tft",
            objective="transformer",
            use_vsn=False,
            attention_levels=["cross", "memory"],
            architecture_config={
                "feature_processing": "vsn",
                "objective": "hybrid",
            },
        )

    assert model._mode == "tft_like"
    assert model.activation_fn_str == "relu"
    assert model.scales == [1]
    assert model.architecture_config == {
        "encoder_type": "hybrid",
        "decoder_attention_stack": ["cross", "memory"],
        "feature_processing": "dense",
    }


def test_base_attentive_docstring_uses_real_doc_components(
    base_attentive_module,
):
    """The rendered class docstring should include imported parameter docs."""
    docstring = base_attentive_module.BaseAttentive.__doc__

    assert docstring is not None
    assert "static_input_dim : int" in docstring
    assert "dynamic_input_dim : int" in docstring
    assert "future_input_dim : int" in docstring
    assert "embed_dim : int, default 32" in docstring


def test_get_config_and_from_config_round_trip_without_mutating_input(
    base_attentive_module,
    make_base_attentive,
):
    """Round-tripping through config should preserve values and input dicts."""
    model = make_base_attentive(
        quantiles=[0.1, 0.5, 0.9],
        scales=[1, 3],
        architecture_config={
            "encoder_type": "transformer",
            "decoder_attention_stack": ["cross"],
        },
    )

    config = model.get_config()
    original_config = copy.deepcopy(config)
    rebuilt = base_attentive_module.BaseAttentive.from_config(config)

    assert config == original_config
    assert (
        rebuilt.get_config()["architecture_config"]
        == original_config["architecture_config"]
    )
    assert rebuilt.quantiles == [0.1, 0.5, 0.9]
    assert rebuilt.scales == [1, 3]


def test_reconfigure_returns_new_model_without_mutating_original(
    make_base_attentive,
):
    """reconfigure should leave the original architecture untouched."""
    model = make_base_attentive(
        architecture_config={
            "encoder_type": "hybrid",
            "decoder_attention_stack": ["cross", "hierarchical"],
            "feature_processing": "vsn",
        }
    )
    original_architecture = copy.deepcopy(model.architecture_config)

    reconfigured = model.reconfigure(
        {
            "encoder_type": "transformer",
            "decoder_attention_stack": ["cross"],
        }
    )

    assert reconfigured is not model
    assert model.architecture_config == original_architecture
    assert reconfigured.architecture_config["encoder_type"] == "transformer"
    assert reconfigured.architecture_config["decoder_attention_stack"] == ["cross"]


def test_call_validates_tft_future_span_and_returns_decoder_output(
    base_attentive_module,
    make_base_attentive,
    monkeypatch,
):
    """tft_like mode should require past-window plus forecast-horizon steps."""
    monkeypatch.setattr(
        base_attentive_module,
        "validate_model_inputs",
        lambda **kwargs: tuple(kwargs["inputs"]),
        raising=False,
    )
    model = make_base_attentive(
        mode="tft_like",
        forecast_horizon=4,
        max_window_size=6,
        quantiles=None,
    )
    decoded = FakeTensor((2, 4, 1), name="decoded")

    model.run_encoder_decoder_core = lambda **kwargs: FakeTensor(
        (2, 4, 8), name="encoded"
    )
    model.multi_decoder = lambda final_features, training=False: decoded

    result = model.call(
        [
            FakeTensor((2, 2), name="static"),
            FakeTensor((2, 6, 3), name="dynamic"),
            FakeTensor((2, 10, 4), name="future"),
        ]
    )

    assert result is decoded


def test_call_uses_quantile_distribution_when_quantiles_are_enabled(
    base_attentive_module,
    make_base_attentive,
    monkeypatch,
):
    """Quantile-enabled models should route decoder output through the head."""
    monkeypatch.setattr(
        base_attentive_module,
        "validate_model_inputs",
        lambda **kwargs: tuple(kwargs["inputs"]),
        raising=False,
    )
    model = make_base_attentive(
        mode="pihal_like",
        forecast_horizon=3,
        quantiles=[0.1, 0.5, 0.9],
    )
    decoded = FakeTensor((2, 3, 1), name="decoded")
    quantiles = FakeTensor((2, 3, 3, 1), name="quantiles")

    model.run_encoder_decoder_core = lambda **kwargs: FakeTensor(
        (2, 3, 8), name="encoded"
    )
    model.multi_decoder = lambda final_features, training=False: decoded
    model.quantile_distribution_modeling = lambda outputs, training=False: quantiles

    result = model.call(
        [
            FakeTensor((2, 2), name="static"),
            FakeTensor((2, 6, 3), name="dynamic"),
            FakeTensor((2, 3, 4), name="future"),
        ]
    )

    assert result is quantiles


def test_call_raises_when_future_span_does_not_match_mode(
    base_attentive_module,
    make_base_attentive,
    monkeypatch,
):
    """pihal_like mode should reject future tensors longer than the horizon."""
    monkeypatch.setattr(
        base_attentive_module,
        "validate_model_inputs",
        lambda **kwargs: tuple(kwargs["inputs"]),
        raising=False,
    )
    model = make_base_attentive(
        mode="pihal_like",
        forecast_horizon=4,
        max_window_size=6,
    )
    model.run_encoder_decoder_core = lambda **kwargs: pytest.fail(
        "run_encoder_decoder_core should not run for invalid future spans"
    )

    with pytest.raises(AssertionError, match="Expected time dimension of 4"):
        model.call(
            [
                FakeTensor((2, 2), name="static"),
                FakeTensor((2, 6, 3), name="dynamic"),
                FakeTensor((2, 6, 4), name="future"),
            ]
        )
