"""Coverage-focused tests for BaseAttentive internals with lightweight stubs."""

from __future__ import annotations

import importlib
import sys
import types
import warnings

import numpy as np
import pytest


class _FakeScalar:
    def __init__(self, value: int):
        self.value = value
        self.dtype = int


class _FakeDebugging:
    @staticmethod
    def assert_equal(actual, expected, message=""):
        actual_value = getattr(actual, "value", actual)
        expected_value = getattr(expected, "value", expected)
        if actual_value != expected_value:
            raise AssertionError(
                message
                or f"{actual_value} != {expected_value}"
            )
        return None


class _FakeModel:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def get_config(self):
        return {"name": self.name}


class _FakeKerasDeps:
    Add = object
    Dense = object
    Tensor = object
    MultiHeadAttention = object
    Layer = object
    LayerNormalization = object
    LSTM = object
    Model = _FakeModel
    debugging = _FakeDebugging()

    @staticmethod
    def shape(tensor):
        return [None, _FakeScalar(tensor.shape[1])]

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


class _CallableLayer:
    def __init__(self, fn=None):
        self.fn = fn or (
            lambda *args, **kwargs: args[0] if args else None
        )
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.fn(*args, **kwargs)


class _DynamicShapeTensor:
    def __init__(self, actual_span: int):
        self.actual_span = actual_span
        self.shape = self

    def __getitem__(self, index):
        raise TypeError("dynamic shape")


@pytest.fixture
def base_attentive_module(monkeypatch):
    """Import ``base_attentive.core.base_attentive`` with lightweight deps."""
    monkeypatch.setenv("KERAS_BACKEND", "")
    monkeypatch.setitem(
        sys.modules, "keras", types.ModuleType("keras")
    )

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
    monkeypatch.setattr(
        base_attentive, "KERAS_DEPS", _FakeKerasDeps()
    )
    monkeypatch.setattr(
        base_attentive,
        "dependency_message",
        lambda name: f"{name} dependencies",
    )

    sys.modules.pop(
        "base_attentive.core.base_attentive", None
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        module = importlib.import_module(
            "base_attentive.core.base_attentive"
        )

    monkeypatch.setattr(
        module,
        "Activation",
        lambda activation: types.SimpleNamespace(
            activation_str=activation
        ),
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
    return module


def _patch_layer_builders(monkeypatch, module):
    """Patch layer constructors with callable stubs."""

    def make_factory(fn=None):
        return lambda *args, **kwargs: _CallableLayer(fn)

    monkeypatch.setattr(
        module,
        "VariableSelectionNetwork",
        make_factory(),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "GatedResidualNetwork",
        make_factory(),
        raising=False,
    )
    monkeypatch.setattr(
        module, "Dense", make_factory(), raising=False
    )
    monkeypatch.setattr(
        module,
        "MultiScaleLSTM",
        make_factory(),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "MultiHeadAttention",
        make_factory(),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "LayerNormalization",
        make_factory(),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "PositionalEncoding",
        make_factory(),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "HierarchicalAttention",
        make_factory(),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "CrossAttention",
        make_factory(),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "MemoryAugmentedAttention",
        make_factory(),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "MultiResolutionAttentionFusion",
        make_factory(),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "DynamicTimeWindow",
        make_factory(),
        raising=False,
    )
    monkeypatch.setattr(
        module, "MultiDecoder", make_factory(), raising=False
    )
    monkeypatch.setattr(
        module,
        "QuantileDistributionModeling",
        make_factory(),
        raising=False,
    )
    monkeypatch.setattr(
        module, "Add", make_factory(), raising=False
    )


def test_build_attentive_layers_vsn_hybrid_with_residuals(
    base_attentive_module,
    monkeypatch,
):
    """VSN + hybrid architectures should create the expected layer set."""
    _patch_layer_builders(monkeypatch, base_attentive_module)

    model = base_attentive_module.BaseAttentive(
        static_input_dim=2,
        dynamic_input_dim=3,
        future_input_dim=4,
        forecast_horizon=2,
        objective="hybrid",
        use_vsn=True,
        use_residuals=True,
    )

    assert model.static_vsn is not None
    assert model.dynamic_vsn is not None
    assert model.future_vsn is not None
    assert model.static_dense is None
    assert model.dynamic_dense is None
    assert model.future_dense is None
    assert model.multi_scale_lstm is not None
    assert model.encoder_self_attention is None
    assert model.residual_dense is not None
    assert len(model.decoder_add_norm) == 2
    assert len(model.final_add_norm) == 2


def test_build_attentive_layers_dense_transformer_without_residuals(
    base_attentive_module,
    monkeypatch,
):
    """Dense + transformer configurations should skip VSN and residual layers."""
    _patch_layer_builders(monkeypatch, base_attentive_module)

    model = base_attentive_module.BaseAttentive(
        static_input_dim=0,
        dynamic_input_dim=3,
        future_input_dim=2,
        forecast_horizon=2,
        objective="transformer",
        use_vsn=False,
        use_residuals=False,
        num_encoder_layers=2,
    )

    assert model.static_vsn is None
    assert model.static_dense is None
    assert model.dynamic_dense is not None
    assert model.future_dense is not None
    assert model.multi_scale_lstm is None
    assert len(model.encoder_self_attention) == 2
    assert model.residual_dense is None
    assert model.decoder_add_norm is None
    assert model.final_add_norm is None


def test_run_encoder_decoder_core_covers_vsn_hybrid_and_tft_paths(
    base_attentive_module,
    monkeypatch,
):
    """The encoder/decoder core should handle VSN processing and TFT slicing."""
    monkeypatch.setattr(
        base_attentive_module,
        "tf_shape",
        lambda value: value.shape,
    )
    monkeypatch.setattr(
        base_attentive_module,
        "tf_concat",
        lambda values, axis=-1: np.concatenate(
            values, axis=axis
        ),
    )
    monkeypatch.setattr(
        base_attentive_module,
        "tf_expand_dims",
        np.expand_dims,
    )
    monkeypatch.setattr(
        base_attentive_module, "tf_tile", np.tile
    )
    monkeypatch.setattr(
        base_attentive_module, "tf_zeros", np.zeros
    )
    monkeypatch.setattr(
        base_attentive_module,
        "aggregate_multiscale_on_3d",
        lambda value, mode="concat": value,
        raising=False,
    )
    monkeypatch.setattr(
        base_attentive_module,
        "aggregate_time_window_output",
        lambda value, mode: value,
        raising=False,
    )

    dtw_layer = _CallableLayer(
        lambda value, training=False: value
    )
    fake_model = types.SimpleNamespace(
        architecture_config={
            "feature_processing": "vsn",
            "encoder_type": "hybrid",
            "decoder_attention_stack": [
                "cross",
                "hierarchical",
                "memory",
            ],
        },
        static_vsn=_CallableLayer(
            lambda value, training=False: value
        ),
        static_vsn_grn=_CallableLayer(
            lambda value, training=False: value
        ),
        dynamic_vsn=_CallableLayer(
            lambda value, training=False: value
        ),
        dynamic_vsn_grn=_CallableLayer(
            lambda value, training=False: value
        ),
        future_vsn=_CallableLayer(
            lambda value, training=False: value
        ),
        future_vsn_grn=_CallableLayer(
            lambda value, training=False: value
        ),
        static_dense=None,
        grn_static_non_vsn=None,
        dynamic_dense=None,
        future_dense=None,
        _mode="tft_like",
        encoder_positional_encoding=_CallableLayer(
            lambda value: value
        ),
        multi_scale_lstm=_CallableLayer(
            lambda value, training=False: value
        ),
        encoder_self_attention=None,
        apply_dtw=True,
        dynamic_time_window=dtw_layer,
        decoder_positional_encoding=_CallableLayer(
            lambda value: value
        ),
        decoder_input_projection=_CallableLayer(
            lambda value: value
        ),
        apply_attention_levels=lambda projected_decoder_input,
        encoder_sequences,
        training=False: projected_decoder_input,
        final_agg="last",
        future_input_dim=2,
        static_input_dim=1,
        forecast_horizon=2,
        attention_units=3,
    )

    result = base_attentive_module.BaseAttentive.run_encoder_decoder_core(
        fake_model,
        static_input=np.ones((2, 1), dtype=np.float32),
        dynamic_input=np.ones((2, 3, 2), dtype=np.float32),
        future_input=np.ones((2, 5, 2), dtype=np.float32),
        training=False,
    )

    assert result.shape == (2, 2, 3)
    assert dtw_layer.calls


def test_run_encoder_decoder_core_covers_dense_transformer_and_zero_decoder_parts(
    base_attentive_module,
    monkeypatch,
):
    """The core path should also cover dense processing and zero-filled decoders."""
    monkeypatch.setattr(
        base_attentive_module,
        "tf_shape",
        lambda value: value.shape,
    )
    monkeypatch.setattr(
        base_attentive_module,
        "tf_concat",
        lambda values, axis=-1: np.concatenate(
            values, axis=axis
        ),
    )
    monkeypatch.setattr(
        base_attentive_module,
        "tf_expand_dims",
        np.expand_dims,
    )
    monkeypatch.setattr(
        base_attentive_module, "tf_tile", np.tile
    )
    monkeypatch.setattr(
        base_attentive_module, "tf_zeros", np.zeros
    )
    monkeypatch.setattr(
        base_attentive_module,
        "aggregate_multiscale_on_3d",
        lambda value, mode="concat": value,
        raising=False,
    )
    monkeypatch.setattr(
        base_attentive_module,
        "aggregate_time_window_output",
        lambda value, mode: ("aggregated", value.shape, mode),
        raising=False,
    )

    fake_model = types.SimpleNamespace(
        architecture_config={
            "feature_processing": "dense",
            "encoder_type": "transformer",
            "decoder_attention_stack": [],
        },
        static_dense=None,
        grn_static_non_vsn=None,
        dynamic_dense=_CallableLayer(lambda value: value),
        future_dense=_CallableLayer(lambda value: value),
        static_vsn=None,
        static_vsn_grn=None,
        dynamic_vsn=None,
        dynamic_vsn_grn=None,
        future_vsn=None,
        future_vsn_grn=None,
        _mode="pihal_like",
        encoder_positional_encoding=_CallableLayer(
            lambda value: value
        ),
        multi_scale_lstm=None,
        encoder_self_attention=[
            (
                _CallableLayer(
                    lambda query, value: np.ones_like(query)
                ),
                _CallableLayer(lambda value: value),
            )
        ],
        apply_dtw=False,
        dynamic_time_window=None,
        decoder_positional_encoding=_CallableLayer(
            lambda value: value
        ),
        decoder_input_projection=_CallableLayer(
            lambda value: value
        ),
        apply_attention_levels=lambda projected_decoder_input,
        encoder_sequences,
        training=False: projected_decoder_input,
        final_agg="flatten",
        future_input_dim=0,
        static_input_dim=0,
        forecast_horizon=2,
        attention_units=4,
    )

    result = base_attentive_module.BaseAttentive.run_encoder_decoder_core(
        fake_model,
        static_input=np.ones((2, 0), dtype=np.float32),
        dynamic_input=np.ones((2, 3, 2), dtype=np.float32),
        future_input=np.ones((2, 2, 0), dtype=np.float32),
        training=False,
    )

    assert result == ("aggregated", (2, 2, 4), "flatten")


def test_apply_attention_levels_covers_residual_and_passthrough_paths(
    base_attentive_module,
):
    """Attention fusion should work with or without each attention stage."""
    projected = np.ones((1, 2, 1), dtype=np.float32)
    encoder = np.ones((1, 3, 1), dtype=np.float32)

    model_with_all = types.SimpleNamespace(
        architecture_config={
            "decoder_attention_stack": [
                "cross",
                "hierarchical",
                "memory",
            ]
        },
        cross_attention=_CallableLayer(
            lambda inputs, training=False: inputs[0] + 1
        ),
        attention_processing_grn=_CallableLayer(
            lambda value, training=False: value + 2
        ),
        use_residuals=True,
        decoder_add_norm=[
            _CallableLayer(
                lambda values: values[0] + values[1]
            ),
            _CallableLayer(lambda value: value + 3),
        ],
        hierarchical_attention=_CallableLayer(
            lambda inputs, training=False: inputs[0] * 2
        ),
        memory_augmented_attention=_CallableLayer(
            lambda value, training=False: value + 4
        ),
        multi_resolution_attention_fusion=_CallableLayer(
            lambda value, training=False: value + 5
        ),
        residual_dense=_CallableLayer(
            lambda value: value + 6
        ),
        final_add_norm=[
            _CallableLayer(
                lambda values: values[0] + values[1]
            ),
            _CallableLayer(lambda value: value + 7),
        ],
    )

    result = base_attentive_module.BaseAttentive.apply_attention_levels(
        model_with_all,
        projected,
        encoder,
        training=False,
    )
    assert result.shape == projected.shape
    assert np.all(result > projected)

    passthrough_model = types.SimpleNamespace(
        architecture_config={"decoder_attention_stack": []},
        cross_attention=None,
        attention_processing_grn=None,
        use_residuals=False,
        decoder_add_norm=None,
        hierarchical_attention=None,
        memory_augmented_attention=None,
        multi_resolution_attention_fusion=_CallableLayer(
            lambda value, training=False: value
        ),
        residual_dense=None,
        final_add_norm=None,
    )

    passthrough = base_attentive_module.BaseAttentive.apply_attention_levels(
        passthrough_model,
        projected,
        encoder,
        training=False,
    )
    assert np.array_equal(passthrough, projected)


def test_call_covers_dynamic_future_span_validation(
    base_attentive_module,
    monkeypatch,
):
    """Dynamic future spans should use the ``tf_shape`` assertion path."""
    monkeypatch.setattr(
        base_attentive_module,
        "validate_model_inputs",
        lambda **kwargs: tuple(kwargs["inputs"]),
        raising=False,
    )
    monkeypatch.setattr(
        base_attentive_module,
        "tf_shape",
        lambda tensor: [
            None,
            _FakeScalar(tensor.actual_span),
        ],
    )
    monkeypatch.setattr(
        base_attentive_module,
        "tf_convert_to_tensor",
        lambda value, dtype=None: _FakeScalar(value),
    )

    monkeypatch.setattr(
        base_attentive_module.BaseAttentive,
        "_build_attentive_layers",
        lambda self: None,
    )

    model = base_attentive_module.BaseAttentive(
        static_input_dim=1,
        dynamic_input_dim=1,
        future_input_dim=1,
        forecast_horizon=4,
        mode="pihal_like",
        quantiles=None,
    )
    model.run_encoder_decoder_core = lambda **kwargs: np.ones(
        (2, 4, 3), dtype=np.float32
    )
    model.multi_decoder = (
        lambda features, training=False: np.ones(
            (2, 4, 1), dtype=np.float32
        )
    )

    future = _DynamicShapeTensor(actual_span=4)
    result = model.call(
        [
            np.ones((2, 1), dtype=np.float32),
            np.ones((2, 3, 1), dtype=np.float32),
            future,
        ]
    )

    assert result.shape == (2, 4, 1)
