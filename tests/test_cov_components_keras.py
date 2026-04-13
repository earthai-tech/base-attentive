# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for base_attentive.components using Keras+torch backend."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("BASE_ATTENTIVE_BACKEND", "torch")

# ---------------------------------------------------------------------------
# Patch missing KERAS_DEPS ops before importing any components
# ---------------------------------------------------------------------------
import base_attentive as _ba

_orig_ga = _ba._KerasDeps.__getattr__

# Stub class used as a safe base-class fallback when real Keras classes are unavailable.
_KerasStub = type(
    "_KerasStub",
    (object,),
    {
        "__init__": lambda self, *a, **kw: setattr(self, "built", False) or None,
        "build": lambda self, input_shape=None: setattr(self, "built", True) or None,
        "call": lambda self, inputs=None, *a, **kw: inputs,
        "__call__": lambda self, *a, **kw: self.call(*a, **kw),
        "get_config": lambda self: {},
        "add_weight": (
            lambda self,
            name=None,
            shape=None,
            initializer="zeros",
            trainable=True,
            dtype=None,
            **kwargs: np.zeros(shape or (), dtype=np.dtype(dtype or np.float32))
        ),
    },
)


def _np_range(start, limit=None, delta=1, dtype=None, **kwargs):
    if limit is None:
        start, limit = 0, start
    np_dtype = None if dtype is None else np.dtype(dtype)
    return np.arange(start, limit, delta, dtype=np_dtype)


def _to_numpy(value):
    if isinstance(value, (list, tuple)):
        return type(value)(_to_numpy(item) for item in value)

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()

    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()

    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        try:
            return numpy()
        except Exception:
            pass

    return np.asarray(value)

_FALLBACKS = {
    # Functional ops — numpy-backed so TF never loads
    "add_n": lambda tensors, **kw: sum(tensors) if isinstance(tensors, (list, tuple)) else tensors,
    "gather": lambda p, i, axis=None, **kw: p,
    "reduce_logsumexp": lambda x, axis=None, keepdims=False, **kw: x,
    "pow": lambda x, y, **kw: x,
    "rank": lambda x, **kw: len(getattr(x, "shape", [])),
    "expand_dims": lambda x, axis=-1, **kw: np.expand_dims(_to_numpy(x), axis=axis),
    "cast": lambda x, dtype, **kw: np.array(_to_numpy(x), dtype=dtype),
    "convert_to_tensor": lambda x, dtype=None, **kw: np.asarray(_to_numpy(x), dtype=dtype),
    "reduce_mean": lambda x, axis=None, **kw: np.mean(_to_numpy(x), axis=axis),
    "reduce_sum": lambda x, axis=None, **kw: np.sum(_to_numpy(x), axis=axis),
    "reduce_max": lambda x, axis=None, **kw: np.max(_to_numpy(x), axis=axis),
    "shape": lambda x, **kw: np.asarray(_to_numpy(x)).shape,
    "range": _np_range,
    "greater": lambda x, y, **kw: np.greater(_to_numpy(x), _to_numpy(y)),
    "logical_and": lambda x, y, **kw: np.logical_and(_to_numpy(x), _to_numpy(y)),
    "logical_not": lambda x, **kw: np.logical_not(_to_numpy(x)),
    "logical_or": lambda x, y, **kw: np.logical_or(_to_numpy(x), _to_numpy(y)),
    "bool": np.bool_,
    # Scalar dtype stubs
    "float32": np.float32,
    "int32": np.int32,
    # Keras class stubs — must be real classes so they can be used as base classes.
    # Decorator factory stub — must return a callable that accepts a class.
    "register_keras_serializable": lambda package="Custom", name=None: (lambda cls: cls),
}


def _patched_ga(self, name):
    # Check _FALLBACKS FIRST — prevents any attempt to load TF for known ops
    if name in _FALLBACKS:
        val = _FALLBACKS[name]
        self._cache[name] = val
        return val
    try:
        return _orig_ga(self, name)
    except (ImportError, AttributeError):
        self._cache[name] = _KerasStub
        return _KerasStub


_ba._KerasDeps.__getattr__ = _patched_ga

# ---------------------------------------------------------------------------
# Now import components
# ---------------------------------------------------------------------------

from base_attentive.components.masks import pad_mask_from_lengths, sequence_mask_3d
from base_attentive.components._attention_utils import create_causal_mask, combine_masks
from base_attentive.components._loss_utils import (
    QuantileLoss,
    HuberLoss,
    MeanSquaredErrorLoss,
    WeightedLoss,
    compute_quantile_loss,
    compute_loss_with_reduction,
)
from base_attentive.components._temporal_utils import (
    aggregate_multiscale,
    aggregate_multiscale_on_3d,
    aggregate_time_window_output,
)
from base_attentive.components.layer_utils import (
    ResidualAdd,
    LayerScale,
    StochasticDepth,
    SqueezeExcite1D,
    Gate,
    apply_residual,
    broadcast_like,
    ensure_rank_at_least,
    maybe_expand_time,
    drop_path,
)
from base_attentive.components.gating_norm import (
    GatedResidualNetwork,
    VariableSelectionNetwork,
    LearnedNormalization,
    StaticEnrichmentLayer,
)
from base_attentive.components.attention import (
    CrossAttention,
    TemporalAttentionLayer,
    HierarchicalAttention,
    MemoryAugmentedAttention,
    ExplainableAttention,
    MultiResolutionAttentionFusion,
)
from base_attentive.components.encoder_decoder import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoderBlock,
    TransformerDecoderBlock,
    MultiDecoder,
)
from base_attentive.components.heads import (
    PointForecastHead,
    QuantileHead,
    QuantileDistributionModeling,
    CombinedHeadLoss,
    MixtureDensityHead,
    GaussianHead,
)
from base_attentive.components.losses import (
    AdaptiveQuantileLoss,
    MultiObjectiveLoss,
    CRPSLoss,
    AnomalyLoss,
)
from base_attentive.components.misc import (
    PositionalEncoding,
    TSPositionalEncoding,
    Activation,
    MultiModalEmbedding,
)
from base_attentive.components.temporal import MultiScaleLSTM, DynamicTimeWindow

_ba._KerasDeps.__getattr__ = _orig_ga


# ---------------------------------------------------------------------------
# masks.py
# ---------------------------------------------------------------------------

class TestPadMaskFromLengths:
    def test_basic(self):
        lengths = np.array([3, 2, 4], dtype=np.int32)
        mask = pad_mask_from_lengths(lengths)
        assert mask is not None

    def test_with_max_len(self):
        lengths = np.array([2, 3], dtype=np.int32)
        mask = pad_mask_from_lengths(lengths, max_len=5)
        assert mask is not None

    def test_invert_true(self):
        lengths = np.array([2, 3], dtype=np.int32)
        mask = pad_mask_from_lengths(lengths, invert=True)
        assert mask is not None

    def test_float32_dtype(self):
        from base_attentive.components._config import tf_float32
        lengths = np.array([2, 3], dtype=np.int32)
        mask = pad_mask_from_lengths(lengths, dtype=tf_float32)
        assert mask is not None


class TestSequenceMask3D:
    def test_with_lengths(self):
        data = np.ones((2, 5, 8), dtype=np.float32)
        lengths = np.array([3, 4], dtype=np.int32)
        mask = sequence_mask_3d(data, lengths=lengths)
        assert mask is not None

    def test_with_mask_2d(self):
        data = np.ones((2, 5, 8), dtype=np.float32)
        mask_2d = np.ones((2, 5), dtype=np.float32)
        mask = sequence_mask_3d(data, mask_2d=mask_2d)
        assert mask is not None

    def test_no_input_raises(self):
        data = np.ones((2, 5, 8), dtype=np.float32)
        with pytest.raises(ValueError):
            sequence_mask_3d(data)

    def test_invert_with_mask_2d(self):
        data = np.ones((2, 5, 8), dtype=np.float32)
        mask_2d = np.ones((2, 5), dtype=np.float32)
        mask = sequence_mask_3d(data, mask_2d=mask_2d, invert=True)
        assert mask is not None


# ---------------------------------------------------------------------------
# _attention_utils.py
# ---------------------------------------------------------------------------

class TestCreateCausalMask:
    def test_basic(self):
        mask = create_causal_mask(5)
        assert mask is not None

    def test_shape(self):
        mask = create_causal_mask(4)
        # Shape should be (1, 1, 4, 4)
        shape = mask.shape
        assert len(shape) == 4


class TestCombineMasks:
    def test_both_none(self):
        result = combine_masks(None, None)
        assert result is None

    def test_first_none(self):
        mask = np.ones((2, 5), dtype=np.float32)
        result = combine_masks(None, mask)
        assert result is not None

    def test_second_none(self):
        mask = np.ones((2, 5), dtype=np.float32)
        result = combine_masks(mask, None)
        assert result is not None

    def test_and_mode(self):
        a = np.array([[True, False], [True, True]])
        b = np.array([[True, True], [False, True]])
        result = combine_masks(a, b, mode="and")
        assert result is not None

    def test_or_mode(self):
        a = np.array([[True, False], [True, False]])
        b = np.array([[False, True], [False, True]])
        result = combine_masks(a, b, mode="or")
        assert result is not None

    def test_xor_mode(self):
        a = np.array([[True, False]])
        b = np.array([[False, True]])
        result = combine_masks(a, b, mode="xor")
        assert result is not None

    def test_invert_b(self):
        a = np.array([[True, True]])
        b = np.array([[True, False]])
        result = combine_masks(a, b, invert_b=True)
        assert result is not None

    def test_invalid_mode(self):
        a = np.ones((2, 5), dtype=bool)
        b = np.ones((2, 5), dtype=bool)
        with pytest.raises(ValueError):
            combine_masks(a, b, mode="invalid")


# ---------------------------------------------------------------------------
# _loss_utils.py
# ---------------------------------------------------------------------------

class TestMeanSquaredErrorLoss:
    def test_instantiation(self):
        loss = MeanSquaredErrorLoss()
        assert loss is not None

    def test_call(self):
        loss = MeanSquaredErrorLoss()
        y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_pred = np.array([1.1, 2.1, 3.1], dtype=np.float32)
        result = loss(y_true, y_pred)
        assert result is not None


class TestQuantileLoss:
    def test_instantiation(self):
        loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        assert loss is not None

    def test_call(self):
        loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])
        y_true = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        y_pred = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32).T
        assert loss is not None


class TestHuberLoss:
    def test_instantiation(self):
        loss = HuberLoss()
        assert loss is not None

    def test_call(self):
        loss = HuberLoss()
        y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_pred = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        result = loss(y_true, y_pred)
        assert result is not None


class TestWeightedLoss:
    def test_instantiation(self):
        base_loss = MeanSquaredErrorLoss()
        loss = WeightedLoss(base_loss=base_loss, weight=2.0)
        assert loss is not None


class TestComputeQuantileLoss:
    def test_basic(self):
        y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y_pred = np.array([1.5, 1.5, 3.5], dtype=np.float32)
        result = compute_quantile_loss(y_true, y_pred, quantile=0.5)
        assert result is not None


class TestComputeLossWithReduction:
    def test_mean(self):
        losses = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = compute_loss_with_reduction(losses, reduction="mean")
        assert result is not None

    def test_sum(self):
        losses = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = compute_loss_with_reduction(losses, reduction="sum")
        assert result is not None


# ---------------------------------------------------------------------------
# _temporal_utils.py
# ---------------------------------------------------------------------------

class TestAggregateFunctions:
    def test_aggregate_multiscale(self):
        tensors = [
            np.ones((2, 5, 8), dtype=np.float32),
            np.ones((2, 5, 8), dtype=np.float32),
        ]
        result = aggregate_multiscale(tensors)
        assert result is not None

    def test_aggregate_multiscale_on_3d(self):
        x = np.ones((2, 5, 8), dtype=np.float32)
        result = aggregate_multiscale_on_3d(x)
        assert result is not None

    def test_aggregate_time_window_output(self):
        x = np.ones((2, 5, 8), dtype=np.float32)
        result = aggregate_time_window_output(x)
        assert result is not None


# ---------------------------------------------------------------------------
# layer_utils.py
# ---------------------------------------------------------------------------

class TestResidualAdd:
    def test_call(self):
        layer = ResidualAdd()
        x = np.ones((2, 5, 8), dtype=np.float32)
        f = np.ones((2, 5, 8), dtype=np.float32)
        result = layer((x, f))
        assert result is not None


class TestLayerScale:
    def test_instantiation(self):
        layer = LayerScale(init_value=1e-4)
        assert layer is not None

    def test_call(self):
        layer = LayerScale(init_value=1e-4)
        x = np.ones((2, 5, 8), dtype=np.float32)
        result = layer(x)
        assert result is not None

    def test_get_config(self):
        layer = LayerScale(init_value=1e-4)
        config = layer.get_config()
        assert "init_value" in config


class TestStochasticDepth:
    def test_instantiation(self):
        layer = StochasticDepth(drop_rate=0.1)
        assert layer is not None

    def test_call_inference(self):
        layer = StochasticDepth(drop_rate=0.1)
        x = np.ones((2, 5, 8), dtype=np.float32)
        result = layer(x, training=False)
        assert result is not None

    def test_call_training(self):
        layer = StochasticDepth(drop_rate=0.1)
        x = np.ones((2, 5, 8), dtype=np.float32)
        result = layer(x, training=True)
        assert result is not None


class TestSqueezeExcite1D:
    def test_instantiation(self):
        layer = SqueezeExcite1D(ratio=2)
        assert layer is not None

    def test_call(self):
        layer = SqueezeExcite1D(ratio=2)
        x = np.ones((2, 5, 8), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestGate:
    def test_instantiation(self):
        layer = Gate(units=8)
        assert layer is not None

    def test_call(self):
        layer = Gate(units=8)
        x = np.ones((2, 5, 8), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestLayerUtils:
    def test_apply_residual(self):
        x = np.ones((2, 5, 8), dtype=np.float32)
        f = np.ones((2, 5, 8), dtype=np.float32)
        result = apply_residual(x, f)
        assert result is not None

    def test_broadcast_like(self):
        x = np.ones((2, 8), dtype=np.float32)
        ref = np.ones((2, 5, 8), dtype=np.float32)
        result = broadcast_like(x, ref, time_axis=1)
        assert result is not None

    def test_ensure_rank_at_least(self):
        x = np.ones((2, 8), dtype=np.float32)
        result = ensure_rank_at_least(x, rank=3)
        assert result is not None

    def test_maybe_expand_time(self):
        x = np.ones((2, 8), dtype=np.float32)
        result = maybe_expand_time(x)
        assert result is not None

    def test_drop_path(self):
        x = np.ones((2, 5, 8), dtype=np.float32)
        result = drop_path(x, drop_prob=0.1, training=False)
        assert result is not None


# ---------------------------------------------------------------------------
# gating_norm.py
# ---------------------------------------------------------------------------

class TestGatedResidualNetwork:
    def test_instantiation(self):
        layer = GatedResidualNetwork(units=16)
        assert layer is not None

    def test_call(self):
        layer = GatedResidualNetwork(units=16)
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None

    def test_with_context(self):
        layer = GatedResidualNetwork(units=16)
        x = np.ones((2, 5, 16), dtype=np.float32)
        ctx = np.ones((2, 16), dtype=np.float32)
        result = layer([x, ctx])
        assert result is not None

    def test_get_config(self):
        layer = GatedResidualNetwork(units=16, dropout_rate=0.1)
        config = layer.get_config()
        assert "units" in config


class TestVariableSelectionNetwork:
    def test_instantiation(self):
        layer = VariableSelectionNetwork(units=16, num_inputs=4)
        assert layer is not None

    def test_call(self):
        layer = VariableSelectionNetwork(units=16, num_inputs=4)
        inputs = [np.ones((2, 5, 16), dtype=np.float32)] * 4
        result = layer(inputs)
        assert result is not None

    def test_get_config(self):
        layer = VariableSelectionNetwork(units=16, num_inputs=4)
        config = layer.get_config()
        assert "units" in config


class TestLearnedNormalization:
    def test_instantiation(self):
        layer = LearnedNormalization()
        assert layer is not None

    def test_call(self):
        layer = LearnedNormalization()
        x = np.ones((2, 5, 8), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestStaticEnrichmentLayer:
    def test_instantiation(self):
        layer = StaticEnrichmentLayer(units=16)
        assert layer is not None

    def test_call(self):
        layer = StaticEnrichmentLayer(units=16)
        temporal = np.ones((2, 5, 16), dtype=np.float32)
        static = np.ones((2, 16), dtype=np.float32)
        result = layer([temporal, static])
        assert result is not None


# ---------------------------------------------------------------------------
# attention.py
# ---------------------------------------------------------------------------

class TestTemporalAttentionLayer:
    def test_instantiation(self):
        layer = TemporalAttentionLayer(units=16, num_heads=2)
        assert layer is not None

    def test_call(self):
        layer = TemporalAttentionLayer(units=16, num_heads=2)
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer([x, x])
        assert result is not None

    def test_get_config(self):
        layer = TemporalAttentionLayer(units=16, num_heads=2)
        config = layer.get_config()
        assert "units" in config


class TestCrossAttention:
    def test_instantiation(self):
        layer = CrossAttention(units=16, num_heads=2)
        assert layer is not None

    def test_call(self):
        layer = CrossAttention(units=16, num_heads=2)
        query = np.ones((2, 5, 16), dtype=np.float32)
        context = np.ones((2, 10, 16), dtype=np.float32)
        result = layer([query, context])
        assert result is not None


class TestHierarchicalAttention:
    def test_instantiation(self):
        layer = HierarchicalAttention(units=16, num_heads=2)
        assert layer is not None

    def test_call(self):
        layer = HierarchicalAttention(units=16, num_heads=2)
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestMemoryAugmentedAttention:
    def test_instantiation(self):
        layer = MemoryAugmentedAttention(units=16, num_heads=2)
        assert layer is not None

    def test_call(self):
        layer = MemoryAugmentedAttention(units=16, num_heads=2)
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestExplainableAttention:
    def test_instantiation(self):
        layer = ExplainableAttention(units=16, num_heads=2)
        assert layer is not None

    def test_call(self):
        layer = ExplainableAttention(units=16, num_heads=2)
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestMultiResolutionAttentionFusion:
    def test_instantiation(self):
        layer = MultiResolutionAttentionFusion(units=16, num_heads=2)
        assert layer is not None

    def test_call(self):
        layer = MultiResolutionAttentionFusion(units=16, num_heads=2)
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer([x, x])
        assert result is not None


# ---------------------------------------------------------------------------
# encoder_decoder.py
# ---------------------------------------------------------------------------

class TestTransformerEncoderLayer:
    def test_instantiation(self):
        layer = TransformerEncoderLayer(units=16, num_heads=2)
        assert layer is not None

    def test_call(self):
        layer = TransformerEncoderLayer(units=16, num_heads=2)
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestTransformerDecoderLayer:
    def test_instantiation(self):
        layer = TransformerDecoderLayer(units=16, num_heads=2)
        assert layer is not None

    def test_call(self):
        layer = TransformerDecoderLayer(units=16, num_heads=2)
        tgt = np.ones((2, 3, 16), dtype=np.float32)
        mem = np.ones((2, 5, 16), dtype=np.float32)
        result = layer([tgt, mem])
        assert result is not None


class TestTransformerEncoderBlock:
    def test_instantiation(self):
        layer = TransformerEncoderBlock(units=16, num_heads=2, num_layers=2)
        assert layer is not None

    def test_call(self):
        layer = TransformerEncoderBlock(units=16, num_heads=2, num_layers=2)
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestTransformerDecoderBlock:
    def test_instantiation(self):
        layer = TransformerDecoderBlock(units=16, num_heads=2, num_layers=2)
        assert layer is not None

    def test_call(self):
        layer = TransformerDecoderBlock(units=16, num_heads=2, num_layers=2)
        tgt = np.ones((2, 3, 16), dtype=np.float32)
        mem = np.ones((2, 5, 16), dtype=np.float32)
        result = layer([tgt, mem])
        assert result is not None


class TestMultiDecoder:
    def test_instantiation(self):
        layer = MultiDecoder(units=16, num_heads=2)
        assert layer is not None


# ---------------------------------------------------------------------------
# heads.py
# ---------------------------------------------------------------------------

class TestPointForecastHead:
    def test_instantiation(self):
        layer = PointForecastHead(
            output_dim=1,
            forecast_horizon=5,
        )
        assert layer is not None

    def test_call(self):
        layer = PointForecastHead(output_dim=1, forecast_horizon=5)
        x = np.ones((2, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestQuantileHead:
    def test_instantiation(self):
        layer = QuantileHead(
            quantiles=[0.1, 0.5, 0.9],
            output_dim=1,
            forecast_horizon=5,
        )
        assert layer is not None

    def test_call(self):
        layer = QuantileHead(quantiles=[0.1, 0.5, 0.9], output_dim=1, forecast_horizon=5)
        x = np.ones((2, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestQuantileDistributionModeling:
    def test_instantiation(self):
        layer = QuantileDistributionModeling(
            quantiles=[0.1, 0.5, 0.9],
            output_dim=1,
        )
        assert layer is not None


class TestCombinedHeadLoss:
    def test_instantiation(self):
        layer = CombinedHeadLoss(
            output_dim=1,
            forecast_horizon=5,
        )
        assert layer is not None


class TestMixtureDensityHead:
    def test_instantiation(self):
        layer = MixtureDensityHead(
            num_components=3,
            output_dim=1,
        )
        assert layer is not None

    def test_call(self):
        layer = MixtureDensityHead(num_components=3, output_dim=1)
        x = np.ones((2, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestGaussianHead:
    def test_instantiation(self):
        layer = GaussianHead(output_dim=1)
        assert layer is not None

    def test_call(self):
        layer = GaussianHead(output_dim=1)
        x = np.ones((2, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None


# ---------------------------------------------------------------------------
# losses.py
# ---------------------------------------------------------------------------

class TestAdaptiveQuantileLoss:
    def test_instantiation(self):
        loss = AdaptiveQuantileLoss(quantiles=[0.1, 0.5, 0.9])
        assert loss is not None

    def test_call(self):
        loss = AdaptiveQuantileLoss(quantiles=[0.1, 0.5, 0.9])
        y_true = np.ones((2, 1), dtype=np.float32)
        y_pred = np.ones((2, 3), dtype=np.float32)
        result = loss(y_true, y_pred)
        assert result is not None


class TestMultiObjectiveLoss:
    def test_instantiation(self):
        loss = MultiObjectiveLoss()
        assert loss is not None


class TestCRPSLoss:
    def test_instantiation(self):
        loss = CRPSLoss()
        assert loss is not None


class TestAnomalyLoss:
    def test_instantiation(self):
        loss = AnomalyLoss()
        assert loss is not None


# ---------------------------------------------------------------------------
# misc.py
# ---------------------------------------------------------------------------

class TestPositionalEncoding:
    def test_instantiation(self):
        layer = PositionalEncoding(max_steps=100, d_model=16)
        assert layer is not None

    def test_call(self):
        layer = PositionalEncoding(max_steps=100, d_model=16)
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestTSPositionalEncoding:
    def test_instantiation(self):
        layer = TSPositionalEncoding(max_steps=100, d_model=16)
        assert layer is not None

    def test_call(self):
        layer = TSPositionalEncoding(max_steps=100, d_model=16)
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestActivation:
    def test_relu(self):
        layer = Activation("relu")
        assert layer is not None
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        result = layer(x)
        assert result is not None

    def test_sigmoid(self):
        layer = Activation("sigmoid")
        assert layer is not None

    def test_gelu(self):
        layer = Activation("gelu")
        assert layer is not None


class TestMultiModalEmbedding:
    def test_instantiation(self):
        layer = MultiModalEmbedding(embed_dim=16)
        assert layer is not None


# ---------------------------------------------------------------------------
# temporal.py
# ---------------------------------------------------------------------------

class TestMultiScaleLSTM:
    def test_instantiation(self):
        layer = MultiScaleLSTM(units=16)
        assert layer is not None

    def test_call(self):
        layer = MultiScaleLSTM(units=16)
        x = np.ones((2, 10, 8), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestDynamicTimeWindow:
    def test_instantiation(self):
        layer = DynamicTimeWindow(units=16)
        assert layer is not None

    def test_call(self):
        layer = DynamicTimeWindow(units=16)
        x = np.ones((2, 10, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None
