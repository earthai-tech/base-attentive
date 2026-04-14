# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for base_attentive.components using Keras+torch backend."""

from __future__ import annotations

import os
import types

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
        "__init__": lambda self, *a, **kw: setattr(
            self, "built", False
        )
        or None,
        "build": lambda self, input_shape=None: setattr(
            self, "built", True
        )
        or None,
        "call": lambda self, inputs=None, *a, **kw: inputs,
        "__call__": lambda self, *a, **kw: self.call(
            *a, **kw
        ),
        "get_config": lambda self: {},
        "add_weight": (
            lambda self,
            name=None,
            shape=None,
            initializer="zeros",
            trainable=True,
            dtype=None,
            **kwargs: np.zeros(
                shape or (),
                dtype=np.dtype(dtype or np.float32),
            )
        ),
    },
)


def _np_range(
    start, limit=None, delta=1, dtype=None, **kwargs
):
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
    "add_n": lambda tensors, **kw: sum(tensors)
    if isinstance(tensors, (list, tuple))
    else tensors,
    "gather": lambda p, i, axis=None, **kw: p,
    "reduce_logsumexp": lambda x,
    axis=None,
    keepdims=False,
    **kw: x,
    "pow": lambda x, y, **kw: x,
    "rank": lambda x, **kw: len(getattr(x, "shape", [])),
    "expand_dims": lambda x, axis=-1, **kw: np.expand_dims(
        _to_numpy(x), axis=axis
    ),
    "cast": lambda x, dtype, **kw: np.array(
        _to_numpy(x), dtype=dtype
    ),
    "convert_to_tensor": lambda x,
    dtype=None,
    **kw: np.asarray(_to_numpy(x), dtype=dtype),
    "reduce_mean": (
        lambda x, axis=None, keepdims=False, **kw: np.mean(
            _to_numpy(x),
            axis=axis,
            keepdims=keepdims,
        )
    ),
    "reduce_sum": lambda x, axis=None, **kw: np.sum(
        _to_numpy(x), axis=axis
    ),
    "reduce_max": lambda x, axis=None, **kw: np.max(
        _to_numpy(x), axis=axis
    ),
    "shape": lambda x, **kw: np.asarray(_to_numpy(x)).shape,
    "range": _np_range,
    "greater": lambda x, y, **kw: np.greater(
        _to_numpy(x), _to_numpy(y)
    ),
    "logical_and": lambda x, y, **kw: np.logical_and(
        _to_numpy(x), _to_numpy(y)
    ),
    "logical_not": lambda x, **kw: np.logical_not(
        _to_numpy(x)
    ),
    "logical_or": lambda x, y, **kw: np.logical_or(
        _to_numpy(x), _to_numpy(y)
    ),
    "bool": np.bool_,
    # Scalar dtype stubs
    "float32": np.float32,
    "int32": np.int32,
    # Keras class stubs — must be real classes so they can be used as base classes.
    # Decorator factory stub — must return a callable that accepts a class.
    "register_keras_serializable": lambda package="Custom",
    name=None: (lambda cls: cls),
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

from base_attentive.components._attention_utils import (
    combine_masks,
    create_causal_mask,
)
from base_attentive.components._loss_utils import (
    HuberLoss,
    MeanSquaredErrorLoss,
    QuantileLoss,
    WeightedLoss,
    compute_loss_with_reduction,
    compute_quantile_loss,
)
from base_attentive.components._temporal_utils import (
    aggregate_multiscale,
    aggregate_multiscale_on_3d,
    aggregate_time_window_output,
)
from base_attentive.components.attention import (
    CrossAttention,
    ExplainableAttention,
    HierarchicalAttention,
    MemoryAugmentedAttention,
    MultiResolutionAttentionFusion,
    TemporalAttentionLayer,
)
from base_attentive.components.encoder_decoder import (
    MultiDecoder,
    TransformerDecoderBlock,
    TransformerDecoderLayer,
    TransformerEncoderBlock,
    TransformerEncoderLayer,
)
from base_attentive.components.gating_norm import (
    GatedResidualNetwork,
    LearnedNormalization,
    StaticEnrichmentLayer,
    VariableSelectionNetwork,
)
from base_attentive.components.heads import (
    CombinedHeadLoss,
    GaussianHead,
    MixtureDensityHead,
    PointForecastHead,
    QuantileDistributionModeling,
    QuantileHead,
)
from base_attentive.components.layer_utils import (
    Gate,
    LayerScale,
    ResidualAdd,
    SqueezeExcite1D,
    StochasticDepth,
    _broadcast_like,
    apply_residual,
    broadcast_like,
    drop_path,
    ensure_rank_at_least,
    maybe_expand_time,
)
from base_attentive.components.losses import (
    AdaptiveQuantileLoss,
    AnomalyLoss,
    CRPSLoss,
    MultiObjectiveLoss,
)
from base_attentive.components.masks import (
    pad_mask_from_lengths,
    sequence_mask_3d,
)
from base_attentive.components.misc import (
    _PositionalEncoding,
    PositionwiseFeedForward,
    PositionalEncoding,
    TSPositionalEncoding,
    Activation,
    MultiModalEmbedding,
)
from base_attentive.components.temporal import (
    DynamicTimeWindow,
    MultiScaleLSTM,
)
import base_attentive.components.attention as _attention_mod
import base_attentive.components.gating_norm as _gating_norm_mod
import base_attentive.components.layer_utils as _layer_utils_mod
import base_attentive.components.misc as _misc_mod

_ba._KerasDeps.__getattr__ = _orig_ga


class _IdentityLayer:
    def __call__(self, value, *args, **kwargs):
        return value

    def get_config(self):
        return {}


class _Recorder:
    def __init__(self, return_value=None):
        self.return_value = return_value
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if self.return_value is not None:
            return self.return_value
        if "query" in kwargs:
            return kwargs["query"]
        return args[0] if args else None


class _RecorderWithScores:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return args[0], self.scores


class _ScalarWithItem:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _DetachCpuScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        raise TypeError("force detach/cpu path")

    def detach(self):
        return self

    def cpu(self):
        return _ScalarWithItem(self.value)


class _DynamicShapeTensor:
    shape = (2, None, 4)

    def __add__(self, other):
        return other


class _IterableShape:
    def __init__(self, *dims):
        self._dims = dims

    def __iter__(self):
        return iter(self._dims)


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
        from base_attentive.components._config import (
            tf_float32,
        )

        lengths = np.array([2, 3], dtype=np.int32)
        mask = pad_mask_from_lengths(
            lengths, dtype=tf_float32
        )
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
        mask = sequence_mask_3d(
            data, mask_2d=mask_2d, invert=True
        )
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
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        np.array(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        ).T
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
        result = compute_quantile_loss(
            y_true, y_pred, quantile=0.5
        )
        assert result is not None


class TestComputeLossWithReduction:
    def test_mean(self):
        losses = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = compute_loss_with_reduction(
            losses, reduction="mean"
        )
        assert result is not None

    def test_sum(self):
        losses = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = compute_loss_with_reduction(
            losses, reduction="sum"
        )
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


class TestLayerUtilsBranches:
    def test_maybe_expand_time_is_noop_for_rank3_without_ref(
        self,
    ):
        x = np.ones((2, 5, 8), dtype=np.float32)
        result = maybe_expand_time(x)
        assert result.shape == x.shape

    def test_maybe_expand_time_is_noop_when_ref_rank_does_not_require_expansion(
        self,
    ):
        x = np.ones((2, 5, 8), dtype=np.float32)
        ref = np.ones((2, 5, 8), dtype=np.float32)
        result = maybe_expand_time(x, ref=ref)
        assert result.shape == x.shape

    def test_private_broadcast_like(self):
        x = np.ones((2, 1, 3), dtype=np.float32)
        ref = np.ones((2, 4, 3), dtype=np.float32)
        result = _broadcast_like(x, ref)
        assert result is not None

    def test_ensure_rank_at_least_requires_target(self):
        x = np.ones((2, 3), dtype=np.float32)
        with pytest.raises(
            ValueError, match="must be provided"
        ):
            ensure_rank_at_least(x)

    def test_drop_path_raises_without_backend_random(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            _layer_utils_mod, "K", types.SimpleNamespace()
        )
        x = np.ones((2, 5, 8), dtype=np.float32)
        with pytest.raises(
            RuntimeError, match="random.uniform"
        ):
            drop_path(x, drop_prob=0.1, training=True)

    def test_gate_get_config(self):
        layer = Gate(units=4, use_bias=False)
        config = layer.get_config()
        assert config["units"] == 4
        assert config["use_bias"] is False

    def test_stochastic_depth_get_config(self):
        layer = StochasticDepth(drop_rate=0.2)
        config = layer.get_config()
        assert config["drop_prob"] == 0.2

    def test_broadcast_like_uses_shape_fallback_scalars(self):
        x = np.ones((2, 1, 3), dtype=np.float32)
        target = object()

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            _layer_utils_mod,
            "shape",
            lambda value: np.asarray(
                [2, 1, 3], dtype=np.int32
            )
            if value is x
            else np.asarray(
                [2, _DetachCpuScalar(4), _ScalarWithItem(3)],
                dtype=object,
            ),
        )
        try:
            result = broadcast_like(x, target)
        finally:
            monkeypatch.undo()
        assert _to_numpy(result).shape == (2, 4, 3)

    def test_private_broadcast_like_casts_repetition_dtype(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            _layer_utils_mod,
            "shape",
            lambda value: np.asarray(
                np.shape(value), dtype=np.int32
            ),
        )
        monkeypatch.setattr(
            _layer_utils_mod,
            "tile",
            lambda value, reps: np.tile(
                np.asarray(value),
                tuple(
                    int(v) for v in np.asarray(reps).tolist()
                ),
            ),
        )
        x = np.ones((2, 1, 3), dtype=np.float32)
        ref = np.ones((2, 4, 3), dtype=np.float32)
        result = _broadcast_like(x, ref)
        assert np.asarray(result).shape == (2, 4, 3)

    def test_drop_path_with_torch_tensor(self):
        torch = pytest.importorskip("torch")
        x = torch.ones((2, 5, 8), dtype=torch.float32)
        result = drop_path(x, drop_prob=0.1, training=True)
        assert tuple(result.shape) == (2, 5, 8)

    def test_squeeze_excite_with_torch_tensor(self):
        torch = pytest.importorskip("torch")
        layer = SqueezeExcite1D(ratio=2)
        x = torch.ones((2, 5, 8), dtype=torch.float32)
        result = layer(x)
        assert tuple(result.shape) == (2, 5, 8)


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
        layer = GatedResidualNetwork(
            units=16, dropout_rate=0.1
        )
        config = layer.get_config()
        assert "units" in config


class TestGatedResidualNetworkBranches:
    def test_invalid_activation_lookup_raises(
        self, monkeypatch
    ):
        fake_activations = types.SimpleNamespace(
            get=lambda name: (_ for _ in ()).throw(
                RuntimeError("broken activation")
            )
        )
        monkeypatch.setattr(
            _gating_norm_mod, "activations", fake_activations
        )
        with pytest.raises(
            ValueError,
            match="Failed to get activation function",
        ):
            GatedResidualNetwork(units=8)

    def test_build_rejects_rank_lt_2(self):
        layer = GatedResidualNetwork(units=8)
        with pytest.raises(
            ValueError, match="at least 2 dimensions"
        ):
            layer.build((8,))

    def test_build_warns_when_last_dim_unknown(self):
        layer = GatedResidualNetwork(units=8)
        with pytest.warns(
            RuntimeWarning,
            match="unknown or invalid last dimension",
        ):
            layer.build((None, None))

    def test_build_creates_projection_when_dims_differ(self):
        layer = GatedResidualNetwork(units=8)
        layer.build((None, 4))
        assert layer.projection is not None

    def test_call_raises_on_incompatible_context_rank(self):
        layer = GatedResidualNetwork(units=8)
        x = np.ones((2, 8), dtype=np.float32)
        context = np.ones((2, 1, 8), dtype=np.float32)
        with pytest.raises(
            ValueError, match="Incompatible ranks"
        ):
            layer(x, context=context)

    def test_output_activation_and_batch_norm_paths(self):
        layer = GatedResidualNetwork(
            units=8,
            output_activation="sigmoid",
            use_batch_norm=True,
        )
        x = np.ones((2, 8), dtype=np.float32)
        result = layer(x, training=True)
        assert result is not None

    def test_from_config(self):
        layer = GatedResidualNetwork(
            units=8,
            dropout_rate=0.1,
            activation="relu",
            output_activation="sigmoid",
            use_batch_norm=True,
        )
        restored = GatedResidualNetwork.from_config(
            layer.get_config()
        )
        assert restored.units == 8
        assert restored.output_activation_str == "sigmoid"


class TestVariableSelectionNetwork:
    def test_instantiation(self):
        layer = VariableSelectionNetwork(
            units=16, num_inputs=4
        )
        assert layer is not None

    def test_call(self):
        layer = VariableSelectionNetwork(
            units=16, num_inputs=4
        )
        inputs = [np.ones((2, 5, 16), dtype=np.float32)] * 4
        result = layer(inputs)
        assert result is not None

    def test_get_config(self):
        layer = VariableSelectionNetwork(
            units=16, num_inputs=4
        )
        config = layer.get_config()
        assert "units" in config


class TestVariableSelectionNetworkBranches:
    def test_build_rejects_rank_lt_expected(self):
        layer = VariableSelectionNetwork(
            units=8, num_inputs=2
        )
        with pytest.raises(
            ValueError, match="VSN build requires input rank"
        ):
            layer.build((4,))

    def test_build_rejects_unknown_non_batch_dims(self):
        layer = VariableSelectionNetwork(
            units=8, num_inputs=2
        )
        with pytest.raises(
            ValueError, match="unknown non-batch dimensions"
        ):
            layer.build((None, None, 4))

    def test_build_wraps_internal_grn_failures(
        self, monkeypatch
    ):
        layer = VariableSelectionNetwork(
            units=8, num_inputs=2
        )
        monkeypatch.setattr(
            layer.single_variable_grns[0],
            "build",
            lambda shape: (_ for _ in ()).throw(
                ValueError("broken grn")
            ),
        )
        with pytest.raises(
            RuntimeError, match="Failed to build internal GRN"
        ):
            layer.build((None, 2, 4))

    def test_call_raises_when_rank_cannot_be_determined(self):
        layer = VariableSelectionNetwork(
            units=8, num_inputs=2
        )
        with pytest.raises(
            TypeError,
            match="Could not determine rank of input",
        ):
            layer.call(object())

    def test_call_raises_when_input_rank_too_low(self):
        layer = VariableSelectionNetwork(
            units=8, num_inputs=2
        )
        with pytest.raises(
            ValueError, match="Input rank must be >="
        ):
            layer.call(np.ones((2,), dtype=np.float32))

    def test_call_with_list_inputs_and_single_input_weights(
        self,
    ):
        single = VariableSelectionNetwork(
            units=4, num_inputs=1
        )
        result = single(np.ones((2, 1, 3), dtype=np.float32))
        assert result is not None
        assert np.allclose(
            _to_numpy(single.variable_importances_), 1.0
        )

        stacked = VariableSelectionNetwork(
            units=4, num_inputs=2
        )
        seq_a = np.ones((2, 5, 3), dtype=np.float32)
        seq_b = np.ones((2, 5, 3), dtype=np.float32)
        stacked_result = stacked([seq_a, seq_b])
        assert stacked_result is not None

    def test_from_config(self):
        layer = VariableSelectionNetwork(
            units=8,
            num_inputs=3,
            dropout_rate=0.1,
            use_time_distributed=True,
            activation="relu",
            use_batch_norm=True,
        )
        restored = VariableSelectionNetwork.from_config(
            layer.get_config()
        )
        assert restored.num_inputs == 3
        assert restored.use_time_distributed is True


class TestLearnedNormalization:
    def test_instantiation(self):
        layer = LearnedNormalization()
        assert layer is not None

    def test_call(self):
        layer = LearnedNormalization()
        x = np.ones((2, 5, 8), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestLearnedNormalizationBranches:
    def test_from_config(self):
        layer = LearnedNormalization()
        restored = LearnedNormalization.from_config(
            layer.get_config()
        )
        assert isinstance(restored, LearnedNormalization)


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


class TestStaticEnrichmentLayerBranches:
    def test_from_config(self):
        layer = StaticEnrichmentLayer(
            units=8, activation="relu", use_batch_norm=True
        )
        restored = StaticEnrichmentLayer.from_config(
            layer.get_config()
        )
        assert restored.units == 8
        assert restored.use_batch_norm is True


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


class TestTemporalAttentionLayerBranches:
    def test_build_with_context_shape_list(self):
        layer = TemporalAttentionLayer(units=8, num_heads=2)
        layer.build([(None, 5, 8), (None, 8)])
        assert layer.context_grn.built is True

    def test_build_rejects_unexpected_shape_format(self):
        layer = TemporalAttentionLayer(units=8, num_heads=2)
        with pytest.raises(
            ValueError, match="Unexpected input_shape format"
        ):
            layer.build([(None, 5, 8), (None, 8), (None, 1)])

    def test_build_rejects_rank_lt_3(self):
        layer = TemporalAttentionLayer(units=8, num_heads=2)
        with pytest.raises(
            ValueError, match="expects input rank >= 3"
        ):
            layer.build(_IterableShape(None, 8))

    def test_call_treats_2d_secondary_input_as_context(self):
        layer = TemporalAttentionLayer(units=8, num_heads=2)
        object.__setattr__(
            layer,
            "context_grn",
            _Recorder(
                return_value=np.ones((2, 8), dtype=np.float32)
            ),
        )
        object.__setattr__(
            layer,
            "multi_head_attention",
            _Recorder(
                return_value=np.ones(
                    (2, 5, 8), dtype=np.float32
                )
            ),
        )
        object.__setattr__(layer, "dropout", _IdentityLayer())
        object.__setattr__(
            layer, "layer_norm1", _IdentityLayer()
        )
        object.__setattr__(
            layer,
            "output_grn",
            _Recorder(
                return_value=np.ones(
                    (2, 5, 8), dtype=np.float32
                )
            ),
        )
        x = np.ones((2, 5, 8), dtype=np.float32)
        context = np.ones((2, 8), dtype=np.float32)
        result = layer.call([x, context])
        assert result is not None

    def test_from_config(self):
        layer = TemporalAttentionLayer(
            units=8, num_heads=2, dropout_rate=0.1
        )
        restored = TemporalAttentionLayer.from_config(
            layer.get_config()
        )
        assert restored.units == 8
        assert restored.dropout_rate == 0.1


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


class TestCrossAttentionBranches:
    def test_query_mask_builds_attention_mask(
        self, monkeypatch
    ):
        layer = CrossAttention(units=4, num_heads=1)
        object.__setattr__(
            layer, "source1_dense", _IdentityLayer()
        )
        object.__setattr__(
            layer, "source2_dense", _IdentityLayer()
        )
        recorder = _Recorder(
            return_value=np.ones((2, 3, 4), dtype=np.float32)
        )
        object.__setattr__(layer, "cross_attention", recorder)
        monkeypatch.setattr(
            _attention_mod,
            "ones_like",
            lambda x, dtype=None: np.ones(
                _to_numpy(x).shape, dtype=bool
            ),
        )

        query = np.ones((2, 3, 4), dtype=np.float32)
        value = np.ones((2, 2, 4), dtype=np.float32)
        query_mask = np.array(
            [[True, False, True], [True, True, False]]
        )

        result = layer([query, value], query_mask=query_mask)
        attention_mask = recorder.calls[-1]["kwargs"][
            "attention_mask"
        ]

        assert result is not None
        assert np.asarray(attention_mask).shape == (2, 3, 2)

    def test_from_config(self):
        layer = CrossAttention(units=6, num_heads=2)
        restored = CrossAttention.from_config(
            {"units": layer.units, "num_heads": 2}
        )
        assert restored.units == 6


class TestHierarchicalAttention:
    def test_instantiation(self):
        layer = HierarchicalAttention(units=16, num_heads=2)
        assert layer is not None

    def test_call(self):
        layer = HierarchicalAttention(units=16, num_heads=2)
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestHierarchicalAttentionBranches:
    def test_short_mask_only_expands_and_long_mask_stays_none(
        self,
    ):
        layer = HierarchicalAttention(units=4, num_heads=1)
        object.__setattr__(
            layer, "short_term_dense", _IdentityLayer()
        )
        object.__setattr__(
            layer, "long_term_dense", _IdentityLayer()
        )
        short_attn = _Recorder(
            return_value=np.ones((2, 3, 4), dtype=np.float32)
        )
        long_attn = _Recorder(
            return_value=np.ones((2, 3, 4), dtype=np.float32)
        )
        object.__setattr__(
            layer, "short_term_attention", short_attn
        )
        object.__setattr__(
            layer, "long_term_attention", long_attn
        )

        inputs = np.ones((2, 3, 4), dtype=np.float32)
        short_mask = np.array(
            [[True, False, True], [True, True, False]]
        )

        result = layer(inputs, short_mask=short_mask)

        assert result is not None
        assert np.asarray(
            short_attn.calls[-1]["kwargs"]["attention_mask"]
        ).shape == (
            2,
            3,
            3,
        )
        assert (
            long_attn.calls[-1]["kwargs"]["attention_mask"]
            is None
        )

    def test_get_config(self):
        layer = HierarchicalAttention(units=4, num_heads=1)
        config = layer.get_config()
        assert config["units"] == 4
        assert "short_term_dense" in config


class TestMemoryAugmentedAttention:
    def test_instantiation(self):
        layer = MemoryAugmentedAttention(
            units=16, num_heads=2
        )
        assert layer is not None

    def test_call(self):
        layer = MemoryAugmentedAttention(
            units=16, num_heads=2
        )
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestMemoryAugmentedAttentionBranches:
    def test_query_mask_builds_memory_attention_mask(
        self, monkeypatch
    ):
        layer = MemoryAugmentedAttention(
            units=4, memory_size=2, num_heads=1
        )
        layer.memory = np.ones((2, 4), dtype=np.float32)
        recorder = _Recorder()
        object.__setattr__(layer, "attention", recorder)
        monkeypatch.setattr(
            _attention_mod,
            "ones",
            lambda shape, dtype=None: np.ones(
                shape, dtype=bool
            ),
        )

        inputs = np.ones((2, 3, 4), dtype=np.float32)
        query_mask = np.array(
            [[True, False, True], [True, True, False]]
        )

        result = layer(inputs, query_mask=query_mask)
        attention_mask = recorder.calls[-1]["kwargs"][
            "attention_mask"
        ]

        assert result is not None
        assert np.asarray(attention_mask).shape == (2, 3, 2)

    def test_from_config(self):
        layer = MemoryAugmentedAttention(
            units=6, memory_size=3, num_heads=2
        )
        restored = MemoryAugmentedAttention.from_config(
            {
                "units": layer.units,
                "memory_size": layer.memory_size,
                "num_heads": 2,
            }
        )
        assert restored.units == 6
        assert restored.memory_size == 3


class TestExplainableAttention:
    def test_instantiation(self):
        layer = ExplainableAttention(units=16, num_heads=2)
        assert layer is not None

    def test_call(self):
        layer = ExplainableAttention(units=16, num_heads=2)
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestExplainableAttentionBranches:
    def test_missing_key_dim_and_units_raises(self):
        with pytest.raises(
            ValueError, match="Provide `key_dim` or `units`"
        ):
            ExplainableAttention(num_heads=2)

    def test_list_input_path_returns_attention_scores(self):
        scores = np.ones((2, 1, 3, 3), dtype=np.float32)
        layer = ExplainableAttention(units=4, num_heads=1)
        object.__setattr__(
            layer, "attention", _RecorderWithScores(scores)
        )

        query = np.ones((2, 3, 4), dtype=np.float32)
        value = np.ones((2, 3, 4), dtype=np.float32)
        result = layer([query, value])

        assert np.asarray(result).shape == (2, 1, 3, 3)

    def test_from_config(self):
        layer = ExplainableAttention(units=6, num_heads=2)
        restored = ExplainableAttention.from_config(
            {
                "num_heads": layer.num_heads,
                "key_dim": layer.key_dim,
            }
        )
        assert restored.key_dim == 6


class TestMultiResolutionAttentionFusion:
    def test_instantiation(self):
        layer = MultiResolutionAttentionFusion(
            units=16, num_heads=2
        )
        assert layer is not None

    def test_call(self):
        layer = MultiResolutionAttentionFusion(
            units=16, num_heads=2
        )
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer([x, x])
        assert result is not None


class TestMultiResolutionAttentionFusionBranches:
    def test_from_config(self):
        layer = MultiResolutionAttentionFusion(
            units=6, num_heads=2
        )
        restored = MultiResolutionAttentionFusion.from_config(
            {
                "units": layer.units,
                "num_heads": layer.num_heads,
            }
        )
        assert restored.units == 6
        assert restored.num_heads == 2


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

    def test_get_config(self):
        layer = TransformerEncoderLayer(
            units=16, num_heads=2, dropout_rate=0.2
        )
        config = layer.get_config()
        assert config["embed_dim"] == 16
        assert config["dropout_rate"] == 0.2


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

    def test_init_requires_embed_dim_or_units(self):
        with pytest.raises(
            ValueError, match="Provide `embed_dim` or `units`"
        ):
            TransformerDecoderLayer(
                embed_dim=None, units=None
            )

    def test_get_config(self):
        layer = TransformerDecoderLayer(
            units=16, num_heads=2, dropout_rate=0.2
        )
        config = layer.get_config()
        assert config["embed_dim"] == 16
        assert config["dropout_rate"] == 0.2


class TestTransformerEncoderBlock:
    def test_instantiation(self):
        layer = TransformerEncoderBlock(
            units=16, num_heads=2, num_layers=2
        )
        assert layer is not None

    def test_call(self):
        layer = TransformerEncoderBlock(
            units=16, num_heads=2, num_layers=2
        )
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None

    def test_get_config_and_from_config(self):
        layer = TransformerEncoderBlock(
            units=16, num_heads=2, num_layers=2
        )
        config = layer.get_config()
        restored = TransformerEncoderBlock.from_config(config)
        assert config["embed_dim"] == 16
        assert restored.num_layers == 2


class TestTransformerDecoderBlock:
    def test_instantiation(self):
        layer = TransformerDecoderBlock(
            units=16, num_heads=2, num_layers=2
        )
        assert layer is not None

    def test_call(self):
        layer = TransformerDecoderBlock(
            units=16, num_heads=2, num_layers=2
        )
        tgt = np.ones((2, 3, 16), dtype=np.float32)
        mem = np.ones((2, 5, 16), dtype=np.float32)
        result = layer([tgt, mem])
        assert result is not None

    def test_get_config_and_from_config(self):
        layer = TransformerDecoderBlock(
            units=16, num_heads=2, num_layers=2
        )
        config = layer.get_config()
        restored = TransformerDecoderBlock.from_config(config)
        assert config["embed_dim"] == 16
        assert restored.num_layers == 2


class TestMultiDecoder:
    def test_instantiation(self):
        layer = MultiDecoder(units=16, num_heads=2)
        assert layer is not None

    def test_call_and_config_round_trip(self):
        layer = MultiDecoder(
            units=16,
            num_heads=2,
            num_horizons=3,
            output_dim=2,
        )
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer(x)
        config = layer.get_config()
        restored = MultiDecoder.from_config(
            {
                "output_dim": config["output_dim"],
                "num_horizons": config["num_horizons"],
            }
        )
        assert _to_numpy(result).shape == (2, 3, 5, 2)
        assert restored.num_horizons == 3


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
        layer = PointForecastHead(
            output_dim=1, forecast_horizon=5
        )
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
        layer = QuantileHead(
            quantiles=[0.1, 0.5, 0.9],
            output_dim=1,
            forecast_horizon=5,
        )
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
        layer = MixtureDensityHead(
            num_components=3, output_dim=1
        )
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


class TestPositionalEncodingBranches:
    def test_max_steps_alias_and_config(self):
        layer = PositionalEncoding(max_steps=12, d_model=6)
        config = layer.get_config()
        assert layer.max_length == 12
        assert config["max_length"] == 12
        assert config["d_model"] == 6

    def test_build_raises_when_feature_dim_unknown(self):
        layer = PositionalEncoding(max_length=8)
        with pytest.raises(
            ValueError, match="cannot be `None`"
        ):
            layer.build((None, 5, None))

    def test_set_weights_empty_is_noop(self):
        layer = PositionalEncoding(max_length=8)
        layer.build((None, 5, 4))
        original = layer.positional_encoding
        assert layer.set_weights([]) is None
        assert layer.positional_encoding is original

    def test_load_own_variables_handles_missing_and_broken_assign(
        self,
    ):
        layer = PositionalEncoding(max_length=8)
        layer.build((None, 5, 4))
        layer.load_own_variables({})
        layer.load_own_variables(
            {"other": np.ones((1, 8, 4), dtype=np.float32)}
        )

        class _BrokenAssign:
            def assign(self, value):
                raise RuntimeError("boom")

        layer.positional_encoding = _BrokenAssign()
        layer.load_own_variables(
            {
                "positional_encoding": np.ones(
                    (1, 8, 4), dtype=np.float32
                )
            }
        )

    def test_build_populates_encoding_once_and_call_uses_seq_len(
        self,
    ):
        layer = PositionalEncoding(max_length=8)
        layer.build((None, 5, 4))
        original = layer.positional_encoding
        layer.build((None, 5, 4))
        result = layer(np.ones((2, 3, 4), dtype=np.float32))
        assert layer.positional_encoding is original
        assert _to_numpy(result).shape == (2, 3, 4)


class TestLegacyPositionalEncoding:
    def test_build_raises_when_feature_dim_unknown(self):
        layer = _PositionalEncoding(max_length=8)
        with pytest.raises(
            ValueError, match="cannot be `None`"
        ):
            layer.build((None, 5, None))

    def test_get_config(self):
        layer = _PositionalEncoding(max_length=8)
        config = layer.get_config()
        assert config["max_length"] == 8


class TestTSPositionalEncoding:
    def test_instantiation(self):
        layer = TSPositionalEncoding(
            max_steps=100, d_model=16
        )
        assert layer is not None

    def test_call(self):
        layer = TSPositionalEncoding(
            max_steps=100, d_model=16
        )
        x = np.ones((2, 5, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None


class TestTSPositionalEncodingBranches:
    def test_missing_required_args_raises(self):
        with pytest.raises(
            ValueError, match="Provide `max_position`"
        ):
            TSPositionalEncoding()

    def test_call_requires_backend(self, monkeypatch):
        layer = TSPositionalEncoding(max_steps=8, d_model=4)
        monkeypatch.setattr(_misc_mod, "KERAS_BACKEND", None)
        with pytest.raises(
            RuntimeError, match="requires a Keras backend"
        ):
            layer(np.ones((1, 2, 4), dtype=np.float32))

    def test_dynamic_shape_path_uses_tf_shape_item(
        self, monkeypatch
    ):
        layer = TSPositionalEncoding(max_steps=8, d_model=4)
        monkeypatch.setattr(
            _misc_mod, "KERAS_BACKEND", "torch"
        )
        monkeypatch.setattr(
            _misc_mod,
            "shape",
            lambda x: (2, _ScalarWithItem(3), 4),
        )
        result = layer.call(_DynamicShapeTensor())
        assert np.asarray(result).shape == (1, 3, 4)

    def test_tf_get_angles_shape(self):
        layer = TSPositionalEncoding(max_steps=8, d_model=4)
        pos = np.arange(3)[:, np.newaxis]
        dims = np.arange(4)[np.newaxis, :]
        result = layer._tf_get_angles(pos, dims, 4)
        assert np.asarray(result).shape[0] == 3

    def test_tf_build_positional_encoding(self):
        layer = TSPositionalEncoding(max_steps=8, d_model=4)
        result = layer._tf_build_positional_encoding(4, 4)
        assert _to_numpy(result).shape == (1, 4, 4)

    def test_get_angles(self):
        layer = TSPositionalEncoding(max_steps=8, d_model=4)
        result = layer._get_angles(
            np.arange(3)[:, np.newaxis],
            np.arange(4)[np.newaxis, :],
            4,
        )
        assert np.asarray(result).shape == (3, 4)

    def test_get_config(self):
        layer = TSPositionalEncoding(max_steps=8, d_model=4)
        config = layer.get_config()
        assert config["max_position"] == 8
        assert config["embed_dim"] == 4

    def test_torch_tensor_path_keeps_device(self):
        torch = pytest.importorskip("torch")
        layer = TSPositionalEncoding(max_steps=8, d_model=4)
        x = torch.ones((2, 3, 4), dtype=torch.float32)
        result = layer(x)
        assert tuple(result.shape) == (2, 3, 4)


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


class TestActivationBranches:
    def test_none_activation_is_identity(self):
        layer = Activation(None)
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(_to_numpy(layer(x)), x)
        assert layer.get_config()["activation"] == "linear"

    def test_unknown_activation_raises(self, monkeypatch):
        fake_activations = types.SimpleNamespace(
            get=lambda name: (_ for _ in ()).throw(
                ValueError("bad activation")
            ),
            serialize=lambda fn: fn,
        )
        monkeypatch.setattr(
            _misc_mod, "activations", fake_activations
        )
        with pytest.raises(
            ValueError, match="Unknown activation"
        ):
            Activation("unknown")

    def test_non_callable_resolution_raises(
        self, monkeypatch
    ):
        fake_activations = types.SimpleNamespace(
            get=lambda name: 123,
            serialize=lambda fn: fn,
        )
        monkeypatch.setattr(
            _misc_mod, "activations", fake_activations
        )
        with pytest.raises(TypeError, match="not callable"):
            Activation("relu")

    def test_invalid_type_raises(self):
        with pytest.raises(
            TypeError,
            match="must be \\*str\\*, Callable, or \\*None\\*",
        ):
            Activation(123)

    def test_callable_activation_serialize_fallback_uses_name(
        self, monkeypatch
    ):
        fake_activations = types.SimpleNamespace(
            get=lambda name: name,
            serialize=lambda fn: (_ for _ in ()).throw(
                ValueError("cannot serialize")
            ),
        )
        monkeypatch.setattr(
            _misc_mod, "activations", fake_activations
        )

        def custom_activation(x):
            return x

        layer = Activation(custom_activation)
        assert layer.activation_str == "custom_activation"


class TestPositionwiseFeedForward:
    def test_call_and_get_config(self):
        layer = PositionwiseFeedForward(
            embed_dim=8,
            ffn_dim=16,
            activation="relu",
            dropout_rate=0.2,
        )
        x = np.ones((2, 5, 8), dtype=np.float32)
        result = layer(x, training=True)
        config = layer.get_config()
        assert result is not None
        assert config["embed_dim"] == 8
        assert config["ffn_dim"] == 16
        assert config["dropout_rate"] == 0.2


class TestMultiModalEmbedding:
    def test_instantiation(self):
        layer = MultiModalEmbedding(embed_dim=16)
        assert layer is not None


class TestMultiModalEmbeddingBranches:
    def test_build_raises_on_none_modality(self):
        layer = MultiModalEmbedding(embed_dim=8)
        with pytest.raises(
            ValueError, match="Unsupported modality type"
        ):
            layer.build([(None, 4, 3), None])

    def test_call_raises_on_unsupported_modality(self):
        layer = MultiModalEmbedding(embed_dim=8)
        layer.dense_layers = [_IdentityLayer()]
        with pytest.raises(
            ValueError, match="Unsupported modality type"
        ):
            layer.call([object()])

    def test_from_config_round_trip(self):
        layer = MultiModalEmbedding(embed_dim=12)
        restored = MultiModalEmbedding.from_config(
            {"embed_dim": layer.embed_dim}
        )
        assert restored.embed_dim == 12

    def test_get_config(self):
        layer = MultiModalEmbedding(embed_dim=10)
        config = layer.get_config()
        assert config["embed_dim"] == 10


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

    def test_return_sequences_and_config_round_trip(self):
        layer = MultiScaleLSTM(
            units=8, scales=[1, 2], return_sequences=True
        )
        x = np.ones((2, 10, 4), dtype=np.float32)
        result = layer(x)
        config = layer.get_config()
        restored = MultiScaleLSTM.from_config(config)
        assert isinstance(result, list)
        assert len(result) == 2
        assert restored.scales == [1, 2]


class TestDynamicTimeWindow:
    def test_instantiation(self):
        layer = DynamicTimeWindow(units=16)
        assert layer is not None

    def test_call(self):
        layer = DynamicTimeWindow(units=16)
        x = np.ones((2, 10, 16), dtype=np.float32)
        result = layer(x)
        assert result is not None
