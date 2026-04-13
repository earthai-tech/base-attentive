"""Tests for _keras_fallback.py — the pure-numpy fallback layer.

These tests run without any Keras / torch dependency; they exercise every
class and module-level function in the fallback module so that the overall
coverage rises even when Keras is unavailable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import base_attentive._keras_fallback as fb


# ---------------------------------------------------------------------------
# DTypeProxy
# ---------------------------------------------------------------------------

class TestDTypeProxy:
    def test_float32_callable(self):
        val = fb.float32(3.14)
        assert isinstance(val, (float, np.floating))

    def test_int32_callable(self):
        val = fb.int32(7)
        assert isinstance(val, (int, np.integer))

    def test_bool_callable(self):
        val = fb.bool(True)
        assert val  # truthy

    def test_repr_contains_dtype(self):
        r = repr(fb.float32)
        assert "float32" in r.lower()

    def test_has_numpy_dtype(self):
        assert hasattr(fb.float32, "as_numpy_dtype")


# ---------------------------------------------------------------------------
# TensorShape
# ---------------------------------------------------------------------------

class TestTensorShape:
    def test_create_from_list(self):
        ts = fb.TensorShape([2, 3, 4])
        assert len(ts) == 3

    def test_rank(self):
        assert fb.TensorShape([2, 3]).rank == 2

    def test_as_list(self):
        assert fb.TensorShape([2, 3]).as_list() == [2, 3]

    def test_concatenate(self):
        ts = fb.TensorShape([2])
        result = ts.concatenate([3, 4])
        assert list(result) == [2, 3, 4]

    def test_from_none(self):
        ts = fb.TensorShape(None)
        assert len(ts) == 0

    def test_from_existing_tensor_shape(self):
        ts1 = fb.TensorShape([2, 3])
        ts2 = fb.TensorShape(ts1)
        assert ts2 is ts1


# ---------------------------------------------------------------------------
# Variable
# ---------------------------------------------------------------------------

class TestVariable:
    def test_assign_updates_value(self):
        v = fb.Variable(np.array([1.0, 2.0]))
        v.assign([3.0, 4.0])
        assert v.value[0] == pytest.approx(3.0)

    def test_array_coercion(self):
        v = fb.Variable(np.array([1.0, 2.0]))
        arr = np.array(v)
        assert arr[0] == pytest.approx(1.0)

    def test_array_with_dtype(self):
        v = fb.Variable(np.array([1.0]))
        arr = v.__array__(dtype=np.float64)
        assert arr.dtype == np.float64

    def test_getitem(self):
        v = fb.Variable(np.array([10.0, 20.0, 30.0]))
        assert v[1] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_normalize_shape_list(self):
        assert fb._normalize_shape([2, 3]) == (2, 3)

    def test_normalize_shape_none(self):
        assert fb._normalize_shape(None) == ()

    def test_normalize_shape_scalar(self):
        assert fb._normalize_shape(5) == (5,)

    def test_normalize_shape_ndarray(self):
        assert fb._normalize_shape(np.array([2, 3])) == (2, 3)

    def test_ensure_array_no_dtype(self):
        arr = fb._ensure_array([1.0, 2.0])
        assert isinstance(arr, np.ndarray)

    def test_ensure_array_with_dtype_proxy(self):
        arr = fb._ensure_array([1.0], dtype=fb.float32)
        assert arr.dtype == np.float32

    def test_to_numpy_dtype_proxy(self):
        assert fb._to_numpy_dtype(fb.float32) == np.float32

    def test_to_numpy_dtype_none(self):
        assert fb._to_numpy_dtype(None) == np.float32

    def test_to_numpy_dtype_string(self):
        assert fb._to_numpy_dtype("float64") == np.float64

    def test_to_numpy_dtype_has_as_numpy_dtype_attr(self):
        class FakeDtype:
            as_numpy_dtype = np.int16
        assert fb._to_numpy_dtype(FakeDtype()) == np.int16

    def test_infer_input_shape_array(self):
        shape = fb._infer_input_shape(np.ones((2, 3)))
        assert list(shape) == [2, 3]

    def test_infer_input_shape_list(self):
        shapes = fb._infer_input_shape([np.ones((2, 3)), np.ones((4, 5))])
        assert isinstance(shapes, list)
        assert len(shapes) == 2


# ---------------------------------------------------------------------------
# Constant initializer
# ---------------------------------------------------------------------------

class TestConstant:
    def test_call_with_shape(self):
        c = fb.Constant(1.0)
        arr = c(shape=(3,), dtype=np.float32)
        assert arr.shape == (3,)
        assert arr[0] == pytest.approx(1.0)

    def test_call_no_shape(self):
        c = fb.Constant(2.0)
        arr = c()
        assert float(arr) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _initialize_value
# ---------------------------------------------------------------------------

class TestInitializeValue:
    def test_zeros_string(self):
        v = fb._initialize_value("zeros", (3,), fb.float32)
        assert np.all(v.value == 0)

    def test_ones_string(self):
        v = fb._initialize_value("ones", (3,), fb.float32)
        assert np.all(v.value == 1)

    def test_unknown_string_falls_back_to_zeros(self):
        v = fb._initialize_value("glorot_uniform", (3,), fb.float32)
        assert v.value.shape == (3,)

    def test_constant_initializer(self):
        c = fb.Constant(5.0)
        v = fb._initialize_value(c, (3,), fb.float32)
        assert np.all(v.value == pytest.approx(5.0))

    def test_none_initializer(self):
        v = fb._initialize_value(None, (3,), fb.float32)
        assert np.all(v.value == 0)

    def test_callable_with_dtype(self):
        v = fb._initialize_value(lambda shape, dtype=None: np.ones(shape), (3,), fb.float32)
        assert np.all(v.value == 1)

    def test_callable_without_dtype_arg(self):
        v = fb._initialize_value(lambda shape: np.ones(shape), (3,), fb.float32)
        assert np.all(v.value == 1)

    def test_scalar_broadcast(self):
        v = fb._initialize_value(np.float32(2.0), (4,), fb.float32)
        assert np.all(v.value == pytest.approx(2.0))

    def test_callable_scalar_broadcast(self):
        v = fb._initialize_value(lambda s, d=None: np.float32(3.0), (3,), fb.float32)
        assert v.value.shape == (3,)


# ---------------------------------------------------------------------------
# Layer base class
# ---------------------------------------------------------------------------

class TestLayerBase:
    def test_build_sets_built(self):
        layer = fb.Layer()
        layer.build((10, 5))
        assert layer.built

    def test_call_passthrough(self):
        layer = fb.Layer()
        x = np.ones((3, 4))
        result = layer(x)
        np.testing.assert_array_equal(result, x)

    def test_get_config_contains_name(self):
        layer = fb.Layer(name="my_layer")
        cfg = layer.get_config()
        assert cfg["name"] == "my_layer"

    def test_add_weight(self):
        layer = fb.Layer()
        w = layer.add_weight(name="w", shape=(3,), initializer="zeros")
        assert np.all(w.value == 0)

    def test_first_call_triggers_build(self):
        layer = fb.Layer()
        assert not layer.built
        layer(np.ones((3, 4)))
        assert layer.built

    def test_build_failure_sets_built(self):
        class BadBuild(fb.Layer):
            def build(self, shape):
                raise RuntimeError("oops")
        layer = BadBuild()
        layer(np.ones((2, 3)))  # should not raise
        assert layer.built


# ---------------------------------------------------------------------------
# Loss base class
# ---------------------------------------------------------------------------

class TestLossBase:
    def test_get_config_has_reduction(self):
        loss = fb.Loss(reduction="sum", name="my_loss")
        cfg = loss.get_config()
        assert cfg["reduction"] == "sum"


# ---------------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------------

class TestDense:
    def test_output_shape(self):
        d = fb.Dense(8)
        out = d(np.random.randn(4, 16).astype(np.float32))
        assert out.shape == (4, 8)

    def test_relu_activation(self):
        d = fb.Dense(4, activation="relu")
        out = d(np.ones((2, 4)))
        assert np.all(out >= 0)

    def test_sigmoid_activation(self):
        d = fb.Dense(4, activation="sigmoid")
        out = d(np.ones((2, 4)))
        assert np.all((out >= 0) & (out <= 1))

    def test_tanh_activation(self):
        d = fb.Dense(4, activation="tanh")
        out = d(np.ones((2, 4)))
        assert np.all(np.abs(out) <= 1.0 + 1e-6)

    def test_get_config(self):
        d = fb.Dense(8, name="my_dense")
        cfg = d.get_config()
        assert cfg["units"] == 8

    def test_scalar_input_zero_dim(self):
        d = fb.Dense(4)
        out = d(np.float32(1.0))
        assert out.shape == (4,)


# ---------------------------------------------------------------------------
# Dropout
# ---------------------------------------------------------------------------

class TestDropout:
    def test_passthrough(self):
        d = fb.Dropout(rate=0.5)
        x = np.ones((3, 4))
        out = d(x)
        np.testing.assert_array_equal(out, x)

    def test_get_config(self):
        cfg = fb.Dropout(rate=0.3, name="drop").get_config()
        assert cfg["rate"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# LayerNormalization / BatchNormalization
# ---------------------------------------------------------------------------

class TestLayerNorm:
    def test_output_shape_preserved(self):
        ln = fb.LayerNormalization()
        x = np.random.randn(4, 8).astype(np.float32)
        assert ln(x).shape == x.shape

    def test_get_config_epsilon(self):
        assert fb.LayerNormalization(epsilon=1e-5).get_config()["epsilon"] == pytest.approx(1e-5)

    def test_batch_norm_is_subclass(self):
        bn = fb.BatchNormalization()
        x = np.ones((3, 5))
        out = bn(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_sums_to_one(self):
        sm = fb.Softmax()
        out = sm(np.array([[1.0, 2.0, 3.0]]))
        np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-6)

    def test_get_config(self):
        assert fb.Softmax(axis=-1).get_config()["axis"] == -1


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class TestEmbedding:
    def test_output_shape(self):
        emb = fb.Embedding(100, 16)
        out = emb(np.array([[1, 2, 3]]))
        assert out.shape == (1, 3, 16)


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

class TestFlatten:
    def test_3d_flattened(self):
        out = fb.Flatten()(np.ones((3, 4, 5)))
        assert out.shape == (3, 20)

    def test_1d_input(self):
        out = fb.Flatten()(np.ones((5,)))
        assert out.ndim == 2


# ---------------------------------------------------------------------------
# Concatenate
# ---------------------------------------------------------------------------

class TestConcatenate:
    def test_concat_last_axis(self):
        cat = fb.Concatenate(axis=-1)
        out = cat([np.ones((3, 4)), np.ones((3, 2))])
        assert out.shape == (3, 6)


# ---------------------------------------------------------------------------
# TimeDistributed
# ---------------------------------------------------------------------------

class TestTimeDistributed:
    def test_3d_input(self):
        td = fb.TimeDistributed(fb.Dense(4))
        out = td(np.ones((2, 5, 8)))
        assert out.shape == (2, 5, 4)

    def test_2d_input(self):
        td = fb.TimeDistributed(fb.Dense(4))
        out = td(np.ones((2, 8)))
        assert out.shape == (2, 4)


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------

class TestSequential:
    def test_chain_of_layers(self):
        seq = fb.Sequential([fb.Dense(8), fb.Dense(4)])
        out = seq(np.ones((3, 16)))
        assert out.shape == (3, 4)

    def test_layer_without_training_arg(self):
        class NoTraining:
            def __call__(self, x):
                return x
        seq = fb.Sequential([NoTraining()])
        out = seq(np.ones((3, 4)))
        assert out.shape == (3, 4)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

class TestLSTM:
    def test_no_sequences_shape(self):
        lstm = fb.LSTM(16)
        out = lstm(np.random.randn(4, 10, 8).astype(np.float32))
        assert out.shape == (4, 16)

    def test_return_sequences_shape(self):
        lstm = fb.LSTM(16, return_sequences=True)
        out = lstm(np.random.randn(4, 10, 8).astype(np.float32))
        assert out.shape == (4, 10, 16)

    def test_2d_input(self):
        lstm = fb.LSTM(8)
        out = lstm(np.random.randn(4, 8).astype(np.float32))
        assert out.shape == (4, 8)


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

class TestMHA:
    def test_basic_call(self):
        mha = fb.MultiHeadAttention(num_heads=4, key_dim=8)
        q = np.random.randn(2, 5, 16).astype(np.float32)
        out = mha(q)
        assert out.shape == q.shape

    def test_return_attention_scores(self):
        mha = fb.MultiHeadAttention(num_heads=4, key_dim=8)
        q = np.random.randn(2, 5, 16).astype(np.float32)
        out, scores = mha(q, return_attention_scores=True)
        assert scores.shape[1] == 4  # num_heads

    def test_keyword_query_value(self):
        mha = fb.MultiHeadAttention(num_heads=2, key_dim=4)
        q = np.ones((2, 3, 8), dtype=np.float32)
        v = np.ones((2, 3, 8), dtype=np.float32)
        out = mha(query=q, value=v)
        assert out.shape == (2, 3, 8)

    def test_different_shapes_no_blend(self):
        mha = fb.MultiHeadAttention(num_heads=2, key_dim=4)
        q = np.ones((2, 3, 8), dtype=np.float32)
        v = np.ones((2, 5, 4), dtype=np.float32)
        out = mha(q, v)
        assert out.shape == q.shape

    def test_return_scores_2d_fallback(self):
        mha = fb.MultiHeadAttention(num_heads=2, key_dim=4)
        q = np.ones((4, 8), dtype=np.float32)
        out, scores = mha(q, return_attention_scores=True)
        assert scores is not None

    def test_key_from_args(self):
        mha = fb.MultiHeadAttention(num_heads=2, key_dim=4)
        q = np.ones((2, 3, 8), dtype=np.float32)
        v = np.ones((2, 3, 8), dtype=np.float32)
        k = np.ones((2, 3, 8), dtype=np.float32)
        out = mha(q, v, k)
        assert out is not None


# ---------------------------------------------------------------------------
# Input helper
# ---------------------------------------------------------------------------

class TestInput:
    def test_zeros_shape(self):
        x = fb.Input(shape=(3, 4))
        assert x.shape == (3, 4)

    def test_with_dtype(self):
        x = fb.Input(shape=(3,), dtype=fb.int32)
        assert x.dtype == np.int32


# ---------------------------------------------------------------------------
# register_keras_serializable
# ---------------------------------------------------------------------------

class TestRegisterSerializable:
    def test_returns_class_unchanged(self):
        @fb.register_keras_serializable("pkg")
        class MyClass:
            pass
        assert MyClass is not None


# ---------------------------------------------------------------------------
# _Activations
# ---------------------------------------------------------------------------

class TestActivations:
    @pytest.mark.parametrize("name,x,check", [
        ("relu", [-1.0, 0.0, 1.0], lambda r: np.all(r >= 0)),
        ("sigmoid", [0.0], lambda r: 0 < r[0] < 1),
        ("tanh", [0.0], lambda r: abs(r[0]) < 1e-6),
        ("elu", [-1.0, 1.0], lambda r: r[0] < 0 and r[1] == pytest.approx(1.0)),
        ("selu", [1.0], lambda r: r[0] > 0),
        ("gelu", [0.0], lambda r: abs(r[0]) < 1e-6),
        ("linear", [1.0, 2.0], lambda r: r[0] == pytest.approx(1.0)),
    ])
    def test_named_activations(self, name, x, check):
        f = fb.activations.get(name)
        result = f(np.array(x, dtype=np.float32))
        assert check(result)

    def test_callable_activation(self):
        f = fb.activations.get(lambda x: x * 2)
        assert f(np.array([1.0]))[0] == pytest.approx(2.0)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            fb.activations.get("not_a_real_activation")

    def test_serialize_string(self):
        assert fb.activations.serialize("relu") == "relu"

    def test_serialize_callable(self):
        def my_act(x): return x
        assert fb.activations.serialize(my_act) == "my_act"

    def test_serialize_class_instance(self):
        result = fb.activations.serialize(fb.Softmax())
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _RandomNamespace
# ---------------------------------------------------------------------------

class TestRandom:
    def test_uniform_shape(self):
        assert fb.random.uniform([3, 4]).shape == (3, 4)

    def test_normal_shape(self):
        assert fb.random.normal([2, 5]).shape == (2, 5)

    def test_uniform_range(self):
        r = fb.random.uniform([1000], minval=0.0, maxval=1.0)
        assert r.min() >= 0.0 and r.max() <= 1.0


# ---------------------------------------------------------------------------
# _LinalgNamespace
# ---------------------------------------------------------------------------

class TestLinalg:
    def test_band_part_diagonal(self):
        x = np.ones((4, 4))
        result = fb.linalg.band_part(x, 0, 0)
        np.testing.assert_array_equal(np.diag(result), np.ones(4))
        assert result[0, 1] == 0.0

    def test_band_part_lower(self):
        x = np.ones((3, 3))
        result = fb.linalg.band_part(x, -1, 0)  # lower triangle
        assert result[2, 0] == 1.0
        assert result[0, 2] == 0.0


# ---------------------------------------------------------------------------
# Module-level functional ops
# ---------------------------------------------------------------------------

class TestFunctionalOps:
    def test_constant(self):
        arr = fb.constant([1.0, 2.0], dtype=fb.float32)
        assert arr[0] == pytest.approx(1.0)

    def test_cast(self):
        arr = fb.cast(np.array([1.0, 2.0]), fb.int32)
        assert arr.dtype == np.int32

    def test_shape(self):
        s = fb.shape(np.ones((3, 4)))
        assert list(s) == [3, 4]

    def test_reshape(self):
        arr = fb.reshape(np.arange(6), [2, 3])
        assert arr.shape == (2, 3)

    def test_repeat(self):
        arr = fb.repeat(np.array([1, 2]), 3)
        assert len(arr) == 6

    def test_add(self):
        np.testing.assert_array_equal(fb.add([1, 2], [3, 4]), [4, 6])

    def test_maximum(self):
        np.testing.assert_array_equal(fb.maximum([1, 3], [2, 2]), [2, 3])

    def test_mean(self):
        assert fb.mean([1.0, 3.0]) == pytest.approx(2.0)

    def test_mean_keepdims(self):
        result = fb.mean(np.ones((3, 4)), axis=1, keepdims=True)
        assert result.shape == (3, 1)

    def test_reduce_mean(self):
        assert fb.reduce_mean([2.0, 4.0]) == pytest.approx(3.0)

    def test_add_n(self):
        result = fb.add_n([np.array([1.0, 2.0]), np.array([3.0, 4.0])])
        np.testing.assert_allclose(result, [4.0, 6.0])

    def test_square(self):
        np.testing.assert_array_equal(fb.square([2.0, 3.0]), [4.0, 9.0])

    def test_transpose(self):
        arr = np.arange(6).reshape(2, 3)
        assert fb.transpose(arr).shape == (3, 2)

    def test_transpose_with_perm(self):
        arr = np.ones((2, 3, 4))
        assert fb.transpose(arr, perm=[2, 0, 1]).shape == (4, 2, 3)

    def test_logical_and(self):
        np.testing.assert_array_equal(
            fb.logical_and([True, False], [True, True]), [True, False]
        )

    def test_logical_not(self):
        np.testing.assert_array_equal(fb.logical_not([True, False]), [False, True])

    def test_logical_or(self):
        np.testing.assert_array_equal(
            fb.logical_or([True, False], [False, False]), [True, False]
        )

    def test_get_static_value_scalar(self):
        assert fb.get_static_value(42) == 42

    def test_get_static_value_numpy_scalar(self):
        assert fb.get_static_value(np.float32(3.14)) == pytest.approx(3.14, rel=1e-4)

    def test_get_static_value_array(self):
        assert fb.get_static_value(np.array([1, 2])) is None

    def test_reduce_sum(self):
        assert fb.reduce_sum([1.0, 2.0, 3.0]) == pytest.approx(6.0)

    def test_stack(self):
        result = fb.stack([np.array([1, 2]), np.array([3, 4])])
        assert result.shape == (2, 2)

    def test_unstack(self):
        arr = np.array([[1, 2], [3, 4]])
        parts = fb.unstack(arr, axis=0)
        assert len(parts) == 2

    def test_expand_dims(self):
        arr = np.ones((3, 4))
        assert fb.expand_dims(arr, axis=0).shape == (1, 3, 4)

    def test_tile(self):
        arr = np.array([[1, 2]])
        assert fb.tile(arr, [2, 3]).shape == (2, 6)

    def test_where_two_args(self):
        result = fb.where([True, False], [10, 20], [30, 40])
        np.testing.assert_array_equal(result, [10, 40])

    def test_where_no_args(self):
        result = fb.where([False, True, False])
        assert 1 in result

    def test_assert_returns_condition(self):
        assert fb.Assert(True) is True

    def test_rank(self):
        assert fb.rank(np.ones((2, 3, 4))) == 3

    def test_split_with_int(self):
        parts = fb.split(np.arange(6), 3)
        assert len(parts) == 3
        np.testing.assert_array_equal(parts[0], [0, 1])

    def test_split_with_sizes(self):
        parts = fb.split(np.arange(6), [1, 2, 3])
        assert len(parts) == 3
        np.testing.assert_array_equal(parts[1], [1, 2])

    def test_multiply(self):
        np.testing.assert_array_equal(fb.multiply([1, 2], [3, 4]), [3, 8])

    def test_cond_true_and_false(self):
        assert fb.cond(True, lambda: "yes", lambda: "no") == "yes"
        assert fb.cond(False, lambda: "yes", lambda: "no") == "no"

    def test_equal(self):
        np.testing.assert_array_equal(fb.equal([1, 2], [1, 3]), [True, False])

    def test_pad(self):
        result = fb.pad(np.array([1, 2]), [(1, 1)], constant_values=0)
        np.testing.assert_array_equal(result, [0, 1, 2, 0])

    def test_ones_like(self):
        result = fb.ones_like(np.array([1, 2]), dtype=fb.float32)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, [1.0, 1.0])

    def test_abs(self):
        np.testing.assert_array_equal(fb.abs([-1, 2]), [1, 2])

    def test_pow(self):
        np.testing.assert_array_equal(fb.pow([2, 3], 2), [4, 9])

    def test_sin_cos_exp_log_sigmoid(self):
        values = np.array([0.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(fb.sin(values), np.sin(values))
        np.testing.assert_allclose(fb.cos(values), np.cos(values))
        np.testing.assert_allclose(fb.exp(values), np.exp(values))
        np.testing.assert_allclose(fb.log(np.array([1.0, np.e], dtype=np.float32)), [0.0, 1.0], atol=1e-6)
        sigmoid = fb.sigmoid(values)
        assert np.all((sigmoid > 0.0) & (sigmoid < 1.0))

    def test_cumsum(self):
        np.testing.assert_array_equal(fb.cumsum([1, 2, 3]), [1, 3, 6])

    def test_gather_axis_zero(self):
        params = np.arange(12).reshape(3, 4)
        result = fb.gather(params, [0, 2], axis=0)
        np.testing.assert_array_equal(result, params[[0, 2]])

    def test_gather_with_batch_dims(self):
        params = np.arange(24).reshape(2, 3, 4)
        indices = np.array([[[0], [2]], [[1], [0]]], dtype=np.int32)
        result = fb.gather(params, indices, axis=1, batch_dims=1)
        assert result.shape == (2, 2, 1, 4)
        np.testing.assert_array_equal(result[0, 0, 0], params[0, 0])
        np.testing.assert_array_equal(result[1, 0, 0], params[1, 1])

    def test_softplus(self):
        result = fb.softplus(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
        assert np.all(result > 0.0)

    def test_reduce_logsumexp(self):
        values = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
        reduced = fb.reduce_logsumexp(values, axis=1)
        assert reduced.shape == (2,)
        kept = fb.reduce_logsumexp(values, axis=1, keepdims=True)
        assert kept.shape == (2, 1)

    def test_sqrt(self):
        np.testing.assert_allclose(fb.sqrt([1.0, 4.0]), [1.0, 2.0])

    def test_ones(self):
        result = fb.ones((2, 3), dtype=fb.float32)
        assert result.shape == (2, 3)
        assert result.dtype == np.float32

    def test_floordiv(self):
        np.testing.assert_array_equal(fb.floordiv([5, 9], 2), [2, 4])

    def test_greater(self):
        np.testing.assert_array_equal(fb.greater([1, 3], [2, 2]), [False, True])


class TestFallbackLayerWeightInit:
    def test_add_weight_broadcasts_scalar_initializer(self):
        layer = fb.Layer()
        weight = layer.add_weight(shape=(2, 2), initializer=3.0, dtype=fb.float32)
        np.testing.assert_array_equal(np.asarray(weight), np.full((2, 2), 3.0, dtype=np.float32))


# ---------------------------------------------------------------------------
# _AutographExperimental
# ---------------------------------------------------------------------------

class TestAutograph:
    def test_do_not_convert_direct(self):
        @fb.autograph.experimental.do_not_convert
        def my_fn(x):
            return x + 1
        assert my_fn(2) == 3

    def test_do_not_convert_with_kwargs(self):
        decorator = fb.autograph.experimental.do_not_convert(some_kwarg=True)
        def my_fn(x):
            return x * 2
        wrapped = decorator(my_fn)
        assert wrapped(3) == 6
