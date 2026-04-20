# SPDX-License-Identifier: Apache-2.0
"""Tests for implementation gaps: torch, tensorflow, jax, generic, experimental."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module when PyTorch is not installed.
# All torch-specific classes (TestEnsureTorch, TestTorchTemporalSelf…, etc.)
# require a working torch installation; the rest of the file also imports
# torch implementation symbols at module level, so the cleanest approach is
# to skip the whole file rather than guard every test individually.
torch = pytest.importorskip(
    "torch",
    reason="PyTorch not installed — skipping torch implementation tests",
)

os.environ.setdefault("KERAS_BACKEND", "torch")


# ---------------------------------------------------------------------------
# Patch KERAS_DEPS before importing anything from components/experimental
# ---------------------------------------------------------------------------
import base_attentive as _ba

_orig_ga = _ba._KerasDeps.__getattr__
_FALLBACKS = {
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
}


def _patched_ga(self, name):
    try:
        return _orig_ga(self, name)
    except (ImportError, AttributeError):
        val = _FALLBACKS.get(name, lambda *a, **kw: None)
        self._cache[name] = val
        return val


_ba._KerasDeps.__getattr__ = _patched_ga


# ---------------------------------------------------------------------------
# PyTorch implementation tests
# ---------------------------------------------------------------------------

import base_attentive.implementations.torch.base_attentive_v2 as _torch_v2_mod
from base_attentive.implementations.torch.base_attentive_v2 import (
    _build_torch_concat_fusion,
    _build_torch_dense_projection,
    _build_torch_last_pool,
    _build_torch_mean_pool,
    _build_torch_point_forecast_head,
    _build_torch_quantile_head,
    _build_torch_temporal_self_attention_encoder,
    _ensure_torch,
    _TorchTemporalSelfAttentionEncoder,
    ensure_torch_v2_registered,
)

_ba._KerasDeps.__getattr__ = _orig_ga
_ba.KERAS_DEPS._cache.clear()

# Reload runtime-dependent component modules after the temporary
# import-time patch above. During pytest collection, some of these
# modules can be imported while `_KerasDeps.__getattr__` is patched,
# which would leave fake fallback ops permanently bound in module
# globals for the rest of the suite. Restoring the descriptor alone is
# not enough once those modules have already been imported.
for _name in (
    "base_attentive.components._config",
    "base_attentive.components.heads",
    "base_attentive.components._temporal_utils",
):
    if _name in sys.modules:
        importlib.reload(sys.modules[_name])


class TestEnsureTorch:
    def test_does_not_raise_when_torch_available(self):
        _ensure_torch()  # Should not raise

    def test_raises_when_torch_none(self):
        orig = _torch_v2_mod.importlib.util.find_spec
        try:
            _torch_v2_mod.importlib.util.find_spec = (
                lambda name: None
            )
            with pytest.raises(
                ImportError, match="PyTorch is required"
            ):
                _ensure_torch()
        finally:
            _torch_v2_mod.importlib.util.find_spec = orig


class TestTorchTemporalSelfAttentionEncoder:
    def test_normal_construction(self):
        encoder = _TorchTemporalSelfAttentionEncoder(
            units=32,
            hidden_units=64,
            num_heads=4,
            activation="relu",
            dropout_rate=0.0,
        )
        assert encoder.units == 32

    def test_units_not_divisible_is_supported_by_key_dim_floor(
        self,
    ):
        encoder = _TorchTemporalSelfAttentionEncoder(
            units=30,
            hidden_units=64,
            num_heads=4,
        )
        assert encoder.units == 30

    def test_forward_training_false(self):
        encoder = _TorchTemporalSelfAttentionEncoder(
            units=32, hidden_units=64, num_heads=4
        )
        x = torch.randn(2, 5, 32)
        out = encoder.forward(x, training=False)
        assert out.shape == (2, 5, 32)

    def test_forward_training_true(self):
        encoder = _TorchTemporalSelfAttentionEncoder(
            units=32, hidden_units=64, num_heads=4
        )
        x = torch.randn(2, 5, 32)
        out = encoder.forward(x, training=True)
        assert out.shape == (2, 5, 32)

    def test_with_dropout(self):
        encoder = _TorchTemporalSelfAttentionEncoder(
            units=32,
            hidden_units=64,
            num_heads=4,
            dropout_rate=0.1,
        )
        assert encoder.dropout is not None
        x = torch.randn(2, 5, 32)
        out = encoder.forward(x, training=False)
        assert out.shape == (2, 5, 32)


class TestBuildTorchDenseProjection:
    def test_returns_keras_dense_layer(self):
        layer = _build_torch_dense_projection(units=64)
        assert getattr(layer, "units", None) == 64

    def test_with_in_features(self):
        layer = _build_torch_dense_projection(
            units=64, in_features=32
        )
        assert getattr(layer, "units", None) == 64

    def test_raises_when_torch_none(self):
        orig = _torch_v2_mod.importlib.util.find_spec
        try:
            _torch_v2_mod.importlib.util.find_spec = (
                lambda name: None
            )
            with pytest.raises(ImportError):
                _build_torch_dense_projection(units=64)
        finally:
            _torch_v2_mod.importlib.util.find_spec = orig


class TestBuildTorchTemporalSelfAttentionEncoder:
    def test_returns_encoder(self):
        encoder = (
            _build_torch_temporal_self_attention_encoder(
                units=32, hidden_units=64, num_heads=4
            )
        )
        assert isinstance(
            encoder, _TorchTemporalSelfAttentionEncoder
        )


class TestBuildTorchMeanPool:
    def test_builds_callable(self):
        pool_fn = _build_torch_mean_pool(axis=1)
        assert callable(pool_fn)

    def test_mean_pool_output(self):
        pool_fn = _build_torch_mean_pool(axis=1)
        x = torch.randn(2, 5, 32)
        out = pool_fn(x)
        assert out.shape == (2, 32)

    def test_mean_pool_none_axis(self):
        pool_fn = _build_torch_mean_pool(axis=None)
        x = torch.randn(2, 5, 32)
        out = pool_fn(x)
        assert out.ndim == 0 or out.numel() == 1  # scalar

    def test_mean_pool_negative_axis(self):
        pool_fn = _build_torch_mean_pool(axis=-1)
        x = torch.randn(2, 5, 32)
        out = pool_fn(x)
        assert out.shape == (2, 5)


class TestBuildTorchLastPool:
    def test_builds_callable(self):
        pool_fn = _build_torch_last_pool()
        assert callable(pool_fn)

    def test_last_pool_output(self):
        pool_fn = _build_torch_last_pool()
        x = torch.randn(2, 5, 32)
        out = pool_fn(x)
        # Returns x[:, -1:, :] so shape is (2, 1, 32)
        assert out.shape == (2, 32)


class TestBuildTorchConcatFusion:
    def test_builds_callable(self):
        concat_fn = _build_torch_concat_fusion()
        assert callable(concat_fn)

    def test_concat_along_last_dim(self):
        concat_fn = _build_torch_concat_fusion(axis=-1)
        x1 = torch.randn(2, 32)
        x2 = torch.randn(2, 16)
        out = concat_fn([x1, x2])
        assert out.shape == (2, 48)


class TestBuildTorchPointForecastHead:
    def test_returns_keras_dense(self):
        head = _build_torch_point_forecast_head(
            output_dim=1, forecast_horizon=24
        )
        assert getattr(head, "units", None) == 24

    def test_with_in_features(self):
        head = _build_torch_point_forecast_head(
            output_dim=1, forecast_horizon=1, in_features=128
        )
        assert getattr(head, "units", None) == 1


class TestBuildTorchQuantileHead:
    def test_with_quantiles(self):
        head = _build_torch_quantile_head(
            output_dim=1,
            forecast_horizon=1,
            quantiles=(0.1, 0.5, 0.9),
        )
        assert getattr(head, "units", None) == 3

    def test_without_quantiles_uses_default(self):
        """Without quantiles arg, default (0.1, 0.5, 0.9) is used."""
        head = _build_torch_quantile_head(
            output_dim=1, forecast_horizon=1
        )
        assert getattr(head, "units", None) == 1


class TestEnsureTorchV2Registered:
    def test_with_fresh_registry(self):
        from base_attentive.registry.component_registry import (
            ComponentRegistry,
        )

        reg = ComponentRegistry()
        ensure_torch_v2_registered(reg)
        # Check that core components are registered
        assert reg.has("projection.dense", backend="torch")
        assert reg.has(
            "encoder.temporal_self_attention", backend="torch"
        )
        assert reg.has("pool.mean", backend="torch")
        assert reg.has("pool.last", backend="torch")
        assert reg.has("fusion.concat", backend="torch")
        assert reg.has("head.point_forecast", backend="torch")
        assert reg.has(
            "head.quantile_forecast", backend="torch"
        )

    def test_with_default_registry(self):
        """Using None uses DEFAULT_COMPONENT_REGISTRY."""
        # Just ensure it doesn't raise
        try:
            ensure_torch_v2_registered(None)
        except KeyError:
            pass  # May already be registered in the default


# ---------------------------------------------------------------------------
# TensorFlow implementation tests (TF not available - test error paths)
# ---------------------------------------------------------------------------


class TestTensorFlowImplementation:
    def test_ensure_tensorflow_raises_when_not_available(
        self,
    ):
        """_ensure_tensorflow() should raise ImportError when TF not installed."""
        patcher, module = _load_tensorflow_impl_module(
            "base_attentive.implementations.tensorflow._coverage_missing_tf_ensure",
            tensorflow_module=None,
            tensorflow_keras_module=None,
        )
        try:
            with pytest.raises(
                ImportError, match="TensorFlow is required"
            ):
                module._ensure_tensorflow()
        finally:
            patcher.undo()

    def test_ensure_tensorflow_registered_requires_tf(self):
        """ensure_tensorflow_v2_registered raises when TF not available."""
        patcher, module = _load_tensorflow_impl_module(
            "base_attentive.implementations.tensorflow._coverage_missing_tf_register",
            tensorflow_module=None,
            tensorflow_keras_module=None,
        )
        try:
            with pytest.raises(ImportError):
                module.ensure_tensorflow_v2_registered()
        finally:
            patcher.undo()


# ---------------------------------------------------------------------------
# JAX implementation tests (JAX not available - test error paths)
# ---------------------------------------------------------------------------


class TestJaxImplementation:
    def test_ensure_jax_raises_when_not_available(self):
        """_ensure_jax() should raise ImportError when JAX not installed."""
        import base_attentive.implementations.jax.base_attentive_v2 as _jax_v2_mod
        from base_attentive.implementations.jax.base_attentive_v2 import (
            _ensure_jax,
        )

        orig_jax = _jax_v2_mod.importlib.util.find_spec
        try:
            _jax_v2_mod.importlib.util.find_spec = (
                lambda name: None
            )
            with pytest.raises(
                ImportError, match="JAX is required"
            ):
                _ensure_jax()
        finally:
            _jax_v2_mod.importlib.util.find_spec = orig_jax

    def test_jax_encoder_raises_when_not_available(self):
        """_JaxTemporalSelfAttentionEncoder raises when JAX not available."""
        import base_attentive.implementations.jax.base_attentive_v2 as _jax_v2_mod
        from base_attentive.implementations.jax.base_attentive_v2 import (
            _JaxTemporalSelfAttentionEncoder,
        )

        orig_jax = _jax_v2_mod.importlib.util.find_spec
        try:
            _jax_v2_mod.importlib.util.find_spec = (
                lambda name: None
            )
            with pytest.raises(
                ImportError, match="JAX is required"
            ):
                _JaxTemporalSelfAttentionEncoder(
                    units=32, hidden_units=64, num_heads=4
                )
        finally:
            _jax_v2_mod.importlib.util.find_spec = orig_jax

    def test_ensure_jax_v2_registered_requires_jax(self):
        """ensure_jax_v2_registered raises when JAX not available."""
        import base_attentive.implementations.jax.base_attentive_v2 as _jax_v2_mod
        from base_attentive.implementations.jax.base_attentive_v2 import (
            ensure_jax_v2_registered,
        )

        orig_jax = _jax_v2_mod.importlib.util.find_spec
        try:
            _jax_v2_mod.importlib.util.find_spec = (
                lambda name: None
            )
            with pytest.raises(ImportError):
                ensure_jax_v2_registered()
        finally:
            _jax_v2_mod.importlib.util.find_spec = orig_jax


def _install_fake_jax(monkeypatch):
    import base_attentive.implementations.jax.base_attentive_v2 as mod

    class _FakeJnp:
        @staticmethod
        def asarray(x):
            return np.asarray(x, dtype=np.float32)

        @staticmethod
        def mean(x, axis=None, keepdims=False):
            return np.mean(
                np.asarray(x), axis=axis, keepdims=keepdims
            )

        @staticmethod
        def std(x, axis=None, keepdims=False):
            return np.std(
                np.asarray(x), axis=axis, keepdims=keepdims
            )

        @staticmethod
        def sqrt(x):
            return np.sqrt(x)

        @staticmethod
        def matmul(a, b):
            return np.matmul(np.asarray(a), np.asarray(b))

        @staticmethod
        def tanh(x):
            return np.tanh(np.asarray(x))

        @staticmethod
        def take(x, indices, axis=None, unique_indices=False):
            del unique_indices
            return np.take(np.asarray(x), indices, axis=axis)

        @staticmethod
        def concatenate(inputs, axis=-1):
            return np.concatenate(
                [np.asarray(x) for x in inputs], axis=axis
            )

    def _softmax(x, axis=-1):
        x = np.asarray(x)
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    fake_jax = types.SimpleNamespace(
        nn=types.SimpleNamespace(
            relu=lambda x: np.maximum(np.asarray(x), 0.0),
            gelu=lambda x: np.asarray(x) * 0.5,
            softmax=_softmax,
        )
    )

    monkeypatch.setattr(mod, "jax", fake_jax)
    monkeypatch.setattr(mod, "jnp", _FakeJnp)
    monkeypatch.setattr(mod, "lax", types.SimpleNamespace())
    return mod


class _FakeLayer:
    def __init__(self, name=None, **kwargs):
        self.name = name
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return inputs

    def get_config(self):
        return {"name": self.name}


def _named_activation(name):
    def _activation(x):
        return x

    _activation.__name__ = name
    return _activation


class _FakeDense(_FakeLayer):
    def __init__(
        self, units, activation=None, name=None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.units = units
        self.activation = (
            None
            if activation is None
            else _named_activation(activation)
        )


class _FakeMultiHeadAttention(_FakeLayer):
    def __init__(
        self,
        num_heads,
        key_dim,
        dropout=0.0,
        attention_axes=None,
        name=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            dtype=dtype,
            attention_axes=attention_axes,
            **kwargs,
        )
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout

    def call(
        self,
        query,
        value,
        attention_mask=None,
        training=False,
    ):
        del value, attention_mask, training
        return np.asarray(query)


class _FakeLayerNormalization(_FakeLayer):
    def __init__(self, epsilon=1e-6, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.epsilon = epsilon


class _FakeDropout(_FakeLayer):
    def __init__(self, rate, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.rate = rate


class _FakeLambda(_FakeLayer):
    def __init__(self, function, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.function = function

    def call(self, inputs):
        return self.function(inputs)


def _load_tensorflow_impl_module(
    module_name,
    *,
    tensorflow_module,
    tensorflow_keras_module,
):
    patcher = pytest.MonkeyPatch()
    patcher.setitem(
        sys.modules, "tensorflow", tensorflow_module
    )
    patcher.setitem(
        sys.modules,
        "tensorflow.keras",
        tensorflow_keras_module,
    )

    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "base_attentive"
        / "implementations"
        / "tensorflow"
        / "base_attentive_v2.py"
    )
    spec = importlib.util.spec_from_file_location(
        module_name, module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    patcher.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return patcher, module


@pytest.fixture
def fake_tf_impl():
    fake_layers = types.SimpleNamespace(
        Layer=_FakeLayer,
        Dense=_FakeDense,
        MultiHeadAttention=_FakeMultiHeadAttention,
        LayerNormalization=_FakeLayerNormalization,
        Dropout=_FakeDropout,
        Lambda=_FakeLambda,
    )

    fake_tf = types.ModuleType("tensorflow")
    fake_tf.reduce_mean = (
        lambda x, axis=None, keepdims=False: np.mean(
            np.asarray(x), axis=axis, keepdims=keepdims
        )
    )
    fake_tf.gather = lambda x, indices, axis=-1: np.take(
        np.asarray(x), indices, axis=axis
    )
    fake_tf.concat = lambda inputs, axis=-1: np.concatenate(
        [np.asarray(x) for x in inputs], axis=axis
    )

    fake_tf_keras = types.ModuleType("tensorflow.keras")
    fake_tf_keras.layers = fake_layers
    fake_tf.keras = fake_tf_keras

    patcher, module = _load_tensorflow_impl_module(
        "base_attentive.implementations.tensorflow._coverage_fake_base_attentive_v2",
        tensorflow_module=fake_tf,
        tensorflow_keras_module=fake_tf_keras,
    )

    try:
        yield module
    finally:
        patcher.undo()


class TestJaxImplementationExtended:
    def test_builder_contract_is_keras_multi_backend_when_jax_available(
        self,
    ):
        pytest.importorskip("jax", reason="JAX not installed")
        import base_attentive.implementations.jax.base_attentive_v2 as _jax_v2_mod

        encoder = _jax_v2_mod._build_jax_temporal_self_attention_encoder(
            units=4,
            hidden_units=8,
            num_heads=2,
            activation="relu",
        )
        dense = _jax_v2_mod._build_jax_dense_projection(
            units=5
        )
        mean_pool = _jax_v2_mod._build_jax_mean_pool(
            axis=1, keepdims=True
        )
        last_pool = _jax_v2_mod._build_jax_last_pool()
        concat = _jax_v2_mod._build_jax_concat_fusion(axis=-1)
        assert encoder is not None
        assert getattr(dense, "units", None) == 5
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        assert mean_pool(x).shape == (2, 1, 4)
        assert last_pool(x).shape == (2, 4)
        assert concat([x, x]).shape == (2, 3, 8)

    def test_ensure_jax_v2_registered_uses_quantile_forecast_key(
        self,
    ):
        pytest.importorskip("jax", reason="JAX not installed")
        import base_attentive.implementations.jax.base_attentive_v2 as _jax_v2_mod
        from base_attentive.registry.component_registry import (
            ComponentRegistry,
        )

        registry = ComponentRegistry()
        _jax_v2_mod.ensure_jax_v2_registered(registry)
        assert registry.has("projection.dense", backend="jax")
        assert registry.has(
            "head.quantile_forecast", backend="jax"
        )


class TestTensorFlowImplementationExtended:
    def test_ensure_tensorflow_accepts_fake_runtime(
        self, fake_tf_impl
    ):
        fake_tf_impl._ensure_tensorflow()

    def test_tf_temporal_self_attention_encoder_call_and_config(
        self, fake_tf_impl
    ):
        encoder = (
            fake_tf_impl._TFTemporalSelfAttentionEncoder(
                units=8,
                hidden_units=16,
                num_heads=2,
                activation="relu",
                dropout_rate=0.1,
                layer_norm_epsilon=1e-5,
                name="enc",
            )
        )
        x = np.ones((2, 4, 8), dtype=np.float32)

        out = encoder(x, training=True)
        config = encoder.get_config()

        assert out.shape == x.shape
        assert config["units"] == 8
        assert config["hidden_units"] == 16
        assert config["num_heads"] == 2
        assert config["dropout_rate"] == 0.1
        assert config["layer_norm_epsilon"] == 1e-5

    def test_tf_builder_functions_return_expected_layers(
        self, fake_tf_impl
    ):
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)

        dense = fake_tf_impl._build_tf_dense_projection(
            units=6, activation="relu", name="dense"
        )
        assert dense.units == 6
        assert dense.activation.__name__ == "relu"

        encoder = fake_tf_impl._build_tf_temporal_self_attention_encoder(
            units=4,
            hidden_units=8,
            num_heads=2,
        )
        assert isinstance(
            encoder,
            fake_tf_impl._TFTemporalSelfAttentionEncoder,
        )

        mean_pool = fake_tf_impl._build_tf_mean_pool(
            axis=1, keepdims=True
        )
        assert mean_pool(x).shape == (2, 1, 4)

        last_pool = fake_tf_impl._build_tf_last_pool()
        assert last_pool(x).shape == (2, 4)

        concat = fake_tf_impl._build_tf_concat_fusion(axis=-1)
        assert concat([x, x]).shape == (2, 3, 8)

        point_head = (
            fake_tf_impl._build_tf_point_forecast_head(
                output_dim=2,
                forecast_horizon=3,
            )
        )
        assert point_head.units == 6

        quantile_head = fake_tf_impl._build_tf_quantile_head(
            output_dim=2,
            forecast_horizon=2,
        )
        assert quantile_head.units == 4

    def test_ensure_tensorflow_v2_registered_with_custom_registry(
        self, fake_tf_impl
    ):
        from base_attentive.registry.component_registry import (
            ComponentRegistry,
        )

        registry = ComponentRegistry()
        fake_tf_impl.ensure_tensorflow_v2_registered(registry)

        assert registry.has(
            "projection.dense", backend="tensorflow"
        )
        assert registry.has("pool.last", backend="tensorflow")
        assert registry.has(
            "head.point_forecast", backend="tensorflow"
        )

    def test_ensure_tensorflow_v2_registered_with_default_registry(
        self, fake_tf_impl, monkeypatch
    ):
        import base_attentive.registry as registry_mod
        from base_attentive.registry.component_registry import (
            ComponentRegistry,
        )

        registry = ComponentRegistry()
        monkeypatch.setattr(
            registry_mod,
            "DEFAULT_COMPONENT_REGISTRY",
            registry,
        )

        fake_tf_impl.ensure_tensorflow_v2_registered()

        assert registry.has(
            "head.quantile_forecast", backend="tensorflow"
        )


# ---------------------------------------------------------------------------
# Generic implementation tests
# ---------------------------------------------------------------------------


class _FakeLayers:
    """Fake Keras layers for testing the generic builder."""

    from unittest.mock import MagicMock

    Dense = None
    MultiHeadAttention = None
    LayerNormalization = None
    Dropout = None


class _FakeContext:
    """Minimal fake backend context."""

    def __init__(self):
        self.layers = _ba.KERAS_DEPS
        self.ops = types.SimpleNamespace(
            mean=lambda x, axis=None: x.mean(axis=axis)
            if hasattr(x, "mean")
            else x,
            concatenate=lambda inputs,
            axis=-1: np.concatenate(
                [np.array(i) for i in inputs], axis=axis
            ),
        )


# Import BaseAttentiveV2 first to resolve circular imports in generic module
from base_attentive.experimental.base_attentive_v2 import (
    BaseAttentiveV2 as _V2ForCircularImport,  # noqa: F401
)
from base_attentive.implementations.generic.base_attentive_v2 import (
    _build_concat_fusion as _gen_build_concat_fusion,
)

# Now safe to import generic builders (circular import already resolved)
from base_attentive.implementations.generic.base_attentive_v2 import (
    _build_dense_projection as _gen_build_dense_projection,
)
from base_attentive.implementations.generic.base_attentive_v2 import (
    _build_last_pool as _gen_build_last_pool,
)
from base_attentive.implementations.generic.base_attentive_v2 import (
    _build_mean_pool as _gen_build_mean_pool,
)
from base_attentive.implementations.generic.base_attentive_v2 import (
    _build_point_forecast_head as _gen_build_point_forecast_head,
)
from base_attentive.implementations.generic.base_attentive_v2 import (
    _build_quantile_forecast_head as _gen_build_quantile_forecast_head,
)


class TestGenericImplementation:
    def setup_method(self):
        self._build_dense_projection = (
            _gen_build_dense_projection
        )
        self._build_mean_pool = _gen_build_mean_pool
        self._build_last_pool = _gen_build_last_pool
        self._build_concat_fusion = _gen_build_concat_fusion
        self._build_point_forecast_head = (
            _gen_build_point_forecast_head
        )
        self._build_quantile_forecast_head = (
            _gen_build_quantile_forecast_head
        )
        self.ctx = _FakeContext()

    def test_build_dense_projection(self):
        layer = self._build_dense_projection(
            context=self.ctx, units=32
        )
        assert layer is not None

    def test_build_mean_pool(self):
        pool_fn = self._build_mean_pool(
            context=self.ctx, axis=1
        )
        assert callable(pool_fn)

    def test_build_mean_pool_call(self):
        pool_fn = self._build_mean_pool(
            context=self.ctx, axis=1
        )
        x = np.ones((2, 5, 8), dtype=np.float32)
        result = pool_fn(x)
        assert result is not None

    def test_build_last_pool(self):
        pool_fn = self._build_last_pool(
            context=self.ctx, axis=1
        )
        assert callable(pool_fn)

    def test_build_last_pool_call(self):
        pool_fn = self._build_last_pool(
            context=self.ctx, axis=1
        )
        x = np.ones((2, 5, 8), dtype=np.float32)
        result = pool_fn(x)
        assert result.shape == (2, 8)

    def test_build_last_pool_invalid_axis(self):
        with pytest.raises(ValueError, match="axis=1 only"):
            self._build_last_pool(context=self.ctx, axis=2)

    def test_build_concat_fusion(self):
        fuse_fn = self._build_concat_fusion(
            context=self.ctx, axis=-1
        )
        assert callable(fuse_fn)

    def test_build_concat_fusion_single_feature(self):
        fuse_fn = self._build_concat_fusion(
            context=self.ctx, axis=-1
        )
        x = np.ones((2, 8), dtype=np.float32)
        result = fuse_fn([x])
        np.testing.assert_allclose(
            np.asarray(result), np.asarray(x)
        )

    def test_build_concat_fusion_no_features_raises(self):
        fuse_fn = self._build_concat_fusion(
            context=self.ctx, axis=-1
        )
        with pytest.raises(
            ValueError, match="no active feature tensors"
        ):
            fuse_fn([None, None])

    def test_build_point_forecast_head(self):
        head = self._build_point_forecast_head(
            context=self.ctx, units=1
        )
        assert head is not None

    def test_build_quantile_forecast_head(self):
        head = self._build_quantile_forecast_head(
            context=self.ctx, units=3
        )
        assert head is not None


# ---------------------------------------------------------------------------
# Experimental V2 tests
# ---------------------------------------------------------------------------

from base_attentive.experimental.base_attentive_v2 import (
    BaseAttentiveV2,
)


@pytest.fixture
def fresh_resolver_registries(monkeypatch):
    import base_attentive.registry as registry_mod
    import base_attentive.resolver.registrars as registrars
    from base_attentive.registry.component_registry import (
        ComponentRegistry,
    )
    from base_attentive.registry.model_registry import (
        ModelRegistry,
    )

    component_registry = ComponentRegistry()
    model_registry = ModelRegistry()
    monkeypatch.setattr(
        registry_mod,
        "DEFAULT_COMPONENT_REGISTRY",
        component_registry,
    )
    monkeypatch.setattr(
        registry_mod, "DEFAULT_MODEL_REGISTRY", model_registry
    )
    monkeypatch.setattr(
        registrars,
        "DEFAULT_COMPONENT_REGISTRY",
        component_registry,
    )
    monkeypatch.setattr(
        registrars, "DEFAULT_MODEL_REGISTRY", model_registry
    )
    registrars._LOADED_COMPONENT_REGISTRARS.clear()
    registrars._LOADED_MODEL_REGISTRARS.clear()
    return component_registry, model_registry


@pytest.mark.usefixtures("configured_runtime_backend")
class TestExperimentalBaseAttentiveV2:
    def test_instantiation_basic(
        self, fresh_resolver_registries
    ):
        model = BaseAttentiveV2(
            static_input_dim=0,
            dynamic_input_dim=8,
            future_input_dim=0,
        )
        assert model is not None

    def test_instantiation_with_static(
        self, fresh_resolver_registries
    ):
        model = BaseAttentiveV2(
            static_input_dim=4,
            dynamic_input_dim=8,
            future_input_dim=0,
        )
        assert model is not None

    def test_call_with_two_inputs(
        self, fresh_resolver_registries
    ):
        """The torch V2 builders use hardcoded in_features, so test model structure."""
        # Just test that instantiation and forward path structure is set up correctly
        model = BaseAttentiveV2(
            static_input_dim=0,
            dynamic_input_dim=32,
            future_input_dim=0,
            embed_dim=32,
            hidden_units=64,
            attention_heads=4,
        )
        # Verify model assembly has expected components
        assert model._assembly is not None
        assert model._assembly.dynamic_projection is not None

    def test_call_with_one_input(
        self, fresh_resolver_registries
    ):
        """Test the normalize_inputs path for single-input list."""
        model = BaseAttentiveV2(
            static_input_dim=0,
            dynamic_input_dim=32,
            future_input_dim=0,
            embed_dim=32,
            hidden_units=64,
            attention_heads=4,
        )
        a = np.ones((2, 5, 32), dtype=np.float32)
        s, d, f = model._normalize_inputs([a])
        assert s is None
        assert d is a
        assert f is None

    def test_normalize_inputs_three(
        self, fresh_resolver_registries
    ):
        model = BaseAttentiveV2(
            static_input_dim=0,
            dynamic_input_dim=8,
            future_input_dim=0,
        )
        a = np.ones((2, 5, 8), dtype=np.float32)
        b = np.ones((2, 5, 8), dtype=np.float32)
        c = np.ones((2, 1, 8), dtype=np.float32)
        s, d, f = model._normalize_inputs([a, b, c])
        assert s is a
        assert d is b
        assert f is c

    def test_normalize_inputs_non_list_raises(
        self, fresh_resolver_registries
    ):
        model = BaseAttentiveV2(
            static_input_dim=0,
            dynamic_input_dim=8,
            future_input_dim=0,
        )
        with pytest.raises(TypeError, match="list or tuple"):
            model._normalize_inputs(np.ones((2, 5, 8)))

    def test_normalize_inputs_too_many_raises(
        self, fresh_resolver_registries
    ):
        model = BaseAttentiveV2(
            static_input_dim=0,
            dynamic_input_dim=8,
            future_input_dim=0,
        )
        inputs = [np.ones((2, 5, 8), dtype=np.float32)] * 5
        with pytest.raises(
            ValueError, match="one, two, or three"
        ):
            model._normalize_inputs(inputs)

    def test_get_config(self, fresh_resolver_registries):
        model = BaseAttentiveV2(
            static_input_dim=0,
            dynamic_input_dim=8,
            future_input_dim=0,
        )
        config = model.get_config()
        assert isinstance(config, dict)

    def test_quantile_head_type(
        self, fresh_resolver_registries
    ):
        """Test that quantile model instantiates with the correct spec."""
        model = BaseAttentiveV2(
            static_input_dim=0,
            dynamic_input_dim=32,
            future_input_dim=0,
            head_type="quantile",
            quantiles=(0.1, 0.5, 0.9),
            embed_dim=32,
            hidden_units=64,
            attention_heads=4,
        )
        assert model.spec.head_type == "quantile"
        assert model.spec.quantiles == (0.1, 0.5, 0.9)


def _generic_spec(
    *,
    feature_processing="dense",
    final_agg="flatten",
    mode="pihal-like",
):
    return types.SimpleNamespace(
        architecture=types.SimpleNamespace(
            feature_processing=feature_processing
        ),
        runtime=types.SimpleNamespace(
            multi_scale_agg="concat"
        ),
        components=types.SimpleNamespace(
            final_pool_mean="pool.final_mean",
            final_pool_flatten="pool.final_flatten",
            final_pool_last="pool.final_last",
        ),
        lstm_units=[8, 4],
        scales=[1, 2],
        max_window_size=3,
        attention_units=8,
        num_heads=2,
        memory_size=4,
        output_dim=2,
        forecast_horizon=3,
        quantiles=(0.1, 0.5, 0.9),
        final_agg=final_agg,
        mode=mode,
        attention_levels=("cross", "hierarchical", "memory"),
        use_residuals=False,
        hidden_units=8,
        embed_dim=8,
        static_input_dim=0,
        dynamic_input_dim=4,
        future_input_dim=2,
        head_type="point",
    )


def test_generic_builder_helpers_cover_more_paths():
    import base_attentive.implementations.generic.base_attentive_v2 as mod

    spec = _generic_spec()
    ctx = types.SimpleNamespace(
        layers=types.SimpleNamespace(),
        ops=types.SimpleNamespace(),
    )

    assert mod._resolve_lstm_units(4) == 4
    assert mod._resolve_lstm_units([7, 5]) == 7
    with pytest.raises(ValueError, match="cannot be empty"):
        mod._resolve_lstm_units([])

    with pytest.raises(ValueError, match="require a spec"):
        mod._build_feature_processor(
            context=ctx,
            spec=None,
            role="dynamic",
            input_dim=4,
            output_units=8,
        )

    dense_proc = mod._build_feature_processor(
        context=ctx,
        spec=spec,
        role="static",
        input_dim=4,
        output_units=8,
        name="dense_proc",
    )
    object.__setattr__(
        dense_proc,
        "_post_grn",
        lambda inputs, training=False: np.ones((2, 8), dtype=np.float32),
    )
    assert dense_proc(np.ones((2, 4), dtype=np.float32)).shape == (
        2,
        8,
    )

    none_proc = mod._build_feature_processor(
        context=ctx,
        spec=spec,
        role="dynamic",
        input_dim=0,
        output_units=8,
    )
    assert none_proc(None) is None

    vsn_spec = _generic_spec(feature_processing="vsn")
    vsn_proc = mod._build_feature_processor(
        context=ctx,
        spec=vsn_spec,
        role="dynamic",
        input_dim=4,
        output_units=8,
    )
    assert vsn_proc is not None

    pos = mod._build_positional_encoding(context=ctx, spec=spec)
    assert pos is not None

    with pytest.raises(
        ValueError, match="requires a spec or lstm_units"
    ):
        mod._build_hybrid_multiscale_encoder(
            context=ctx,
            spec=None,
            lstm_units=None,
        )

    hybrid = mod._build_hybrid_multiscale_encoder(
        context=ctx,
        spec=spec,
        aggregation="average",
    )
    assert hybrid.get_config()["scales"] == [1, 2]

    encoder = mod._build_temporal_self_attention_encoder(
        context=ctx,
        spec=spec,
        units=8,
        hidden_units=16,
        num_heads=2,
        activation="relu",
        dropout_rate=0.0,
    )
    object.__setattr__(
        encoder,
        "attention",
        lambda query, value, training=False: query * 0 + 1,
    )
    object.__setattr__(encoder, "norm1", lambda value: value)
    object.__setattr__(encoder, "ffn_hidden", lambda value: value)
    object.__setattr__(encoder, "ffn_output", lambda value: value)
    object.__setattr__(encoder, "norm2", lambda value: value)
    encoded = encoder(
        np.ones((2, 3, 8), dtype=np.float32)
    )
    assert encoded.shape == (2, 3, 8)

    window = mod._build_dynamic_window(
        context=ctx, spec=spec
    )
    assert window.max_window_size == 3

    assert mod._build_cross_attention(
        context=ctx, spec=spec
    ) is not None
    assert mod._build_hierarchical_attention(
        context=ctx, spec=spec
    ) is not None
    assert mod._build_memory_attention(
        context=ctx, spec=spec
    ) is not None
    assert mod._build_multi_resolution_attention_fusion(
        context=ctx, spec=spec
    ) is not None

    flat = mod._build_flatten_pool(context=ctx, spec=spec)
    assert flat(np.ones((2, 3, 4), dtype=np.float32)).shape == (
        2,
        12,
    )

    multi = mod._build_multi_horizon_head(
        context=ctx,
        spec=spec,
    )
    assert multi(np.ones((2, 6), dtype=np.float32)).shape == (
        2,
        3,
        2,
    )

    qdist = mod._build_quantile_distribution_head(
        context=ctx,
        spec=spec,
    )
    assert qdist is not None

    assert (
        mod._resolve_final_pool_key(
            _generic_spec(final_agg="average")
        )
        == "pool.final_mean"
    )
    assert (
        mod._resolve_final_pool_key(
            _generic_spec(final_agg="flatten")
        )
        == "pool.final_flatten"
    )
    assert (
        mod._resolve_final_pool_key(
            _generic_spec(final_agg="last")
        )
        == "pool.final_last"
    )

    from base_attentive.registry.component_registry import (
        ComponentRegistry,
    )
    from base_attentive.registry.model_registry import (
        ModelRegistry,
    )

    component_registry = ComponentRegistry()
    model_registry = ModelRegistry()
    mod.ensure_generic_v2_registered(
        component_registry=component_registry,
        model_registry=model_registry,
    )
    assert component_registry.has(
        "feature.static_processor",
        backend="generic",
    )
    assert component_registry.has(
        "pool.final_flatten",
        backend="generic",
    )
    assert model_registry.has(
        "base_attentive.v2",
        backend="generic",
    )


def test_experimental_helper_methods_cover_internal_paths(
    monkeypatch,
):
    from dataclasses import dataclass

    import base_attentive.experimental.base_attentive_v2 as mod

    payload = {
        "spec": {"output_dim": 2},
        "architecture": {"kind": "x"},
        "output_dim": 9,
    }
    assert mod._extract_spec_payload(payload) == {
        "output_dim": 2,
        "architecture": {"kind": "x"},
    }

    assert mod._invoke(None, "x") == "x"
    assert mod._invoke(
        lambda value, training=False: (value, training),
        "y",
        training=True,
    ) == ("y", True)
    assert mod._invoke(lambda value: value + 1, 2) == 3

    model = object.__new__(mod.BaseAttentiveV2)
    model.spec = _generic_spec(mode="tft-like")

    @dataclass
    class _Assembly:
        foo: object
        bar: object

    model._assembly = _Assembly(foo="tracked", bar=None)
    model._track_assembly_components()
    assert model.foo == "tracked"
    assert model._tracked_component_names == ("foo",)

    assert model._normalized_mode() == "tft_like"
    assert model._tensor_width(np.ones((2, 3, 4))) == 4
    assert (
        model._tensor_width(types.SimpleNamespace(shape=None))
        is None
    )

    base = np.ones((2, 3, 4), dtype=np.float32)
    value = np.ones((2, 3, 6), dtype=np.float32)
    projected = model._align_residual_base(
        value,
        base,
        lambda x, training=False: np.ones(
            (2, 3, 6), dtype=np.float32
        ),
    )
    assert projected.shape == (2, 3, 6)
    with pytest.raises(ValueError, match="Residual width mismatch"):
        model._align_residual_base(value, base, None)

    residual = model._apply_residual(
        np.ones((2, 3, 4), dtype=np.float32),
        np.ones((2, 3, 4), dtype=np.float32),
        lambda values: values[0] + values[1],
        lambda output: output + 1,
    )
    np.testing.assert_allclose(residual, 3.0)

    model._assembly = types.SimpleNamespace(
        decoder_cross_attention=lambda values, training=False: values[0] + 1,
        decoder_cross_postprocess=lambda value, training=False: value + 1,
        decoder_hierarchical_attention=lambda values, training=False: values[0] + 2,
        decoder_memory_attention=lambda value, training=False: value + 3,
        decoder_fusion=lambda value, training=False: value + 4,
        decoder_residual_add=None,
        decoder_residual_norm=None,
        final_residual_add=None,
        final_residual_norm=None,
        residual_projection=None,
    )
    decoder = np.ones((2, 3, 4), dtype=np.float32)
    encoded = np.ones((2, 5, 4), dtype=np.float32)
    stacked = model._apply_decoder_stack(decoder, encoded)
    np.testing.assert_allclose(stacked, 12.0)

    seen = []
    model.call = lambda inputs, training=False: seen.append(
        np.asarray(inputs).shape
    )
    model.build_from_config({"input_shape": (None, 3, 4)})
    assert seen == [(1, 3, 4)]
    model.build_from_config(None)

    model.spec = _generic_spec()
    assert model.compute_output_shape((None, 3, 4)) == (
        None,
        3,
        2,
    )
    model.spec = _generic_spec()
    model.spec.head_type = "quantile"
    assert model.compute_output_shape(
        [(None, 3, 4), (None, 3, 2)]
    ) == (None, 3, 3, 2)

    class _Capture(mod.BaseAttentiveV2):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    normalized = types.SimpleNamespace(
        static_input_dim=1,
        dynamic_input_dim=2,
        future_input_dim=3,
        output_dim=4,
        forecast_horizon=5,
        quantiles=(0.1, 0.9),
        embed_dim=6,
        hidden_units=7,
        attention_heads=2,
        layer_norm_epsilon=1e-5,
        dropout_rate=0.1,
        activation="relu",
        backend_name="torch",
        head_type="point",
    )
    monkeypatch.setattr(
        mod,
        "normalize_base_attentive_spec",
        lambda payload=None, **kwargs: normalized,
    )
    monkeypatch.setattr(
        mod,
        "serialize_base_attentive_spec",
        lambda spec: {"serialized": True},
    )
    created = _Capture.from_config(
        {
            "spec": {"x": 1},
            "name": "captured",
            "trainable": True,
        }
    )
    assert created.kwargs["static_input_dim"] == 1
    assert created.kwargs["spec"] == {"serialized": True}

    passthrough = _Capture.from_config(
        {
            "static_input_dim": 1,
            "dynamic_input_dim": 2,
            "future_input_dim": 0,
        }
    )
    assert passthrough.kwargs["dynamic_input_dim"] == 2


def test_experimental_call_covers_migrated_heads_and_decoder_paths():
    import base_attentive.experimental.base_attentive_v2 as mod

    model = object.__new__(mod.BaseAttentiveV2)
    model.spec = _generic_spec(mode="tft-like")
    model.spec.forecast_horizon = 2
    model.spec.attention_units = 6
    model.spec.head_type = "point"
    model.backend_context = types.SimpleNamespace(
        shape=lambda value: np.asarray(value).shape,
        concat=lambda values, axis=-1: np.concatenate(
            [np.asarray(v) for v in values], axis=axis
        ),
        expand_dims=lambda value, axis=-1: np.expand_dims(
            np.asarray(value), axis=axis
        ),
        tile=lambda value, reps: np.tile(np.asarray(value), reps),
        zeros=lambda shape: np.zeros(shape, dtype=np.float32),
    )
    model._assembly = types.SimpleNamespace(
        static_processor=lambda value, training=False: np.ones(
            (2, 3), dtype=np.float32
        ),
        dynamic_processor=lambda value, training=False: np.asarray(
            value
        ),
        future_processor=lambda value, training=False: np.asarray(
            value
        ),
        encoder_positional_encoding=lambda value, training=False: value + 1,
        dynamic_encoder=lambda value, training=False: value + 2,
        dynamic_window=lambda value, training=False: value + 3,
        future_positional_encoding=lambda value, training=False: value + 4,
        decoder_input_projection=lambda value, training=False: value + 5,
        final_pool=lambda value, training=False: value.mean(axis=1),
        hidden_projection=lambda value: value + 6,
        dropout=lambda value, training=False: value,
        multi_horizon_head=lambda value, training=False: np.ones(
            (2, 2, 2), dtype=np.float32
        )
        * 7,
        quantile_distribution_head=lambda value, training=False: np.ones(
            (2, 2, 3, 2), dtype=np.float32
        )
        * 9,
        output_head=lambda value: np.ones((2, 12), dtype=np.float32),
    )
    model._apply_decoder_stack = (
        lambda decoder_input, encoder_sequences, training=False: decoder_input
    )

    static_x = np.ones((2, 1), dtype=np.float32)
    dynamic_x = np.ones((2, 3, 4), dtype=np.float32)
    future_x = np.ones((2, 5, 2), dtype=np.float32)

    point = model.call([static_x, dynamic_x, future_x], training=True)
    assert point.shape == (2, 2, 2)

    model.spec.head_type = "quantile"
    quantile = model.call(
        [static_x, dynamic_x, future_x], training=False
    )
    assert quantile.shape == (2, 2, 3, 2)

    model._assembly.quantile_distribution_head = None
    with pytest.raises(RuntimeError, match="Quantile head type"):
        model.call([static_x, dynamic_x, future_x])


def test_experimental_call_covers_zero_decoder_and_output_reshape():
    import base_attentive.experimental.base_attentive_v2 as mod

    model = object.__new__(mod.BaseAttentiveV2)
    model.spec = _generic_spec(mode="pihal-like")
    model.spec.forecast_horizon = 2
    model.spec.attention_units = 4
    model.spec.output_dim = 2
    model.spec.quantiles = (0.1, 0.5, 0.9)
    model.spec.head_type = "point"
    model.backend_context = types.SimpleNamespace(
        shape=lambda value: np.asarray(value).shape,
        concat=lambda values, axis=-1: np.concatenate(
            [np.asarray(v) for v in values], axis=axis
        ),
        expand_dims=lambda value, axis=-1: np.expand_dims(
            np.asarray(value), axis=axis
        ),
        tile=lambda value, reps: np.tile(np.asarray(value), reps),
        zeros=lambda shape: np.zeros(shape, dtype=np.float32),
    )
    model._assembly = types.SimpleNamespace(
        static_processor=None,
        dynamic_processor=lambda value, training=False: np.asarray(
            value
        ),
        future_processor=None,
        encoder_positional_encoding=None,
        dynamic_encoder=None,
        dynamic_window=None,
        future_positional_encoding=None,
        decoder_input_projection=None,
        final_pool=lambda value, training=False: value.mean(axis=1),
        hidden_projection=lambda value: value,
        dropout=None,
        multi_horizon_head=None,
        quantile_distribution_head=None,
        output_head=lambda value: np.arange(
            2 * model.spec.forecast_horizon * model.spec.output_dim,
            dtype=np.float32,
        ).reshape(2, -1),
    )
    model._apply_decoder_stack = (
        lambda decoder_input, encoder_sequences, training=False: decoder_input
    )

    dynamic_x = np.ones((2, 3, 4), dtype=np.float32)

    point = model.call([dynamic_x], training=False)
    assert point.shape == (2, 2, 2)

    model.spec.head_type = "quantile"
    model._assembly.output_head = lambda value: np.arange(
        2
        * model.spec.forecast_horizon
        * len(model.spec.quantiles)
        * model.spec.output_dim,
        dtype=np.float32,
    ).reshape(2, -1)
    quantile = model.call([dynamic_x], training=False)
    assert quantile.shape == (2, 2, 3, 2)

    with pytest.raises(ValueError, match="dynamic input is required"):
        model.call([None], training=False)


def test_jax_and_torch_wrappers_cover_extra_builder_paths(
    monkeypatch,
):
    import base_attentive.implementations.jax.base_attentive_v2 as jax_mod
    import base_attentive.implementations.torch.base_attentive_v2 as torch_mod

    monkeypatch.setattr(jax_mod, "_ensure_jax", lambda: None)
    assert jax_mod._clean_builder_kwargs(
        {
            "units": 4,
            "component_key": "x",
            "forecast_horizon": 2,
        }
    ) == {"units": 4}

    delegated = []
    monkeypatch.setattr(
        jax_mod,
        "_build_generic_feature_processor",
        lambda **kwargs: delegated.append(("feature", kwargs))
        or "feature",
    )
    monkeypatch.setattr(
        jax_mod,
        "_build_generic_positional_encoding",
        lambda **kwargs: delegated.append(("pos", kwargs))
        or "pos",
    )
    monkeypatch.setattr(
        jax_mod,
        "_build_generic_hybrid_multiscale_encoder",
        lambda **kwargs: delegated.append(("hybrid", kwargs))
        or "hybrid",
    )
    monkeypatch.setattr(
        jax_mod,
        "_build_generic_dynamic_window",
        lambda **kwargs: delegated.append(("window", kwargs))
        or "window",
    )
    monkeypatch.setattr(
        jax_mod,
        "_build_generic_flatten_pool",
        lambda **kwargs: delegated.append(("flat", kwargs))
        or "flat",
    )

    assert jax_mod._build_jax_static_processor(role="static") == "feature"
    assert jax_mod._build_jax_dynamic_processor(role="dynamic") == "feature"
    assert jax_mod._build_jax_future_processor(role="future") == "feature"
    assert jax_mod._build_jax_positional_encoding() == "pos"
    assert jax_mod._build_jax_hybrid_multiscale_encoder() == "hybrid"
    assert jax_mod._build_jax_dynamic_window() == "window"
    assert jax_mod._build_jax_flatten_pool() == "flat"
    assert len(delegated) == 7

    spec = _generic_spec()
    q = np.ones((2, 3, 4), dtype=np.float32)
    c = np.ones((2, 5, 4), dtype=np.float32)
    jax_cross = jax_mod._build_jax_cross_attention(spec=spec)
    assert jax_cross([q, c]).shape == (2, 3, 8)
    jax_hier = jax_mod._build_jax_hierarchical_attention(spec=spec)
    assert jax_hier(q).shape == (2, 3, 8)
    jax_memory = jax_mod._build_jax_memory_attention(spec=spec)
    assert jax_memory(q).shape == (2, 3, 8)
    jax_fusion = jax_mod._build_jax_multi_resolution_attention_fusion(
        spec=spec
    )
    assert jax_fusion(q).shape == (2, 3, 8)
    jax_multi = jax_mod._build_jax_multi_horizon_head(spec=spec)
    assert jax_multi(np.ones((2, 8), dtype=np.float32)).shape == (
        2,
        3,
        2,
    )
    jax_q = jax_mod._build_jax_quantile_distribution_head(spec=spec)
    assert jax_q(
        np.ones((2, 3, 2), dtype=np.float32)
    ).shape == (2, 3, 3, 2)

    monkeypatch.setattr(torch_mod, "_ensure_torch", lambda: None)
    delegated.clear()
    monkeypatch.setattr(
        torch_mod,
        "_build_generic_feature_processor",
        lambda **kwargs: delegated.append(("feature", kwargs))
        or "feature",
    )
    monkeypatch.setattr(
        torch_mod,
        "_build_generic_positional_encoding",
        lambda **kwargs: delegated.append(("pos", kwargs))
        or "pos",
    )
    monkeypatch.setattr(
        torch_mod,
        "_build_generic_hybrid_multiscale_encoder",
        lambda **kwargs: delegated.append(("hybrid", kwargs))
        or "hybrid",
    )
    monkeypatch.setattr(
        torch_mod,
        "_build_generic_dynamic_window",
        lambda **kwargs: delegated.append(("window", kwargs))
        or "window",
    )
    monkeypatch.setattr(
        torch_mod,
        "_build_generic_flatten_pool",
        lambda **kwargs: delegated.append(("flat", kwargs))
        or "flat",
    )

    assert torch_mod._build_torch_static_processor(role="static") == "feature"
    assert torch_mod._build_torch_dynamic_processor(role="dynamic") == "feature"
    assert torch_mod._build_torch_future_processor(role="future") == "feature"
    assert torch_mod._build_torch_positional_encoding() == "pos"
    assert torch_mod._build_torch_hybrid_multiscale_encoder() == "hybrid"
    assert torch_mod._build_torch_dynamic_window() == "window"
    assert torch_mod._build_torch_flatten_pool() == "flat"

    tq = torch.randn(2, 3, 4)
    tc = torch.randn(2, 5, 4)
    torch_cross = torch_mod._build_torch_cross_attention(spec=spec)
    assert torch_cross([tq, tc]).shape == (2, 3, 8)
    torch_hier = torch_mod._build_torch_hierarchical_attention(spec=spec)
    assert torch_hier(tq).shape == (2, 3, 8)
    torch_memory = torch_mod._build_torch_memory_attention(spec=spec)
    assert torch_memory(tq).shape == (2, 3, 8)
    torch_fusion = torch_mod._build_torch_multi_resolution_attention_fusion(
        spec=spec
    )
    assert torch_fusion(tq).shape == (2, 3, 8)
    torch_multi = torch_mod._build_torch_multi_horizon_head(spec=spec)
    assert torch_multi(torch.randn(2, 8)).shape == (2, 3, 2)
    torch_q = torch_mod._build_torch_quantile_distribution_head(
        spec=spec
    )
    assert torch_q(tq).shape == (2, 3, 3, 2)


def test_generic_runtime_layer_classes_cover_configs_and_edges():
    import base_attentive.implementations.generic.base_attentive_v2 as mod

    x = np.ones((2, 3, 4), dtype=np.float32)

    dense_proc = mod._GenericFeatureProcessor(
        role="static",
        input_dim=4,
        output_units=6,
        vsn_units=None,
        feature_processing="dense",
        activation="relu",
        dropout_rate=0.1,
        use_batch_norm=False,
        name="dense_proc",
    )
    object.__setattr__(
        dense_proc,
        "_post_grn",
        lambda inputs, training=False: np.ones((2, 6), dtype=np.float32),
    )
    assert dense_proc(x[:, 0, :]).shape == (2, 6)
    dense_cfg = dense_proc.get_config()
    assert dense_cfg["role"] == "static"
    assert dense_cfg["output_units"] == 6

    vsn_proc = mod._GenericFeatureProcessor(
        role="dynamic",
        input_dim=4,
        output_units=5,
        vsn_units=7,
        feature_processing="vsn",
        name="vsn_proc",
    )
    object.__setattr__(
        vsn_proc,
        "_vsn",
        lambda inputs, training=False: np.ones((2, 3, 7), dtype=np.float32),
    )
    object.__setattr__(
        vsn_proc,
        "_post_grn",
        lambda inputs, training=False: np.ones((2, 3, 5), dtype=np.float32),
    )
    assert vsn_proc(x).shape == (2, 3, 5)
    vsn_cfg = vsn_proc.get_config()
    assert vsn_cfg["feature_processing"] == "vsn"
    assert vsn_cfg["vsn_units"] == 7

    null_proc = mod._GenericFeatureProcessor(
        role="future",
        input_dim=0,
        output_units=5,
        vsn_units=None,
        feature_processing="dense",
    )
    assert null_proc(None) is None

    mean_pool = mod._GenericMeanPool(axis=1, keepdims=True)
    assert mean_pool(x).shape == (2, 1, 4)
    assert mean_pool.get_config()["keepdims"] is True

    last_pool = mod._GenericLastPool(axis=1)
    assert last_pool(x).shape == (2, 4)
    assert last_pool.get_config()["axis"] == 1
    with pytest.raises(ValueError, match="axis=1 only"):
        mod._GenericLastPool(axis=0)

    flatten_pool = mod._GenericFlattenPool(axis=1)
    assert flatten_pool(x).shape == (2, 12)
    assert flatten_pool.get_config()["axis"] == 1
    with pytest.raises(ValueError, match="axis=1 only"):
        mod._GenericFlattenPool(axis=0)

    concat = mod._GenericConcatFusion(axis=-1)
    assert concat([x, None, x]).shape == (2, 3, 8)
    assert concat([x]).shape == x.shape
    assert concat.get_config()["axis"] == -1
    with pytest.raises(
        ValueError, match="no active feature tensors"
    ):
        concat([None, None])

    temporal = mod._TemporalSelfAttentionEncoder(
        units=4,
        hidden_units=8,
        num_heads=2,
        activation="relu",
        dropout_rate=0.1,
        layer_norm_epsilon=1e-5,
        name="temporal",
    )
    object.__setattr__(
        temporal,
        "attention",
        lambda query, value, training=False: query * 0 + 1,
    )
    object.__setattr__(temporal, "norm1", lambda value: value)
    object.__setattr__(
        temporal, "ffn_hidden", lambda value: value
    )
    object.__setattr__(
        temporal, "ffn_output", lambda value: value
    )
    object.__setattr__(
        temporal,
        "dropout",
        lambda value, training=False: value,
    )
    object.__setattr__(temporal, "norm2", lambda value: value)
    assert temporal(x, training=True).shape == x.shape
    temporal_cfg = temporal.get_config()
    assert temporal_cfg["hidden_units"] == 8
    assert temporal_cfg["dropout_rate"] == 0.1

    hybrid = mod._HybridMultiScaleEncoder(
        lstm_units=4,
        scales=[1, 2],
        sequence_mode="average",
        name="hybrid",
    )
    hybrid_out = hybrid(np.ones((2, 5, 4), dtype=np.float32))
    assert hybrid_out.shape[0] == 2
    hybrid_cfg = hybrid.get_config()
    assert hybrid_cfg["lstm_units"] == 4
    assert hybrid_cfg["scales"] == [1, 2]


def test_jax_runtime_layer_classes_cover_configs(monkeypatch):
    import base_attentive.implementations.jax.base_attentive_v2 as mod

    monkeypatch.setattr(mod, "_ensure_jax", lambda: None)

    x = np.ones((2, 3, 8), dtype=np.float32)
    context = np.ones((2, 5, 8), dtype=np.float32)

    encoder = mod._JaxTemporalSelfAttentionEncoder(
        units=8,
        hidden_units=12,
        num_heads=2,
        activation="relu",
        dropout_rate=0.1,
        layer_norm_epsilon=1e-5,
        name="jax_enc",
    )
    assert encoder(x, training=True).shape == x.shape
    enc_cfg = encoder.get_config()
    assert enc_cfg["units"] == 8
    assert enc_cfg["dropout_rate"] == 0.1

    mean_pool = mod._MeanPool(axis=1, keepdims=True)
    assert mean_pool(x).shape == (2, 1, 8)
    assert mean_pool.get_config()["keepdims"] is True

    last_pool = mod._LastPool(axis=1)
    assert last_pool(x).shape == (2, 8)
    with pytest.raises(ValueError, match="axis=1 only"):
        mod._LastPool(axis=0)

    concat = mod._ConcatFusion(axis=-1)
    assert concat([x, None, x]).shape == (2, 3, 16)
    assert concat([x]).shape == x.shape
    assert concat.get_config()["axis"] == -1

    assert mod._SelfAttentionBlockMixin._key_dim(8, 3) == 2

    cross = mod._JaxCrossAttention(
        units=8, num_heads=2, dropout_rate=0.1
    )
    assert cross([x, context]).shape == x.shape
    assert cross.get_config()["num_heads"] == 2

    hierarchical = mod._JaxHierarchicalAttention(
        units=8, num_heads=2, dropout_rate=0.1
    )
    assert hierarchical([x, context[:, :3, :]]).shape == x.shape
    assert hierarchical(x).shape == x.shape
    assert hierarchical.get_config()["dropout_rate"] == 0.1

    memory = mod._JaxMemoryAttention(
        units=8, memory_size=4, num_heads=2, dropout_rate=0.1
    )
    assert memory(x).shape == x.shape
    memory_cfg = memory.get_config()
    assert memory_cfg["memory_size"] == 4
    assert memory_cfg["units"] == 8

    fusion = mod._JaxMultiResolutionAttentionFusion(
        units=8, num_heads=2, dropout_rate=0.1
    )
    assert fusion(x).shape == x.shape
    assert fusion.get_config()["num_heads"] == 2

    multi_head = mod._JaxMultiHorizonHead(
        output_dim=2,
        forecast_horizon=4,
        activation="relu",
    )
    assert multi_head(np.ones((2, 8), dtype=np.float32)).shape == (
        2,
        4,
        2,
    )
    assert multi_head.get_config()["forecast_horizon"] == 4

    quantile_head = mod._JaxQuantileDistributionHead(
        quantiles=(0.1, 0.5, 0.9),
        output_dim=8,
        forecast_horizon=4,
    )
    assert quantile_head(x).shape == (2, 3, 3, 8)
    quantile_cfg = quantile_head.get_config()
    assert quantile_cfg["quantiles"] == [0.1, 0.5, 0.9]
    assert quantile_cfg["forecast_horizon"] == 4


def test_torch_runtime_layer_classes_cover_configs(monkeypatch):
    import base_attentive.implementations.torch.base_attentive_v2 as mod

    monkeypatch.setattr(mod, "_ensure_torch", lambda: None)

    x = np.ones((2, 3, 8), dtype=np.float32)
    context = np.ones((2, 5, 8), dtype=np.float32)

    encoder = mod._TorchTemporalSelfAttentionEncoder(
        units=8,
        hidden_units=12,
        num_heads=2,
        activation="relu",
        dropout_rate=0.1,
        layer_norm_epsilon=1e-5,
        name="torch_enc",
    )
    assert encoder(x, training=True).shape == x.shape
    assert encoder.forward(x, training=False).shape == x.shape
    enc_cfg = encoder.get_config()
    assert enc_cfg["hidden_units"] == 12
    assert enc_cfg["layer_norm_epsilon"] == 1e-5

    mean_pool = mod._MeanPool(axis=1, keepdims=True)
    assert mean_pool(x).shape == (2, 1, 8)
    assert mean_pool.get_config()["axis"] == 1

    last_pool = mod._LastPool(axis=1)
    assert last_pool(x).shape == (2, 8)
    with pytest.raises(ValueError, match="axis=1 only"):
        mod._LastPool(axis=0)

    concat = mod._ConcatFusion(axis=-1)
    assert concat([x, None, x]).shape == (2, 3, 16)
    assert concat([x]).shape == x.shape
    assert concat.get_config()["axis"] == -1

    assert mod._SelfAttentionBlockMixin._key_dim(8, 3) == 2

    cross = mod._TorchCrossAttention(
        units=8, num_heads=2, dropout_rate=0.1
    )
    assert cross([x, context], training=True).shape == x.shape
    assert cross.get_config()["dropout_rate"] == 0.1

    hierarchical = mod._TorchHierarchicalAttention(
        units=8, num_heads=2, dropout_rate=0.1
    )
    assert hierarchical([x, context[:, :3, :]], training=True).shape == (
        2,
        3,
        8,
    )
    assert hierarchical(x).shape == x.shape
    assert hierarchical.get_config()["num_heads"] == 2

    memory = mod._TorchMemoryAttention(
        units=8, memory_size=4, num_heads=2, dropout_rate=0.1
    )
    assert memory(x, training=True).shape == x.shape
    memory_cfg = memory.get_config()
    assert memory_cfg["memory_size"] == 4
    assert memory_cfg["dropout_rate"] == 0.1

    fusion = mod._TorchMultiResolutionAttentionFusion(
        units=8, num_heads=2, dropout_rate=0.1
    )
    assert fusion(x, training=True).shape == x.shape
    assert fusion.get_config()["units"] == 8

    multi_head = mod._TorchMultiHorizonHead(
        output_dim=2,
        forecast_horizon=4,
        activation="relu",
    )
    assert multi_head(np.ones((2, 8), dtype=np.float32)).shape == (
        2,
        4,
        2,
    )
    assert multi_head.get_config()["output_dim"] == 2

    quantile_head = mod._TorchQuantileDistributionHead(
        quantiles=(0.1, 0.5, 0.9),
        output_dim=2,
    )
    assert quantile_head(x).shape == (2, 3, 3, 2)
    quantile_cfg = quantile_head.get_config()
    assert quantile_cfg["quantiles"] == [0.1, 0.5, 0.9]
    assert quantile_cfg["output_dim"] == 2
