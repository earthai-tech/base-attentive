"""Tests for backend-agnostic validation helpers."""

from __future__ import annotations

import numpy as np
import pytest


class _FakeKerasDeps:
    """Small numpy-backed stand-in for Keras ops used by validation helpers."""

    float32 = np.float32

    @staticmethod
    def convert_to_tensor(value):
        if isinstance(value, str) and value == "__boom__":
            raise TypeError("boom")
        return np.asarray(value)

    @staticmethod
    def reduce_mean(value, axis):
        return np.mean(np.asarray(value), axis=axis)

    @staticmethod
    def reduce_sum(value, axis):
        return np.sum(np.asarray(value), axis=axis)

    @staticmethod
    def expand_dims(value, axis=-1):
        return np.expand_dims(np.asarray(value), axis=axis)

    @staticmethod
    def cast(value, dtype):
        return np.asarray(value, dtype=dtype)


@pytest.fixture
def validation_module():
    """Import the validation module once for monkeypatch-driven tests."""
    import base_attentive.validation as validation_module

    return validation_module


@pytest.fixture
def fake_runtime(validation_module, monkeypatch):
    """Patch validation helpers to use a lightweight fake runtime."""
    monkeypatch.setattr(validation_module, "_has_runtime", lambda: True)
    monkeypatch.setattr(validation_module, "KERAS_DEPS", _FakeKerasDeps())
    return validation_module


class TestValidationModule:
    """Test tensor validation utilities without importing TensorFlow."""

    def test_validate_model_inputs_none_returns_empty_triplet(self, validation_module):
        """None inputs should normalize to a predictable empty triplet."""
        static, dynamic, future = validation_module.validate_model_inputs(None)

        assert (static, dynamic, future) == (None, None, None)

    def test_validate_model_inputs_passthrough_without_runtime(
        self, validation_module, monkeypatch
    ):
        """Without a runtime, triplet inputs should be returned unchanged."""
        raw_inputs = [object(), object(), object()]
        monkeypatch.setattr(validation_module, "_has_runtime", lambda: False)

        static, dynamic, future = validation_module.validate_model_inputs(raw_inputs)

        assert static is raw_inputs[0]
        assert dynamic is raw_inputs[1]
        assert future is raw_inputs[2]

    def test_validate_model_inputs_converts_runtime_values(self, fake_runtime):
        """The active runtime should convert each non-empty input slot."""
        inputs = [
            [[1.0, 2.0], [3.0, 4.0]],
            np.ones((2, 3, 4)),
            np.zeros((2, 5, 6)),
        ]

        static, dynamic, future = fake_runtime.validate_model_inputs(inputs)

        assert isinstance(static, np.ndarray)
        assert static.shape == (2, 2)
        assert dynamic.shape == (2, 3, 4)
        assert future.shape == (2, 5, 6)

    def test_validate_model_inputs_single_tensor_expands_triplet(self, fake_runtime):
        """Single inputs should be mapped to the static slot."""
        static, dynamic, future = fake_runtime.validate_model_inputs(np.ones((4, 8)))

        assert static.shape == (4, 8)
        assert dynamic is None
        assert future is None

    def test_validate_model_inputs_wraps_conversion_errors(self, fake_runtime):
        """Conversion failures should surface as validation errors."""
        with pytest.raises(ValueError, match="Could not convert input"):
            fake_runtime.validate_model_inputs("__boom__")

    def test_maybe_reduce_quantiles_bh_passthrough(self, fake_runtime):
        """Two-dimensional arrays should remain unchanged."""
        x = np.ones((32, 10))

        result = fake_runtime.maybe_reduce_quantiles_bh(x)

        assert result.shape == x.shape

    def test_maybe_reduce_quantiles_bh_reduction_3d(self, fake_runtime):
        """Three-dimensional arrays with Q > 1 should be reduced."""
        x = np.arange(32 * 10 * 5, dtype=np.float32).reshape(32, 10, 5)

        result = fake_runtime.maybe_reduce_quantiles_bh(x, reduction="mean")

        assert result.shape == (32, 10)

    def test_ensure_bh1_2d_to_3d_and_cast(self, fake_runtime):
        """Two-dimensional arrays should become ``(B, H, 1)``."""
        x = np.ones((32, 10), dtype=np.float64)

        result = fake_runtime.ensure_bh1(x, dtype=np.float32)

        assert result.shape == (32, 10, 1)
        assert result.dtype == np.float32

    def test_ensure_bh1_preserves_rank_3(self, fake_runtime):
        """Existing ``(B, H, 1)`` arrays should keep their trailing axis."""
        x = np.ones((32, 10, 1), dtype=np.float32)

        result = fake_runtime.ensure_bh1(x)

        assert result.shape == (32, 10, 1)

    # ------------------------------------------------------------------
    # validate_model_inputs – additional branches
    # ------------------------------------------------------------------

    def test_validate_model_inputs_verbose_logs_shapes(self, fake_runtime):
        """verbose=1 should execute info-logger calls without raising."""
        inputs = [np.ones((2, 3)), np.ones((2, 4, 5)), np.ones((2, 6, 7))]
        static, dynamic, future = fake_runtime.validate_model_inputs(
            inputs, verbose=1
        )
        assert static.shape == (2, 3)
        assert dynamic.shape == (2, 4, 5)
        assert future.shape == (2, 6, 7)

    def test_validate_model_inputs_error_warn_swallows_exception(
        self, fake_runtime
    ):
        """With error='warn', a bad input is kept as-is instead of raising."""
        static, dynamic, future = fake_runtime.validate_model_inputs(
            "__boom__", error="warn"
        )
        # The raw (unconverted) value is stored in the static slot.
        assert static == "__boom__"
        assert dynamic is None
        assert future is None

    # ------------------------------------------------------------------
    # maybe_reduce_quantiles_bh – additional branches
    # ------------------------------------------------------------------

    def test_maybe_reduce_quantiles_bh_no_runtime_passthrough(
        self, validation_module, monkeypatch
    ):
        """Without a runtime the function must return the original object."""
        monkeypatch.setattr(validation_module, "_has_runtime", lambda: False)
        x = np.ones((4, 5))
        result = validation_module.maybe_reduce_quantiles_bh(x)
        assert result is x

    def test_maybe_reduce_quantiles_bh_rank4_mean(self, fake_runtime):
        """Rank-4 tensor with reduction='mean' should collapse axis 2."""
        x = np.ones((2, 5, 3, 1), dtype=np.float32)
        result = fake_runtime.maybe_reduce_quantiles_bh(x, reduction="mean")
        assert result.shape == (2, 5, 1)

    def test_maybe_reduce_quantiles_bh_rank4_sum(self, fake_runtime):
        """Rank-4 tensor with reduction='sum' should collapse axis 2."""
        x = np.ones((2, 5, 3, 1), dtype=np.float32)
        result = fake_runtime.maybe_reduce_quantiles_bh(x, reduction="sum")
        assert result.shape == (2, 5, 1)

    def test_maybe_reduce_quantiles_bh_rank4_callable(self, fake_runtime):
        """Rank-4 tensor should work with a callable reduction."""
        x = np.ones((2, 5, 3, 1), dtype=np.float32)
        result = fake_runtime.maybe_reduce_quantiles_bh(
            x, reduction=lambda t, axis: np.mean(t, axis=axis)
        )
        assert result.shape == (2, 5, 1)

    def test_maybe_reduce_quantiles_bh_3d_last_dim_1_passthrough(
        self, fake_runtime
    ):
        """3D tensor with last dim == 1 should be returned unchanged."""
        x = np.ones((2, 5, 1), dtype=np.float32)
        result = fake_runtime.maybe_reduce_quantiles_bh(x, reduction="mean")
        assert result.shape == (2, 5, 1)

    def test_maybe_reduce_quantiles_bh_3d_sum(self, fake_runtime):
        """3D tensor with last dim > 1 and reduction='sum' should collapse."""
        x = np.arange(32 * 10 * 5, dtype=np.float32).reshape(32, 10, 5)
        result = fake_runtime.maybe_reduce_quantiles_bh(x, reduction="sum")
        assert result.shape == (32, 10)

    def test_maybe_reduce_quantiles_bh_3d_callable(self, fake_runtime):
        """3D tensor should work with a callable reduction."""
        x = np.arange(32 * 10 * 5, dtype=np.float32).reshape(32, 10, 5)
        result = fake_runtime.maybe_reduce_quantiles_bh(
            x, reduction=lambda t, axis: np.mean(t, axis=axis)
        )
        assert result.shape == (32, 10)

    # ------------------------------------------------------------------
    # ensure_bh1 – additional branches
    # ------------------------------------------------------------------

    def test_ensure_bh1_no_runtime_passthrough(
        self, validation_module, monkeypatch
    ):
        """Without a runtime the function must return the original object."""
        monkeypatch.setattr(validation_module, "_has_runtime", lambda: False)
        x = np.ones((4, 5))
        result = validation_module.ensure_bh1(x)
        assert result is x

    def test_ensure_bh1_reduce_axis_mean(self, fake_runtime):
        """reduce_axis with 'mean' should collapse the specified axis."""
        x = np.ones((2, 5, 4), dtype=np.float32)
        result = fake_runtime.ensure_bh1(x, reduce_axis=2, reduction="mean")
        assert result.shape == (2, 5)

    def test_ensure_bh1_reduce_axis_sum(self, fake_runtime):
        """reduce_axis with 'sum' should collapse the specified axis."""
        x = np.ones((2, 5, 4), dtype=np.float32)
        result = fake_runtime.ensure_bh1(x, reduce_axis=2, reduction="sum")
        assert result.shape == (2, 5)

    def test_ensure_bh1_reduce_axis_callable(self, fake_runtime):
        """reduce_axis with a callable should apply it to the axis."""
        x = np.ones((2, 5, 4), dtype=np.float32)
        result = fake_runtime.ensure_bh1(
            x,
            reduce_axis=2,
            reduction=lambda t, axis: np.mean(t, axis=axis),
        )
        assert result.shape == (2, 5)
