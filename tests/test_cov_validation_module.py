# SPDX-License-Identifier: Apache-2.0
"""Tests for base_attentive/validation/__init__.py."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")

from base_attentive.validation import (
    validate_model_inputs,
    maybe_reduce_quantiles_bh,
    ensure_bh1,
)
import base_attentive.validation as _val_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_runtime():
    return _val_mod._has_runtime()


# ---------------------------------------------------------------------------
# validate_model_inputs
# ---------------------------------------------------------------------------


class TestValidateModelInputs:
    """Tests for validate_model_inputs."""

    def test_list_of_three_returns_tuple_of_three(self):
        a = np.ones((2, 4), dtype=np.float32)
        b = np.ones((2, 5, 4), dtype=np.float32)
        c = np.ones((2, 3, 2), dtype=np.float32)
        result = validate_model_inputs([a, b, c])
        assert len(result) == 3

    def test_single_input_pads_to_three(self):
        a = np.ones((2, 4), dtype=np.float32)
        s, d, f = validate_model_inputs(a)
        assert d is None
        assert f is None

    def test_none_input_returns_three_nones(self):
        s, d, f = validate_model_inputs(None)
        assert s is None
        assert d is None
        assert f is None

    def test_list_of_one_pads_to_three(self):
        a = np.ones((2, 4), dtype=np.float32)
        result = validate_model_inputs([a])
        assert len(result) == 3
        assert result[1] is None
        assert result[2] is None

    def test_list_of_five_truncates_to_three(self):
        arrays = [np.ones((2, 4), dtype=np.float32)] * 5
        result = validate_model_inputs(arrays)
        assert len(result) == 3

    def test_list_with_none_inputs(self):
        a = np.ones((2, 4), dtype=np.float32)
        result = validate_model_inputs([a, None, None])
        assert len(result) == 3
        assert result[1] is None

    def test_tuple_input_works(self):
        a = np.ones((2, 4), dtype=np.float32)
        b = np.ones((2, 5, 4), dtype=np.float32)
        result = validate_model_inputs((a, b))
        assert len(result) == 3

    def test_verbose_logging(self, caplog):
        import logging
        a = np.ones((2, 4), dtype=np.float32)
        b = np.ones((2, 5, 4), dtype=np.float32)
        c = np.ones((2, 3, 2), dtype=np.float32)
        # Just ensure no exception with verbose=1
        result = validate_model_inputs([a, b, c], verbose=1)
        assert len(result) == 3

    def test_no_runtime_returns_raw_values(self):
        """When no runtime, inputs are returned as-is."""
        orig_backend = _val_mod.KERAS_BACKEND
        orig_deps = _val_mod.KERAS_DEPS
        try:
            _val_mod.KERAS_BACKEND = ""
            _val_mod.KERAS_DEPS = None
            a = np.ones((2, 4), dtype=np.float32)
            b = np.ones((2, 5, 4), dtype=np.float32)
            c = np.ones((2, 3, 2), dtype=np.float32)
            result = validate_model_inputs([a, b, c])
            assert len(result) == 3
        finally:
            _val_mod.KERAS_BACKEND = orig_backend
            _val_mod.KERAS_DEPS = orig_deps

    def test_no_runtime_single_input(self):
        orig_backend = _val_mod.KERAS_BACKEND
        orig_deps = _val_mod.KERAS_DEPS
        try:
            _val_mod.KERAS_BACKEND = ""
            _val_mod.KERAS_DEPS = None
            a = np.ones((2, 4), dtype=np.float32)
            result = validate_model_inputs(a)
            assert len(result) == 3
        finally:
            _val_mod.KERAS_BACKEND = orig_backend
            _val_mod.KERAS_DEPS = orig_deps

    def test_convert_error_raise_mode(self):
        """Error path: convert_to_tensor raises, error='raise' → ValueError."""
        if not _has_runtime():
            pytest.skip("No runtime available")

        from unittest.mock import MagicMock, patch

        original_deps = _val_mod.KERAS_DEPS
        try:
            mock_deps = MagicMock()
            mock_deps.convert_to_tensor = MagicMock(side_effect=Exception("bad input"))
            _val_mod.KERAS_DEPS = mock_deps

            with pytest.raises(ValueError, match="Could not convert input to tensor"):
                validate_model_inputs(
                    [np.ones((2, 4), dtype=np.float32)],
                    error="raise",
                )
        finally:
            _val_mod.KERAS_DEPS = original_deps

    def test_convert_error_warn_mode(self):
        """Error path: convert_to_tensor raises, error='warn' → returns original."""
        if not _has_runtime():
            pytest.skip("No runtime available")

        from unittest.mock import MagicMock

        original_deps = _val_mod.KERAS_DEPS
        try:
            a = np.ones((2, 4), dtype=np.float32)
            mock_deps = MagicMock()
            mock_deps.convert_to_tensor = MagicMock(side_effect=Exception("bad input"))
            _val_mod.KERAS_DEPS = mock_deps

            result = validate_model_inputs([a], error="warn")
            assert len(result) == 3
        finally:
            _val_mod.KERAS_DEPS = original_deps


# ---------------------------------------------------------------------------
# maybe_reduce_quantiles_bh
# ---------------------------------------------------------------------------


class TestMaybeReduceQuantilesBh:
    """Tests for maybe_reduce_quantiles_bh."""

    def test_no_runtime_returns_unchanged(self):
        orig_backend = _val_mod.KERAS_BACKEND
        orig_deps = _val_mod.KERAS_DEPS
        try:
            _val_mod.KERAS_BACKEND = ""
            _val_mod.KERAS_DEPS = None
            x = np.ones((2, 3, 4))
            result = maybe_reduce_quantiles_bh(x)
            assert result is x
        finally:
            _val_mod.KERAS_BACKEND = orig_backend
            _val_mod.KERAS_DEPS = orig_deps

    def test_rank4_mean_reduction(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        x = np.ones((2, 5, 3, 4), dtype=np.float32)
        result = maybe_reduce_quantiles_bh(x, reduction="mean")
        assert result is not None

    def test_rank4_sum_reduction(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        x = np.ones((2, 5, 3, 4), dtype=np.float32)
        result = maybe_reduce_quantiles_bh(x, reduction="sum")
        assert result is not None

    def test_rank4_callable_reduction(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        x = np.ones((2, 5, 3, 4), dtype=np.float32)
        called = {}

        def my_reduce(t, axis=None):
            called["axis"] = axis
            return t.mean(axis=axis) if hasattr(t, "mean") else t

        result = maybe_reduce_quantiles_bh(x, reduction=my_reduce)
        assert result is not None

    def test_rank3_with_last_dim_greater_1_mean(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        x = np.ones((2, 5, 4), dtype=np.float32)
        result = maybe_reduce_quantiles_bh(x, reduction="mean")
        assert result is not None

    def test_rank3_with_last_dim_greater_1_sum(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        x = np.ones((2, 5, 4), dtype=np.float32)
        result = maybe_reduce_quantiles_bh(x, reduction="sum")
        assert result is not None

    def test_rank3_with_last_dim_greater_1_callable(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        x = np.ones((2, 5, 4), dtype=np.float32)

        def my_reduce(t, axis=None):
            return t

        result = maybe_reduce_quantiles_bh(x, reduction=my_reduce)
        assert result is not None

    def test_rank3_with_last_dim_1_unchanged(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        x = np.ones((2, 5, 1), dtype=np.float32)
        result = maybe_reduce_quantiles_bh(x, reduction="mean")
        # Last dim is 1, should be returned unchanged
        assert result is not None

    def test_rank2_unchanged(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        x = np.ones((2, 5), dtype=np.float32)
        result = maybe_reduce_quantiles_bh(x, reduction="mean")
        assert result is not None


# ---------------------------------------------------------------------------
# ensure_bh1
# ---------------------------------------------------------------------------


class TestEnsureBh1:
    """Tests for ensure_bh1."""

    def test_no_runtime_returns_unchanged(self):
        orig_backend = _val_mod.KERAS_BACKEND
        orig_deps = _val_mod.KERAS_DEPS
        try:
            _val_mod.KERAS_BACKEND = ""
            _val_mod.KERAS_DEPS = None
            x = np.ones((2, 5))
            result = ensure_bh1(x)
            assert result is x
        finally:
            _val_mod.KERAS_BACKEND = orig_backend
            _val_mod.KERAS_DEPS = orig_deps

    def test_rank1_expands_to_rank3(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        x = np.ones((5,), dtype=np.float32)
        result = ensure_bh1(x)
        assert len(result.shape) == 3

    def test_rank2_expands_to_rank3(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        x = np.ones((2, 5), dtype=np.float32)
        result = ensure_bh1(x)
        assert len(result.shape) == 3

    def test_rank3_unchanged(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        x = np.ones((2, 5, 1), dtype=np.float32)
        result = ensure_bh1(x)
        assert len(result.shape) == 3

    def test_with_dtype_cast(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        import keras
        x = np.ones((2, 5, 1), dtype=np.float32)
        result = ensure_bh1(x, dtype="float32")
        assert result is not None

    def test_with_reduce_axis_mean(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        x = np.ones((2, 5, 4), dtype=np.float32)
        result = ensure_bh1(x, reduce_axis=2, reduction="mean")
        assert result is not None

    def test_with_reduce_axis_sum(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        x = np.ones((2, 5, 4), dtype=np.float32)
        result = ensure_bh1(x, reduce_axis=2, reduction="sum")
        assert result is not None

    def test_with_reduce_axis_callable(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        x = np.ones((2, 5, 4), dtype=np.float32)

        def my_reduce(t, axis=None):
            return t

        result = ensure_bh1(x, reduce_axis=2, reduction=my_reduce)
        assert result is not None

    def test_rank3_with_high_last_dim_no_reduce(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        # rank=3 but last dim > 1 (no reduce) → stays as-is
        x = np.ones((2, 5, 10), dtype=np.float32)
        result = ensure_bh1(x)
        assert result is not None

    def test_none_reduce_axis_skips_reduction(self):
        if not _has_runtime():
            pytest.skip("No runtime available")
        x = np.ones((2, 5, 1), dtype=np.float32)
        result = ensure_bh1(x, reduce_axis=None)
        assert result is not None
