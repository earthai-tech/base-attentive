"""Test validation module."""

from __future__ import annotations

import pytest


class TestValidationModule:
    """Test tensor validation utilities."""

    def test_validate_model_inputs_none(self):
        """Test validate_model_inputs with None inputs."""
        from base_attentive.validation import validate_model_inputs

        static, dynamic, future = validate_model_inputs(None)
        assert static is not None or dynamic is not None or future is not None

    def test_validate_model_inputs_list(self):
        """Test validate_model_inputs with list."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not installed")

        from base_attentive.validation import validate_model_inputs

        inputs = [
            tf.random.normal([32, 4]),  # static
            tf.random.normal([32, 10, 8]),  # dynamic
            tf.random.normal([32, 24, 6]),  # future
        ]
        static, dynamic, future = validate_model_inputs(inputs)
        assert static is not None
        assert dynamic is not None
        assert future is not None

    def test_validate_model_inputs_single_tensor(self):
        """Test validate_model_inputs with single tensor."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not installed")

        from base_attentive.validation import validate_model_inputs

        single_input = tf.random.normal([32, 10, 8])
        result = validate_model_inputs(single_input)
        assert result is not None

    def test_maybe_reduce_quantiles_bh_passthrough(self):
        """Test maybe_reduce_quantiles_bh with 2D tensor."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not installed")

        from base_attentive.validation import maybe_reduce_quantiles_bh

        # 2D tensor should pass through unchanged
        x = tf.random.normal([32, 10])
        result = maybe_reduce_quantiles_bh(x)
        assert result.shape == x.shape

    def test_maybe_reduce_quantiles_bh_reduction_3d(self):
        """Test maybe_reduce_quantiles_bh reduces 3D with Q > 1."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not installed")

        from base_attentive.validation import maybe_reduce_quantiles_bh

        # (B, H, Q) with Q > 1 should be reduced
        x = tf.random.normal([32, 10, 5])  # Q = 5
        result = maybe_reduce_quantiles_bh(x, reduction="mean")
        assert result.shape[-1] != 5 or len(result.shape) == 2

    def test_ensure_bh1_2d_to_3d(self):
        """Test ensure_bh1 converts 2D to 3D."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not installed")

        from base_attentive.validation import ensure_bh1

        # (B, H) -> (B, H, 1)
        x = tf.random.normal([32, 10])
        result = ensure_bh1(x)
        assert len(result.shape) == 3
        assert result.shape[-1] == 1

    def test_ensure_bh1_already_3d(self):
        """Test ensure_bh1 preserves 3D."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not installed")

        from base_attentive.validation import ensure_bh1

        # (B, H, 1) should remain unchanged
        x = tf.random.normal([32, 10, 1])
        result = ensure_bh1(x)
        assert len(result.shape) == 3
        assert result.shape[-1] == 1

    def test_ensure_bh1_dtype_cast(self):
        """Test ensure_bh1 casts dtype."""
        try:
            import tensorflow as tf
        except ImportError:
            pytest.skip("TensorFlow not installed")

        from base_attentive.validation import ensure_bh1

        x = tf.random.normal([32, 10])
        result = ensure_bh1(x, dtype=tf.float32)
        assert result.dtype == tf.float32
