"""Extended coverage tests for components/_temporal_utils.py.

Exercises every aggregation mode of:
  - aggregate_multiscale
  - aggregate_multiscale_on_3d
  - aggregate_time_window_output
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("BASE_ATTENTIVE_BACKEND", "torch")


def _to_numpy(x):
    detach = getattr(x, "detach", None)
    if callable(detach):
        x = detach()
    cpu = getattr(x, "cpu", None)
    if callable(cpu):
        x = cpu()
    numpy = getattr(x, "numpy", None)
    if callable(numpy):
        try:
            return numpy()
        except Exception:
            pass
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------


def _load_temporal_utils():
    pytest.importorskip("keras", reason="Keras not installed")
    from base_attentive.components._temporal_utils import (
        aggregate_multiscale,
        aggregate_multiscale_on_3d,
        aggregate_time_window_output,
    )

    return {
        "aggregate_multiscale": aggregate_multiscale,
        "aggregate_multiscale_on_3d": (
            aggregate_multiscale_on_3d
        ),
        "aggregate_time_window_output": (
            aggregate_time_window_output
        ),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def aggregate_multiscale(*args, **kwargs):
    return _load_temporal_utils()["aggregate_multiscale"](
        *args,
        **kwargs,
    )


def aggregate_multiscale_on_3d(*args, **kwargs):
    return _load_temporal_utils()[
        "aggregate_multiscale_on_3d"
    ](*args, **kwargs)


def aggregate_time_window_output(*args, **kwargs):
    return _load_temporal_utils()[
        "aggregate_time_window_output"
    ](*args, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, T, U = 4, 10, 16


@pytest.fixture
def seq_list():
    """Three scale outputs of different time lengths."""
    return [
        np.random.randn(B, T, U).astype(np.float32),
        np.random.randn(B, T // 2, U).astype(np.float32),
        np.random.randn(B, T // 5, U).astype(np.float32),
    ]


@pytest.fixture
def uniform_seq_list():
    """Three scale outputs of equal time length (needed for flatten/concat)."""
    return [
        np.random.randn(B, T, U).astype(np.float32)
        for _ in range(3)
    ]


# ---------------------------------------------------------------------------
# aggregate_multiscale
# ---------------------------------------------------------------------------


class TestAggregateMultiscale:
    def test_mode_none_passthrough(self):
        tensor = np.random.randn(B, U * 3).astype(np.float32)
        result = aggregate_multiscale(tensor, mode=None)
        assert _to_numpy(result).shape == (B, U * 3)

    def test_mode_average(self, seq_list):
        result = _to_numpy(
            aggregate_multiscale(seq_list, mode="average")
        )
        assert result.shape == (B, U * 3)

    def test_mode_sum(self, seq_list):
        result = _to_numpy(
            aggregate_multiscale(seq_list, mode="sum")
        )
        assert result.shape == (B, U * 3)

    def test_mode_flatten(self, uniform_seq_list):
        result = _to_numpy(
            aggregate_multiscale(
                uniform_seq_list, mode="flatten"
            )
        )
        assert result.shape[0] == B
        assert result.ndim == 2

    def test_mode_concat(self, uniform_seq_list):
        result = _to_numpy(
            aggregate_multiscale(
                uniform_seq_list, mode="concat"
            )
        )
        assert result.shape == (B, U * 3)

    def test_mode_last(self, seq_list):
        result = _to_numpy(
            aggregate_multiscale(seq_list, mode="last")
        )
        assert result.shape == (B, U * 3)

    def test_mode_auto(self, seq_list):
        result = _to_numpy(
            aggregate_multiscale(seq_list, mode="auto")
        )
        assert result.shape == (B, U * 3)

    def test_single_scale(self):
        single = [np.random.randn(B, T, U).astype(np.float32)]
        result = _to_numpy(
            aggregate_multiscale(single, mode="last")
        )
        assert result.shape == (B, U)


# ---------------------------------------------------------------------------
# aggregate_multiscale_on_3d
# ---------------------------------------------------------------------------


class TestAggregateMultiscaleOn3D:
    def test_non_list_passthrough(self):
        tensor = np.random.randn(B, U * 3).astype(np.float32)
        result = _to_numpy(
            aggregate_multiscale_on_3d(tensor, mode="last")
        )
        assert result.shape == (B, U * 3)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            aggregate_multiscale_on_3d([], mode="last")

    def test_mode_last(self, seq_list):
        result = _to_numpy(
            aggregate_multiscale_on_3d(seq_list, mode="last")
        )
        assert result.shape == (B, U * 3)

    def test_mode_auto(self, seq_list):
        result = _to_numpy(
            aggregate_multiscale_on_3d(seq_list, mode="auto")
        )
        assert result.shape == (B, U * 3)

    def test_mode_average(self, seq_list):
        result = _to_numpy(
            aggregate_multiscale_on_3d(
                seq_list, mode="average"
            )
        )
        assert result.shape == (B, U * 3)

    def test_mode_sum(self, seq_list):
        result = _to_numpy(
            aggregate_multiscale_on_3d(seq_list, mode="sum")
        )
        assert result.shape == (B, U * 3)

    def test_mode_flatten(self, uniform_seq_list):
        result = _to_numpy(
            aggregate_multiscale_on_3d(
                uniform_seq_list, mode="flatten"
            )
        )
        assert result.shape[0] == B
        assert result.ndim == 2

    def test_mode_concat_3d_output(self, seq_list):
        import keras

        # Convert to keras tensors so tensor.shape.ndims works
        tensors = [
            keras.ops.convert_to_tensor(t) for t in seq_list
        ]
        result = _to_numpy(
            aggregate_multiscale_on_3d(tensors, mode="concat")
        )
        # Should produce 3D: (B, T_max, U * num_scales)
        assert result.ndim == 3
        assert result.shape[0] == B
        assert result.shape[2] == U * 3

    def test_mode_concat_non_3d_raises(self):
        import keras

        bad = [
            keras.ops.convert_to_tensor(
                np.random.randn(B, U).astype(np.float32)
            )
        ]
        with pytest.raises(ValueError):
            aggregate_multiscale_on_3d(bad, mode="concat")


# ---------------------------------------------------------------------------
# aggregate_time_window_output
# ---------------------------------------------------------------------------


class TestAggregateTimeWindowOutput:
    @pytest.fixture
    def twx(self):
        return np.random.randn(B, T, U).astype(np.float32)

    def test_mode_last(self, twx):
        result = _to_numpy(
            aggregate_time_window_output(twx, mode="last")
        )
        assert result.shape == (B, U)

    def test_mode_average(self, twx):
        result = _to_numpy(
            aggregate_time_window_output(twx, mode="average")
        )
        assert result.shape == (B, U)

    def test_mode_flatten(self, twx):
        result = _to_numpy(
            aggregate_time_window_output(twx, mode="flatten")
        )
        assert result.shape == (B, T * U)

    def test_mode_none_defaults_to_flatten(self, twx):
        result = _to_numpy(
            aggregate_time_window_output(twx, mode=None)
        )
        assert result.shape == (B, T * U)

    def test_invalid_mode_raises(self, twx):
        with pytest.raises(
            ValueError, match="Unsupported mode"
        ):
            aggregate_time_window_output(
                twx, mode="unknown_mode"
            )
