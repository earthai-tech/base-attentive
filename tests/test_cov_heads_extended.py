"""Extended coverage tests for base_attentive.components.heads.

Covers GaussianHead, MixtureDensityHead, PointForecastHead,
QuantileHead, CombinedHeadLoss, QuantileDistributionModeling,
and the _append_shape helper.
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


def _t(arr):
    """Convert numpy array to the active Keras backend tensor."""
    import keras as _keras

    return _keras.ops.convert_to_tensor(arr)


def _load_heads():
    pytest.importorskip("keras", reason="Keras not installed")
    import keras
    from base_attentive.components.heads import (
        CombinedHeadLoss,
        GaussianHead,
        MixtureDensityHead,
        PointForecastHead,
        QuantileDistributionModeling,
        QuantileHead,
        _append_shape,
    )

    return {
        "keras": keras,
        "CombinedHeadLoss": CombinedHeadLoss,
        "GaussianHead": GaussianHead,
        "MixtureDensityHead": MixtureDensityHead,
        "PointForecastHead": PointForecastHead,
        "QuantileDistributionModeling": QuantileDistributionModeling,
        "QuantileHead": QuantileHead,
        "_append_shape": _append_shape,
    }


# ---------------------------------------------------------------------------
# _append_shape helper
# ---------------------------------------------------------------------------


def _append_shape(*args, **kwargs):
    return _load_heads()["_append_shape"](*args, **kwargs)


def _GaussianHead(*args, **kwargs):
    return _load_heads()["GaussianHead"](*args, **kwargs)


def _MixtureDensityHead(*args, **kwargs):
    return _load_heads()["MixtureDensityHead"](
        *args, **kwargs
    )


def _PointForecastHead(*args, **kwargs):
    return _load_heads()["PointForecastHead"](*args, **kwargs)


def _QuantileDistributionModeling(*args, **kwargs):
    return _load_heads()["QuantileDistributionModeling"](
        *args,
        **kwargs,
    )


def _QuantileHead(*args, **kwargs):
    return _load_heads()["QuantileHead"](*args, **kwargs)


def _CombinedHeadLoss(*args, **kwargs):
    return _load_heads()["CombinedHeadLoss"](*args, **kwargs)


# ---------------------------------------------------------------------------
# _append_shape helper
# ---------------------------------------------------------------------------


class TestAppendShape:
    def test_tuple_input(self):
        result = _append_shape((4, 10, 32), 3, 1)
        assert result == (4, 10, 3, 1)

    def test_list_input(self):
        result = _append_shape([4, 10, 32], 3, 1)
        assert result == (4, 10, 3, 1)


# ---------------------------------------------------------------------------
# GaussianHead
# ---------------------------------------------------------------------------


class TestGaussianHead:
    def test_2d_input(self):
        head = _GaussianHead(output_dim=1)
        features = np.random.randn(4, 16).astype(np.float32)
        out = head(features)
        assert "mean" in out and "scale" in out
        assert _to_numpy(out["scale"]).min() > 0

    def test_3d_input(self):
        head = _GaussianHead(output_dim=2)
        features = np.random.randn(4, 10, 16).astype(
            np.float32
        )
        out = head(features)
        mean_shape = _to_numpy(out["mean"]).shape
        assert mean_shape[-1] == 2

    def test_nll_scalar(self):
        head = _GaussianHead(output_dim=1)
        features = np.random.randn(4, 16).astype(np.float32)
        out = head(features)
        y_true = _t(np.random.randn(4, 1).astype(np.float32))
        nll = head.nll(y_true, out["mean"], out["scale"])
        assert np.isfinite(float(_to_numpy(nll)))

    def test_get_config(self):
        cfg = _GaussianHead(
            output_dim=3, min_scale=1e-3
        ).get_config()
        assert cfg["output_dim"] == 3
        assert cfg["min_scale"] == pytest.approx(1e-3)

    def test_from_config(self):
        orig = _GaussianHead(output_dim=2, forecast_horizon=5)
        clone = _load_heads()["GaussianHead"].from_config(orig.get_config())
        assert clone.output_dim == 2
        assert clone.forecast_horizon == 5


# ---------------------------------------------------------------------------
# MixtureDensityHead
# ---------------------------------------------------------------------------


class TestMixtureDensityHead:
    def test_basic_call(self):
        head = _MixtureDensityHead(
            output_dim=1, num_components=3
        )
        features = np.random.randn(4, 16).astype(np.float32)
        out = head(features)
        assert (
            "weights" in out
            and "means" in out
            and "scales" in out
        )

    def test_scales_positive(self):
        head = _MixtureDensityHead(
            output_dim=1, num_components=2
        )
        features = np.random.randn(2, 8).astype(np.float32)
        out = head(features)
        assert _to_numpy(out["scales"]).min() > 0

    def test_single_component(self):
        head = _MixtureDensityHead(
            output_dim=1, num_components=1
        )
        features = np.random.randn(2, 8).astype(np.float32)
        out = head(features)
        assert out is not None

    def test_multi_output_dim(self):
        head = _MixtureDensityHead(
            output_dim=2, num_components=2
        )
        features = np.random.randn(4, 16).astype(np.float32)
        out = head(features)
        assert _to_numpy(out["means"]).shape[-1] == 2

    def test_nll_scalar(self):
        head = _MixtureDensityHead(
            output_dim=1, num_components=2
        )
        features = np.random.randn(4, 16).astype(np.float32)
        out = head(features)
        y_true = _t(np.random.randn(4, 1).astype(np.float32))
        nll = head.nll(
            y_true,
            out["weights"],
            out["means"],
            out["scales"],
        )
        assert np.isfinite(float(_to_numpy(nll)))

    def test_invalid_num_components_raises(self):
        with pytest.raises(ValueError):
            _MixtureDensityHead(
                output_dim=1, num_components=0
            )

    def test_get_config(self):
        cfg = _MixtureDensityHead(
            output_dim=1, num_components=4
        ).get_config()
        assert cfg["num_components"] == 4

    def test_from_config(self):
        orig = _MixtureDensityHead(
            output_dim=2, num_components=3
        )
        clone = _load_heads()["MixtureDensityHead"].from_config(
            orig.get_config()
        )
        assert clone.num_components == 3


# ---------------------------------------------------------------------------
# PointForecastHead
# ---------------------------------------------------------------------------


class TestPointForecastHead:
    def test_2d_input(self):
        head = _PointForecastHead(output_dim=1)
        features = np.random.randn(4, 16).astype(np.float32)
        out = _to_numpy(head(features))
        assert out.shape == (4, 1)

    def test_3d_input(self):
        head = _PointForecastHead(output_dim=3)
        features = np.random.randn(4, 10, 16).astype(
            np.float32
        )
        out = _to_numpy(head(features))
        assert out.shape[-1] == 3

    def test_get_config(self):
        cfg = _PointForecastHead(
            output_dim=2, forecast_horizon=7
        ).get_config()
        assert cfg["output_dim"] == 2
        assert cfg["forecast_horizon"] == 7

    def test_from_config(self):
        orig = _PointForecastHead(output_dim=4)
        clone = _load_heads()["PointForecastHead"].from_config(
            orig.get_config()
        )
        assert clone.output_dim == 4


# ---------------------------------------------------------------------------
# QuantileHead
# ---------------------------------------------------------------------------


class TestQuantileHead:
    def test_output_shape_2d(self):
        head = _QuantileHead(
            quantiles=[0.1, 0.5, 0.9], output_dim=1
        )
        features = np.random.randn(4, 16).astype(np.float32)
        out = _to_numpy(head(features))
        assert out.shape[-2] == 3  # Q=3
        assert out.shape[-1] == 1  # O=1

    def test_output_shape_3d(self):
        head = _QuantileHead(
            quantiles=[0.1, 0.9], output_dim=2
        )
        features = np.random.randn(4, 10, 16).astype(
            np.float32
        )
        out = _to_numpy(head(features))
        assert out.shape[-2] == 2  # Q=2
        assert out.shape[-1] == 2  # O=2

    def test_empty_quantiles_raises(self):
        with pytest.raises(ValueError):
            _QuantileHead(quantiles=[], output_dim=1)

    def test_get_config(self):
        cfg = _QuantileHead(
            quantiles=[0.1, 0.5, 0.9], output_dim=1
        ).get_config()
        assert cfg["quantiles"] == [0.1, 0.5, 0.9]

    def test_from_config(self):
        orig = _QuantileHead(
            quantiles=[0.25, 0.75], output_dim=2
        )
        clone = _load_heads()["QuantileHead"].from_config(orig.get_config())
        assert clone.quantiles == [0.25, 0.75]


# ---------------------------------------------------------------------------
# CombinedHeadLoss
# ---------------------------------------------------------------------------


class TestCombinedHeadLoss:
    def _mse(self, y_true, y_pred):
        return _to_numpy(
            np.mean(
                (np.asarray(y_true) - np.asarray(y_pred)) ** 2
            )
        )

    def test_default_loss(self):
        loss_fn = _CombinedHeadLoss()
        y = _t(np.ones((2, 5, 1), dtype=np.float32))
        result = loss_fn({"default": y}, {"default": y})
        assert np.isfinite(float(_to_numpy(result)))

    def test_custom_heads_sum(self):
        from base_attentive.components.losses import (
            AnomalyLoss,
        )

        heads = {
            "a": (AnomalyLoss(weight=1.0), 1.0),
            "b": (AnomalyLoss(weight=2.0), 0.5),
        }
        loss_fn = _CombinedHeadLoss(heads, reduction="sum")
        y = _t(np.ones((2, 4, 2), dtype=np.float32))
        result = loss_fn({"a": y, "b": y}, {"a": y, "b": y})
        assert np.isfinite(float(_to_numpy(result)))

    def test_mean_reduction(self):
        from base_attentive.components.losses import (
            AnomalyLoss,
        )

        heads = {"x": (AnomalyLoss(), 1.0)}
        loss_fn = _CombinedHeadLoss(heads, reduction="mean")
        y = _t(np.ones((2, 3, 1), dtype=np.float32))
        result = loss_fn({"x": y}, {"x": y})
        assert np.isfinite(float(_to_numpy(result)))

    def test_invalid_reduction_raises(self):
        from base_attentive.components.losses import (
            AnomalyLoss,
        )

        heads = {"x": (AnomalyLoss(), 1.0)}
        loss_fn = _CombinedHeadLoss(
            heads, reduction="invalid"
        )
        y = _t(np.ones((2, 3, 1), dtype=np.float32))
        with pytest.raises(ValueError):
            loss_fn({"x": y}, {"x": y})

    def test_missing_key_raises(self):
        from base_attentive.components.losses import (
            AnomalyLoss,
        )

        heads = {"a": (AnomalyLoss(), 1.0)}
        loss_fn = _CombinedHeadLoss(heads)
        y = _t(np.ones((2, 3, 1), dtype=np.float32))
        with pytest.raises(KeyError):
            loss_fn({"wrong_key": y}, {"wrong_key": y})

    def test_weight_only_in_tuple(self):
        from base_attentive.components.losses import (
            AnomalyLoss,
        )

        # Supply single-element tuple (no explicit weight)
        heads = {"a": (AnomalyLoss(),)}
        loss_fn = _CombinedHeadLoss(heads)
        y = _t(np.ones((2, 3, 1), dtype=np.float32))
        result = loss_fn({"a": y}, {"a": y})
        assert np.isfinite(float(_to_numpy(result)))

    def test_bare_callable_value(self):
        from base_attentive.components.losses import (
            AnomalyLoss,
        )

        # Supply bare callable (no tuple wrapper)
        heads = {"a": AnomalyLoss()}
        loss_fn = _CombinedHeadLoss(heads)
        y = _t(np.ones((2, 3, 1), dtype=np.float32))
        result = loss_fn({"a": y}, {"a": y})
        assert np.isfinite(float(_to_numpy(result)))

    def test_get_config(self):
        loss_fn = _CombinedHeadLoss()
        cfg = loss_fn.get_config()
        assert "heads_losses" in cfg


# ---------------------------------------------------------------------------
# QuantileDistributionModeling
# ---------------------------------------------------------------------------


class TestQuantileDistributionModeling:
    def test_with_quantiles(self):
        qdm = _QuantileDistributionModeling(
            quantiles=[0.1, 0.5, 0.9], output_dim=1
        )
        features = np.random.randn(4, 10, 16).astype(
            np.float32
        )
        out = _to_numpy(qdm(features))
        assert out.shape[-2] == 3  # Q
        assert out.shape[-1] == 1  # O

    def test_without_quantiles(self):
        qdm = _QuantileDistributionModeling(
            quantiles=None, output_dim=2
        )
        features = np.random.randn(4, 10, 16).astype(
            np.float32
        )
        out = _to_numpy(qdm(features))
        assert out.shape[-1] == 2

    def test_auto_quantiles(self):
        qdm = _QuantileDistributionModeling(
            quantiles="auto", output_dim=1
        )
        assert qdm.quantiles == [0.1, 0.5, 0.9]

    def test_get_config(self):
        cfg = _QuantileDistributionModeling(
            quantiles=[0.1, 0.9], output_dim=1
        ).get_config()
        assert cfg["quantiles"] == [0.1, 0.9]

    def test_from_config(self):
        orig = _QuantileDistributionModeling(
            quantiles=[0.5], output_dim=3
        )
        clone = _load_heads()["QuantileDistributionModeling"].from_config(
            orig.get_config()
        )
        assert clone.output_dim == 3
