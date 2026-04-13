"""Extended coverage tests for base_attentive.components.losses.

Covers CRPSLoss, AdaptiveQuantileLoss, AnomalyLoss, MultiObjectiveLoss,
CRPSLossWrapper, and the _std_normal_pdf/_std_normal_cdf helpers.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("BASE_ATTENTIVE_BACKEND", "torch")


def _to_numpy(x):
    """Safely convert a backend tensor to numpy."""
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
# Imports – skip entire module if Keras not available
# ---------------------------------------------------------------------------

pytest.importorskip("keras", reason="Keras not installed; skipping loss tests")

from base_attentive.components.losses import (  # noqa: E402
    AdaptiveQuantileLoss,
    AnomalyLoss,
    CRPSLoss,
    CRPSLossWrapper,
    MultiObjectiveLoss,
    _std_normal_cdf,
    _std_normal_pdf,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _y(shape, fill=0.5):
    return np.full(shape, fill, dtype=np.float32)


# ---------------------------------------------------------------------------
# _std_normal_pdf / _std_normal_cdf
# ---------------------------------------------------------------------------

class TestStdNormalHelpers:
    def test_pdf_at_zero(self):
        from base_attentive.components._config import tf_constant, tf_float32
        z = tf_constant(np.array([0.0]), dtype=tf_float32)
        result = _to_numpy(_std_normal_pdf(z))
        expected = 1.0 / np.sqrt(2.0 * np.pi)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_cdf_at_zero(self):
        from base_attentive.components._config import tf_constant, tf_float32
        z = tf_constant(np.array([0.0]), dtype=tf_float32)
        result = _to_numpy(_std_normal_cdf(z))
        np.testing.assert_allclose(result, 0.5, atol=1e-5)

    def test_cdf_positive(self):
        from base_attentive.components._config import tf_constant, tf_float32
        z = tf_constant(np.array([2.0]), dtype=tf_float32)
        result = _to_numpy(_std_normal_cdf(z))
        assert result[0] > 0.97


# ---------------------------------------------------------------------------
# AdaptiveQuantileLoss
# ---------------------------------------------------------------------------

class TestAdaptiveQuantileLoss:
    def test_none_quantiles_returns_zero(self):
        loss_fn = AdaptiveQuantileLoss(quantiles=None)
        result = loss_fn(_y((4, 10, 1)), _y((4, 10, 1)))
        assert float(_to_numpy(result)) == pytest.approx(0.0)

    def test_auto_quantiles(self):
        loss_fn = AdaptiveQuantileLoss(quantiles="auto")
        assert loss_fn.quantiles == [0.1, 0.5, 0.9]

    def test_4d_y_pred(self):
        loss_fn = AdaptiveQuantileLoss(quantiles=[0.1, 0.5, 0.9])
        y_true = _y((2, 5, 1))
        y_pred = _y((2, 5, 3, 1))
        result = loss_fn(y_true, y_pred)
        assert np.isfinite(float(_to_numpy(result)))

    def test_2d_y_pred(self):
        loss_fn = AdaptiveQuantileLoss(quantiles=[0.1, 0.9])
        y_true = _y((4, 1))
        y_pred = _y((4, 2))
        result = loss_fn(y_true, y_pred)
        assert np.isfinite(float(_to_numpy(result)))

    def test_get_config(self):
        loss_fn = AdaptiveQuantileLoss(quantiles=[0.2, 0.8])
        cfg = loss_fn.get_config()
        assert cfg["quantiles"] == [0.2, 0.8]

    def test_from_config(self):
        original = AdaptiveQuantileLoss(quantiles=[0.3, 0.7])
        cfg = original.get_config()
        cfg.pop("reduction", None)  # base Loss.get_config adds 'reduction'
        clone = AdaptiveQuantileLoss.from_config(cfg)
        assert clone.quantiles == [0.3, 0.7]


# ---------------------------------------------------------------------------
# AnomalyLoss
# ---------------------------------------------------------------------------

class TestAnomalyLoss:
    def test_basic_call_returns_scalar(self):
        loss_fn = AnomalyLoss(weight=1.0)
        scores = _y((4, 10, 8))
        # Call .call() directly to bypass Keras Loss.__call__ tensor conversion
        result = loss_fn.call(scores)
        assert np.isfinite(float(_to_numpy(result)))

    def test_weight_scales_loss(self):
        scores = _y((2, 5, 4))
        r1 = float(_to_numpy(AnomalyLoss(weight=1.0).call(scores)))
        r2 = float(_to_numpy(AnomalyLoss(weight=2.0).call(scores)))
        assert r2 == pytest.approx(r1 * 2.0, rel=1e-5)

    def test_get_config(self):
        cfg = AnomalyLoss(weight=3.0).get_config()
        assert cfg["weight"] == pytest.approx(3.0)

    def test_from_config(self):
        orig = AnomalyLoss(weight=4.0)
        cfg = orig.get_config()
        cfg.pop("reduction", None)  # base Loss.get_config adds 'reduction'
        clone = AnomalyLoss.from_config(cfg)
        assert clone.weight == pytest.approx(4.0)

    def test_with_y_pred_none(self):
        loss_fn = AnomalyLoss()
        # AnomalyLoss.call takes (anomaly_scores, y_pred=None)
        result = loss_fn.call(_y((2, 3, 4)), y_pred=None)
        assert np.isfinite(float(_to_numpy(result)))


# ---------------------------------------------------------------------------
# MultiObjectiveLoss
# ---------------------------------------------------------------------------

class TestMultiObjectiveLoss:
    def test_without_anomaly_scores(self):
        q_fn = AdaptiveQuantileLoss(quantiles=[0.1, 0.5, 0.9])
        a_fn = AnomalyLoss()
        loss_fn = MultiObjectiveLoss(q_fn, a_fn)
        y_true = _y((2, 5, 1))
        y_pred = _y((2, 5, 3, 1))
        result = loss_fn(y_true, y_pred)
        assert np.isfinite(float(_to_numpy(result)))

    def test_with_anomaly_scores(self):
        import torch
        q_fn = AdaptiveQuantileLoss(quantiles=[0.1, 0.5, 0.9])
        a_fn = AnomalyLoss(weight=0.5)
        anomaly = torch.tensor(_y((2, 5, 8)))
        loss_fn = MultiObjectiveLoss(q_fn, a_fn, anomaly_scores=anomaly)
        y_true = torch.tensor(_y((2, 5, 1)))
        y_pred = torch.tensor(_y((2, 5, 3, 1)))
        result = loss_fn(y_true, y_pred)
        assert np.isfinite(float(_to_numpy(result)))

    def test_default_sub_losses(self):
        loss_fn = MultiObjectiveLoss()
        assert isinstance(loss_fn.quantile_loss_fn, AdaptiveQuantileLoss)
        assert isinstance(loss_fn.anomaly_loss_fn, AnomalyLoss)

    def test_get_config(self):
        loss_fn = MultiObjectiveLoss()
        cfg = loss_fn.get_config()
        assert "quantile_loss_fn" in cfg
        assert "anomaly_loss_fn" in cfg

    def test_from_config(self):
        original = MultiObjectiveLoss()
        cfg = original.get_config()
        # Nested configs may include 'reduction' from base Loss.get_config
        for sub_key in ("quantile_loss_fn", "anomaly_loss_fn"):
            if isinstance(cfg.get(sub_key), dict):
                cfg[sub_key].pop("reduction", None)
        clone = MultiObjectiveLoss.from_config(cfg)
        assert clone is not None


# ---------------------------------------------------------------------------
# CRPSLoss
# ---------------------------------------------------------------------------

class TestCRPSLoss:
    def test_quantile_mode_tensor(self):
        loss_fn = CRPSLoss(mode="quantile", quantiles=[0.1, 0.5, 0.9])
        y_true = _y((2, 5, 1))
        y_pred = _y((2, 5, 3, 1))
        result = loss_fn(y_true, y_pred)
        assert np.isfinite(float(_to_numpy(result)))

    def test_quantile_mode_dict(self):
        loss_fn = CRPSLoss(mode="quantile", quantiles=[0.1, 0.5, 0.9])
        y_true = _y((2, 5, 1))
        y_pred = {"quantiles": _y((2, 5, 3, 1)), "q_values": [0.1, 0.5, 0.9]}
        result = loss_fn(y_true, y_pred)
        assert np.isfinite(float(_to_numpy(result)))

    def test_quantile_mode_dict_no_q_values(self):
        loss_fn = CRPSLoss(mode="quantile", quantiles=[0.1, 0.5, 0.9])
        y_true = _y((2, 5, 1))
        y_pred = {"quantiles": _y((2, 5, 3, 1))}
        result = loss_fn(y_true, y_pred)
        assert np.isfinite(float(_to_numpy(result)))

    def test_gaussian_mode_dict(self):
        loss_fn = CRPSLoss(mode="gaussian")
        y_true = _y((2, 5, 1))
        y_pred = {"loc": _y((2, 5, 1)), "scale": np.full((2, 5, 1), 0.5, dtype=np.float32)}
        result = loss_fn(y_true, y_pred)
        assert np.isfinite(float(_to_numpy(result)))

    def test_gaussian_mode_tensor(self):
        loss_fn = CRPSLoss(mode="gaussian")
        y_true = _y((2, 5))
        y_pred = _y((2, 5, 2))  # last dim = 2 => [loc, scale]
        result = loss_fn(y_true, y_pred)
        assert np.isfinite(float(_to_numpy(result)))

    def test_auto_mode_detects_quantile(self):
        loss_fn = CRPSLoss(mode="auto", quantiles=[0.1, 0.5, 0.9])
        y_pred = {"quantiles": _y((2, 5, 3, 1))}
        mode = loss_fn._infer_mode(y_pred)
        assert mode == "quantile"

    def test_auto_mode_detects_mixture(self):
        loss_fn = CRPSLoss(mode="auto")
        y_pred = {
            "loc": _y((2, 5, 2, 1)),
            "scale": _y((2, 5, 2, 1)),
            "weights": _y((2, 5, 2, 1)),
        }
        mode = loss_fn._infer_mode(y_pred)
        assert mode == "mixture"

    def test_auto_mode_detects_gaussian_from_dict(self):
        loss_fn = CRPSLoss(mode="auto")
        y_pred = {"loc": _y((2, 5, 1)), "scale": _y((2, 5, 1))}
        mode = loss_fn._infer_mode(y_pred)
        assert mode == "gaussian"

    def test_auto_mode_fallback_quantile_when_quantiles_set(self):
        loss_fn = CRPSLoss(mode="auto", quantiles=[0.5])
        mode = loss_fn._infer_mode(_y((2, 5)))  # not a dict, not dim-2
        assert mode == "quantile"

    def test_auto_mode_default_gaussian(self):
        loss_fn = CRPSLoss(mode="auto")
        mode = loss_fn._infer_mode(_y((2, 5)))
        assert mode == "gaussian"

    def test_unsupported_mode_raises(self):
        loss_fn = CRPSLoss(mode="quantile", quantiles=[0.5])
        loss_fn.mode = "invalid"
        with pytest.raises(ValueError):
            loss_fn(_y((2, 5, 1)), _y((2, 5, 1, 1)))

    def test_get_config(self):
        loss_fn = CRPSLoss(mode="gaussian", mc_samples=128)
        cfg = loss_fn.get_config()
        assert cfg["mode"] == "gaussian"
        assert cfg["mc_samples"] == 128

    def test_from_config(self):
        orig = CRPSLoss(mode="gaussian", quantiles=None, mc_samples=64)
        clone = CRPSLoss.from_config(orig.get_config())
        assert clone.mc_samples == 64

    def test_quantile_missing_raises(self):
        loss_fn = CRPSLoss(mode="quantile")  # no quantiles
        with pytest.raises((ValueError, TypeError, Exception)):
            loss_fn(_y((2, 5, 1)), _y((2, 5, 3, 1)))


# ---------------------------------------------------------------------------
# CRPSLossWrapper
# ---------------------------------------------------------------------------

class TestCRPSLossWrapper:
    def test_quantile_mode(self):
        loss_fn = CRPSLossWrapper(mode="quantile", quantiles=[0.1, 0.5, 0.9])
        y_true = _y((2, 5, 1))
        y_pred = _y((2, 5, 3, 1))
        result = loss_fn(y_true, y_pred)
        assert np.isfinite(float(_to_numpy(result)))

    def test_gaussian_mode_dict(self):
        loss_fn = CRPSLossWrapper(mode="gaussian")
        y_true = _y((2, 5, 1))
        y_pred = {"loc": _y((2, 5, 1)), "scale": np.full((2, 5, 1), 0.5, dtype=np.float32)}
        result = loss_fn(y_true, y_pred)
        assert np.isfinite(float(_to_numpy(result)))

    def test_gaussian_mode_tensor(self):
        loss_fn = CRPSLossWrapper(mode="gaussian")
        y_true = _y((2, 5))
        y_pred = _y((2, 5, 2))
        result = loss_fn(y_true, y_pred)
        assert np.isfinite(float(_to_numpy(result)))

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            CRPSLossWrapper(mode="mixture")

    def test_quantile_mode_without_quantiles_raises(self):
        with pytest.raises(ValueError):
            CRPSLossWrapper(mode="quantile", quantiles=None)

    def test_get_config(self):
        loss_fn = CRPSLossWrapper(mode="quantile", quantiles=[0.5])
        cfg = loss_fn.get_config()
        assert cfg["mode"] == "quantile"
        assert cfg["quantiles"] == [0.5]

    def test_from_config(self):
        orig = CRPSLossWrapper(mode="quantile", quantiles=[0.1, 0.9])
        clone = CRPSLossWrapper.from_config(orig.get_config())
        assert clone.quantiles == [0.1, 0.9]
