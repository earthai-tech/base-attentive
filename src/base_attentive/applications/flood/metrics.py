"""Flood metrics for PADR-Net examples and diagnostics."""

from __future__ import annotations

import numpy as np


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def nash_sutcliffe_efficiency(
    y_true,
    y_pred,
    *,
    eps: float = 1e-12,
) -> float:
    """Return the Nash-Sutcliffe efficiency coefficient."""

    true = _as_float_array(y_true)
    pred = _as_float_array(y_pred)
    denom = np.sum((true - np.mean(true)) ** 2)
    if denom <= eps:
        return 0.0
    error = np.sum((true - pred) ** 2)
    return float(1.0 - error / (denom + eps))


def delta_mass(
    y_true,
    y_pred,
    *,
    eps: float = 1e-12,
) -> float:
    """Return percentage mass bias."""

    true = _as_float_array(y_true)
    pred = _as_float_array(y_pred)
    total = np.sum(true)
    if abs(total) <= eps:
        return 0.0
    return float(
        100.0 * (np.sum(pred) - total) / (total + eps)
    )


def critical_success_index(
    y_true,
    y_pred,
    *,
    threshold: float,
) -> float:
    """Return CSI for threshold exceedance."""

    true = _as_float_array(y_true) >= threshold
    pred = _as_float_array(y_pred) >= threshold
    hits = np.logical_and(true, pred).sum()
    misses = np.logical_and(true, ~pred).sum()
    false_alarms = np.logical_and(~true, pred).sum()
    denom = hits + misses + false_alarms
    if denom == 0:
        return 0.0
    return float(hits / denom)


def true_skill_statistic(
    y_true,
    y_pred,
    *,
    threshold: float,
) -> float:
    """Return TSS = hit rate - false-alarm rate."""

    true = _as_float_array(y_true) >= threshold
    pred = _as_float_array(y_pred) >= threshold
    hits = np.logical_and(true, pred).sum()
    misses = np.logical_and(true, ~pred).sum()
    false_alarms = np.logical_and(~true, pred).sum()
    correct_negatives = np.logical_and(~true, ~pred).sum()
    hit_rate = (
        hits / (hits + misses) if hits + misses else 0.0
    )
    false_alarm_rate = (
        false_alarms / (false_alarms + correct_negatives)
        if false_alarms + correct_negatives
        else 0.0
    )
    return float(hit_rate - false_alarm_rate)
