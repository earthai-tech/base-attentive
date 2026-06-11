"""Backend-neutral hydrological helpers for PADR-Net."""

from __future__ import annotations

import numpy as np


def linear_reservoir_response(
    precipitation,
    *,
    tau: float = 24.0,
    gain: float = 0.01,
    initial_depth: float = 0.0,
) -> np.ndarray:
    """Compute a simple linear-reservoir depth response.

    This helper is lightweight. It is useful for examples,
    diagnostics, and tests, while full training loops
    may replace it with a richer hydrodynamic operator.
    """

    precip = np.asarray(precipitation, dtype=float)
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if gain < 0:
        raise ValueError("gain cannot be negative.")
    response = np.zeros_like(precip, dtype=float)
    if response.size == 0:
        return response
    response[..., 0] = initial_depth
    decay = np.exp(-1.0 / tau)
    for i in range(1, precip.shape[-1]):
        response[..., i] = (
            decay * response[..., i - 1]
            + (1.0 - decay) * gain * precip[..., i - 1]
        )
    return np.maximum(response, 0.0)


def mass_balance_residual(
    precipitation,
    depth,
    *,
    tau: float = 24.0,
    gain: float | None = None,
) -> np.ndarray:
    """Return a discrete rainfall-storage residual.

    The residual is
    ``dh/dt - (gain * precipitation - depth / tau)``.  If
    ``gain`` is absent, it is estimated by least squares.
    """

    precip = np.asarray(precipitation, dtype=float)
    h = np.asarray(depth, dtype=float)
    if precip.shape != h.shape:
        raise ValueError(
            "precipitation and depth shapes must match."
        )
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if h.size == 0:
        return np.asarray(h, dtype=float)

    dh = np.gradient(h, axis=-1)
    if gain is None:
        target = dh + h / tau
        denom = np.sum(precip * precip)
        gain = (
            0.0
            if denom <= 1e-12
            else float(np.sum(target * precip) / denom)
        )
    return dh - (float(gain) * precip - h / tau)


def exceedance_probability(
    depth,
    *,
    threshold: float,
    scale: float | None = None,
) -> np.ndarray:
    """Convert depth to flood-threshold probability."""

    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    h = np.asarray(depth, dtype=float)
    if scale is None:
        scale = max(0.004, 0.08 * threshold)
    if scale <= 0:
        raise ValueError("scale must be positive.")
    logits = np.clip((h - threshold) / scale, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))
