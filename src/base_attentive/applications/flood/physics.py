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
    """Convert depth to flood-threshold probability.

    Implements the smooth logistic wet/dry indicator from
    Eq. (18) of the paper:
    pi = sigma((h - h_0) / s_h).
    """

    if threshold <= 0:
        raise ValueError("threshold must be positive.")
    h = np.asarray(depth, dtype=float)
    if scale is None:
        scale = max(0.004, 0.08 * threshold)
    if scale <= 0:
        raise ValueError("scale must be positive.")
    logits = np.clip((h - threshold) / scale, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


def recover_velocity(
    h,
    uh,
    vh,
    *,
    epsilon_h: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover depth-averaged velocities from conservative momentum.

    Applies Eq. (16) of the paper:
    u = uh / (h + epsilon_h),  v = vh / (h + epsilon_h).

    Parameters
    ----------
    h:
        Water depth array.
    uh:
        x-momentum component (u * h).
    vh:
        y-momentum component (v * h).
    epsilon_h:
        Small constant preventing division by zero in nearly
        dry cells.

    Returns
    -------
    u, v : tuple of np.ndarray
        Depth-averaged velocities in the x and y directions.
    """

    h_arr = np.asarray(h, dtype=float)
    uh_arr = np.asarray(uh, dtype=float)
    vh_arr = np.asarray(vh, dtype=float)
    if epsilon_h <= 0:
        raise ValueError("epsilon_h must be positive.")
    denom = h_arr + epsilon_h
    return uh_arr / denom, vh_arr / denom


def swe_residual(
    h,
    uh,
    vh,
    *,
    precipitation,
    infiltration=None,
    drainage=None,
    lateral_inflow=None,
    bed_slope_x: float = 0.0,
    bed_slope_y: float = 0.0,
    manning_n: float = 0.04,
    gravity: float = 9.81,
    epsilon_h: float = 1e-6,
    dt: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute the event-scale lumped SWE residual.

    Evaluates the shallow-water system residuals at each time
    step using finite differences in time.  The spatial
    representation is event-scale lumped (no explicit spatial
    grid), so advective flux divergence terms are omitted; bed
    slope and Manning friction source terms are included.

    The continuity residual is (Eq. 20 of the paper)::

        F_h = dh/dt - (P - I - D + Q_lat)

    The momentum residuals (lumped, no advection) are::

        F_x = d(uh)/dt + g_0 * h * db/dx + tau_bx / rho
        F_y = d(vh)/dt + g_0 * h * db/dy + tau_by / rho

    where the Manning friction stresses are::

        tau_bx / rho = C_f * u * sqrt(u^2 + v^2)
        C_f = g_0 * n^2 / h^{1/3}

    Parameters
    ----------
    h:
        Predicted water depth; shape (..., T).
    uh:
        Predicted x-momentum (u*h); same shape as h.
    vh:
        Predicted y-momentum (v*h); same shape as h.
    precipitation:
        Precipitation rate; same shape as h.
    infiltration:
        Infiltration loss rate; defaults to zero.
    drainage:
        Drainage loss rate; defaults to zero.
    lateral_inflow:
        Lateral inflow rate Q_lat; defaults to zero.
    bed_slope_x:
        Representative bed slope db/dx (scalar or array).
    bed_slope_y:
        Representative bed slope db/dy (scalar or array).
    manning_n:
        Manning roughness coefficient n.
    gravity:
        Gravitational acceleration g_0 (m s^{-2}).
    epsilon_h:
        Small constant for velocity recovery and friction
        computation in nearly dry cells.
    dt:
        Time step in hours (default 1.0, consistent with
        hourly ERA5 forcing).

    Returns
    -------
    dict with keys ``"continuity"``, ``"momentum_x"``,
    ``"momentum_y"``, each an ndarray of the same shape as h.
    """

    h_arr = np.asarray(h, dtype=float)
    uh_arr = np.asarray(uh, dtype=float)
    vh_arr = np.asarray(vh, dtype=float)
    precip = np.asarray(precipitation, dtype=float)

    zeros = np.zeros_like(h_arr)
    infil = (
        np.asarray(infiltration, dtype=float)
        if infiltration is not None
        else zeros
    )
    drain = (
        np.asarray(drainage, dtype=float)
        if drainage is not None
        else zeros
    )
    q_lat = (
        np.asarray(lateral_inflow, dtype=float)
        if lateral_inflow is not None
        else zeros
    )

    # Time derivatives via central differences (numpy gradient)
    dh_dt = np.gradient(h_arr, dt, axis=-1)
    duh_dt = np.gradient(uh_arr, dt, axis=-1)
    dvh_dt = np.gradient(vh_arr, dt, axis=-1)

    # Velocity recovery
    u, v = recover_velocity(
        h_arr, uh_arr, vh_arr, epsilon_h=epsilon_h
    )

    # Manning friction coefficient C_f = g_0 * n^2 / h^{1/3}
    h_safe = np.maximum(h_arr, epsilon_h)
    C_f = gravity * manning_n**2 / h_safe ** (1.0 / 3.0)
    speed = np.sqrt(u**2 + v**2)
    tau_bx_rho = C_f * u * speed
    tau_by_rho = C_f * v * speed

    # Continuity residual: dh/dt - (P - I - D + Q_lat)
    F_h = dh_dt - (precip - infil - drain + q_lat)

    # x-momentum residual (lumped, no advection divergence)
    F_x = duh_dt + gravity * h_arr * bed_slope_x + tau_bx_rho

    # y-momentum residual (lumped, no advection divergence)
    F_y = dvh_dt + gravity * h_arr * bed_slope_y + tau_by_rho

    return {
        "continuity": F_h,
        "momentum_x": F_x,
        "momentum_y": F_y,
    }
