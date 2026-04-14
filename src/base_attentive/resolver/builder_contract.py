"""Shared builder-contract helpers for resolver components."""

from __future__ import annotations

from typing import Any


def resolve_head_units(
    *,
    units: int | None = None,
    output_dim: int | None = None,
    forecast_horizon: int | None = None,
    quantiles: tuple[float, ...] | list[float] | None = None,
    is_quantile: bool = False,
) -> int:
    """Resolve the flattened output width for forecast heads."""
    if units is not None:
        return int(units)

    output_dim = int(output_dim or 1)
    forecast_horizon = int(forecast_horizon or 1)
    total = output_dim * forecast_horizon

    if is_quantile:
        active_quantiles = tuple(quantiles or ())
        total *= (
            len(active_quantiles) if active_quantiles else 1
        )

    return total


def build_head_kwargs(
    *,
    units: int | None = None,
    output_dim: int | None = None,
    forecast_horizon: int | None = None,
    quantiles: tuple[float, ...] | list[float] | None = None,
    activation: str | None = None,
) -> dict[str, Any]:
    """Return normalized keyword arguments for head builders."""
    return {
        "units": units,
        "output_dim": output_dim,
        "forecast_horizon": forecast_horizon,
        "quantiles": tuple(quantiles or ()),
        "activation": activation,
    }


__all__ = [
    "resolve_head_units",
    "build_head_kwargs",
]
