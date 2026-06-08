"""Configuration objects for PADR-Net flood forecasting."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral, Real

from ...compat.sklearn import Interval, validate_params


@dataclass(frozen=True, init=False)
class PADRNetConfig:
    """Configuration for PADR-Net flood forecasting.

    Parameters
    ----------
    input_dim:
        Number of dynamic forcing/covariate features per time
        step.
    static_dim:
        Optional number of static basin/region descriptors.
    hidden_dim:
        Latent feature dimension used by the temporal encoder.
    num_heads:
        Number of self-attention heads.
    num_layers:
        Number of recurrent-attention encoder blocks.
    forecast_horizon:
        Number of future time steps predicted by the depth
        head.
    dropout:
        Dropout rate used in backend implementations.
    lambda_physics:
        Weight intended for physics-residual regularization in
        training loops.
    lambda_mass:
        Weight intended for mass-conservation regularization.
    lambda_smooth:
        Weight intended for temporal smoothness
        regularization.
    flood_threshold:
        Water-depth threshold used by the exceedance
        probability head.
    reservoir_tau:
        Characteristic response time for simple storage
        residual diagnostics.
    """

    input_dim: int
    static_dim: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    forecast_horizon: int
    dropout: float
    lambda_physics: float
    lambda_mass: float
    lambda_smooth: float
    flood_threshold: float
    reservoir_tau: float

    @validate_params(
        {
            "input_dim": [
                Interval(Integral, 1, None, closed="left")
            ],
            "static_dim": [
                Interval(Integral, 0, None, closed="left")
            ],
            "hidden_dim": [
                Interval(Integral, 1, None, closed="left")
            ],
            "num_heads": [
                Interval(Integral, 1, None, closed="left")
            ],
            "num_layers": [
                Interval(Integral, 1, None, closed="left")
            ],
            "forecast_horizon": [
                Interval(Integral, 1, None, closed="left")
            ],
            "dropout": [
                Interval(Real, 0, 1, closed="left")
            ],
            "lambda_physics": [
                Interval(Real, 0, None, closed="left")
            ],
            "lambda_mass": [
                Interval(Real, 0, None, closed="left")
            ],
            "lambda_smooth": [
                Interval(Real, 0, None, closed="left")
            ],
            "flood_threshold": [
                Interval(Real, 0, None, closed="left")
            ],
            "reservoir_tau": [
                Interval(Real, 0, None, closed="left")
            ],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        forecast_horizon: int = 1,
        dropout: float = 0.0,
        lambda_physics: float = 0.1,
        lambda_mass: float = 0.0,
        lambda_smooth: float = 0.0,
        flood_threshold: float = 0.05,
        reservoir_tau: float = 24.0,
    ):
        object.__setattr__(self, "input_dim", int(input_dim))
        object.__setattr__(
            self, "static_dim", int(static_dim)
        )
        object.__setattr__(
            self, "hidden_dim", int(hidden_dim)
        )
        object.__setattr__(self, "num_heads", int(num_heads))
        object.__setattr__(
            self, "num_layers", int(num_layers)
        )
        object.__setattr__(
            self, "forecast_horizon", int(forecast_horizon)
        )
        object.__setattr__(self, "dropout", float(dropout))
        object.__setattr__(
            self, "lambda_physics", float(lambda_physics)
        )
        object.__setattr__(
            self, "lambda_mass", float(lambda_mass)
        )
        object.__setattr__(
            self, "lambda_smooth", float(lambda_smooth)
        )
        object.__setattr__(
            self, "flood_threshold", float(flood_threshold)
        )
        object.__setattr__(
            self, "reservoir_tau", float(reservoir_tau)
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads."
            )

    def with_updates(self, **updates) -> "PADRNetConfig":
        """Return a copy with selected fields updated."""

        return replace(self, **updates)
