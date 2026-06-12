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
        step (N_in in the paper).  These include precipitation,
        antecedent wetness, river-state indicators, terrain
        gradient, flow accumulation, seasonal variables, and
        exposure descriptors.
    static_dim:
        Optional number of static basin/region descriptors
        appended to the dynamic input at each time step.
    reservoir_dim:
        Dimension N_res of the echo-state reservoir state
        vector x_{i,t,g}.
    spectral_radius:
        Spectral radius rho(W_res) of the fixed reservoir
        weight matrix.  Must lie strictly in (0, 1) so that
        the contractivity condition L_sigma * rho < 1 is
        satisfied (tanh has Lipschitz constant L_sigma = 1).
    input_scaling:
        Scale factor applied when randomly initialising the
        input weight matrix W_in.
    gravity:
        Gravitational acceleration g_0 (m s^{-2}).
    manning_n:
        Manning roughness coefficient n used in the SWE
        friction source term.
    flood_threshold:
        Water-depth threshold h_0 (m) for the smooth wet/dry
        indicator (exceedance probability head).
    slope_h:
        Softness parameter s_h for the logistic wet/dry
        transition.  Smaller values give a sharper threshold.
    epsilon_h:
        Small positive constant added to depth when recovering
        velocity from momentum, preventing division by zero in
        nearly dry cells.
    reservoir_tau:
        Characteristic response time tau (hours) used by the
        simple linear-reservoir diagnostic helper.
    lambda_physics:
        Weight lambda_phys on the shallow-water residual loss
        L_phys in the training objective.
    lambda_readout:
        Ridge penalty omega on the Frobenius norm of the
        trainable readout matrix W_out.
    lambda_severity:
        Ridge penalty omega_s on the event-severity head
        coefficients beta_s.
    w_ext:
        Weight w_ext for the satellite-extent binary
        cross-entropy loss channel L_ext.
    w_depth:
        Weight w_depth for the continuous-depth MSE loss
        channel L_depth.
    w_impact:
        Weight w_impact for the event-level impact MSE loss
        channel L_impact.
    dropout:
        Dropout rate applied in the event-severity head.
    """

    input_dim: int
    static_dim: int
    reservoir_dim: int
    spectral_radius: float
    input_scaling: float
    gravity: float
    manning_n: float
    flood_threshold: float
    slope_h: float
    epsilon_h: float
    reservoir_tau: float
    lambda_physics: float
    lambda_readout: float
    lambda_severity: float
    w_ext: float
    w_depth: float
    w_impact: float
    dropout: float

    @validate_params(
        {
            "input_dim": [
                Interval(Integral, 1, None, closed="left")
            ],
            "static_dim": [
                Interval(Integral, 0, None, closed="left")
            ],
            "reservoir_dim": [
                Interval(Integral, 1, None, closed="left")
            ],
            "spectral_radius": [
                Interval(Real, 0, 1, closed="neither")
            ],
            "input_scaling": [
                Interval(Real, 0, None, closed="neither")
            ],
            "gravity": [
                Interval(Real, 0, None, closed="neither")
            ],
            "manning_n": [
                Interval(Real, 0, None, closed="neither")
            ],
            "flood_threshold": [
                Interval(Real, 0, None, closed="neither")
            ],
            "slope_h": [
                Interval(Real, 0, None, closed="neither")
            ],
            "epsilon_h": [
                Interval(Real, 0, None, closed="neither")
            ],
            "reservoir_tau": [
                Interval(Real, 0, None, closed="neither")
            ],
            "lambda_physics": [
                Interval(Real, 0, None, closed="left")
            ],
            "lambda_readout": [
                Interval(Real, 0, None, closed="left")
            ],
            "lambda_severity": [
                Interval(Real, 0, None, closed="left")
            ],
            "w_ext": [
                Interval(Real, 0, None, closed="left")
            ],
            "w_depth": [
                Interval(Real, 0, None, closed="left")
            ],
            "w_impact": [
                Interval(Real, 0, None, closed="left")
            ],
            "dropout": [
                Interval(Real, 0, 1, closed="left")
            ],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        reservoir_dim: int = 500,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.1,
        gravity: float = 9.81,
        manning_n: float = 0.04,
        flood_threshold: float = 0.05,
        slope_h: float = 0.01,
        epsilon_h: float = 1e-6,
        reservoir_tau: float = 24.0,
        lambda_physics: float = 0.1,
        lambda_readout: float = 1e-4,
        lambda_severity: float = 1e-4,
        w_ext: float = 1.0,
        w_depth: float = 1.0,
        w_impact: float = 1.0,
        dropout: float = 0.0,
    ):
        object.__setattr__(self, "input_dim", int(input_dim))
        object.__setattr__(
            self, "static_dim", int(static_dim)
        )
        object.__setattr__(
            self, "reservoir_dim", int(reservoir_dim)
        )
        object.__setattr__(
            self, "spectral_radius", float(spectral_radius)
        )
        object.__setattr__(
            self, "input_scaling", float(input_scaling)
        )
        object.__setattr__(self, "gravity", float(gravity))
        object.__setattr__(
            self, "manning_n", float(manning_n)
        )
        object.__setattr__(
            self, "flood_threshold", float(flood_threshold)
        )
        object.__setattr__(self, "slope_h", float(slope_h))
        object.__setattr__(
            self, "epsilon_h", float(epsilon_h)
        )
        object.__setattr__(
            self, "reservoir_tau", float(reservoir_tau)
        )
        object.__setattr__(
            self, "lambda_physics", float(lambda_physics)
        )
        object.__setattr__(
            self, "lambda_readout", float(lambda_readout)
        )
        object.__setattr__(
            self, "lambda_severity", float(lambda_severity)
        )
        object.__setattr__(self, "w_ext", float(w_ext))
        object.__setattr__(self, "w_depth", float(w_depth))
        object.__setattr__(
            self, "w_impact", float(w_impact)
        )
        object.__setattr__(self, "dropout", float(dropout))

    def with_updates(self, **updates) -> "PADRNetConfig":
        """Return a copy with selected fields updated."""

        return replace(self, **updates)
