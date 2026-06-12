"""PyTorch implementation of PADR-Net."""

from __future__ import annotations

import numpy as np

from ...api.property import NNLearner
from ...applications.flood.config import PADRNetConfig

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


def _ensure_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required for TorchPADRNet. "
            "Install it with: pip install torch"
        )


def _init_reservoir_matrices(
    input_dim: int,
    reservoir_dim: int,
    *,
    spectral_radius: float,
    input_scaling: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (W_in, W_res, b_res) as numpy float32 arrays.

    W_res is scaled so its spectral radius matches the
    requested value, satisfying the contractivity condition
    L_sigma * rho(W_res) < 1 for tanh (L_sigma = 1).
    """

    rng = np.random.default_rng(seed)
    W_in = (
        rng.standard_normal((reservoir_dim, input_dim)).astype(
            np.float32
        )
        * input_scaling
    )
    W_raw = rng.standard_normal(
        (reservoir_dim, reservoir_dim)
    ).astype(np.float32)
    eigvals = np.linalg.eigvals(W_raw)
    current_sr = float(np.abs(eigvals).max())
    W_res = (
        W_raw * (spectral_radius / current_sr)
        if current_sr > 1e-10
        else W_raw
    )
    b_res = (
        rng.standard_normal(reservoir_dim).astype(np.float32)
        * 0.1
    )
    return W_in, W_res.astype(np.float32), b_res


_TORCH_MODULE_BASE = nn.Module if nn else object


class TorchPADRNet(NNLearner, _TORCH_MODULE_BASE):
    """Echo-state PADR-Net for flood forecasting.

    The reservoir evolves as (Eq. 13 of the paper)::

        x_t = tanh(W_in @ phi_t + W_res @ x_{t-1} + b_res)

    where W_in and W_res are fixed at initialisation.  Only
    the linear readout W_out and the event-severity head
    are optimised.  The readout produces the conservative
    state (h_raw, uh, vh) and depth is recovered via the
    log-transform inverse h = expm1(h_raw).clamp(0).
    """

    def __init__(
        self,
        config: PADRNetConfig,
        *,
        name: str = "padr_net",
        **kwargs,
    ):
        del kwargs
        _ensure_torch()
        super().__init__()
        self.config = config
        self.name = name

        N_in = config.input_dim + config.static_dim
        N_res = config.reservoir_dim

        W_in_np, W_res_np, b_res_np = _init_reservoir_matrices(
            N_in,
            N_res,
            spectral_radius=config.spectral_radius,
            input_scaling=config.input_scaling,
        )
        self.register_buffer(
            "W_in", torch.from_numpy(W_in_np)
        )
        self.register_buffer(
            "W_res", torch.from_numpy(W_res_np)
        )
        self.register_buffer(
            "b_res", torch.from_numpy(b_res_np)
        )

        # Trainable linear readout: x -> (h_raw, uh, vh)
        self.W_out = nn.Linear(N_res, 3, bias=True)

        # Event-severity head: [x_T; mean(x)] -> scalar impact
        self.severity_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(2 * N_res, 1),
        )

    def forward(self, inputs, static_inputs=None):
        """Run the ESN reservoir and compute all output heads.

        Parameters
        ----------
        inputs:
            Dynamic forcing tensor of shape (B, T, input_dim).
        static_inputs:
            Optional static descriptor tensor of shape
            (B, static_dim).  Concatenated to dynamic inputs
            at each time step.

        Returns
        -------
        dict with keys:
        - ``"depth"``            : (B, T, 1), non-negative
        - ``"momentum_x"``       : (B, T, 1)
        - ``"momentum_y"``       : (B, T, 1)
        - ``"velocity_x"``       : (B, T, 1)
        - ``"velocity_y"``       : (B, T, 1)
        - ``"exceedance_probability"`` : (B, T, 1) in [0, 1]
        - ``"reservoir_states"`` : (B, T, N_res)
        - ``"severity"``         : (B, 1)
        """

        B, T, _ = inputs.shape

        # Concatenate static descriptors (replicated over time)
        if (
            self.config.static_dim > 0
            and static_inputs is not None
        ):
            static_exp = static_inputs.unsqueeze(1).expand(
                B, T, -1
            )
            phi = torch.cat([inputs, static_exp], dim=-1)
        else:
            phi = inputs

        # Run fixed reservoir recurrence
        x = torch.zeros(
            B,
            self.config.reservoir_dim,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        states = []
        for t in range(T):
            x = torch.tanh(
                phi[:, t, :] @ self.W_in.T
                + x @ self.W_res.T
                + self.b_res
            )
            states.append(x)

        # states: (B, T, N_res)
        states_t = torch.stack(states, dim=1)

        # Linear readout -> conservative state (h_raw, uh, vh)
        q_hat = self.W_out(states_t)  # (B, T, 3)

        # Depth: softplus readout h = log(1+exp(eta_h)) ≥ 0
        # matches Eq. padr_positive_depth in the paper
        h = torch.nn.functional.softplus(q_hat[..., 0])
        uh = q_hat[..., 1]
        vh = q_hat[..., 2]

        # Velocity recovery: u = uh / (h + eps)
        eps = self.config.epsilon_h
        u = uh / (h + eps)
        v = vh / (h + eps)

        # Smooth exceedance probability
        h0 = self.config.flood_threshold
        sh = self.config.slope_h
        pi = torch.sigmoid((h - h0) / sh)

        # Event-level severity: concat last state and mean state
        s_event = torch.cat(
            [states_t[:, -1, :], states_t.mean(dim=1)], dim=-1
        )
        severity = self.severity_head(s_event)

        return {
            "depth": h.unsqueeze(-1),
            "momentum_x": uh.unsqueeze(-1),
            "momentum_y": vh.unsqueeze(-1),
            "velocity_x": u.unsqueeze(-1),
            "velocity_y": v.unsqueeze(-1),
            "exceedance_probability": pi.unsqueeze(-1),
            "reservoir_states": states_t,
            "severity": severity,
        }


__all__ = ["TorchPADRNet"]
