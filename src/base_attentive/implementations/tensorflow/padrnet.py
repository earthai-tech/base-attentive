"""TensorFlow implementation of PADR-Net."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..._bootstrap import KERAS_DEPS
from ...api.property import NNLearner
from ...applications.flood.config import PADRNetConfig

try:
    import tensorflow as tf

    try:
        layers = tf.keras.layers
        Model = tf.keras.Model
        keras_register = (
            tf.keras.utils.register_keras_serializable
        )
    except (ImportError, ModuleNotFoundError):
        from tensorflow.python import keras as _tf_keras
        from tensorflow.python.keras import layers
        from tensorflow.python.keras.utils import (
            generic_utils as _generic_utils,
        )

        keras_register = (
            _generic_utils.register_keras_serializable
        )
        Model = _tf_keras.Model
except ImportError:  # pragma: no cover
    tf = None
    layers = None
    Model = object
    keras_register = None


def _ensure_tensorflow() -> None:
    if tf is None or layers is None:
        raise ImportError(
            "TensorFlow is required for TensorFlowPADRNet. "
            "Install it with: pip install tensorflow"
        )


def _register_keras_serializable(name: str):
    if keras_register is None:
        return lambda cls: cls
    return keras_register(package=__name__, name=name)


def _init_reservoir_matrices(
    input_dim: int,
    reservoir_dim: int,
    *,
    spectral_radius: float,
    input_scaling: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (W_in, W_res, b_res) as numpy float32 arrays."""

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


_TF_PADR_BASES = (Model, NNLearner) if tf else (NNLearner,)


@_register_keras_serializable("TensorFlowPADRNet")
class TensorFlowPADRNet(*_TF_PADR_BASES):
    """Echo-state PADR-Net for flood forecasting (TensorFlow).

    The reservoir evolves as (Eq. 13 of the paper)::

        x_t = tanh(W_in @ phi_t + W_res @ x_{t-1} + b_res)

    W_in and W_res are stored as non-trainable weights.  Only
    the linear readout W_out and the event-severity head are
    optimised.
    """

    def __init__(
        self,
        config: PADRNetConfig,
        *,
        name: str = "padr_net",
        **kwargs: Any,
    ):
        _ensure_tensorflow()
        super().__init__(name=name, **kwargs)
        self.config = config

        N_in = config.input_dim + config.static_dim
        N_res = config.reservoir_dim

        W_in_np, W_res_np, b_res_np = _init_reservoir_matrices(
            N_in,
            N_res,
            spectral_radius=config.spectral_radius,
            input_scaling=config.input_scaling,
        )

        # Fixed (non-trainable) reservoir matrices stored as
        # tf.Variable so they are tracked by the Keras model
        # without relying on the keras.initializers path that
        # conflicts with standalone Keras 3.x installations.
        self._W_in = tf.Variable(
            W_in_np, trainable=False, name="W_in"
        )
        self._W_res = tf.Variable(
            W_res_np, trainable=False, name="W_res"
        )
        self._b_res = tf.Variable(
            b_res_np, trainable=False, name="b_res"
        )

        # Trainable readout: x -> (h_raw, uh, vh)
        self._W_out = layers.Dense(
            3, use_bias=True, name="W_out"
        )

        # Event-severity head
        self._dropout = layers.Dropout(
            config.dropout, name="severity_dropout"
        )
        self._severity = layers.Dense(
            1, name="severity_head"
        )

    def call(
        self,
        inputs,
        static_inputs=None,
        training: bool = False,  # noqa: FBT002
    ):
        """Run the ESN reservoir and compute all output heads.

        Parameters
        ----------
        inputs:
            Dynamic forcing tensor of shape (B, T, input_dim).
        static_inputs:
            Optional static descriptor tensor of shape
            (B, static_dim).
        training:
            Passed to dropout layers.

        Returns
        -------
        dict with keys ``"depth"``, ``"momentum_x"``,
        ``"momentum_y"``, ``"velocity_x"``, ``"velocity_y"``,
        ``"exceedance_probability"``, ``"reservoir_states"``,
        ``"severity"``.
        """

        # Concatenate static descriptors (replicated over time)
        if (
            self.config.static_dim > 0
            and static_inputs is not None
        ):
            T = tf.shape(inputs)[1]
            static_exp = tf.tile(
                KERAS_DEPS.expand_dims(static_inputs, axis=1),
                [1, T, 1],
            )
            phi = tf.concat([inputs, static_exp], axis=-1)
        else:
            phi = inputs

        B = tf.shape(phi)[0]
        T_static = phi.shape[1]  # static dim for unstack
        N_res = self.config.reservoir_dim

        # Run fixed reservoir recurrence over time axis
        x = tf.zeros((B, N_res), dtype=phi.dtype)
        phi_steps = tf.unstack(phi, axis=1)
        states = []
        W_in = tf.cast(tf.transpose(self._W_in), phi.dtype)
        W_res = tf.cast(tf.transpose(self._W_res), phi.dtype)
        b_res = tf.cast(self._b_res, phi.dtype)
        for phi_t in phi_steps:
            x = tf.math.tanh(
                phi_t @ W_in + x @ W_res + b_res
            )
            states.append(x)

        # states_t: (B, T, N_res)
        states_t = tf.stack(states, axis=1)

        # Linear readout -> (h_raw, uh, vh)
        q_hat = self._W_out(states_t)  # (B, T, 3)

        # Depth: softplus readout h = log(1+exp(eta_h)) ≥ 0
        # matches Eq. padr_positive_depth in the paper
        h_raw = q_hat[..., 0]
        h = tf.math.softplus(h_raw)
        uh = q_hat[..., 1]
        vh = q_hat[..., 2]

        # Velocity recovery
        eps = tf.cast(self.config.epsilon_h, h.dtype)
        u = uh / (h + eps)
        v = vh / (h + eps)

        # Smooth exceedance probability
        h0 = tf.cast(
            self.config.flood_threshold, h.dtype
        )
        sh = tf.cast(self.config.slope_h, h.dtype)
        pi = KERAS_DEPS.sigmoid((h - h0) / sh)

        # Event-level severity: [x_T; mean(x)]
        x_last = states_t[:, -1, :]
        x_mean = tf.reduce_mean(states_t, axis=1)
        s_event = tf.concat([x_last, x_mean], axis=-1)
        s_event = self._dropout(s_event, training=training)
        severity = self._severity(s_event)

        return {
            "depth": KERAS_DEPS.expand_dims(h, axis=-1),
            "momentum_x": KERAS_DEPS.expand_dims(
                uh, axis=-1
            ),
            "momentum_y": KERAS_DEPS.expand_dims(
                vh, axis=-1
            ),
            "velocity_x": KERAS_DEPS.expand_dims(u, axis=-1),
            "velocity_y": KERAS_DEPS.expand_dims(v, axis=-1),
            "exceedance_probability": KERAS_DEPS.expand_dims(
                pi, axis=-1
            ),
            "reservoir_states": states_t,
            "severity": severity,
        }

    def get_config(self) -> dict[str, Any]:
        cfg = self.config.__dict__.copy()
        return {"name": self.name, "padrnet_config": cfg}

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        padrnet_cfg = config.pop("padrnet_config")
        return cls(PADRNetConfig(**padrnet_cfg), **config)


__all__ = ["TensorFlowPADRNet"]
