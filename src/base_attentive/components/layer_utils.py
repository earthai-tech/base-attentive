# SPDX-License-Identifier: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>
# Adapted from: earthai-tech/fusionlab-learn — https://github.com/earthai-tech/gofast
# Modified for GeoPrior-v3 API.

"""
geoprior/nn/components/layer_utils.py

Small generic helpers + micro‑layers (residual, gating, etc.).

"""

from __future__ import annotations

from ..api.property import NNLearner
from ..utils.deps_utils import ensure_pkg
from ._config import (
    DEP_MSG,
    KERAS_BACKEND,
    KERAS_DEPS,
    Dense,
    Layer,
    Tensor,
    register_keras_serializable,
    add,
    do_not_convert,
    bool_dtype,
    cast,
    expand_dims,
    float32,
    multiply,
    rank as tensor_rank,
    reduce_mean,
    shape,
    stack,
    tile,
    where,
)

K = KERAS_DEPS

__all__ = [
    "ResidualAdd",
    "LayerScale",
    "StochasticDepth",
    "SqueezeExcite1D",
    "Gate",
    "maybe_expand_time",
    "broadcast_like",
    "ensure_rank_at_least",
    "apply_residual",
    "drop_path",
]
SERIALIZATION_PACKAGE = __name__


@register_keras_serializable(
    SERIALIZATION_PACKAGE, name="ResidualAdd"
)
class ResidualAdd(Layer, NNLearner):
    """Y = X + F(X). Assumes shapes match."""

    def call(
        self, inputs: tuple[Tensor, Tensor], training=False
    ) -> Tensor:
        x, f = inputs
        return add(x, f)


@register_keras_serializable(
    SERIALIZATION_PACKAGE, name="LayerScale"
)
class LayerScale(Layer, NNLearner):
    """
    Per-channel trainable scale vector (like ConvNeXt).

    Parameters
    ----------
    init_value : float
        Small value to start with (e.g. 1e-4).
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, init_value: float = 1e-4, **kw):
        super().__init__(**kw)
        self.init_value = init_value

    def build(self, input_shape):
        gamma_shape = input_shape[-1:]
        dtype_factory = getattr(
            float32, "as_numpy_dtype", float32
        )
        self.gamma = self.add_weight(
            name="gamma",
            shape=gamma_shape,
            initializer=lambda shape,
            dtype=None: dtype_factory(self.init_value),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: Tensor, training=False) -> Tensor:
        return x * self.gamma

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"init_value": self.init_value})
        return cfg


@register_keras_serializable(
    SERIALIZATION_PACKAGE, name="StochasticDepth"
)
class StochasticDepth(Layer, NNLearner):
    """
    Wrap a branch with DropPath (stochastic depth).

    Parameters
    ----------
    drop_prob : float
        Probability of dropping the path.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        drop_prob: float = 0.1,
        *,
        drop_rate: float | None = None,
        **kw,
    ):
        super().__init__(**kw)
        if drop_rate is not None:
            drop_prob = drop_rate
        self.drop_prob = drop_prob

    def call(self, x: Tensor, training=False) -> Tensor:
        return drop_path(x, self.drop_prob, training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"drop_prob": self.drop_prob})
        return cfg


@register_keras_serializable(
    SERIALIZATION_PACKAGE, name="SqueezeExcite1D"
)
class SqueezeExcite1D(Layer, NNLearner):
    """
    Simple SE block for (B,T,C) or (B,C).

    Parameters
    ----------
    ratio : int
        Reduction ratio, e.g. 16.
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, ratio: int = 16, **kw):
        super().__init__(**kw)
        self.ratio = ratio

    def build(self, input_shape):
        c = input_shape[-1]
        mid = max(c // self.ratio, 1)
        self.fc1 = Dense(
            mid, activation="relu", name="se_fc1"
        )
        self.fc2 = Dense(
            c, activation="sigmoid", name="se_fc2"
        )
        super().build(input_shape)

    def call(self, x: Tensor, training=False) -> Tensor:
        # Global squeeze
        if tensor_rank(x) == 3:  # (B,T,C) -> (B,C)
            z = reduce_mean(x, axis=1)
        else:  # (B,C)
            z = x

        s = self.fc2(self.fc1(z))  # (B,C)
        if tensor_rank(x) == 3:
            s = expand_dims(s, 1)  # (B,1,C)
        # Ensure s is on the same device as x (handles torch tensor and
        # numpy-array cases; numpy has no .to() so the duck-type guard fails).
        try:
            import numpy as _np
            import torch as _torch

            if isinstance(x, _torch.Tensor):
                if isinstance(s, _torch.Tensor):
                    s = s.to(x.device)
                else:
                    s = _torch.tensor(
                        _np.asarray(s, dtype=_np.float32),
                        device=x.device,
                    )
        except ImportError:
            pass
        return x * s

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"ratio": self.ratio})
        return cfg


@register_keras_serializable(
    SERIALIZATION_PACKAGE, name="Gate"
)
class Gate(Layer, NNLearner):
    """
    Generic gating layer: y = x * σ(Wx + b).

    Parameters
    ----------
    units : int
        Output units = input dim unless overridden.
    use_bias : bool
    """

    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        units: int | None = None,
        use_bias: bool = True,
        **kw,
    ):
        super().__init__(**kw)
        self.units = units
        self.use_bias = use_bias

    def build(self, input_shape):
        feat = (
            input_shape[-1]
            if self.units is None
            else self.units
        )
        self.proj = Dense(
            feat,
            activation="sigmoid",
            use_bias=self.use_bias,
            name="gate_proj",
        )
        super().build(input_shape)

    def call(self, x: Tensor, training=False) -> Tensor:
        g = self.proj(x)
        return multiply(x, g)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {"units": self.units, "use_bias": self.use_bias}
        )
        return cfg


# ------------------------- Helper functions -----------------------


def maybe_expand_time(
    x: Tensor,
    ref: Tensor | None = None,
    axis: int = 1,
) -> Tensor:
    """
    If `x` lacks a time dim but `ref` has one, expand `x` on that axis.

    Parameters
    ----------
    x   : Tensor  (B, F) or (B,T,F)
    ref : Tensor  reference for time dim (B,T,?)
    axis: int     where to insert time dim

    Returns
    -------
    Tensor
        (B,T,F) if expanded, else original `x`.
    """
    if ref is None:
        if tensor_rank(x) == 2:
            return expand_dims(x, axis=axis)
        return x

    xr = tensor_rank(x)
    rr = tensor_rank(ref)
    if rr >= 3 and xr == rr - 1:
        return expand_dims(x, axis=axis)
    return x


def broadcast_like(
    x: Tensor,
    target: Tensor,
    time_axis: int | None = None,
) -> Tensor:
    """
    Broadcast `x` so it matches `target` along leading dims.

    If last dim already matches, tile/expand others.
    Example: x:(B,1,F) target:(B,T,F) -> repeat T times.

    NOTE: Uses dynamic shapes; avoid for huge dims.
    """
    del time_axis

    def _to_python_scalar(value):
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if hasattr(value, "item"):
            try:
                return int(value.item())
            except Exception:
                pass
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        if hasattr(value, "item"):
            try:
                return int(value.item())
            except Exception:
                pass
        return int(value)

    def _shape_as_list(value):
        value_shape = getattr(value, "shape", None)
        if value_shape is not None:
            dims = []
            for dim in value_shape:
                if dim is None:
                    dims = []
                    break
                try:
                    dims.append(_to_python_scalar(dim))
                except Exception:
                    dims = []
                    break
            if dims:
                return dims

        return [
            _to_python_scalar(dim) for dim in shape(value)
        ]

    x_shape = _shape_as_list(x)
    t_shape = _shape_as_list(target)

    while len(x_shape) < len(t_shape):
        x = expand_dims(x, 1)
        x_shape = _shape_as_list(x)

    reps = tuple(
        1 if tgt == cur else tgt
        for cur, tgt in zip(x_shape, t_shape)
    )
    return tile(x, reps)


def _broadcast_like(x: Tensor, target: Tensor) -> Tensor:
    """
    Broadcast `x` so it matches `target` along leading dims.

    If last dim already matches, tile/expand others.
    Example: x:(B,1,F) target:(B,T,F) -> repeat T times.

    NOTE: Uses dynamic shapes; avoid for huge dims.

    Returns
    -------
    Tensor
    """
    x_shape = shape(x)
    t_shape = shape(target)

    # Pad ranks by inserting singleton dims until ranks match
    while tensor_rank(x) < tensor_rank(target):
        x = expand_dims(x, 1)
        x_shape = shape(x)

    # Build repetition vector for tile
    reps = []
    tgt_rank = tensor_rank(target)
    for i in range(tgt_rank):
        same = cast(t_shape[i] == x_shape[i], bool_dtype)
        # if same -> 1 else -> t_shape[i]
        reps_i = where(same, 1, t_shape[i])
        reps.append(reps_i)

    # Convert each scalar rep to a Python int so that keras.ops.tile
    # receives a plain sequence — avoids MPS tensor .numpy() failures
    # in the Torch backend (repeats.int().numpy() on mps:0 raises).
    def _scalar_to_int(v):
        if hasattr(v, "cpu"):  # torch MPS / CUDA tensor
            v = v.cpu()
        if hasattr(v, "item"):
            return int(v.item())
        if hasattr(v, "numpy"):
            return int(v.numpy())
        return int(v)

    try:
        reps = tuple(_scalar_to_int(v) for v in reps)
    except Exception:
        reps = stack(reps)

    return tile(x, reps)


def ensure_rank_at_least(
    x: Tensor,
    min_rank: int | None = None,
    axis_to_expand: int = -1,
    *,
    rank: int | None = None,
) -> Tensor:
    """
    Pad dimensions (=1) until rank >= min_rank.

    Parameters
    ----------
    x : Tensor
    min_rank : int
    axis_to_expand : int
        Where to insert singleton dims.

    Returns
    -------
    Tensor
    """
    if min_rank is None:
        min_rank = rank
    if min_rank is None:
        raise ValueError(
            "`min_rank` or `rank` must be provided."
        )

    r = tensor_rank(x)
    while r < min_rank:
        x = expand_dims(x, axis=axis_to_expand)
        r += 1
    return x


def apply_residual(x: Tensor, y: Tensor) -> Tensor:
    """
    Add & return residual. Shape check is user's job.
    """
    return add(x, y)


@do_not_convert
def drop_path(
    x: Tensor, drop_prob: float, training: bool
) -> Tensor:
    """
    Stochastic depth on residual branches. If training, randomly
    zero the entire sample (per-batch item) path with prob p.

    Parameters
    ----------
    x : Tensor (B, ... , F)
    drop_prob : float
    training : bool

    Returns
    -------
    Tensor
    """
    if (not training) or drop_prob <= 0.0:
        return x

    b = shape(x)[0]
    keep_prob = 1.0 - drop_prob
    # mask shape: (B,1,1,...)
    mask_shape = [b] + [1] * (tensor_rank(x) - 1)
    # uniform in [0,1)
    if hasattr(K, "random"):
        rnd = K.random.uniform(mask_shape)
    else:
        raise RuntimeError(
            "drop_path requires a backend random.uniform implementation."
        )

    mask = cast(rnd < keep_prob, float32)
    # Ensure mask is on the same device as x (handles both torch tensor
    # and numpy-array cases; the numpy fallback has no .to() method).
    try:
        import numpy as _np
        import torch as _torch

        if isinstance(x, _torch.Tensor):
            if isinstance(mask, _torch.Tensor):
                mask = mask.to(x.device)
            else:
                mask = _torch.tensor(
                    _np.asarray(mask, dtype=_np.float32),
                    device=x.device,
                )
    except ImportError:
        pass
    # rescale to preserve expected value
    return x * mask / keep_prob
