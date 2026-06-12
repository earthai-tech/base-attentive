"""Public PADR-Net factory for flood forecasting."""

from __future__ import annotations

from typing import Any

from ...api.docs import DocstringComponents, _padrnet_params
from ...api.property import NNLearner
from ...backend import normalize_backend_name
from ...compat.sklearn import StrOptions, validate_params
from .config import PADRNetConfig

_param_docs = DocstringComponents.from_nested_components(
    padrnet=DocstringComponents(_padrnet_params),
)


@validate_params(
    {
        "config": [PADRNetConfig],
        "backend": [
            StrOptions(
                {"tensorflow", "tf", "torch", "pytorch"}
            ),
            None,
        ],
    },
    prefer_skip_nested_validation=True,
)
def create_padrnet(
    config: PADRNetConfig,
    *,
    backend: str | None = None,
    **kwargs: Any,
) -> Any:
    """Create a backend-specific PADR-Net model.

    Parameters
    ----------
    config:
        PADR-Net model configuration.
    backend:
        Backend name. Supported values are ``"tensorflow"``,
        ``"tf"``, ``"torch"``, and ``"pytorch"``. If omitted,
        TensorFlow is used.
    **kwargs:
        Extra keyword arguments passed to the backend model
        constructor.
    """

    normalized = normalize_backend_name(backend)
    if normalized == "tensorflow":
        from ...implementations.tensorflow.padrnet import (
            TensorFlowPADRNet,
        )  # noqa: PLC0415

        return TensorFlowPADRNet(config, **kwargs)
    if normalized == "torch":
        from ...implementations.torch.padrnet import (
            TorchPADRNet,
        )  # noqa: PLC0415

        return TorchPADRNet(config, **kwargs)
    raise ValueError(
        "PADR-Net supports tensorflow and torch backends. "
        f"Received: {backend!r}."
    )


class PADRNet(NNLearner):
    """Callable factory namespace for PADR-Net."""

    @validate_params(
        {
            "config": [PADRNetConfig],
            "backend": [
                StrOptions(
                    {"tensorflow", "tf", "torch", "pytorch"}
                ),
                None,
            ],
        },
        prefer_skip_nested_validation=True,
    )
    def __new__(
        cls,
        config: PADRNetConfig,
        *,
        backend: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return create_padrnet(
            config, backend=backend, **kwargs
        )


__all__ = ["PADRNet", "create_padrnet"]


_PADRNET_DOC = r"""
PADR-Net physics-aware flood forecasting model.

``PADRNet`` is the public factory for the Physics-Aware Deep
Reservoir Network used by the flood-forecasting application
module.  The factory returns a backend-specific model while
keeping one stable user-facing API.  TensorFlow backends
return a ``TensorFlowPADRNet`` model and PyTorch backends
return a ``TorchPADRNet`` module.

Architecture
~~~~~~~~~~~~
PADR-Net is an Echo State Network (ESN) with a physics-
informed readout.  At each event :math:`i`, time :math:`t`,
and grid cell :math:`g`, the local input vector
:math:`\boldsymbol{\phi}_{i,t,g}` (dynamic forcings plus
static descriptors) drives a contractive shared reservoir:

.. math::

   \mathbf{x}_{i,t,g}
   =
   \tanh\!\bigl(
     \mathbf{W}_{\mathrm{in}}\boldsymbol{\phi}_{i,t,g}
     +
     \mathbf{W}_{\mathrm{res}}\mathbf{x}_{i,t-1,g}
     +
     \mathbf{b}_{\mathrm{res}}
   \bigr),

where :math:`\mathbf{W}_{\mathrm{in}}` and
:math:`\mathbf{W}_{\mathrm{res}}` are **fixed** at
initialisation.  The contractivity condition
:math:`\|\mathbf{W}_{\mathrm{res}}\|_2 < 1` (tanh has
Lipschitz constant 1) guarantees fading memory.

Only the linear readout and the event-severity head are
optimised.  The readout produces the conservative
shallow-water state:

.. math::

   \hat{\mathbf{q}}_{i,t,g}
   =
   \mathbf{W}_{\mathrm{out}}\,\mathbf{x}_{i,t,g}
   =
   \bigl(\hat{h},\,\widehat{uh},\,\widehat{vh}\bigr)^\top.

Depth is recovered via the log-transform inverse
:math:`h = \exp(\hat{y}_h) - 1 \ge 0`, and velocities via
:math:`u = \widehat{uh}/(h + \varepsilon_h)`.

The smooth wet/dry exceedance probability is

.. math::

   \pi_t
   =
   \sigma\!\left(
     \frac{h_t - h_0}{s_h}
   \right).

Training objective
~~~~~~~~~~~~~~~~~~
The composite loss combines data channels and the SWE
residual penalty:

.. math::

   \mathcal{L}
   =
   w_{\mathrm{ext}}\mathcal{L}_{\mathrm{ext}}
   +
   w_{\mathrm{depth}}\mathcal{L}_{\mathrm{depth}}
   +
   w_{\mathrm{impact}}\mathcal{L}_{\mathrm{impact}}
   +
   \lambda_{\mathrm{phys}}\mathcal{L}_{\mathrm{phys}}
   +
   \omega\lVert\mathbf{W}_{\mathrm{out}}\rVert_F^2
   +
   \omega_s\lVert\boldsymbol{\beta}_s\rVert_2^2,

where :math:`\mathcal{L}_{\mathrm{phys}}` is the sum of
squared shallow-water residuals evaluated at collocation
points [PADR5]_.

Parameters
----------
__PADRNET_CONFIG_DOC__
__PADRNET_BACKEND_DOC__
__PADRNET_KWARGS_DOC__

Returns
-------
TensorFlowPADRNet or TorchPADRNet
    Backend-specific PADR-Net model.  Calling it produces a
    dictionary with the following keys:

    ``"depth"``
        Predicted water depth h ≥ 0; shape (B, T, 1).

    ``"momentum_x"``
        x-momentum component uh; shape (B, T, 1).

    ``"momentum_y"``
        y-momentum component vh; shape (B, T, 1).

    ``"velocity_x"``
        Depth-averaged x-velocity; shape (B, T, 1).

    ``"velocity_y"``
        Depth-averaged y-velocity; shape (B, T, 1).

    ``"exceedance_probability"``
        Smooth wet/dry probability π in [0, 1]; shape
        (B, T, 1).

    ``"reservoir_states"``
        Full reservoir state sequence; shape (B, T, N_res).
        Used to evaluate the SWE readout loss.

    ``"severity"``
        Event-level impact prediction from the severity head;
        shape (B, 1).

Notes
-----
PADR-Net is implemented as a backend factory.  This allows
the same application API to support native TensorFlow and
PyTorch implementations while preserving the parameter-
management behaviour inherited from
:class:`~base_attentive.api.property.NNLearner`.

Input tensors use shape ``(batch, time, input_dim)``.  If
static descriptors are configured, ``static_inputs`` should
have shape ``(batch, static_dim)``; they are replicated at
each time step and concatenated to the dynamic input before
the reservoir update.

The physics helpers
:func:`~base_attentive.applications.flood.physics.swe_residual`
and
:func:`~base_attentive.applications.flood.physics.recover_velocity`
operate on the readout outputs and are backend-neutral.

Examples
--------
Create a PyTorch PADR-Net model:

>>> from base_attentive import PADRNet, PADRNetConfig
>>> config = PADRNetConfig(
...     input_dim=8,
...     static_dim=3,
...     reservoir_dim=64,
...     spectral_radius=0.9,
...     flood_threshold=0.05,
... )
>>> model = PADRNet(config, backend="torch")

Run a forward pass:

>>> import torch
>>> x = torch.zeros(2, 48, 8)
>>> s = torch.zeros(2, 3)
>>> outputs = model(x, s)
>>> outputs["depth"].shape
torch.Size([2, 48, 1])
>>> outputs["reservoir_states"].shape
torch.Size([2, 48, 64])

Create the TensorFlow implementation:

>>> import tensorflow as tf
>>> model = PADRNet(config, backend="tensorflow")
>>> outputs = model(tf.zeros((2, 48, 8)), tf.zeros((2, 3)))
>>> tuple(outputs["depth"].shape)
(2, 48, 1)

Use the hydrological helper functions:

>>> from base_attentive.applications.flood import (
...     critical_success_index,
...     delta_mass,
...     recover_velocity,
...     swe_residual,
... )
>>> score = critical_success_index(
...     [0.0, 0.1, 0.2],
...     [0.0, 0.08, 0.18],
...     threshold=0.05,
... )
>>> bias = delta_mass([0.0, 0.1, 0.2], [0.0, 0.08, 0.18])

See Also
--------
PADRNetConfig
    Validated configuration object for PADR-Net.
create_padrnet
    Functional factory equivalent to ``PADRNet(...)``.
base_attentive.applications.flood.physics
    SWE residual, velocity recovery, and exceedance helpers.
base_attentive.applications.flood.metrics
    Flood metrics such as NSE, CSI, TSS, and mass bias.
BaseAttentive
    General attentive sequence model used by the main package
    API.

References
----------
.. [PADR1] Nash, J. E. and Sutcliffe, J. V. (1970). River flow
   forecasting through conceptual models part I: A discussion
   of principles. *Journal of Hydrology*, 10(3), 282--290.

.. [PADR2] Beven, K. J. (2012). *Rainfall-Runoff Modelling:
   The Primer*.  Wiley-Blackwell.

.. [PADR3] Jaeger, H. (2001). The ``echo state'' approach to
   analysing and training recurrent neural networks. *GMD
   Report 148*, German National Research Center for
   Information Technology.

.. [PADR4] Kratzert, F., Klotz, D., Brenner, C., Schulz, K.,
   and Herrnegger, M. (2018). Rainfall-runoff modelling using
   Long Short-Term Memory networks. *Hydrology and Earth
   System Sciences*, 22, 6005--6022.

.. [PADR5] Raissi, M., Perdikaris, P., and Karniadakis, G. E.
   (2019). Physics-informed neural networks: A deep learning
   framework for solving forward and inverse problems
   involving nonlinear partial differential equations.
   *Journal of Computational Physics*, 378, 686--707.
"""

PADRNet.__doc__ = (
    _PADRNET_DOC.replace(
        "__PADRNET_CONFIG_DOC__",
        _param_docs.padrnet.config,
    )
    .replace(
        "__PADRNET_BACKEND_DOC__",
        _param_docs.padrnet.backend,
    )
    .replace(
        "__PADRNET_KWARGS_DOC__",
        _param_docs.padrnet.kwargs,
    )
)
