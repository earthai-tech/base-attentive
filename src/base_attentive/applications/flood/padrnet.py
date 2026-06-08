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

``PADRNet`` is the public factory for the Physics-Aware
Depth-Response Network used by the flood-forecasting
application module. The factory returns a backend-specific
model while keeping one stable user-facing API. TensorFlow
backends return a ``TensorFlowPADRNet`` model and PyTorch
backends return a ``TorchPADRNet`` module.

PADR-Net maps a dynamic forcing sequence
:math:`\mathbf{X}_{1:T}` and optional static descriptors
:math:`\mathbf{s}` to a multi-step water-depth forecast
:math:`\hat{\mathbf{h}}_{1:H}`. In compact form,

.. math::

  \mathbf{z}_{1:T}
  = f_{\theta}(\mathbf{X}_{1:T}, \mathbf{s}),
  \qquad
  \hat{\mathbf{h}}_{1:H}
   = \operatorname{softplus}
     (g_{\theta}(\mathbf{z}_{1:T})).

The flood-exceedance head converts depth to a smooth threshold
probability,

.. math::

   p_t
   =
   \sigma\left(
   \frac{\hat{h}_t - h_{\mathrm{crit}}}{\alpha}
   \right),

where :math:`h_{\mathrm{crit}}` is the configured flood
threshold and :math:`\alpha` controls the transition
sharpness.

The physics-aware training objective is intended to combine a
forecasting loss with hydrological consistency terms. The
diagnostics follow common hydrological skill measures [PADR1]_
and rainfall-runoff response concepts [PADR2]_, while the
temporal encoder follows the attention formulation [PADR3]_.

.. math::

   \mathcal{L}
   =
   \mathcal{L}_{\mathrm{pred}}
   + \lambda_{\mathrm{phys}}
     \lVert r_{\mathrm{phys}} \rVert_2^2
   + \lambda_{\mathrm{mass}}
     \lvert \Delta M \rvert
   + \lambda_{\mathrm{smooth}}
     \lVert \nabla_t \hat{\mathbf{h}} \rVert_2^2 .

For a simple rainfall-storage diagnostic, the residual can be
written as

.. math::

   r_t
   =
   \frac{d h_t}{d t}
   -
   \left(
   \gamma P_t - \frac{h_t}{\tau}
   \right),

with rainfall forcing :math:`P_t`, gain :math:`\gamma`, and
response time scale :math:`\tau`. This style of regularization
is closely related to regional rainfall-runoff learning
[PADR4]_ and physics-informed learning [PADR5]_.

Parameters
----------
__PADRNET_CONFIG_DOC__
__PADRNET_BACKEND_DOC__
__PADRNET_KWARGS_DOC__

Returns
-------
TensorFlowPADRNet or TorchPADRNet
    Backend-specific PADR-Net model. Calling it produces a
    dictionary with the following keys:

    ``"depth"``
        Forecast water depth with shape
        ``(batch, forecast_horizon, 1)``.

    ``"exceedance_probability"``
        Smooth probability of exceeding the configured flood
        threshold, with the same shape as ``"depth"``.

    ``"features"``
        Latent event representation produced by the temporal
        encoder.

Notes
-----
PADR-Net is implemented as a backend factory rather than a
single framework class. This allows the same application API
to support native TensorFlow and PyTorch implementations while
preserving the parameter-management behaviour inherited from
:class:`~base_attentive.api.property.NNLearner`.

The current module provides the model architecture and
reusable physics/metric helpers. Full training loops may
choose how to combine the prediction loss and physics
penalties depending on available observations, flood
thresholds, and hydrodynamic constraints.

Input tensors are expected to use shape
``(batch, time, input_dim)``. If static descriptors are
configured, ``static_inputs`` should have shape
``(batch, static_dim)``.

Examples
--------
Create a PyTorch PADR-Net model:

>>> from base_attentive import PADRNet, PADRNetConfig
>>> config = PADRNetConfig(
...     input_dim=8,
...     static_dim=3,
...     hidden_dim=64,
...     num_heads=4,
...     forecast_horizon=24,
...     flood_threshold=0.05,
... )
>>> model = PADRNet(config, backend="torch")

Run a forward pass with PyTorch tensors:

>>> import torch
>>> x = torch.zeros(2, 48, 8)
>>> s = torch.zeros(2, 3)
>>> outputs = model(x, s)
>>> outputs["depth"].shape
torch.Size([2, 24, 1])

Create the TensorFlow implementation:

>>> import tensorflow as tf
>>> model = PADRNet(config, backend="tensorflow")
>>> outputs = model(tf.zeros((2, 48, 8)), tf.zeros((2, 3)))
>>> tuple(outputs["depth"].shape)
(2, 24, 1)

Use the hydrological helper functions:

>>> from base_attentive.applications.flood import (
...     critical_success_index,
...     delta_mass,
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
    Hydrological residual and exceedance-probability helpers.
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
   The Primer*.
   Wiley-Blackwell.

.. [PADR3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J.,
   Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I.
   (2017).
   Attention is all you need. *Advances in Neural Information
   Processing Systems*, 30.

.. [PADR4] Kratzert, F., Klotz, D., Brenner, C., Schulz, K.,
   and
   Herrnegger, M. (2018). Rainfall-runoff modelling using
   Long Short-Term Memory networks. *Hydrology and Earth
   System Sciences*, 22, 6005--6022.

.. [PADR5] Karniadakis, G. E., Kevrekidis, I. G., Lu, L.,
   Perdikaris, P., Wang, S., and Yang, L. (2021).
   Physics-informed machine learning. *Nature Reviews
   Physics*, 3, 422--440.
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
