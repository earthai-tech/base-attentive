PADR-Net Flood Forecasting
==========================

``PADR-Net`` is the flood-forecasting application module in
BaseAttentive. It provides a small, validated public API for building a
Physics-Aware Depth-Response Network with either TensorFlow or PyTorch
while keeping the same configuration object, output contract, and
hydrological helper functions.

The model is intended for event-scale flood forecasting where dynamic
forcing histories, optional static basin descriptors, and physics-aware
diagnostics must be handled together. Typical inputs include rainfall,
antecedent precipitation, soil-moisture proxies, runoff or river-stage
history, topographic descriptors, land-cover summaries, and basin area
or slope metadata.

.. contents:: On this page
   :local:
   :depth: 2

Why PADR-Net?
-------------

BaseAttentive is a general sequence model. PADR-Net is more specific:
it is designed around flood depth, flood-threshold exceedance, and
hydrological consistency. This makes it useful when the result should
be interpreted as a water-depth forecast rather than an arbitrary
sequence prediction.

PADR-Net is useful when you need to combine attention-based sequence
modelling [PADRG3]_ with hydrological response diagnostics
[PADRG1]_, [PADRG2]_. In practice, this means you can:

* predict multi-step flood depth from forcing histories;
* keep a stable public API across TensorFlow and PyTorch;
* expose exceedance probabilities from a configured flood threshold;
* combine statistical accuracy with physics-aware penalties;
* report hydrological metrics such as NSE, CSI, TSS, and mass bias.

Conceptual Model
----------------

Let :math:`\mathbf{X}_{1:T}` denote a dynamic sequence of forcing and
state variables, and let :math:`\mathbf{s}` denote optional static
descriptors for the basin, region, or grid cell. PADR-Net learns a
latent temporal representation

.. math::

   \mathbf{z}_{1:T}
   =
   f_{\theta}(\mathbf{X}_{1:T}, \mathbf{s}),

then decodes it into a non-negative multi-horizon depth forecast:

.. math::

   \hat{\mathbf{h}}_{1:H}
   =
   \operatorname{softplus}
   \left(g_{\theta}(\mathbf{z}_{1:T})\right).

The model also computes a smooth flood-threshold exceedance
probability,

.. math::

   p_t
   =
   \sigma
   \left(
   \frac{\hat{h}_t - h_{\mathrm{crit}}}{\alpha}
   \right),

where :math:`h_{\mathrm{crit}}` is the configured flood threshold and
:math:`\alpha` controls how sharp the transition is around the
threshold.

Physics-Aware Objective
-----------------------

PADR-Net itself returns model predictions. Training code can combine
the prediction loss with physics-aware regularization terms:

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

For a simple rainfall-storage response, a residual can be written as

.. math::

   r_t
   =
   \frac{d h_t}{d t}
   -
   \left(
   \gamma P_t - \frac{h_t}{\tau}
   \right),

with rainfall forcing :math:`P_t`, rainfall-depth gain
:math:`\gamma`, and reservoir response time :math:`\tau`. The helper
functions in ``base_attentive.applications.flood.physics`` provide
reusable forms of these diagnostics.

Input and Output Contract
-------------------------

The public model factory is ``PADRNet``. It accepts a
``PADRNetConfig`` and returns a concrete backend model.

.. list-table::
   :header-rows: 1
   :widths: 24 32 44

   * - Object
     - Shape
     - Meaning
   * - ``dynamic_inputs``
     - ``(batch, time, input_dim)``
     - Time-varying forcing and state features.
   * - ``static_inputs``
     - ``(batch, static_dim)``
     - Optional basin, region, or grid-cell descriptors.
   * - ``outputs["depth"]``
     - ``(batch, forecast_horizon, 1)``
     - Forecast flood depth.
   * - ``outputs["exceedance_probability"]``
     - ``(batch, forecast_horizon, 1)``
     - Smooth probability of exceeding the flood threshold.
   * - ``outputs["features"]``
     - Backend-dependent latent shape
     - Learned temporal event representation.

Configuration
-------------

``PADRNetConfig`` is a validated dataclass. It follows the same
``validate_params`` pattern used by the main BaseAttentive API, so
invalid dimensionality, negative horizons, incompatible attention
heads, and invalid probability-like weights are caught early.

.. code-block:: python

   from base_attentive import PADRNetConfig

   config = PADRNetConfig(
       input_dim=8,
       static_dim=3,
       hidden_dim=64,
       num_heads=4,
       num_layers=2,
       forecast_horizon=24,
       dropout=0.1,
       lambda_physics=0.1,
       lambda_mass=0.05,
       lambda_smooth=0.01,
       flood_threshold=0.05,
       reservoir_tau=24.0,
   )

The most important parameters are:

``input_dim``
   Number of dynamic forcing and state variables at each historical
   time step.

``static_dim``
   Number of optional time-invariant descriptors. Set this to ``0``
   when no static stream is used.

``hidden_dim``
   Latent channel width used by the temporal encoder and decoder.

``num_heads``
   Number of attention heads. ``hidden_dim`` must be divisible by
   ``num_heads``.

``forecast_horizon``
   Number of future steps predicted by the depth head.

``lambda_physics``, ``lambda_mass``, ``lambda_smooth``
   Weights intended for training-time physics, mass-balance, and
   temporal smoothness penalties.

``flood_threshold``
   Water-depth threshold used to convert depth into exceedance
   probability.

``reservoir_tau``
   Response time scale for simple storage-style hydrological
   diagnostics.

Backend Examples
----------------

PyTorch
~~~~~~~

.. code-block:: python

   import torch
   from base_attentive import PADRNet, PADRNetConfig

   config = PADRNetConfig(
       input_dim=8,
       static_dim=3,
       hidden_dim=64,
       num_heads=4,
       forecast_horizon=24,
   )

   model = PADRNet(config, backend="torch")

   x = torch.zeros(2, 48, 8)
   s = torch.zeros(2, 3)
   outputs = model(x, s)

   depth = outputs["depth"]
   exceedance = outputs["exceedance_probability"]

TensorFlow
~~~~~~~~~~

.. code-block:: python

   import tensorflow as tf
   from base_attentive import PADRNet, PADRNetConfig

   config = PADRNetConfig(
       input_dim=8,
       static_dim=3,
       hidden_dim=64,
       num_heads=4,
       forecast_horizon=24,
   )

   model = PADRNet(config, backend="tensorflow")

   x = tf.zeros((2, 48, 8))
   s = tf.zeros((2, 3))
   outputs = model(x, s)

   depth = outputs["depth"]
   exceedance = outputs["exceedance_probability"]

Metrics and Diagnostics
-----------------------

PADR-Net is usually evaluated with both continuous depth skill and
threshold-event skill. The flood application module exposes helpers for
common diagnostics:

.. code-block:: python

   from base_attentive.applications.flood import (
       critical_success_index,
       delta_mass,
       nash_sutcliffe_efficiency,
       true_skill_statistic,
   )

   observed = [0.0, 0.1, 0.2, 0.15]
   predicted = [0.0, 0.08, 0.18, 0.12]

   nse = nash_sutcliffe_efficiency(observed, predicted)
   csi = critical_success_index(
       observed,
       predicted,
       threshold=0.05,
   )
   tss = true_skill_statistic(
       observed,
       predicted,
       threshold=0.05,
   )
   mass_bias = delta_mass(observed, predicted)

``NSE`` summarizes continuous depth agreement. ``CSI`` and ``TSS``
evaluate flooded-threshold detection. ``delta_mass`` summarizes the
signed difference between predicted and reference water volume or
depth accumulation.

Physics Helpers
---------------

The physics helpers are intentionally separate from the model class so
that training loops can combine them in different ways:

.. code-block:: python

   from base_attentive.applications.flood import (
       exceedance_probability,
       linear_reservoir_response,
       mass_balance_residual,
   )

   prob = exceedance_probability(
       depth=[0.0, 0.04, 0.08],
       threshold=0.05,
   )

   response = linear_reservoir_response(
       precipitation=[1.0, 3.0, 2.0],
       tau=24.0,
   )

   residual = mass_balance_residual(
       precipitation=[1.0, 3.0, 2.0],
       depth=[0.0, 0.04, 0.08],
       tau=24.0,
   )

Interpretation Workflow
-----------------------

A strong flood-forecasting result should usually report three views:

1. **Hydrodynamic agreement**: compare PADR-Net peak depth against a
   reference hydrodynamic or hydrological-model response using a shared
   water-depth color scale.
2. **Threshold-event skill**: report flooded / non-flooded detection
   with CSI, TSS, hit rate, false-alarm rate, and threshold boundary
   overlays.
3. **Physical consistency**: show whether the prediction respects
   plausible temporal response, mass-balance tendency, and smoothness
   constraints. This follows the broader idea that neural flood
   models should be evaluated not only by point accuracy, but also by
   whether the learned response remains hydrologically plausible
   [PADRG4]_, [PADRG5]_.

For regional transfer experiments, use the same configuration and
evaluation protocol across regions, then report source-region,
target-region, and leave-one-region-out skill. This is often more
convincing than a single in-region test because it tests whether the
learned response transfers beyond the calibration domain.

API Reference Links
-------------------

The generated API reference for the flood application is available in
:doc:`api_reference`. The most important public objects are
:class:`base_attentive.applications.flood.PADRNet`,
:class:`base_attentive.applications.flood.PADRNetConfig`,
:func:`base_attentive.applications.flood.create_padrnet`,
``base_attentive.applications.flood.metrics``, and
``base_attentive.applications.flood.physics``.

See Also
--------

* :doc:`applications` for broader domain application patterns.
* :doc:`api_reference` for package-wide API documentation.
* :doc:`backends/index` for TensorFlow, Torch, and JAX backend notes.
* :doc:`notebooks/index` for executable examples.

References
----------

.. [PADRG1] Nash, J. E. and Sutcliffe, J. V. (1970). River flow
   forecasting through conceptual models part I: A discussion of
   principles. *Journal of Hydrology*, 10(3), 282--290.

.. [PADRG2] Beven, K. J. (2012). *Rainfall-Runoff Modelling: The Primer*.
   Wiley-Blackwell.

.. [PADRG3] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J.,
   Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017).
   Attention is all you need. *Advances in Neural Information
   Processing Systems*, 30.

.. [PADRG4] Kratzert, F., Klotz, D., Brenner, C., Schulz, K., and
   Herrnegger, M. (2018). Rainfall-runoff modelling using Long
   Short-Term Memory networks. *Hydrology and Earth System Sciences*,
   22, 6005--6022.

.. [PADRG5] Karniadakis, G. E., Kevrekidis, I. G., Lu, L., Perdikaris, P.,
   Wang, S., and Yang, L. (2021). Physics-informed machine learning.
   *Nature Reviews Physics*, 3, 422--440.
