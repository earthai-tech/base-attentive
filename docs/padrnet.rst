PADR-Net Flood Forecasting
==========================

``PADR-Net`` is the flood-forecasting application module in
BaseAttentive.  It implements a **Physics-Aware Deep Reservoir Network**
whose hydrodynamic predictions are constrained toward the admissible
shallow-water solution manifold, while an event-level severity head
provides impact ranking.  The model is grounded in the paper
*Physics-Informed Reservoir Learning for Shallow-Water Flood Modelling*
(Mathematical Geosciences, 2026) [PADRG6]_.

The backbone is a **fixed Echo State Network (ESN)** — a contractive
reservoir whose weights are never updated by gradient descent.  Only
two linear readouts (hydrodynamic head and severity head) are
optimised, via Ridge regression.  This keeps training fast (≈ 3 s for
127 events on a single CPU core), supports closed-form hyperparameter
selection, and gives a clear fading-memory guarantee.

.. contents:: On this page
   :local:
   :depth: 2

Why PADR-Net?
-------------

BaseAttentive is a general sequence model.  PADR-Net is more specific:
it is designed around flood depth, flood-threshold exceedance, and
hydrological consistency.  The model adds three things that a bare
sequence model does not provide.

* **Physics constraint** — a shallow-water residual penalty pushes the
  depth trajectory toward a water-balance-consistent state, bounded by
  :math:`O(\lambda_{\mathrm{phys}}^{-1/2})` under local residual
  stability.
* **Fading memory** — the ESN contractivity condition guarantees that
  the influence of arbitrary initial reservoir states decays
  geometrically, so the prediction depends on forcing history rather
  than initialization.
* **Head decoupling** — because the reservoir is fixed, tuning the
  physics weight changes depth skill and SWE-residual norm while
  leaving Spearman rank correlation, PR-AUC, and all other impact
  metrics exactly unchanged.

In practice this means you can:

* predict flood-depth trajectories from forcing histories;
* obtain event-level severity scores and high-impact PR-AUC without
  retraining for every physics-weight choice;
* evaluate source-term sensitivity (friction, infiltration, lateral
  inflow) without touching the readout;
* compare reliably against random-weight LSTM and GRU baselines,
  because all three architectures share the same readout paradigm.

Architecture
------------

The reservoir update at event :math:`i`, time :math:`t`, and grid cell
:math:`g` is

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
:math:`\mathbf{W}_{\mathrm{res}}` are **fixed** at initialisation and
never trained.  The spectral radius satisfies
:math:`\rho(\mathbf{W}_{\mathrm{res}}) < 1`, so the contractivity
condition :math:`L_\sigma\,\rho < 1` (with :math:`L_\sigma = 1` for
tanh) holds.

The hydrodynamic readout produces the conservative shallow-water state:

.. math::

   \hat{\mathbf{q}}
   =
   \mathbf{W}_{\mathrm{out}}\,\mathbf{x}
   =
   \bigl(\hat{h},\;\widehat{uh},\;\widehat{vh}\bigr)^\top.

Physical water depth is recovered as
:math:`h = \operatorname{softplus}(\hat{y}_h) \ge 0`, and
depth-averaged velocities as
:math:`u = \widehat{uh}/(h + \varepsilon_h)`.

The smooth wet/dry exceedance probability is

.. math::

   \pi_t
   =
   \sigma\!\left(
     \frac{h_t - h_0}{s_h}
   \right).

An event-level severity head :math:`\hat{Y}^A_i
= \boldsymbol{\beta}_s^\top\mathbf{z}_i` is fitted independently of
the physics weight sweep, so impact metrics are invariant to
:math:`\lambda_{\mathrm{phys}}`.

Physics-Aware Objective
-----------------------

Training minimizes a composite loss over data channels and the
shallow-water residual:

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
   \omega_s\lVert\boldsymbol{\beta}_s\rVert_2^2 .

:math:`\mathcal{L}_{\mathrm{phys}}` is the sum of squared
shallow-water residuals.  The physics weight is selected from
validation depth diagnostics (NSE, mass error) without touching the
test set.  For a linearised 1-D storage balance,

.. math::

   r_t
   =
   \frac{d h_t}{d t}
   -
   \bigl(p_s P_t - c_f h_t\bigr),

with rainfall-scaling :math:`p_s = 10^{-3}` and default friction
coefficient :math:`c_f = 0.05\,\mathrm{h}^{-1}`.

Input and Output Contract
-------------------------

``PADRNet`` accepts a ``PADRNetConfig`` and returns a concrete backend
model.

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Object
     - Shape
     - Meaning
   * - ``dynamic_inputs``
     - ``(batch, time, input_dim)``
     - Time-varying forcing: precipitation, antecedent wetness,
       river-state indicators, terrain, exposure.
   * - ``static_inputs``
     - ``(batch, static_dim)``
     - Optional time-invariant descriptors (basin, region, grid cell).
   * - ``outputs["depth"]``
     - ``(batch, time, 1)``
     - Predicted water depth :math:`h \ge 0`.
   * - ``outputs["momentum_x"]``
     - ``(batch, time, 1)``
     - x-momentum component :math:`\widehat{uh}`.
   * - ``outputs["momentum_y"]``
     - ``(batch, time, 1)``
     - y-momentum component :math:`\widehat{vh}`.
   * - ``outputs["velocity_x"]``
     - ``(batch, time, 1)``
     - Depth-averaged x-velocity :math:`u`.
   * - ``outputs["velocity_y"]``
     - ``(batch, time, 1)``
     - Depth-averaged y-velocity :math:`v`.
   * - ``outputs["exceedance_probability"]``
     - ``(batch, time, 1)``
     - Smooth wet/dry probability :math:`\pi \in [0,1]`.
   * - ``outputs["reservoir_states"]``
     - ``(batch, time, N_\mathrm{res})``
     - Full reservoir state sequence; used for SWE readout loss.
   * - ``outputs["severity"]``
     - ``(batch, 1)``
     - Event-level impact score from the severity head.

Configuration
-------------

``PADRNetConfig`` is a validated frozen dataclass.  Invalid spectral
radii (outside :math:`(0,1)`), non-positive dimensions, and
out-of-range probabilities are caught at construction time.

.. code-block:: python

   from base_attentive import PADRNetConfig

   config = PADRNetConfig(
       input_dim=8,
       static_dim=3,
       reservoir_dim=200,
       spectral_radius=0.90,
       input_scaling=0.1,
       flood_threshold=0.10,
       slope_h=0.05,
       lambda_physics=0.10,
       lambda_readout=1e-4,
       lambda_severity=1e-4,
       w_ext=1.0,
       w_depth=1.0,
       w_impact=1.0,
   )

The most important parameters are:

``input_dim``
   Number of dynamic forcing features per time step.

``static_dim``
   Number of time-invariant descriptors.  Set to ``0`` if unused.

``reservoir_dim``
   Echo-state reservoir size :math:`N_\mathrm{res}` (default 500 in
   the library; 200 in the Mathematical Geosciences experiments).

``spectral_radius``
   Spectral radius of :math:`\mathbf{W}_{\mathrm{res}}`; must lie
   strictly in :math:`(0, 1)`.  The recommended value is 0.9.

``input_scaling``
   Scale factor for random initialisation of
   :math:`\mathbf{W}_{\mathrm{in}}`.

``flood_threshold``
   Wet-cell depth threshold :math:`h_0` (metres) for the smooth
   wet/dry indicator.

``lambda_physics``
   Weight on the SWE residual loss.  The reference value
   :math:`\lambda^* = 0.10` was selected by validation depth NSE in
   the African archive experiments.

``lambda_readout``, ``lambda_severity``
   Ridge penalties on the hydrodynamic readout
   :math:`\mathbf{W}_{\mathrm{out}}` and the severity head
   :math:`\boldsymbol{\beta}_s`, respectively.

``w_ext``, ``w_depth``, ``w_impact``
   Per-channel loss weights for satellite extent, continuous depth,
   and event-level impact.

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
       reservoir_dim=200,
       spectral_radius=0.90,
       flood_threshold=0.10,
       lambda_physics=0.10,
   )

   model = PADRNet(config, backend="torch")

   x = torch.zeros(2, 168, 8)   # 168-hour event window
   s = torch.zeros(2, 3)
   outputs = model(x, s)

   depth       = outputs["depth"]             # (2, 168, 1)
   exceedance  = outputs["exceedance_probability"]
   reservoir   = outputs["reservoir_states"]  # (2, 168, 200)
   severity    = outputs["severity"]          # (2, 1)

TensorFlow
~~~~~~~~~~

.. code-block:: python

   import tensorflow as tf
   from base_attentive import PADRNet, PADRNetConfig

   config = PADRNetConfig(
       input_dim=8,
       static_dim=3,
       reservoir_dim=200,
       spectral_radius=0.90,
       flood_threshold=0.10,
       lambda_physics=0.10,
   )

   model = PADRNet(config, backend="tensorflow")

   x = tf.zeros((2, 168, 8))
   s = tf.zeros((2, 3))
   outputs = model(x, s)

   depth    = outputs["depth"]               # (2, 168, 1)
   severity = outputs["severity"]            # (2, 1)

Metrics and Diagnostics
-----------------------

PADR-Net results should be reported across three skill categories.

Hydrodynamic reconstruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from base_attentive.applications.flood import (
       nash_sutcliffe_efficiency,
       delta_mass,
   )

   nse      = nash_sutcliffe_efficiency(h_ref, h_hat)
   mass_err = delta_mass(h_ref, h_hat)          # % bias

``NSE_depth`` is the primary depth metric (NSE on per-event maximum
depth).  A Δ\ *M* < 5 % indicates good water-balance consistency.

Threshold-event skill
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from base_attentive.applications.flood import (
       critical_success_index,
       true_skill_statistic,
   )

   csi = critical_success_index(h_ref, h_hat, threshold=0.10)
   tss = true_skill_statistic(h_ref, h_hat, threshold=0.10)

Impact ranking and classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from scipy.stats import spearmanr
   from sklearn.metrics import average_precision_score

   rho_s  = spearmanr(y_true, y_score).statistic    # Spearman ρ_s
   pr_auc = average_precision_score(y_bin, y_score)  # PR-AUC

On the 243-event African archive, the full PADR-Net (M6) achieves:

.. list-table::
   :header-rows: 1
   :widths: 36 20 20 20

   * - Model
     - NSE\ :sub:`depth`
     - ρ\ :sub:`s`
     - PR-AUC
   * - Rainfall only (M0)
     - 0.380
     - 0.194
     - 0.403
   * - + Memory (M2)
     - 0.380
     - 0.614
     - 0.620
   * - + Exposure (M4)
     - 0.380
     - 0.699
     - 0.675
   * - Full PADR-Net M6 (λ\* = 0.10)
     - 0.643
     - 0.671
     - 0.703

Source-Closure Sensitivity
--------------------------

Reviewer-requested experiments perturb the uncertain SWE source terms
(friction coefficient :math:`c_f`, precipitation scaling
:math:`\alpha_p`, and lateral inflow :math:`Q_\mathrm{lat}`) to bound
residual sensitivity without refitting the readout.  Table results from
``07_source_closure_sensitivity.py``:

.. list-table::
   :header-rows: 1
   :widths: 30 32 16 12 14

   * - Scenario
     - Perturbation
     - NSE\ :sub:`depth`
     - CSI
     - Δ\ *M* (%)
   * - S0 Baseline
     - :math:`c_f=0.05,\;\alpha_p=1.0`
     - 0.925
     - 0.760
     - 0.2
   * - S1 Low infiltration
     - :math:`\alpha_p \times 0.5`
     - −8.2
     - 0.798
     - 23.0
   * - S2 High infiltration
     - :math:`\alpha_p \times 1.5`
     - 0.521
     - 0.292
     - 22.7
   * - S3 Low friction
     - :math:`c_f \times 0.8`
     - 0.871
     - 0.625
     - 6.6
   * - S4 High friction
     - :math:`c_f \times 1.2`
     - 0.884
     - 0.774
     - 5.2
   * - S5 Lateral inflow
     - :math:`Q_\mathrm{lat}=0.10\,\bar{P}` (omitted baseline)
     - −11.0
     - 0.000
     - 93.0

Friction perturbations of ±20 % change NSE\ :sub:`depth` by at most
±0.054 (S3–S4).  Large mass errors occur only when an unmodelled
source term (lateral inflow, S5) is substantial — the readout detects
source-closure inconsistency through elevated mass error before it
degrades depth NSE.

Architectural Baseline Comparison
----------------------------------

To isolate the contribution of the reservoir architecture from the SWE
penalty, PADR-Net is compared against random-weight LSTM and GRU
baselines that share the same readout paradigm: fixed random recurrent
weights, event-level feature summary, and Ridge regression readout
(``08_lstm_gru_baseline.py``).  This comparison differs from
backpropagation-trained LSTM because it holds the training protocol
constant and varies only the memory architecture.

.. list-table::
   :header-rows: 1
   :widths: 26 20 16 16 16

   * - Architecture
     - NSE\ :sub:`depth`
     - ρ\ :sub:`s`
     - PR-AUC
     - MAE
   * - RAND-LSTM
     - 0.947
     - 0.426
     - 0.617
     - 0.315
   * - RAND-GRU
     - 0.933
     - 0.529
     - 0.836
     - 0.286
   * - PADR-Net (M6)
     - 0.925
     - 0.487
     - 0.736
     - 0.300

PADR-Net is competitive in depth reconstruction and balanced across
all four metrics.  The random-weight LSTM has higher depth skill but
lower rank correlation and PR-AUC.  Crucially, none of these
architectures includes the SWE residual penalty; the ablation between
M7 (λ = 0) and M6 (λ\ * = 0.10) remains the primary test of the
physics constraint.

Reliability and Calibration
----------------------------

The reliability (calibration) curve evaluates whether the predicted
high-impact probability matches observed event rates.
``09_reliability_curve.py`` fits isotonic regression on the M6
severity scores, bins the test set into ten equal-frequency groups, and
reports the Brier score:

.. math::

   \mathrm{BS}
   = \frac{1}{N}\sum_{i=1}^{N}\!\bigl(\hat{p}_i - y_i\bigr)^2,
   \qquad
   \mathrm{BSS}
   = 1 - \frac{\mathrm{BS}_{\mathrm{model}}}
              {\mathrm{BS}_{\mathrm{climatology}}}.

On the 243-event African archive:

* Brier score = **0.086** (lower is better)
* Brier Skill Score = **0.550** (positive = better than climatology)

.. code-block:: python

   import numpy as np
   from sklearn.isotonic import IsotonicRegression

   # convert severity scores to calibrated probabilities
   p_raw = (score - score.min()) / (score.max() - score.min() + 1e-12)
   iso   = IsotonicRegression(out_of_bounds="clip")
   iso.fit(p_raw_train, y_bin_train)
   p_hat = iso.predict(p_raw_test)

   bs  = np.mean((p_hat - y_bin) ** 2)
   bss = 1.0 - bs / np.mean((y_bin - y_bin.mean()) ** 2)

Physics Helpers
---------------

The physics helpers are separate from the model class so training loops
can combine them freely:

.. code-block:: python

   from base_attentive.applications.flood import (
       exceedance_probability,
       linear_reservoir_response,
       mass_balance_residual,
       recover_velocity,
       swe_residual,
   )

   prob = exceedance_probability(depth=[0.0, 0.04, 0.08], threshold=0.05)

   response = linear_reservoir_response(precipitation=[1.0, 3.0, 2.0], tau=24.0)

   residual = mass_balance_residual(
       precipitation=[1.0, 3.0, 2.0],
       depth=[0.0, 0.04, 0.08],
       tau=24.0,
   )

   u, v = recover_velocity(uh=outputs["momentum_x"], vh=outputs["momentum_y"],
                           h=outputs["depth"])

   f_h, f_x, f_y = swe_residual(h=outputs["depth"], uh=outputs["momentum_x"],
                                 vh=outputs["momentum_y"], P=precip, dt=1.0)

Interpretation Workflow
-----------------------

A complete PADR-Net evaluation reports four views:

1. **Hydrodynamic agreement** — compare peak depth against a reference
   SWE response using NSE, CSI, and mass-balance error.  Report the
   physics-weight sensitivity curve to show the depth–constraint
   trade-off.

2. **Threshold-event skill** — report CSI and TSS with the same
   :math:`h_0` used during training.  Sensitivity runs at
   :math:`h_0 \in \{0.05, 0.10, 0.20\}` m should show modest changes
   in CSI (±0.03) with stable NSE.

3. **Impact ranking and classification** — report Spearman ρ\ :sub:`s`
   and PR-AUC for the event-level severity head.  Because the severity
   head is invariant to :math:`\lambda_{\mathrm{phys}}`, these metrics
   should be flat across the physics-weight grid.

4. **Calibration** — compute the Brier score and reliability diagram
   after isotonic calibration.  A positive Brier Skill Score confirms
   that the model is more informative than climatology for high-impact
   event detection.

For regional transfer, report leave-one-region-out and
leave-one-year-out scores separately.  Depth response typically
transfers more reliably than impact ranking, because depth is
constrained by the physics residual while ranking also depends on
regional exposure structure and inventory completeness.

Reproducibility Archive
-----------------------

The Mathematical Geosciences paper experiments are fully reproducible
via the scripts in
`earthai-tech/padrnet <https://github.com/earthai-tech/padrnet>`_
(v1.1.0-matg-rebuild, archived at
`Zenodo DOI 10.5281/zenodo.20651388 <https://doi.org/10.5281/zenodo.20651388>`_).

The pipeline is:

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Script
     - Purpose
   * - ``00``
     - Generate synthetic event table (offline / reproducibility mode)
   * - ``04``
     - PADR-Net training, ablation M0–M8, lambda sensitivity
   * - ``07``
     - Source-closure sensitivity (friction, infiltration, lateral inflow)
   * - ``08``
     - RAND-LSTM / RAND-GRU architectural baseline comparison
   * - ``09``
     - Reliability / calibration curve and Brier score

Run the full pipeline from the ``padrnet/`` root:

.. code-block:: bash

   python scripts/run_all.py              # all scripts in order
   python scripts/run_all.py --from 07   # Reviewer 3 experiments only
   python scripts/run_all.py --only 09   # reliability curve only

API Reference Links
-------------------

The generated API reference for the flood application is in
:doc:`api_reference`.  The key public objects are
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

.. [PADRG2] Beven, K. J. (2012). *Rainfall-Runoff Modelling: The
   Primer*. Wiley-Blackwell.

.. [PADRG3] Jaeger, H. (2001). The "echo state" approach to analysing
   and training recurrent neural networks. *GMD Report 148*, German
   National Research Center for Information Technology.

.. [PADRG4] Lukoševičius, M. and Jaeger, H. (2009). Reservoir
   computing approaches to recurrent neural network training.
   *Computer Science Review*, 3(3), 127--149.
   doi:10.1016/j.cosrev.2009.03.005

.. [PADRG5] Gallicchio, C. and Micheli, A. (2017). Echo state property
   of deep reservoir computing networks. *Cognitive Computation*,
   9(3), 337--350.  doi:10.1007/s12559-017-9461-9

.. [PADRG6] Kouadio, K. L. (2026). Physics-Informed Reservoir Learning
   for Shallow-Water Flood Modelling. *Mathematical Geosciences*.
   Reproducibility archive: Zenodo
   `10.5281/zenodo.20651388 <https://doi.org/10.5281/zenodo.20651388>`_.

.. [PADRG7] Raissi, M., Perdikaris, P., and Karniadakis, G. E. (2019).
   Physics-informed neural networks. *Journal of Computational
   Physics*, 378, 686--707.

.. [PADRG8] Kratzert, F., Klotz, D., Brenner, C., Schulz, K., and
   Herrnegger, M. (2018). Rainfall-runoff modelling using Long
   Short-Term Memory networks. *Hydrology and Earth System Sciences*,
   22, 6005--6022.
