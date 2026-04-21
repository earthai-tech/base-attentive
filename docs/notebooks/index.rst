.. _examples-notebooks:

Example Notebooks
=================

The notebooks below are rendered directly from the
`examples/ <https://github.com/earthai-tech/base-attentive/tree/master/examples>`_
folder in the repository.  Each one can be run interactively on Binder
(no local installation required) or downloaded and run locally.

.. admonition:: Before you run — Backend & Import Order
   :class: important

   Every notebook (and any script using ``BaseAttentive``) requires the backend to be
   set **before** Keras or BaseAttentive are imported.  Follow this order every time:

   .. code-block:: python

      # 1. Set the backend environment variables FIRST
      import os
      os.environ["BASE_ATTENTIVE_BACKEND"] = "tensorflow"   # or "torch" / "jax"
      os.environ["KERAS_BACKEND"]          = "tensorflow"   # must match above

      # 2. Import Keras SECOND
      import keras

      # 3. Import BaseAttentive THIRD
      import base_attentive
      from base_attentive import BaseAttentive

   If you skip step 1 or import in a different order, BaseAttentive will raise a
   ``BackendConfigurationError`` because it cannot detect the backend after Keras
   has already initialised.

   **Binder users**: open a notebook, execute the very first cell (it already
   contains the environment setup), and then run the remaining cells in order.
   Do not skip the first cell.

   **Local users**: the ``os.environ`` calls work only when they run *before*
   any Keras import in the same Python process.  If you have already imported
   Keras in a prior cell or session, restart the kernel and re-run from the top.

.. list-table::
   :header-rows: 1
   :widths: 5 45 50

   * - #
     - Notebook
     - Topics covered
   * - 01
     - :doc:`01_quickstart`
     - Model creation, configuration inspection, save/load
   * - 02
     - :doc:`02_hybrid_vs_transformer`
     - Hybrid vs. Transformer objective comparison
   * - 03
     - :doc:`03_attention_stack_configuration`
     - Attention levels, cross / hierarchical / memory-augmented
   * - 04
     - :doc:`04_standalone_applications`
     - Domain application patterns, multi-output forecasting
   * - 05
     - :doc:`05_kernel_robust_networks`
     - Kernel-robust training, DTW alignment, regularisation
   * - 06
     - :doc:`06_crps_probabilistic_forecasting`
     - **CRPSLoss** — quantile / gaussian / mixture modes, MC sampling
   * - 07
     - :doc:`07_v2_spec_registry`
     - **V2 Spec & Registry** — ``BaseAttentiveSpec``, ``ComponentRegistry``, custom encoders
   * - 08
     - :doc:`08_financial_forecasting`
     - **Financial ML** — walk-forward validation, IC/ICIR/Sharpe/drawdown, regime analysis, gradient saliency
   * - 09
     - :doc:`09_attention_interpretability`
     - **Interpretability** — VSN weights, cross/hierarchical attention heatmaps, integrated gradients, multi-head diversity
   * - 10
     - :doc:`10_benchmarking`
     - **Benchmarking** — 7 architecture variants vs baselines, efficiency frontier, hyperparameter sensitivity, noise robustness, statistical significance
   * - 11
     - :doc:`11_landslide_susceptibility`
     - **Landslide Susceptibility** — physics-informed FS regularisation, depth-profile attention, ensemble uncertainty, scenario-conditioned hazard curves, method comparison (LR/RF/BA)

.. admonition:: Run on Binder

   Launch any notebook interactively (no local install needed):

   .. image:: https://mybinder.org/badge_logo.svg
      :target: https://mybinder.org/v2/gh/earthai-tech/base-attentive/master?filepath=examples
      :alt: Launch on Binder

.. toctree::
   :hidden:
   :maxdepth: 1

   01_quickstart
   02_hybrid_vs_transformer
   03_attention_stack_configuration
   04_standalone_applications
   05_kernel_robust_networks
   06_crps_probabilistic_forecasting
   07_v2_spec_registry
   08_financial_forecasting
   09_attention_interpretability
   10_benchmarking
   11_landslide_susceptibility
