.. _examples-notebooks:

Example Notebooks
=================

The notebooks below are rendered directly from the
`examples/ <https://github.com/earthai-tech/base-attentive/tree/master/examples>`_
folder in the repository.  Each one can be run interactively on Binder
(no local installation required) or downloaded and run locally.

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
