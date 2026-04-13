BaseAttentive Documentation
============================

.. image:: https://img.shields.io/badge/python-3.10%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.10+

.. image:: https://img.shields.io/pypi/v/base-attentive.svg
   :target: https://pypi.org/project/base-attentive/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/version-2.0.0-brightgreen.svg
   :target: release_notes/index.html
   :alt: Version 2.0.0

.. image:: https://img.shields.io/badge/semver-2.0.0-blueviolet.svg
   :target: https://semver.org/
   :alt: Semantic Versioning

.. image:: https://img.shields.io/badge/keras-%E2%89%A53.0-FF6F00.svg
   :target: https://keras.io/
   :alt: Keras ≥ 3.0

.. image:: https://img.shields.io/badge/backends-TF%20%7C%20JAX%20%7C%20Torch-7B2D8B.svg
   :target: backends.html
   :alt: Backends: TensorFlow, JAX, PyTorch

.. image:: https://img.shields.io/badge/code%20style-ruff-46a758.svg
   :target: https://docs.astral.sh/ruff/
   :alt: Code Style: ruff

.. image:: https://github.com/earthai-tech/base-attentive/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/earthai-tech/base-attentive/actions/workflows/tests.yml
   :alt: Tests

.. image:: https://codecov.io/gh/earthai-tech/base-attentive/graph/badge.svg?token=YEJY7PEREM
   :target: https://codecov.io/gh/earthai-tech/base-attentive
   :alt: Coverage

.. image:: https://img.shields.io/github/last-commit/earthai-tech/base-attentive.svg
   :target: https://github.com/earthai-tech/base-attentive/commits/master
   :alt: Last Commit

.. image:: https://img.shields.io/github/issues/earthai-tech/base-attentive.svg
   :target: https://github.com/earthai-tech/base-attentive/issues
   :alt: Open Issues

.. image:: https://img.shields.io/badge/PRs-welcome-brightgreen.svg
   :target: https://github.com/earthai-tech/base-attentive/pulls
   :alt: PRs Welcome

.. image:: https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-4a5568.svg
   :alt: Platform: Linux, macOS, Windows

A modular encoder-decoder architecture for sequence-to-sequence time series
forecasting with layered attention mechanisms — **version 2.0.0**.

**BaseAttentive** is a modular encoder-decoder architecture designed to process
three distinct types of inputs:

- **Static features** — constant across time (e.g., geographical coordinates, site properties)
- **Dynamic past features** — historical time series (e.g., sensor readings, observations)
- **Known future features** — forecast-period exogenous variables (e.g., weather forecasts)

It combines these inputs through a configurable attention stack for forecasting
experiments and applied workflows. The v2 architecture introduces a registry /
resolver / assembly system making every component pluggable and backend-neutral.

Main Elements
=============

**Architecture options**
   - Hybrid mode: Multi-scale LSTM with attention (``objective="hybrid"``)
   - Transformer mode: self-attention encoder (``objective="transformer"``)
   - Operational mode shortcuts: TFT-like (``mode="tft"``), PIHALNet-like (``mode="pihal"``)
   - Declarative attention stack via ``attention_levels``

**Core components**
   - Variable selection networks for feature weighting
   - Multi-scale LSTM for temporal aggregation (``scales``, ``multi_scale_agg``)
   - Cross, hierarchical, and memory-augmented attention
   - Transformer encoder/decoder blocks
   - Quantile and probabilistic forecast heads

**V2 system**
   - ``BaseAttentiveSpec`` / ``BaseAttentiveComponentSpec`` for backend-neutral config
   - ``ComponentRegistry`` and ``ModelRegistry`` for pluggable components
   - ``BaseAttentiveV2Assembly`` resolver/assembler pattern
   - Multi-backend: TensorFlow, Torch, and JAX (all supported in v2)

**Runtime support**
   - Keras 3 based implementation
   - ``make_fast_predict_fn`` for traced TF inference
   - Input validation utilities
   - ``TorchDeviceManager`` for CUDA/MPS device management

Quick Example
=============

.. code-block:: python

   import numpy as np
   from base_attentive import BaseAttentive

   # Create a model
   model = BaseAttentive(
       static_input_dim=4,        # Site properties
       dynamic_input_dim=8,       # Historical observations
       future_input_dim=6,        # Known future features
       output_dim=2,              # Forecast targets
       forecast_horizon=24,       # 24-step forecast
       quantiles=[0.1, 0.5, 0.9], # Uncertainty quantiles
       embed_dim=32,
       num_heads=8,
       dropout_rate=0.15,
   )

   # Prepare inputs
   batch_size = 32
   x_static  = np.random.randn(batch_size, 4).astype('float32')
   x_dynamic = np.random.randn(batch_size, 100, 8).astype('float32')
   x_future  = np.random.randn(batch_size, 24, 6).astype('float32')

   # Get predictions
   predictions = model([x_static, x_dynamic, x_future])
   # shape: (32, 24, 3, 2) — batch, horizon, quantiles, output_dim

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quick_start

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   backends/index
   usage
   architecture_guide
   configuration_guide
   applications

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference
   components_reference

.. toctree::
   :maxdepth: 2
   :caption: Resources

   release_notes/index

.. toctree::
   :maxdepth: 2
   :caption: Development

   development
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
