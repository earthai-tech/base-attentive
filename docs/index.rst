BaseAttentive Documentation
===========================

.. image:: https://img.shields.io/badge/python-3.10%2B-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/license-Apache%202.0-green.svg
   :target: https://github.com/earthai-tech/base-attentive/blob/main/LICENSE

A modular encoder-decoder architecture for sequence-to-sequence time series forecasting with layered attention mechanisms.

**BaseAttentive** is a modular encoder-decoder architecture designed to process three distinct types of inputs:

- **Static features** — constant across time (e.g., geographical coordinates, site properties)
- **Dynamic past features** — historical time series (e.g., sensor readings, observations)
- **Known future features** — forecast-period exogenous variables (e.g., weather forecasts)

It combines these inputs through a configurable attention stack for forecasting experiments and applied workflows.

Main Elements
=============

**Architecture options**
   - Hybrid mode: Multi-scale LSTM with attention
   - Transformer mode: self-attention encoder
   - Configurable decoder attention stack

**Core components**
   - Variable selection networks for feature weighting
   - Multi-scale LSTM for temporal aggregation
   - Cross-attention for encoder-decoder interaction
   - Memory-augmented attention for long-range context
   - Quantile modeling for uncertainty-aware outputs

**Runtime support**
   - Keras 3 based implementation
   - Model serialization and configuration export
   - Input validation utilities
   - Logging and debugging hooks

Quick Example
=============

.. code-block:: python

   import tensorflow as tf
   from base_attentive import BaseAttentive

   # Create a model
   model = BaseAttentive(
       static_input_dim=4,        # Site properties
       dynamic_input_dim=8,       # Historical observations
       future_input_dim=6,        # Known future features
       output_dim=2,              # Forecast targets
       forecast_horizon=24,       # 24-hour forecast
       quantiles=[0.1, 0.5, 0.9], # Uncertainty quantiles
       embed_dim=32,
       num_heads=8,
       dropout_rate=0.15,
   )

   # Prepare inputs
   batch_size = 32
   x_static = tf.random.normal([batch_size, 4])
   x_dynamic = tf.random.normal([batch_size, 100, 8])
   x_future = tf.random.normal([batch_size, 24, 6])

   # Get predictions
   predictions = model([x_static, x_dynamic, x_future])
   # shape: (32, 24, 3, 2) — batch, horizon, quantiles, output

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quick_start

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   backends
   torch_backend_guide
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

   release_notes

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
