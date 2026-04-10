BaseAttentive Documentation
===========================

.. image:: https://img.shields.io/badge/python-3.9%2B-blue.svg
   :target: https://www.python.org/downloads/

.. image:: https://img.shields.io/badge/license-Apache%202.0-green.svg
   :target: https://github.com/earthai-tech/base-attentive/blob/main/LICENSE

A foundational blueprint for building powerful, data-driven, sequence-to-sequence time series forecasting models with advanced attention mechanisms.

**BaseAttentive** is a sophisticated encoder-decoder architecture designed to process three distinct types of inputs:

- **Static features** — constant across time (e.g., geographical coordinates, site properties)
- **Dynamic past features** — historical time series (e.g., sensor readings, observations)
- **Known future features** — forecast-period exogenous variables (e.g., weather forecasts)

It fuses these inputs using a modular stack of attention mechanisms and serves as the core engine for advanced forecasting models.

Key Features
============

✨ **Flexible Architecture**
   - Hybrid mode: Multi-scale LSTM + Attention
   - Transformer mode: Pure self-attention
   - Configurable attention stack (cross, hierarchical, memory-augmented)

📊 **Advanced Components**
   - Variable Selection Networks for learnable feature selection
   - Multi-scale LSTM for hierarchical temporal patterns
   - Cross-attention for encoder-decoder interaction
   - Memory-augmented attention for long-term dependencies
   - Dynamic time warping for time-series alignment
   - Quantile distribution modeling for uncertainty quantification

🔧 **Production-Ready**
   - Keras 3 backed with configurable runtimes
   - Serializable (save/load models)
   - Input validation and parameter checking
   - Comprehensive logging and debugging support

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

   usage
   architecture_guide
   configuration_guide

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference
   components_reference

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
