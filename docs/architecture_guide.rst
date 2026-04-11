Architecture Guide
==================

Overview
--------

BaseAttentive is an encoder-decoder neural network architecture designed for 
sequence-to-sequence time series forecasting. It combines three distinct feature streams 
into a unified forecasting kernel:

1. **Static features** - Time-invariant properties (shape: ``batch_size, static_dim``)
   
   Examples: geographical coordinates, site elevation, installation date
   
2. **Dynamic features** - Historical time series data (shape: ``batch_size, history_steps, dynamic_dim``)
   
   Examples: sensor readings, temperature, humidity, historical observations
   
3. **Future features** - Known exogenous variables for forecast period (shape: ``batch_size, forecast_horizon, future_dim``)
   
   Examples: weather forecasts, calendar features, scheduled maintenance

The architecture processes these three input streams through a configurable encoder-decoder 
pipeline that supports multiple architectural choices, attention mechanisms, and output modes.

Conceptual Flow
---------------

At a high level, the model follows this processing path:

1. **Ingest** - Accept static, historical, and future feature tensors
2. **Project** - Transform features into a shared embedding space
3. **Encode** - Process temporal context with hybrid or transformer-style encoder
4. **Attend** - Apply configured decoder attention stack (cross, hierarchical, memory)
5. **Forecast** - Generate point or probabilistic forecasts over horizon

This flow keeps the three feature types available throughout the model while
allowing different temporal encoding and fusion choices.

Input/Output Contract
---------------------

.. code-block:: text

   ┌─────────────────────────────────────┐
   │           Inputs (3 types)          │
   ├─────────────────────────────────────┤
   │ static:   (batch, static_dim)       │
   │ dynamic:  (batch, T, dynamic_dim)   │
   │ future:   (batch, H, future_dim)    │
   └────────────────┬────────────────────┘
                    │
                    ▼
           ┌─────────────────────┐
           │   Encoder-Decoder   │
           │   Architecture      │
           └────────────┬────────┘
                        │
                ┌───────┴────────┐
                │                │
                ▼                ▼
           Point Forecast      With Quantiles
         (B, H, output_dim)  (B, H, Q, output_dim)

Where:
- B = batch size
- T = time steps (history)
- H = forecast horizon
- Q = quantiles

Core Configuration Parameters
------------------------------

BaseAttentive exposes most architectural choices directly on the model constructor.

**Required Parameters:**

- ``static_input_dim`` - Number of static features
- ``dynamic_input_dim`` - Number of dynamic features
- ``future_input_dim`` - Number of future features
- ``output_dim`` - Number of output variables
- ``forecast_horizon`` - Forecast length in time steps

**Important Hyperparameters:**

- ``embed_dim`` - Embedding dimension (default: 32)
- ``attention_units`` - Attention mechanism dimension (default: 64)
- ``num_heads`` - Multi-head attention heads (default: 4)
- ``dropout_rate`` - Dropout probability (default: 0.2)
- ``quantiles`` - Quantiles for probabilistic output (default: None)

**Architecture-level choices** live in the ``architecture_config`` dictionary:

.. code-block:: python

   architecture_config = {
       "encoder_type": "hybrid",  # or "transformer"
       "decoder_attention_stack": ["cross", "hierarchical", "memory"],
       "feature_processing": "vsn",  # or "dense"
   }

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       architecture_config=architecture_config,
       embed_dim=32,
       attention_units=64,
       num_heads=8,
       dropout_rate=0.2,
   )

Core Components
---------------

Feature Extraction & Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Raw input features are first projected into a shared embedding space using one of two methods:

**Variable Selection Network (VSN)** - Default method

Dynamically learns feature importance:

.. code-block:: python

   model = BaseAttentive(
       use_vsn=True,  # Learns which features matter
       ...
   )

- Uses gating mechanisms to weight features
- Often useful for high-dimensional inputs
- Slightly more parameters and computation
- Can make feature weighting easier to inspect

**Dense Processing** - Alternative method

Simple linear transformation:

.. code-block:: python

   model = BaseAttentive(
       use_vsn=False,  # Simple linear embedding
       ...
   )

- Standard dense layer projection
- Fewer parameters
- Lower computational cost
- Simpler structure with fewer moving parts

Encoder Architecture
~~~~~~~~~~~~~~~~~~~~

The encoder processes temporal context with one of two architectural approaches:

**Hybrid Mode** (Default)

Combines Multi-Scale LSTM with attention:

.. code-block:: python

   model = BaseAttentive(
       architecture_config={
           "encoder_type": "hybrid"
       },
       ...
   )

Typical characteristics:
- Multi-scale LSTM captures hierarchical patterns
- Lower computational cost than pure attention
- Often used for longer sequences

**Transformer Mode**

Pure self-attention based:

.. code-block:: python

   model = BaseAttentive(
       architecture_config={
           "encoder_type": "transformer"
       },
       ...
   )

Typical characteristics:
- Full temporal dependency modeling
- Often used for shorter sequences
- Parallelizable computation

Attention Types
~~~~~~~~~~~~~~~

The decoder can stack different attention mechanisms:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Type
     - Purpose
     - Use Case
   * - Cross-Attention
     - Bridge encoder-decoder
     - Always included by default
   * - Hierarchical
     - Multi-level aggregation
     - Seasonal patterns
   * - Memory-Augmented
     - Historical pattern retrieval
     - Long-range dependencies


.. code-block:: python

   # Use all attention types
   model = BaseAttentive(
       architecture_config={
           "decoder_attention_stack": ["cross", "hierarchical", "memory"]
       },
       ...
   )

Data Flow
---------

Complete processing pipeline:

.. code-block:: text

   Input tensors
        │
        ├─► Static (B, S_dim) ──┐
        ├─► Dynamic (B, T, D_dim) ──┬─► Feature Extraction (VSN or Dense)
        └─► Future (B, H, F_dim) ──┘
                                │
                        ┌───────▼────────┐
                        │ Embedded Input │
                        └───────┬────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼──────┐  ┌─────▼──────┐  ┌───▼──────────┐
        │Static Encoder│ │Dyn Encoder│ │Future Context│
        │(Dense)       │ │(LSTM/Attn)│ │(Dense)       │
        └───────┬──────┘  └─────┬──────┘  └───┬──────────┘
                │               │            │
                └───────────────┼────────────┘
                                │
                ┌───────────────▼────────────┐
                │  Attention Stack          │
                │ • Cross-attention         │
                │ • Hierarchical-attention  │
                │ • Memory-attention        │
                └───────────────┬────────────┘
                                │
                        ┌───────▼──────┐
                        │ Time Agg.    │
                        │ Last/Avg     │
                        └───────┬──────┘
                                │
                        ┌───────▼──────┐
                        │ Multi-Decoder│
                        │ Per-horizon  │
                        │ heads        │
                        └───────┬──────┘
                                │
                    ┌───────────┴──────────┐
                    │                      │
         Point Forecast          Quantile Modeling
            (B,H,D)                (B,H,Q,D)

Configuration Hierarchy
-----------------------

Parameters are applied in order of precedence:

1. **architecture_config** dict (highest)
   
   .. code-block:: python
   
      architecture_config = {
          "encoder_type": "transformer",
          "decoder_attention_stack": ["cross"]
      }

2. **Explicit keyword arguments**
   
   .. code-block:: python
   
      model = BaseAttentive(
          embed_dim=64,  # Overrides default
          ...
      )

3. **Default values** (lowest)
   
   .. code-block:: python
   
      DEFAULT_ARCHITECTURE = {
          "encoder_type": "hybrid",
          "decoder_attention_stack": ["cross", "hierarchical", "memory"],
          "feature_processing": "vsn"
      }

Example Configuration Merging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # User configuration
   config = {
       'static_input_dim': 4,
       'dynamic_input_dim': 8,
       'future_input_dim': 6,
       'output_dim': 2,
       'forecast_horizon': 24,
       'embed_dim': 64,  # Overrides default (32)
       'architecture_config': {
           'encoder_type': 'transformer'  # Overrides default (hybrid)
       }
   }

   model = BaseAttentive(**config)

   # Result:
   # - embed_dim: 64 (from keyword argument)
   # - encoder_type: 'transformer' (from architecture_config)
   # - decoder_attention_stack: ["cross", "hierarchical", "memory"] (from default)

Output Modes
~~~~~~~~~~~~

The model supports two output modes depending on the ``quantiles`` parameter:

**Point Forecast** (Default)

When ``quantiles`` is not set, returns single point predictions:

.. code-block:: python

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       # quantiles not specified
   )

   predictions = model([static, dynamic, future])
   # Shape: (batch_size, forecast_horizon, output_dim)
   # Example: (32, 24, 2)

Use when you need a single deterministic forecast.

**Probabilistic Forecast** (Quantile-based)

When ``quantiles`` are specified, returns uncertainty estimates:

.. code-block:: python

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       quantiles=[0.1, 0.5, 0.9],  # Lower, median, upper
   )

   predictions = model([static, dynamic, future])
   # Shape: (batch_size, forecast_horizon, num_quantiles, output_dim)
   # Example: (32, 24, 3, 2)
   # predictions[:, :, 0, :] = 10th percentile (lower bound)
   # predictions[:, :, 1, :] = 50th percentile (median)
   # predictions[:, :, 2, :] = 90th percentile (upper bound)

Use when you need quantile ranges and uncertainty estimates.

Optional Features
-----------------

Residual Connections
~~~~~~~~~~~~~~~~~~~~

Skip connections for gradient flow:

.. code-block:: python

   model = BaseAttentive(
       use_residuals=True,  # Default
       ...
   )

Typical effects:
- Easier training of deep models
- Mitigates vanishing gradient
- Can improve optimization stability

Batch Normalization
~~~~~~~~~~~~~~~~~~~

Normalize layer activations:

.. code-block:: python

   model = BaseAttentive(
       use_batch_norm=True,  # Default: False
       ...
   )

Typical effects:
- Faster training
- May improve optimization stability
- Reduced internal covariate shift

Quantile Modeling
~~~~~~~~~~~~~~~~~

Probabilistic forecasts with uncertainty:

.. code-block:: python

   model = BaseAttentive(
       quantiles=[0.1, 0.5, 0.9],  # Default: None (point forecast)
       ...
   )

Output:
- Shape: (batch, horizon, num_quantiles, output_dim)
- Allows confidence intervals
- Useful for risk-aware applications

Performance Characteristics
---------------------------

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Performance Characteristics by Architecture
   :header-rows: 1
   :widths: 15 20 20 22

   * - Mode
     - Encoder
     - Complexity
     - Time/Inference
   * - Hybrid
     - MultiScale LSTM
     - O(T·h²)
     - ~10ms (RTX 3090, batch=32)
   * - Transformer
     - Self-Attention
     - O(T²·h)
     - ~25ms (RTX 3090, batch=32)


Memory Usage
~~~~~~~~~~~~

Per batch (batch_size=32, embed_dim=64):

- Model weights: ~2-5 MB
- Activations: ~10-20 MB
- Attention matrices: ~1-10 MB

Total: ~15-35 MB per batch

Extensibility
--------------

Custom Encoders
~~~~~~~~~~~~~~~

Implement custom encoder logic:

.. code-block:: python

   from keras import layers

   class CustomEncoder(layers.Layer):
       def call(self, inputs):
           # Your encoding logic
           return encoded


Custom Loss Functions
~~~~~~~~~~~~~~~~~~~~~

Plug in custom loss:

.. code-block:: python

   from keras import losses

   class QuantileLoss(losses.Loss):
       def call(self, y_true, y_pred):
           # Quantile-specific loss
           return loss

   model.compile(loss=QuantileLoss())

See Also
--------

- :doc:`configuration_guide` - Detailed parameter reference
- :doc:`api_reference` - Complete API documentation
- :doc:`usage` - Extended usage patterns
