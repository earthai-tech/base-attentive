=======================
Configuration Guide
=======================

Parameter Reference
====================

Required Parameters
-------------------

These must be specified when creating a BaseAttentive model:

.. code-block:: python

   model = BaseAttentive(
       static_input_dim=4,      # Number of static features
       dynamic_input_dim=8,     # Number of dynamic features
       future_input_dim=6,      # Number of future features
       output_dim=2,            # Number of output variables
       forecast_horizon=24,     # Forecast length
   )

.. list-table:: Required Parameters
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Constraints
     - Description
   * - ``static_input_dim``
     - int
     - ≥ 0
     - Static feature dimension
   * - ``dynamic_input_dim``
     - int
     - ≥ 0
     - Dynamic feature dimension
   * - ``future_input_dim``
     - int
     - ≥ 0
     - Future feature dimension
   * - ``output_dim``
     - int
     - ≥ 1
     - Number of output variables
   * - ``forecast_horizon``
     - int
     - ≥ 1
     - Forecast horizon

Architectural Parameters
------------------------

.. code-block:: python

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       # Architectural params:
       embed_dim=32,           # Default embedding dimension
       hidden_units=64,        # Dense/LSTM hidden units
       lstm_units=64,          # LSTM-specific units
       attention_units=64,     # Attention mechanism units
       num_heads=4,            # Multi-head attention heads
   )

.. list-table:: Architectural Parameters
   :header-rows: 1
   :widths: 20 12 12 15 41

   * - Parameter
     - Type
     - Default
     - Range
     - Description
   * - ``embed_dim``
     - int
     - 32
     - [8, 512]
     - Embedding dimension
   * - ``hidden_units``
     - int
     - 64
     - [16, 1024]
     - Dense hidden units
   * - ``lstm_units``
     - int
     - 64
     - [16, 1024]
     - LSTM units
   * - ``attention_units``
     - int
     - 64
     - [16, 1024]
     - Attention dimension
   * - ``num_heads``
     - int
     - 4
     - [1, 16]
     - Multi-head attention heads

**Guidance:**

- Increase for complex patterns: 64 → 128 → 256
- Decrease for faster inference: 32 → 16 → 8
- Keep embed_dim ≥ num_heads (for attention)

Regularization Parameters
--------------------------

.. code-block:: python

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       # Regularization:
       dropout_rate=0.2,       # Dropout probability
       activation='relu',      # Activation function
       use_batch_norm=False,   # Batch normalization
       use_residuals=True,     # Residual connections
   )

.. list-table:: Regularization Parameters
   :header-rows: 1
   :widths: 20 12 12 56

   * - Parameter
     - Type
     - Default
     - Description
   * - ``dropout_rate``
     - float
     - 0.2
     - Dropout probability [0-1]
   * - ``activation``
     - str
     - 'relu'
     - Activation function
   * - ``use_batch_norm``
     - bool
     - False
     - Apply batch normalization
   * - ``use_residuals``
     - bool
     - True
     - Use residual connections

**Supported Activations:**

- ``'relu'`` - ReLU (recommended)
- ``'elu'`` - ELU
- ``'selu'`` - SELU
- ``'sigmoid'`` - Sigmoid
- ``'tanh'`` - Hyperbolic tangent
- ``'linear'`` - Linear (no activation)

Feature Processing Parameters
------------------------------

.. code-block:: python

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       # Feature processing:
       use_vsn=True,           # Variable Selection Network
       feature_processing='vsn',  # Method: 'vsn' or 'dense'
   )

.. list-table:: Feature Processing Parameters
   :header-rows: 1
   :widths: 20 13 12 15 40

   * - Parameter
     - Type
     - Default
     - Options
     - Description
   * - ``use_vsn``
     - bool
     - True
     - —
     - Enable Variable Selection Network
   * - ``feature_processing``
     - str
     - 'vsn'
     - 'vsn', 'dense'
     - Feature processing method

**Variable Selection Network (VSN):**

- Learns feature importance dynamically
- Good for high-dimensional data
- Slightly more parameters

**Dense Processing:**

- Simple linear transformation
- Fewer parameters
- Faster training

Optional Output Parameters
---------------------------

.. code-block:: python

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       # Optional outputs:
       quantiles=[0.1, 0.5, 0.9],  # Probabilistic forecasts
   )

.. list-table:: Optional Output Parameters
   :header-rows: 1
   :widths: 20 12 15 53

   * - Parameter
     - Type
     - Default
     - Description
   * - ``quantiles``
     - list[float]
     - None
     - Quantiles for uncertainty

**Common Quantile Sets:**

- ``None`` - Point forecast only
- ``[0.5]`` - Median only
- ``[0.1, 0.5, 0.9]`` - 80% confidence interval
- ``[0.025, 0.5, 0.975]`` - 95% confidence interval

Architecture Configuration
==========================

Use the ``architecture_config`` dictionary for structural choices:

.. code-block:: python

   config = {
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
       architecture_config=config,
   )

Encoder Type
------------

**Hybrid (Default)**

.. code-block:: python

   architecture_config = {
       "encoder_type": "hybrid"
   }

- Multi-scale LSTM + Attention
- Lower computational cost
- Good for long sequences (T > 500)
- Recommended for real-time inference

**Transformer**

.. code-block:: python

   architecture_config = {
       "encoder_type": "transformer"
   }

- Pure self-attention
- Full temporal dependency modeling
- Good for short-medium sequences (T < 500)
- Better expressiveness

Decoder Attention Stack
-----------------------

.. code-block:: python

   # Default stack
   architecture_config = {
       "decoder_attention_stack": ["cross", "hierarchical", "memory"]
   }

   # Custom combinations
   architecture_config = {
       "decoder_attention_stack": ["cross"]  # Just cross-attention
   }

   architecture_config = {
       "decoder_attention_stack": ["cross", "attention"]  # For transformer
   }

**Available Attention Types:**

.. list-table:: 
   :header-rows: 1
   :widths: 15 25 60

   * - Type
     - Purpose
     - Use Case
   * - ``cross``
     - Encoder-decoder interaction
     - Always recommended
   * - ``hierarchical``
     - Multi-level temporal patterns
     - Seasonal data
   * - ``memory``
     - Historical pattern retrieval
     - Long-range dependencies
   * - ``attention``
     - Generic self-attention
     - Transformer mode

Feature Processing
------------------

.. code-block:: python

   architecture_config = {
       "feature_processing": "vsn"  # or "dense"
   }

- ``"vsn"`` - Variable Selection Network (learns importance)
- ``"dense"`` - Simple linear embedding

Configuration Presets
=====================

Save and reuse common configurations:

Minimal Configuration
---------------------

Smallest model for testing:

.. code-block:: python

   MINIMAL = {
       "embed_dim": 8,
       "hidden_units": 16,
       "lstm_units": 16,
       "attention_units": 16,
       "num_heads": 1,
       "dropout_rate": 0.1,
   }

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       **MINIMAL
   )

Standard Configuration
----------------------

Balanced for accuracy and speed:

.. code-block:: python

   STANDARD = {
       "embed_dim": 32,
       "hidden_units": 64,
       "lstm_units": 64,
       "attention_units": 64,
       "num_heads": 4,
       "dropout_rate": 0.2,
       "use_residuals": True,
   }

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       **STANDARD
   )

Large Configuration
-------------------
High-capacity model for complex problems:

.. code-block:: python

   LARGE = {
       "embed_dim": 128,
       "hidden_units": 256,
       "lstm_units": 256,
       "attention_units": 128,
       "num_heads": 8,
       "dropout_rate": 0.3,
       "use_batch_norm": True,
       "use_residuals": True,
   }

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       **LARGE
   )

Hybrid vs Transformer Preset
-----------------------------

.. code-block:: python

   # Hybrid mode (recommended)
   HYBRID = {
       "architecture_config": {
           "encoder_type": "hybrid",
           "decoder_attention_stack": ["cross", "hierarchical"],
           "feature_processing": "vsn"
       },
       "embed_dim": 32,
       "num_heads": 4,
       "dropout_rate": 0.2,
   }

   # Transformer mode
   TRANSFORMER = {
       "architecture_config": {
           "encoder_type": "transformer",
           "decoder_attention_stack": ["cross", "attention"],
           "feature_processing": "vsn"
       },
       "embed_dim": 64,
       "num_heads": 8,
       "dropout_rate": 0.15,
   }

Tuning Guidelines
=================

For Longer Sequences (T > 500)
------------------------------

.. code-block:: python

   model = BaseAttentive(
       ...,
       architecture_config={"encoder_type": "hybrid"},
       embed_dim=32,  # Lower to save memory
       dropout_rate=0.15,
   )

For Complex Patterns
---------------------

.. code-block:: python

   model = BaseAttentive(
       ...,
       embed_dim=64,
       num_heads=8,
       use_batch_norm=True,
       use_residuals=True,
       dropout_rate=0.2,
   )

For Fast Inference
-------------------

.. code-block:: python

   model = BaseAttentive(
       ...,
       architecture_config={"encoder_type": "hybrid"},
       embed_dim=16,
       hidden_units=32,
       dropout_rate=0.1,
   )

For Probabilistic Forecasts
----------------------------

.. code-block:: python

   model = BaseAttentive(
       ...,
       quantiles=[0.1, 0.5, 0.9],
       dropout_rate=0.2,  # Higher for uncertainty
       use_residuals=True,
   )

Configuration Management
=========================

Get Configuration
------------------

.. code-block:: python

   config = model.get_config()
   print(config)
   # {'static_input_dim': 4, 'dynamic_input_dim': 8, ...}

Create from Configuration
--------------------------

.. code-block:: python

   # Save
   config = model.get_config()

   # Later: recreate model
   new_model = BaseAttentive.from_config(config)

Reconfigure Model
-----------------

Create variant with updated parameters:

.. code-block:: python

   # Original
   model1 = BaseAttentive(..., embed_dim=32)

   # Variant with larger embedding
   model2 = model1.reconfigure(embed_dim=64)

   # Original unchanged
   print(model1.config['embed_dim'])  # 32
   print(model2.config['embed_dim'])  # 64

Common Mistakes
===============

❌ **Mismatched input dimensions**

.. code-block:: python

   # Wrong - features don't match
   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       ...
   )
   static = np.random.randn(32, 5)  # 5 features, but model expects 4

✅ **Correct**

.. code-block:: python

   static = np.random.randn(32, 4)  # Matches static_input_dim=4

❌ **Too many model parameters**

.. code-block:: python

   # Wrong - model too large
   model = BaseAttentive(
       ...,
       embed_dim=512,
       hidden_units=1024,
       lstm_units=1024,
       num_heads=16,
   )

✅ **Correct - scale appropriately**

.. code-block:: python

   model = BaseAttentive(
       ...,
       embed_dim=64,
       hidden_units=128,
       lstm_units=128,
       num_heads=4,
   )

See Also
========

- :doc:`quick_start` - Quick start guide
- :doc:`architecture_guide` - Architecture details
- :doc:`api_reference` - Full API reference
