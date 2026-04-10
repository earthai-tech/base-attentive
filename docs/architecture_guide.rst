Architecture Guide
==================

Overview
--------

BaseAttentive is an encoder-decoder architecture designed for sequence-to-sequence forecasting with three input types:

1. **Static features** - Time-invariant properties
2. **Dynamic features** - Historical time series
3. **Future features** - Known exogenous variables

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

Core Components
---------------

Feature Extraction
~~~~~~~~~~~~~~~~~~

Processes raw inputs into embeddings:

- **Variable Selection Network (VSN)**: Learns feature importance dynamically
- **Dense Processing**: Simple linear transformation

.. code-block:: python

   # Enable VSN (default)
   model = BaseAttentive(
       use_vsn=True,  # Learns which features matter
       ...
   )

   # Or use simple dense
   model = BaseAttentive(
       use_vsn=False,  # Linear embedding
       ...
   )

Encoder Types
~~~~~~~~~~~~~

**Hybrid Mode** (Default)

Combines Multi-Scale LSTM with attention:

.. code-block:: python

   model = BaseAttentive(
       architecture_config={
           "encoder_type": "hybrid"
       },
       ...
   )

Features:
- Multi-scale LSTM captures hierarchical patterns
- Lower computational cost than pure attention
- Good for long sequences

**Transformer Mode**

Pure self-attention based:

.. code-block:: python

   model = BaseAttentive(
       architecture_config={
           "encoder_type": "transformer"
       },
       ...
   )

Features:
- Full temporal dependency modeling
- Better for shorter sequences
- Parallelizable computation

Attention Types
~~~~~~~~~~~~~~~

The decoder can stack different attention mechanisms:

| Type | Purpose | Use Case |
|------|---------|----------|
| Cross-Attention | Bridge encoder-decoder | Always included by default |
| Hierarchical | Multi-level aggregation | Seasonal patterns |
| Memory-Augmented | Historical pattern retrieval | Long-range dependencies |

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

Advanced Features
------------------

Residual Connections
~~~~~~~~~~~~~~~~~~~~

Skip connections for gradient flow:

.. code-block:: python

   model = BaseAttentive(
       use_residuals=True,  # Default
       ...
   )

Benefits:
- Easier training of deep models
- Mitigates vanishing gradient
- Faster convergence

Batch Normalization
~~~~~~~~~~~~~~~~~~~

Normalize layer activations:

.. code-block:: python

   model = BaseAttentive(
       use_batch_norm=True,  # Default: False
       ...
   )

Benefits:
- Faster training
- Better generalization
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

| Mode | Encoder | Complexity | Time/Inference |
|------|---------|-----------|-----------------|
| Hybrid | MultiScale LSTM | O(T·h²) | ~10ms (RTX 3090, batch=32) |
| Transformer | Self-Attention | O(T²·h) | ~25ms (RTX 3090, batch=32) |

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
~~~~~~~~~~~~~~~~~~~~

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
- :doc:`usage` - Advanced usage patterns
