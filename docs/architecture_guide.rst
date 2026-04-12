Architecture Guide
==================

Overview
--------

BaseAttentive is an encoder-decoder neural network for sequence-to-sequence
time series forecasting. It combines three distinct feature streams:

1. **Static features** — time-invariant properties ``(batch, static_dim)``
2. **Dynamic features** — historical time series ``(batch, T, dynamic_dim)``
3. **Future features** — known future exogenous variables ``(batch, H, future_dim)``

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
           └────────────┬────────┘
                        │
                ┌───────┴────────┐
                │                │
                ▼                ▼
           Point Forecast      With Quantiles
         (B, H, output_dim)  (B, H, Q, output_dim)

Conceptual Flow
---------------

1. **Select** — Variable Selection Networks weight each feature
2. **Project** — Transform features into shared embedding space
3. **Encode** — Process temporal context (hybrid LSTM or transformer)
4. **Attend** — Apply decoder attention stack (cross / hierarchical / memory)
5. **Aggregate** — Pool sequence representation
6. **Forecast** — Generate point or probabilistic outputs

Encoder Architecture
--------------------

Hybrid Mode (Default)
~~~~~~~~~~~~~~~~~~~~~

Multi-scale LSTM with attention — best for long sequences (T > 500):

.. code-block:: python

   model = BaseAttentive(
       ...,
       objective="hybrid",
       scales=[1, 2, 4],
       multi_scale_agg="last",
   )

Transformer Mode
~~~~~~~~~~~~~~~~

Pure self-attention — better parallelism on shorter sequences:

.. code-block:: python

   model = BaseAttentive(
       ...,
       objective="transformer",
       num_encoder_layers=4,
   )

Attention Types
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Type
     - Purpose
     - Use Case
   * - ``cross``
     - Bridge encoder-decoder
     - Default; always useful
   * - ``hierarchical``
     - Multi-level temporal patterns
     - Seasonal / structured data
   * - ``memory``
     - Historical pattern retrieval
     - Long-range dependencies

Operational Modes
-----------------

The ``mode`` parameter applies a named configuration profile:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode value
     - Effect
   * - ``None`` (default)
     - Manual configuration — use ``objective``, ``architecture_config``, etc.
   * - ``"tft"`` / ``"tft_like"``
     - Temporal Fusion Transformer style: VSN, gated residual, cross attention
   * - ``"pihal"`` / ``"pihal_like"``
     - Physics-Informed HAL style: memory-augmented + hierarchical stack

Attention Level Control
-----------------------

``attention_levels`` accepts several forms:

.. code-block:: python

   model = BaseAttentive(..., attention_levels=None)              # all three
   model = BaseAttentive(..., attention_levels="cross")
   model = BaseAttentive(..., attention_levels=["cross", "memory"])
   model = BaseAttentive(..., attention_levels=1)   # cross
   model = BaseAttentive(..., attention_levels=2)   # hierarchical
   model = BaseAttentive(..., attention_levels=3)   # memory

Multi-Scale Aggregation
-----------------------

.. code-block:: python

   model = BaseAttentive(
       ...,
       scales=[1, 2, 4],
       multi_scale_agg="average",  # 'last','average','flatten','sum','concat'
       final_agg="last",           # 'last','average','flatten'
   )

Feature Extraction
------------------

**Variable Selection Network (VSN)** — learns feature importance:

.. code-block:: python

   model = BaseAttentive(..., use_vsn=True, vsn_units=64)

**Dense Processing** — simple linear projection:

.. code-block:: python

   model = BaseAttentive(..., use_vsn=False)

Output Modes
------------

.. code-block:: python

   # Point forecast
   model = BaseAttentive(..., output_dim=2, forecast_horizon=24)
   # predictions shape: (batch, 24, 2)

   # Probabilistic
   model = BaseAttentive(..., quantiles=[0.1, 0.5, 0.9])
   # predictions shape: (batch, 24, 3, 2)

V2 Architecture: Registry / Resolver / Assembly
------------------------------------------------

Version 1.0.0 introduces a pluggable component system for backend-neutral
model construction.

ComponentRegistry
~~~~~~~~~~~~~~~~~

Stores builder functions keyed by ``(component_key, backend)``:

.. code-block:: python

   from base_attentive.registry import DEFAULT_COMPONENT_REGISTRY

   def my_encoder_builder(*, context, units, hidden_units, **kw):
       return MyCustomEncoder(units=units)

   DEFAULT_COMPONENT_REGISTRY.register(
       "encoder.my_custom",
       my_encoder_builder,
       backend="generic",
       description="My custom temporal encoder.",
   )

ModelRegistry
~~~~~~~~~~~~~

Stores assembler functions:

.. code-block:: python

   from base_attentive.registry import DEFAULT_MODEL_REGISTRY

   DEFAULT_MODEL_REGISTRY.register(
       "base_attentive.v2",
       my_assembler_fn,
       backend="generic",
   )

Default component keys (``generic`` backend):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Registry Key
     - Purpose
   * - ``projection.static``
     - Static feature linear projection
   * - ``projection.dynamic``
     - Dynamic sequence projection
   * - ``projection.future``
     - Future covariate projection
   * - ``projection.hidden``
     - Post-fusion hidden projection
   * - ``projection.dense``
     - Generic dense projection (fallback)
   * - ``encoder.temporal_self_attention``
     - Temporal self-attention encoder
   * - ``pool.mean``
     - Sequence mean pooling
   * - ``pool.last``
     - Last-step pooling
   * - ``fusion.concat``
     - Feature concatenation
   * - ``head.point_forecast``
     - Point forecast head
   * - ``head.quantile_forecast``
     - Quantile forecast head

BaseAttentiveSpec
~~~~~~~~~~~~~~~~~

A frozen dataclass that fully describes a V2 model:

.. code-block:: python

   from base_attentive.config import BaseAttentiveSpec, BaseAttentiveComponentSpec

   spec = BaseAttentiveSpec(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       embed_dim=32,
       hidden_units=64,
       attention_heads=4,
       layer_norm_epsilon=1e-6,
       dropout_rate=0.1,
       activation="relu",
       backend_name="tensorflow",
       head_type="point",
       quantiles=(),
       components=BaseAttentiveComponentSpec(
           sequence_pooling="pool.last",  # override one component
       ),
   )

Configuration Hierarchy
-----------------------

Precedence (lowest to highest):

1. Default values (``DEFAULT_ARCHITECTURE``)
2. Explicit keyword arguments (``objective``, ``mode``, ``attention_levels``, …)
3. ``architecture_config`` dict (overrides all)

.. code-block:: python

   model = BaseAttentive(
       ...,
       objective="hybrid",          # step 2
       architecture_config={
           "encoder_type": "transformer",  # step 3 — wins
       },
   )

Data Flow Diagram
-----------------

.. code-block:: text

   Static (B,S)  Dynamic (B,T,D)  Future (B,H,F)
        │               │                │
        └───────────────┴────────────────┘
                        │
             ┌──────────▼──────────┐
             │  Feature Selection  │
             │  (VSN or Dense)     │
             └──────────┬──────────┘
                        │
              ┌─────────┴──────────┐
              │                    │
   ┌──────────▼──────┐  ┌──────────▼──────┐
   │ Dynamic Encoder  │  │ Future Context  │
   │ (LSTM / Attn)    │  │ (Dense proj.)   │
   └──────────┬──────┘  └──────────┬──────┘
              │                    │
              └─────────┬──────────┘
                        │
             ┌──────────▼──────────┐
             │   Attention Stack   │
             │ cross / hier / mem  │
             └──────────┬──────────┘
                        │
             ┌──────────▼──────────┐
             │  Sequence Pooling   │
             └──────────┬──────────┘
                        │
             ┌──────────▼──────────┐
             │  Hidden Projection  │
             └──────────┬──────────┘
                        │
             ┌──────────┴──────────┐
             │                     │
       Point Forecast        Quantile Forecast
         (B, H, D)             (B, H, Q, D)

Performance Notes
-----------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 22 43

   * - Mode
     - Encoder
     - Complexity
     - Notes
   * - Hybrid
     - Multi-scale LSTM
     - O(T·h²)
     - Recommended for T > 500
   * - Transformer
     - Self-attention
     - O(T²·h)
     - Recommended for T < 500

See Also
--------

- :doc:`configuration_guide` — Detailed parameter reference
- :doc:`api_reference` — Complete API
- :doc:`usage` — Extended usage patterns
- :doc:`components_reference` — Component library
