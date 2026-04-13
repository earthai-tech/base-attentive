Architecture Guide
==================

This guide explains how BaseAttentive works internally, what changed in
**v2.0.0**, and how to use the new registry / resolver / assembly system.
If you are migrating from v1.0.0, read the
:ref:`breaking changes <arch-breaking-changes>` section first.

.. contents:: On this page
   :local:
   :depth: 2

----

Overview
--------

BaseAttentive is an encoder-decoder neural network for sequence-to-sequence
time series forecasting.  It accepts three distinct feature streams:

1. **Static features** — time-invariant properties ``(batch, static_dim)``
2. **Dynamic features** — historical time series ``(batch, T, dynamic_dim)``
3. **Future features** — known future exogenous variables ``(batch, H, future_dim)``

.. code-block:: text

   ┌─────────────────────────────────────┐
   │           Inputs (3 types)          │
   ├─────────────────────────────────────┤
   │ static:   (batch, S)                │
   │ dynamic:  (batch, T, D)             │
   │ future:   (batch, H, F)             │
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

Conceptual flow:

1. **Select** — Variable Selection Networks (VSN) weight each input feature
2. **Project** — Transform features into a shared embedding space
3. **Encode** — Process temporal context (hybrid LSTM or pure transformer)
4. **Attend** — Apply the decoder attention stack (cross / hierarchical / memory)
5. **Pool** — Collapse the sequence representation into a fixed vector
6. **Forecast** — Generate point or probabilistic outputs

----

Encoder Architectures
---------------------

Hybrid Mode (``objective="hybrid"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-scale LSTM with attention.  Each LSTM processes a down-sampled version
of the sequence at scale ``s``, then the outputs are aggregated before
entering the decoder:

.. code-block:: python

   import numpy as np
   from base_attentive import BaseAttentive

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       objective="hybrid",
       scales=[1, 2, 4],          # sequence sub-sampled at ×1, ×2, ×4
       multi_scale_agg="average", # how to merge the scale outputs
       embed_dim=32,
   )

``scales=[1, 2, 4]`` creates three parallel LSTMs.  At scale ``s``, every
``s``-th time step is kept, so the LSTM at scale 4 sees a quarter of the
full history.  This lets the model capture both fine-grained and coarse
temporal patterns simultaneously.

``multi_scale_agg`` choices:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Effect
   * - ``"last"``
     - Keep the final hidden state of each scale; concatenate then project
   * - ``"average"``
     - Average all hidden states across time, then merge
   * - ``"flatten"``
     - Flatten the full output sequence of each scale, then project
   * - ``"sum"``
     - Sum hidden states element-wise across time
   * - ``"concat"``
     - Concatenate all time-step outputs end-to-end

Transformer Mode (``objective="transformer"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pure self-attention encoder — better parallelism on shorter sequences
(T < 500):

.. code-block:: python

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       objective="transformer",
       num_encoder_layers=4,
       num_heads=8,
       embed_dim=64,
   )

----

Decoder Attention Stack
-----------------------

After encoding, a configurable stack of attention mechanisms bridges the
encoded history with the future feature context.

Attention types:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Type
     - Purpose
     - Use case
   * - ``"cross"``
     - Bridge encoder outputs to future context
     - Default; works for all forecasting tasks
   * - ``"hierarchical"``
     - Multi-level temporal patterns in the decoder
     - Seasonal / structured data with nested cycles
   * - ``"memory"``
     - Retrieve patterns from a learned memory bank
     - Long-range dependencies, repeated anomalies

Controlling the stack with ``attention_levels``:

.. code-block:: python

   # All three levels
   model = BaseAttentive(..., attention_levels=None)

   # Single level by name
   model = BaseAttentive(..., attention_levels="cross")

   # Two levels by list
   model = BaseAttentive(..., attention_levels=["cross", "memory"])

   # Single level by integer (1=cross, 2=hierarchical, 3=memory)
   model = BaseAttentive(..., attention_levels=2)

----

Operational Mode Shortcuts
--------------------------

The ``mode`` parameter applies a named configuration profile, wiring up
encoder type, attention stack, and decoder in one step:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Effect
   * - ``None`` (default)
     - Manual configuration — use ``objective``, ``architecture_config``, etc.
   * - ``"tft"`` / ``"tft_like"``
     - Temporal Fusion Transformer style: VSN + gated residuals + cross attention
   * - ``"pihal"`` / ``"pihal_like"``
     - Physics-Informed HAL style: memory-augmented + hierarchical stack

.. code-block:: python

   # TFT-like mode — no need to specify objective or attention_levels
   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       mode="tft",
       embed_dim=32,
   )

----

Output Modes
------------

.. code-block:: python

   # Point forecast — shape (batch, H, output_dim)
   model = BaseAttentive(..., output_dim=2, forecast_horizon=24)

   # Quantile forecast — shape (batch, H, Q, output_dim)
   model = BaseAttentive(..., quantiles=[0.1, 0.5, 0.9])

   # Probabilistic (Gaussian mixture, for CRPSLoss)
   model = BaseAttentive(..., output_mode="gaussian_mixture")

----

.. _arch-v2-system:

V2 Architecture: Registry / Resolver / Assembly
------------------------------------------------

Version 2.0.0 replaces the monolithic class hierarchy of v1.0.0 with a
**registry / resolver / assembly** system.  Every model component is now
registered under a string key and resolved at build time.  This makes the
model fully pluggable and backend-neutral.

Why this matters
~~~~~~~~~~~~~~~~

In v1.0.0 the encoder, attention heads, and forecast head were hard-coded
inside ``BaseAttentive``.  Customising them required subclassing internal
layers and overriding private methods — fragile and backend-specific.

In v2.0.0:

- Each component is a **builder function** stored in a registry.
- ``BaseAttentiveSpec`` / ``BaseAttentiveComponentSpec`` describe the model
  purely as data (no Keras imports required at spec-creation time).
- ``BaseAttentiveV2Assembly`` reads the spec, resolves each component from
  the registry, and wires everything together.
- Swapping a component is a one-line registry call — no subclassing.

The Three Registries
~~~~~~~~~~~~~~~~~~~~

``ComponentRegistry``
    Stores builder functions for individual layers
    (encoders, projections, attention heads, pooling, forecast heads).
    Key format: ``"<category>.<name>"``.

``ModelRegistry``
    Stores assembler functions that construct the full model from a spec.

Both registries are available as singletons:

.. code-block:: python

   from base_attentive.registry import (
       DEFAULT_COMPONENT_REGISTRY,
       DEFAULT_MODEL_REGISTRY,
   )

Registering a custom encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from base_attentive.registry import DEFAULT_COMPONENT_REGISTRY

   def wavenet_encoder_builder(*, context, units, hidden_units, **kw):
       """
       A WaveNet-style dilated causal encoder.
       context: BaseAttentiveSpec — gives access to embed_dim, dropout_rate, etc.
       """
       from my_layers import WaveNetBlock
       return WaveNetBlock(
           units=units,
           dilation_rates=[1, 2, 4, 8],
           dropout=context.dropout_rate,
       )

   DEFAULT_COMPONENT_REGISTRY.register(
       "encoder.wavenet",
       wavenet_encoder_builder,
       backend="generic",          # works across TF / Torch / JAX
       description="WaveNet dilated causal encoder.",
   )

Then use the key in a spec:

.. code-block:: python

   from base_attentive.config import BaseAttentiveSpec, BaseAttentiveComponentSpec

   spec = BaseAttentiveSpec(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       embed_dim=64,
       components=BaseAttentiveComponentSpec(
           temporal_encoder="encoder.wavenet",   # <-- custom component
       ),
   )

   from base_attentive.assembly import BaseAttentiveV2Assembly
   assembler = BaseAttentiveV2Assembly()
   model = assembler.build(spec)

BaseAttentiveSpec
~~~~~~~~~~~~~~~~~

A frozen dataclass that fully describes a model without any framework
imports.  All fields have defaults.

.. code-block:: python

   from base_attentive.config import BaseAttentiveSpec, BaseAttentiveComponentSpec

   spec = BaseAttentiveSpec(
       # ── Input dimensions ────────────────────────────────────────────
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,

       # ── Model capacity ──────────────────────────────────────────────
       embed_dim=32,
       hidden_units=64,
       attention_heads=4,
       dropout_rate=0.1,
       activation="relu",
       layer_norm_epsilon=1e-6,

       # ── Backend / head ──────────────────────────────────────────────
       backend_name="tensorflow",   # or "torch" / "jax"
       head_type="point",           # or "quantile"
       quantiles=(),                # e.g. (0.1, 0.5, 0.9)

       # ── Component overrides ─────────────────────────────────────────
       components=BaseAttentiveComponentSpec(
           sequence_pooling="pool.last",      # override pooling
           temporal_encoder="encoder.wavenet",# override encoder
       ),
   )

``BaseAttentiveComponentSpec`` accepts the following keys
(all optional — omitted keys use the registry default):

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Field
     - Registry key resolved
   * - ``static_projection``
     - ``"projection.static"``
   * - ``dynamic_projection``
     - ``"projection.dynamic"``
   * - ``future_projection``
     - ``"projection.future"``
   * - ``hidden_projection``
     - ``"projection.hidden"``
   * - ``temporal_encoder``
     - ``"encoder.temporal_self_attention"``
   * - ``sequence_pooling``
     - ``"pool.mean"``
   * - ``feature_fusion``
     - ``"fusion.concat"``
   * - ``forecast_head``
     - ``"head.point_forecast"`` or ``"head.quantile_forecast"``

Default component keys (built-in, ``"generic"`` backend):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Registry key
     - Purpose
   * - ``"projection.static"``
     - Static feature linear projection
   * - ``"projection.dynamic"``
     - Dynamic sequence projection
   * - ``"projection.future"``
     - Future covariate projection
   * - ``"projection.hidden"``
     - Post-fusion hidden projection
   * - ``"projection.dense"``
     - Generic dense projection (fallback)
   * - ``"encoder.temporal_self_attention"``
     - Temporal self-attention encoder
   * - ``"pool.mean"``
     - Sequence mean pooling
   * - ``"pool.last"``
     - Last-step pooling
   * - ``"fusion.concat"``
     - Feature concatenation
   * - ``"head.point_forecast"``
     - Point forecast head
   * - ``"head.quantile_forecast"``
     - Quantile forecast head

Inspecting the registry
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from base_attentive.registry import DEFAULT_COMPONENT_REGISTRY

   # List all registered keys
   for key in DEFAULT_COMPONENT_REGISTRY.list_keys():
       print(key)

   # Check if a key exists
   if DEFAULT_COMPONENT_REGISTRY.has("encoder.wavenet"):
       print("custom encoder registered")

   # Retrieve builder metadata
   info = DEFAULT_COMPONENT_REGISTRY.get_info("encoder.temporal_self_attention")
   print(info["description"])

Full v2 build-from-spec example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from base_attentive.config import BaseAttentiveSpec
   from base_attentive.assembly import BaseAttentiveV2Assembly

   spec = BaseAttentiveSpec(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       embed_dim=32,
       hidden_units=64,
       attention_heads=4,
       backend_name="tensorflow",
       head_type="quantile",
       quantiles=(0.1, 0.5, 0.9),
   )

   model = BaseAttentiveV2Assembly().build(spec)
   model.compile(optimizer="adam", loss="mse")

   x_static  = np.random.randn(16, 4).astype("float32")
   x_dynamic = np.random.randn(16, 100, 8).astype("float32")
   x_future  = np.random.randn(16, 24, 6).astype("float32")
   y         = np.random.randn(16, 24, 1).astype("float32")

   model.fit([x_static, x_dynamic, x_future], y, epochs=2)

Using ``BaseAttentive`` (facade)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``BaseAttentive`` class is a convenience facade that builds the model
from keyword arguments without requiring you to construct a spec manually.
It delegates to the same registry/assembly system under the hood:

.. code-block:: python

   from base_attentive import BaseAttentive

   # This is equivalent to building through BaseAttentiveSpec + Assembly
   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       embed_dim=32,
       num_heads=4,
       quantiles=[0.1, 0.5, 0.9],
   )

----

.. _arch-breaking-changes:

Breaking Changes in v2.0.0
--------------------------

v2.0.0 is a **major release**.  If you are upgrading from v1.0.0, the
following changes require action.

.. note::

   These changes are intentional.  The v1.0.0 API was tightly coupled to
   TensorFlow; v2.0.0 achieves full backend neutrality through these
   structural changes.

1. Keras 3 required
~~~~~~~~~~~~~~~~~~~

v1.0.0 used ``tensorflow.keras`` directly.  v2.0.0 uses
`Keras 3 <https://keras.io/>`_ (``import keras``) as the framework
abstraction layer.

**What breaks:** Any code that imports from ``tensorflow.keras`` or passes
``tf.Tensor`` objects to model inputs may need updating.

**Migration:**

.. code-block:: python

   # v1.0.0 — TensorFlow-coupled
   import tensorflow as tf
   model = BaseAttentive(...)
   x = tf.random.normal([32, 100, 8])

   # v2.0.0 — backend-neutral
   import numpy as np
   model = BaseAttentive(...)
   x = np.random.randn(32, 100, 8).astype("float32")
   # or use the active backend's tensor type directly

2. Internal layer paths removed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In v1.0.0, internal layer classes were importable from
``base_attentive.layers.*`` and ``base_attentive.models.components.*``.
These paths no longer exist in v2.0.0.  All components are accessed
through the registry.

**What breaks:** Direct imports of internal layer classes.

**Migration:**

.. code-block:: python

   # v1.0.0 (breaks in v2.0.0)
   from base_attentive.layers import HierarchicalAttention

   # v2.0.0 — use registry or components_reference API
   from base_attentive.registry import DEFAULT_COMPONENT_REGISTRY
   builder = DEFAULT_COMPONENT_REGISTRY.get("attention.hierarchical")

3. ``architecture_config`` dict keys changed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several ``architecture_config`` keys were renamed for clarity:

.. list-table::
   :header-rows: 1
   :widths: 35 35 30

   * - v1.0.0 key
     - v2.0.0 key
     - Notes
   * - ``"encoder_units"``
     - ``"embed_dim"``
     - Unified dimension name
   * - ``"decoder_heads"``
     - ``"num_heads"``
     - Consistent with Keras naming
   * - ``"use_attention"``
     - ``"attention_levels"``
     - Now accepts name, list, or int
   * - ``"temporal_mode"``
     - ``"objective"``
     - ``"hybrid"`` or ``"transformer"``

**Migration:**

.. code-block:: python

   # v1.0.0
   model = BaseAttentive(
       ...,
       architecture_config={"encoder_units": 64, "use_attention": True},
   )

   # v2.0.0
   model = BaseAttentive(
       ...,
       embed_dim=64,
       attention_levels=["cross"],
   )

4. ``output_mode`` default changed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

v1.0.0 default was ``"quantile"`` when ``quantiles`` was set.
v2.0.0 always infers the output mode from the combination of
``quantiles`` and ``output_mode``.  Passing ``quantiles`` without
``output_mode`` now produces a quantile forecast as before, but
the internal tensor layout changed:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Setting
     - v1.0.0 output shape
     - v2.0.0 output shape
   * - ``output_dim=2``, no quantiles
     - ``(B, H, 2)``
     - ``(B, H, 2)`` (unchanged)
   * - ``output_dim=2``, ``quantiles=[0.1,0.5,0.9]``
     - ``(B, H, 2, 3)``  ← Q last
     - ``(B, H, 3, 2)``  ← Q before output_dim

**Migration:** If you index the quantile axis, update from ``[..., i]``
(v1) to ``[:, :, i, :]`` (v2).

----

Data Flow Diagram (v2)
-----------------------

.. code-block:: text

   Static (B,S)  Dynamic (B,T,D)  Future (B,H,F)
        │               │                │
        │        ┌──────▼──────┐         │
        │        │  VSN / Dense │         │
        │        └──────┬──────┘         │
        │               │                │
   ┌────▼────┐   ┌──────▼──────┐  ┌──────▼──────┐
   │ Static  │   │  Temporal   │  │  Future     │
   │ Proj.   │   │  Encoder    │  │  Proj.      │
   │ (Dense) │   │ (LSTM/Attn) │  │  (Dense)    │
   └────┬────┘   └──────┬──────┘  └──────┬──────┘
        │               │                │
        └───────────────┴────────────────┘
                        │
             ┌──────────▼──────────┐
             │   Feature Fusion    │
             │   (concat + proj)   │
             └──────────┬──────────┘
                        │
             ┌──────────▼──────────┐
             │   Attention Stack   │
             │   (cross → hier     │
             │    → memory)        │
             └──────────┬──────────┘
                        │
             ┌──────────▼──────────┐
             │  Sequence Pooling   │
             │  (mean / last)      │
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

----

Configuration Hierarchy
-----------------------

Precedence (lowest → highest):

1. Built-in defaults (``DEFAULT_ARCHITECTURE``)
2. Explicit keyword arguments (``objective``, ``mode``, ``attention_levels``, …)
3. ``architecture_config`` dict (overrides all)

.. code-block:: python

   model = BaseAttentive(
       ...,
       objective="hybrid",            # step 2
       architecture_config={
           "encoder_type": "transformer",  # step 3 — wins over step 2
       },
   )

----

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

----

See Also
--------

- :doc:`configuration_guide` — Full parameter reference
- :doc:`api_reference` — Complete API docs
- :doc:`usage` — Extended usage patterns
- :doc:`components_reference` — Component library
- :doc:`release_notes/v2.0.0` — v2.0.0 stable release notes
- :doc:`release_notes/v1.0.0` — v1.0.0 release notes
