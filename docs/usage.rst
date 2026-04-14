Usage
=====

This page explains how to configure, train, and extend BaseAttentive in v2.
It covers every configuration path in detail — from the one-liner quick-start
through to the full spec-based assembly API — with worked examples at each
level.

.. contents:: On this page
   :local:
   :depth: 2

----

Quick Start
-----------

The shortest possible working example:

.. code-block:: python

   import numpy as np
   from base_attentive import BaseAttentive

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
   )
   model.compile(optimizer="adam", loss="mse")

   x_static  = np.random.randn(32, 4).astype("float32")
   x_dynamic = np.random.randn(32, 100, 8).astype("float32")
   x_future  = np.random.randn(32, 24, 6).astype("float32")
   y         = np.random.randn(32, 24, 1).astype("float32")

   model.fit([x_static, x_dynamic, x_future], y, epochs=5)
   preds = model.predict([x_static, x_dynamic, x_future])
   # shape: (32, 24, 1)

----

Input Contract
--------------

``BaseAttentive`` expects three inputs in fixed order:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Slot
     - Shape
     - Content
   * - ``static``
     - ``(batch, static_input_dim)``
     - Time-invariant properties (e.g. coordinates, site type)
   * - ``dynamic``
     - ``(batch, T, dynamic_input_dim)``
     - Historical time series (any T ≥ 1)
   * - ``future``
     - ``(batch, forecast_horizon, future_input_dim)``
     - Known future exogenous variables

Pass them as a list:

.. code-block:: python

   predictions = model([x_static, x_dynamic, x_future])

Any slot can be omitted by setting its ``*_input_dim`` to ``0``; in that case
pass a zero-column array or ``None`` for the missing slot.

Output Shapes
~~~~~~~~~~~~~

.. code-block:: python

   # Point forecast
   model = BaseAttentive(..., output_dim=2, forecast_horizon=24)
   # → (batch, 24, 2)

   # Quantile forecast — quantile axis is second-to-last
   model = BaseAttentive(..., output_dim=2, quantiles=[0.1, 0.5, 0.9])
   # → (batch, 24, 3, 2)   ← (batch, horizon, Q, output_dim)

----

V2 Configuration System
------------------------

v2.0.0 introduces three ways to configure a model.  They can be combined:
higher-precedence levels override lower ones.

.. code-block:: text

   Precedence (low → high)
   ──────────────────────
   1. Built-in defaults
   2. Keyword arguments (embed_dim, objective, mode, …)
   3. architecture_config dict
   ──────────────────────
   (optional) BaseAttentiveSpec + BaseAttentiveComponentSpec  ← full declarative path

Path 1: Keyword arguments
~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest and most common path.  Pass parameters directly to
``BaseAttentive``:

.. code-block:: python

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       # ── capacity ────────────────────
       embed_dim=64,
       hidden_units=128,
       num_heads=8,
       num_encoder_layers=3,
       # ── architecture ────────────────
       objective="hybrid",
       scales=[1, 2, 4],
       multi_scale_agg="average",
       attention_levels=["cross", "hierarchical"],
       # ── regularisation ──────────────
       dropout_rate=0.1,
       use_vsn=True,
       # ── output ──────────────────────
       quantiles=[0.1, 0.5, 0.9],
   )

Path 2: ``architecture_config`` dict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this dict for **structural** overrides that do not have a dedicated
keyword argument, or to enforce settings regardless of the keyword values:

.. code-block:: python

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       objective="hybrid",                  # step-2 keyword
       architecture_config={
           "encoder_type": "transformer",   # step-3 override — wins
           "decoder_attention_stack": ["cross", "memory"],
           "feature_processing": "dense",   # skip VSN
       },
   )

Common ``architecture_config`` keys:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Key
     - Effect
   * - ``"encoder_type"``
     - ``"hybrid"`` or ``"transformer"``
   * - ``"decoder_attention_stack"``
     - list from ``["cross", "hierarchical", "memory"]``
   * - ``"feature_processing"``
     - ``"vsn"`` (Variable Selection Network) or ``"dense"``
   * - ``"pooling"``
     - ``"last"`` or ``"mean"``
   * - ``"use_residuals"``
     - ``True`` / ``False``
   * - ``"use_batch_norm"``
     - ``True`` / ``False``

Path 3: ``BaseAttentiveSpec`` + Assembly (declarative)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fully declarative v2 path.  Build a frozen configuration object
first, then assemble the model from it.  No Keras imports are needed to
create the spec — useful for config files, hyperparameter search, or
experiment tracking systems.

.. code-block:: python

   from base_attentive.config import BaseAttentiveSpec, BaseAttentiveComponentSpec
   from base_attentive.assembly import BaseAttentiveV2Assembly

   spec = BaseAttentiveSpec(
       # ── dimensions ──────────────────────────────────────────────
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       # ── capacity ────────────────────────────────────────────────
       embed_dim=64,
       hidden_units=128,
       attention_heads=8,
       dropout_rate=0.1,
       activation="relu",
       layer_norm_epsilon=1e-6,
       # ── backend + head ──────────────────────────────────────────
       backend_name="tensorflow",   # "tensorflow" | "torch" | "jax"
       head_type="quantile",
       quantiles=(0.1, 0.5, 0.9),
       # ── component overrides (all optional) ──────────────────────
       components=BaseAttentiveComponentSpec(
           sequence_pooling="pool.last",
           temporal_encoder="encoder.temporal_self_attention",
       ),
   )

   model = BaseAttentiveV2Assembly().build(spec)
   model.compile(optimizer="adam", loss="mse")

Serialising a spec to JSON and reloading it:

.. code-block:: python

   import json
   from base_attentive.config import BaseAttentiveSpec

   # Save
   with open("my_spec.json", "w") as f:
       json.dump(spec.__dict__, f, indent=2)   # spec is a dataclass

   # Reload
   with open("my_spec.json") as f:
       data = json.load(f)
   reloaded_spec = BaseAttentiveSpec(**data)
   model2 = BaseAttentiveV2Assembly().build(reloaded_spec)

``BaseAttentiveComponentSpec`` field reference:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Field
     - Default registry key
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

Omit any field to use the registry default.

----

Architecture Parameters
-----------------------

Required parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 12 64

   * - Parameter
     - Type
     - Description
   * - ``static_input_dim``
     - int ≥ 0
     - Number of static (time-invariant) features
   * - ``dynamic_input_dim``
     - int ≥ 1
     - Number of historical time-series features
   * - ``future_input_dim``
     - int ≥ 0
     - Number of known future covariate features
   * - ``output_dim``
     - int ≥ 1
     - Number of output variables (default: 1)
   * - ``forecast_horizon``
     - int ≥ 1
     - Number of future steps to predict (default: 1)

Capacity parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 10 10 56

   * - Parameter
     - Type
     - Default
     - Description
   * - ``embed_dim``
     - int
     - 32
     - Shared embedding dimension for all projections
   * - ``hidden_units``
     - int
     - 64
     - Dense hidden layer width
   * - ``lstm_units``
     - int
     - 64
     - LSTM hidden size (hybrid mode only)
   * - ``attention_units``
     - int
     - 32
     - Attention projection dimension
   * - ``num_heads``
     - int
     - 4
     - Multi-head attention head count
   * - ``num_encoder_layers``
     - int
     - 2
     - Stacked encoder layer count
   * - ``memory_size``
     - int
     - 100
     - Memory bank entries (memory-augmented attention)

Encoder parameters
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 10 12 54

   * - Parameter
     - Type
     - Default
     - Description
   * - ``objective``
     - str
     - ``"hybrid"``
     - Encoder type: ``"hybrid"`` (multi-scale LSTM) or ``"transformer"``
   * - ``scales``
     - list[int] or ``"auto"``
     - None
     - Temporal scales for multi-scale LSTM, e.g. ``[1, 2, 4]``
   * - ``multi_scale_agg``
     - str
     - ``"last"``
     - How to merge multi-scale outputs: ``"last"``, ``"average"``, ``"flatten"``, ``"sum"``, ``"concat"``
   * - ``final_agg``
     - str
     - ``"last"``
     - Final sequence aggregation: ``"last"``, ``"average"``, ``"flatten"``
   * - ``max_window_size``
     - int
     - 10
     - Maximum dynamic time window size

Attention and decoder
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 14 14 48

   * - Parameter
     - Type
     - Default
     - Description
   * - ``mode``
     - str or None
     - None
     - Mode shortcut: ``"tft"``, ``"pihal"``, ``"tft_like"``, ``"pihal_like"``
   * - ``attention_levels``
     - str, list, int, None
     - None (all three)
     - Decoder attention stack: name, list, integer shortcut, or None
   * - ``architecture_config``
     - dict or None
     - None
     - Structural overrides (highest precedence)

Regularisation
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 10 10 56

   * - Parameter
     - Type
     - Default
     - Description
   * - ``dropout_rate``
     - float
     - 0.1
     - Dropout probability ``[0, 1)``
   * - ``activation``
     - str
     - ``"relu"``
     - ``"relu"``, ``"elu"``, ``"selu"``, ``"gelu"``, ``"swish"``, ``"tanh"``, ``"sigmoid"``, ``"linear"``
   * - ``use_batch_norm``
     - bool
     - False
     - Apply batch normalisation
   * - ``use_residuals``
     - bool
     - True
     - Residual connections throughout

Feature processing
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 10 10 56

   * - Parameter
     - Type
     - Default
     - Description
   * - ``use_vsn``
     - bool
     - True
     - Enable Variable Selection Network (VSN)
   * - ``vsn_units``
     - int or None
     - None
     - Override VSN projection units (falls back to ``embed_dim``)
   * - ``apply_dtw``
     - bool
     - True
     - Apply Dynamic Time Warping alignment

Output
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 14 14 48

   * - Parameter
     - Type
     - Default
     - Description
   * - ``quantiles``
     - list[float] or None
     - None
     - Quantile levels, e.g. ``[0.1, 0.5, 0.9]``
   * - ``verbose``
     - int
     - 0
     - Logging verbosity

----

Mode Shortcuts
--------------

``mode`` wires up encoder, attention stack, and decoder in one step so you
do not need to set ``objective``, ``attention_levels``, etc. individually:

.. code-block:: python

   # TFT-like: VSN + gated residuals + cross attention
   model = BaseAttentive(..., mode="tft")

   # PIHALNet-like: multi-scale LSTM + memory + hierarchical attention
   model = BaseAttentive(..., mode="pihal")

You can still override individual parameters after setting ``mode``:

.. code-block:: python

   model = BaseAttentive(
       ...,
       mode="tft",
       embed_dim=128,          # larger capacity
       dropout_rate=0.2,       # more regularisation
       quantiles=[0.05, 0.5, 0.95],
   )

----

Attention Level Control
-----------------------

The decoder attention stack is set with ``attention_levels``.  All four
forms below are equivalent:

.. code-block:: python

   # All three levels (default)
   model = BaseAttentive(..., attention_levels=None)
   model = BaseAttentive(..., attention_levels=["cross", "hierarchical", "memory"])

   # Single level
   model = BaseAttentive(..., attention_levels="cross")
   model = BaseAttentive(..., attention_levels=1)    # integer: 1=cross

   # Two levels
   model = BaseAttentive(..., attention_levels=["cross", "memory"])
   model = BaseAttentive(..., attention_levels=2)    # 2=hierarchical

Integer shortcuts: ``1`` → ``cross``, ``2`` → ``hierarchical``,
``3`` → ``memory``.

When to use each level:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Level
     - When to use it
   * - ``"cross"``
     - Always include; it bridges encoded history and future context
   * - ``"hierarchical"``
     - Add when the data has nested temporal patterns (daily + weekly + seasonal)
   * - ``"memory"``
     - Add for long-range dependencies or when repeated patterns need retrieval

----

Multi-Scale Aggregation
-----------------------

``scales`` activates the multi-scale LSTM encoder (``objective="hybrid"``
required).  Each integer in the list defines a temporal stride:

.. code-block:: python

   model = BaseAttentive(
       ...,
       objective="hybrid",
       scales=[1, 2, 4],           # three LSTMs at stride ×1, ×2, ×4
       multi_scale_agg="average",  # merge strategy
       final_agg="last",           # sequence → vector
   )

``multi_scale_agg`` choices:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Value
     - Effect
   * - ``"last"``
     - Keep the final hidden state of each scale; concatenate then project
   * - ``"average"``
     - Average all hidden states across time; merge
   * - ``"flatten"``
     - Flatten the full output sequence of each scale; project
   * - ``"sum"``
     - Sum hidden states element-wise across time
   * - ``"concat"``
     - Concatenate all time-step outputs end-to-end

Use ``scales="auto"`` to let the model choose scales based on the
sequence length.

----

Training Patterns
-----------------

Standard compile/fit
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from base_attentive import BaseAttentive

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       quantiles=[0.1, 0.5, 0.9],
       embed_dim=64,
       num_heads=8,
   )

   model.compile(
       optimizer="adam",
       loss="mse",
       metrics=["mae"],
   )

   history = model.fit(
       [x_static, x_dynamic, x_future],
       y,
       epochs=50,
       batch_size=64,
       validation_split=0.2,
       callbacks=[
           keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
       ],
   )

Using CRPSLoss
~~~~~~~~~~~~~~

For probabilistic training with the mixture mode:

.. code-block:: python

   from base_attentive.components import CRPSLoss

   # Quantile CRPS (pinball approximation)
   model.compile(
       optimizer="adam",
       loss=CRPSLoss(mode="quantile", quantiles=[0.1, 0.5, 0.9]),
   )

   # Gaussian closed-form CRPS
   model.compile(
       optimizer="adam",
       loss=CRPSLoss(mode="gaussian"),
   )

   # Gaussian Mixture Monte-Carlo CRPS
   model.compile(
       optimizer="adam",
       loss=CRPSLoss(mode="mixture", mc_samples=50),
   )

Custom training loop
~~~~~~~~~~~~~~~~~~~~

When you need per-step control (e.g. physics constraints, gradient clipping,
or multi-task losses):

.. code-block:: python

   import keras
   import numpy as np
   from base_attentive import BaseAttentive

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
   )

   optimizer = keras.optimizers.Adam(learning_rate=1e-3)
   loss_fn   = keras.losses.MeanSquaredError()

   @keras.function
   def train_step(x_batch, y_batch):
       with keras.GradientTape() as tape:
           preds = model(x_batch, training=True)
           loss  = loss_fn(y_batch, preds)
       grads = tape.gradient(loss, model.trainable_variables)
       optimizer.apply_gradients(zip(grads, model.trainable_variables))
       return loss

   for epoch in range(50):
       for x_batch, y_batch in dataset:
           loss = train_step(x_batch, y_batch)

----

Selecting a Backend in V2
--------------------------

Set the backend **before** importing BaseAttentive or Keras:

.. code-block:: python

   import os
   os.environ["KERAS_BACKEND"] = "torch"   # or "tensorflow" / "jax"

   from base_attentive import BaseAttentive
   model = BaseAttentive(...)

Or, using the runtime API (also before any model creation):

.. code-block:: python

   from base_attentive import set_backend
   set_backend("jax")

   from base_attentive import BaseAttentive
   model = BaseAttentive(...)

When building via ``BaseAttentiveSpec``, declare the backend in the spec:

.. code-block:: python

   spec = BaseAttentiveSpec(
       ...,
       backend_name="torch",   # "tensorflow" | "torch" | "jax"
   )
   model = BaseAttentiveV2Assembly().build(spec)

See :doc:`backends/index` for installation details and device management.

----

Registering Custom Components
------------------------------

V2 makes every layer pluggable through the component registry.  You can
replace any part of the model without subclassing ``BaseAttentive``.

Step 1 — write the builder function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A builder receives a ``context`` (the ``BaseAttentiveSpec``) plus any kwargs
the spec declares for that component:

.. code-block:: python

   from base_attentive.registry import DEFAULT_COMPONENT_REGISTRY

   def my_dilated_encoder(*, context, units, hidden_units, **kw):
       """
       WaveNet-style dilated causal encoder.
       context: BaseAttentiveSpec gives dropout_rate, embed_dim, etc.
       """
       import keras
       inputs = keras.Input(shape=(None, units))
       x = inputs
       for rate in [1, 2, 4, 8]:
           x = keras.layers.Conv1D(
               filters=hidden_units,
               kernel_size=2,
               dilation_rate=rate,
               padding="causal",
               activation="relu",
           )(x)
           x = keras.layers.Dropout(context.dropout_rate)(x)
       return keras.Model(inputs, x, name="dilated_encoder")

   DEFAULT_COMPONENT_REGISTRY.register(
       "encoder.dilated_causal",
       my_dilated_encoder,
       backend="generic",
       description="WaveNet-style dilated causal encoder.",
   )

Step 2 — reference the key in a spec
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from base_attentive.config import BaseAttentiveSpec, BaseAttentiveComponentSpec
   from base_attentive.assembly import BaseAttentiveV2Assembly

   spec = BaseAttentiveSpec(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       embed_dim=64,
       hidden_units=128,
       components=BaseAttentiveComponentSpec(
           temporal_encoder="encoder.dilated_causal",
       ),
   )
   model = BaseAttentiveV2Assembly().build(spec)

Step 3 — inspect the registry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from base_attentive.registry import DEFAULT_COMPONENT_REGISTRY

   print(DEFAULT_COMPONENT_REGISTRY.list_keys())
   # ['projection.static', 'projection.dynamic', ..., 'encoder.dilated_causal']

   info = DEFAULT_COMPONENT_REGISTRY.get_info("encoder.dilated_causal")
   print(info["description"])   # WaveNet-style dilated causal encoder.

----

Using BaseAttentive as a Keras Kernel
--------------------------------------

Wrapper pattern (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrap ``BaseAttentive`` when you want to add extra heads, safety logic, or
fusion with other models, while keeping the serialised API separate:

.. code-block:: python

   import keras
   import numpy as np
   from base_attentive import BaseAttentive

   class ResidualForecastModel(keras.Model):
       """BaseAttentive kernel + learnable residual correction."""

       def __init__(self, forecast_horizon=24, output_dim=1):
           super().__init__()
           self.kernel = BaseAttentive(
               static_input_dim=4,
               dynamic_input_dim=8,
               future_input_dim=6,
               output_dim=output_dim,
               forecast_horizon=forecast_horizon,
               mode="tft",
           )
           self.context_pool  = keras.layers.GlobalAveragePooling1D()
           self.residual_head = keras.Sequential([
               keras.layers.Dense(64, activation="relu"),
               keras.layers.RepeatVector(forecast_horizon),
               keras.layers.Dense(output_dim),
           ])
           self.gate = keras.layers.Dense(output_dim, activation="sigmoid")

       def call(self, inputs, training=False):
           static_x, dynamic_x, future_x = inputs
           base_forecast = self.kernel(inputs, training=training)
           # (batch, H, output_dim)

           context  = self.context_pool(dynamic_x)           # (batch, D)
           residual = self.residual_head(context)             # (batch, H, output_dim)
           gate     = keras.ops.expand_dims(
               self.gate(context), axis=1
           )                                                  # (batch, 1, output_dim)
           return base_forecast + gate * residual

Direct inheritance (advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inherit when the new behaviour should be part of the same serialised model
(same ``get_config`` / ``from_config`` lifecycle):

.. code-block:: python

   from base_attentive import BaseAttentive
   import keras

   class PhysicsAwareModel(BaseAttentive):
       def __init__(self, physics_weight=0.1, **kwargs):
           super().__init__(**kwargs)
           self.physics_weight  = physics_weight
           self.correction_head = keras.layers.Dense(self.output_dim)

       def call(self, inputs, training=False):
           base_forecast  = super().call(inputs, training=training)
           _, _, future_x = inputs
           correction     = self.correction_head(
               keras.ops.mean(future_x, axis=1)
           )                                      # (batch, output_dim)
           correction = keras.ops.expand_dims(correction, axis=1)
           return base_forecast + self.physics_weight * correction

       def get_config(self):
           cfg = super().get_config()
           cfg["physics_weight"] = self.physics_weight
           return cfg

When to choose which path:

- **Wrapper** — extra outputs, multi-task heads, downstream logic you do not
  want serialised together with the forecasting model.
- **Inheritance** — the new behaviour must survive ``from_config`` round-trips
  and is inseparable from the forecasting logic.

----

Serialisation and Reconfiguration
-----------------------------------

.. code-block:: python

   # Save / restore config
   config = model.get_config()
   cloned = BaseAttentive.from_config(config)

   # Reconfigure without mutating the original
   transformer_variant = model.reconfigure({"encoder_type": "transformer"})

   # Full Keras save (weights + config)
   model.save("my_model.keras")
   loaded = keras.saving.load_model("my_model.keras")

----

Validation Helpers
------------------

.. code-block:: python

   from base_attentive.validation import (
       validate_model_inputs,
       maybe_reduce_quantiles_bh,
       ensure_bh1,
   )

   # Check shapes and return normalised tensors
   static, dynamic, future = validate_model_inputs(
       [x_static, x_dynamic, x_future],
       static_input_dim=4,
       dynamic_input_dim=8,
       verbose=1,
   )

   # Reduce quantile output to (batch, horizon, output_dim)
   point_pred = maybe_reduce_quantiles_bh(predictions)

   # Ensure (batch, horizon, 1) shape
   reshaped = ensure_bh1(predictions)

----

Accelerated Inference (TensorFlow)
------------------------------------

Wrap repeated inference with ``make_fast_predict_fn`` to compile it once
with ``tf.function``:

.. code-block:: python

   from base_attentive import make_fast_predict_fn
   import numpy as np

   fast_predict = make_fast_predict_fn(
       model,
       warmup_inputs=[x_static, x_dynamic, x_future],
   )
   predictions = fast_predict([x_static, x_dynamic, x_future])

Keep input shapes stable across calls for best tracing performance.  For
training, ``model.compile(..., jit_compile="auto")`` may also accelerate
TensorFlow graphs.

----

Runtime Backend Utilities
--------------------------

.. code-block:: python

   from base_attentive import (
       get_backend,
       get_available_backends,
       detect_available_backends,
       select_best_backend,
   )

   print(get_backend())              # "tensorflow"
   print(get_available_backends())   # ["tensorflow", "torch"]

   info = detect_available_backends()
   for name, details in info.items():
       print(f"{name}: {details.get('version')}")

   best = select_best_backend(require_supported=True)

----

Complete Parameter Reference
------------------------------

For the full list of every accepted keyword and its type, range, and
default value, see :doc:`configuration_guide`.

----

See Also
--------

- :doc:`architecture_guide` — How the v2 registry / assembly system works
- :doc:`configuration_guide` — Full parameter reference
- :doc:`applications` — Domain-specific usage patterns
- :doc:`backends/index` — Backend selection and device management
- :doc:`api_reference` — Complete API docs
