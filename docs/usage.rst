Usage
=====

Quick start
-----------

.. code-block:: python

   import numpy as np
   from base_attentive import BaseAttentive

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       quantiles=[0.1, 0.5, 0.9],
       embed_dim=32,
       attention_units=64,
       num_heads=8,
       dropout_rate=0.15,
   )

   batch_size = 32
   x_static  = np.random.randn(batch_size, 4).astype('float32')
   x_dynamic = np.random.randn(batch_size, 10, 8).astype('float32')
   x_future  = np.random.randn(batch_size, 24, 6).astype('float32')

   predictions = model([x_static, x_dynamic, x_future])

Input contract
--------------

``BaseAttentive`` expects up to three inputs in this order:

1. static features
2. dynamic historical features
3. known future covariates

Shape contract:

- static:  ``(batch, static_input_dim)``
- dynamic: ``(batch, past_steps, dynamic_input_dim)``
- future:  ``(batch, forecast_horizon, future_input_dim)``

If you pass a single tensor to the validation helpers, it is treated as the
static slot and the missing slots are normalized to ``None``.

Output shapes
-------------

Point-forecast output:

.. code-block:: text

   (batch, forecast_horizon, output_dim)

Quantile output:

.. code-block:: text

   (batch, forecast_horizon, num_quantiles, output_dim)

Complete Parameter Reference
-----------------------------

.. list-table:: Required Parameters
   :header-rows: 1
   :widths: 22 12 12 54

   * - Parameter
     - Type
     - Constraints
     - Description
   * - ``static_input_dim``
     - int
     - >= 0
     - Number of static (time-invariant) features
   * - ``dynamic_input_dim``
     - int
     - >= 1
     - Number of historical time-series features
   * - ``future_input_dim``
     - int
     - >= 0
     - Number of known future covariate features
   * - ``output_dim``
     - int
     - >= 1
     - Number of output variables (default: 1)
   * - ``forecast_horizon``
     - int
     - >= 1
     - Number of future steps to predict (default: 1)

.. list-table:: Architecture Parameters
   :header-rows: 1
   :widths: 22 10 10 58

   * - Parameter
     - Type
     - Default
     - Description
   * - ``embed_dim``
     - int
     - 32
     - Shared embedding dimension
   * - ``hidden_units``
     - int
     - 64
     - Dense hidden layer width
   * - ``lstm_units``
     - int
     - 64
     - LSTM hidden size (hybrid mode)
   * - ``attention_units``
     - int
     - 32
     - Attention projection dimension
   * - ``num_heads``
     - int
     - 4
     - Multi-head attention heads
   * - ``num_encoder_layers``
     - int
     - 2
     - Stacked encoder layer count
   * - ``max_window_size``
     - int
     - 10
     - Maximum dynamic time window size
   * - ``memory_size``
     - int
     - 100
     - Memory bank size for memory-augmented attention

.. list-table:: Temporal Aggregation
   :header-rows: 1
   :widths: 22 10 12 56

   * - Parameter
     - Type
     - Default
     - Description
   * - ``scales``
     - list[int] or 'auto'
     - None
     - LSTM temporal scales, e.g. ``[1, 2, 4]``
   * - ``multi_scale_agg``
     - str
     - 'last'
     - Merge multi-scale outputs: ``'last'``, ``'average'``, ``'flatten'``, ``'sum'``, ``'concat'``
   * - ``final_agg``
     - str
     - 'last'
     - Final sequence aggregation: ``'last'``, ``'average'``, ``'flatten'``

.. list-table:: Regularization
   :header-rows: 1
   :widths: 22 10 10 58

   * - Parameter
     - Type
     - Default
     - Description
   * - ``dropout_rate``
     - float
     - 0.1
     - Dropout probability [0, 1]
   * - ``activation``
     - str
     - 'relu'
     - ``'relu'``, ``'elu'``, ``'selu'``, ``'sigmoid'``, ``'tanh'``, ``'linear'``, ``'gelu'``, ``'swish'``
   * - ``use_batch_norm``
     - bool
     - False
     - Apply batch normalization
   * - ``use_residuals``
     - bool
     - True
     - Use residual connections

.. list-table:: Feature Processing
   :header-rows: 1
   :widths: 22 10 10 58

   * - Parameter
     - Type
     - Default
     - Description
   * - ``use_vsn``
     - bool
     - True
     - Enable Variable Selection Network
   * - ``vsn_units``
     - int or None
     - None
     - Override VSN projection units (uses ``embed_dim`` when None)
   * - ``apply_dtw``
     - bool
     - True
     - Apply Dynamic Time Warping alignment

.. list-table:: Configuration
   :header-rows: 1
   :widths: 22 10 12 56

   * - Parameter
     - Type
     - Default
     - Description
   * - ``objective``
     - str
     - 'hybrid'
     - Encoder type: ``'hybrid'`` or ``'transformer'``
   * - ``mode``
     - str or None
     - None
     - Mode shortcut: ``'tft'``, ``'pihal'``, ``'tft_like'``, ``'pihal_like'``
   * - ``attention_levels``
     - str, list, int, None
     - None
     - Decoder attention stack control
   * - ``quantiles``
     - list[float] or None
     - None
     - Quantile levels for probabilistic output
   * - ``architecture_config``
     - dict or None
     - None
     - Structural overrides (highest precedence)
   * - ``verbose``
     - int
     - 0
     - Logging verbosity

Architecture configuration
--------------------------

Use ``architecture_config`` for structural choices:

.. code-block:: python

   transformer_config = {
       "encoder_type": "transformer",
       "decoder_attention_stack": ["cross", "hierarchical"],
       "feature_processing": "dense",
   }

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       architecture_config=transformer_config,
   )

Common options:

- ``encoder_type``: ``"hybrid"`` or ``"transformer"``
- ``decoder_attention_stack``: list from ``["cross", "hierarchical", "memory"]``
- ``feature_processing``: ``"vsn"`` or ``"dense"``

Operational Mode (``mode`` parameter)
--------------------------------------

.. code-block:: python

   model = BaseAttentive(..., mode="tft")    # TFT-like profile
   model = BaseAttentive(..., mode="pihal")  # PIHALNet-like profile

Attention Levels
----------------

.. code-block:: python

   model = BaseAttentive(..., attention_levels=None)              # all three
   model = BaseAttentive(..., attention_levels="cross")           # cross only
   model = BaseAttentive(..., attention_levels=["cross", "memory"])
   model = BaseAttentive(..., attention_levels=1)                 # cross (int shortcut)
   model = BaseAttentive(..., attention_levels=2)                 # hierarchical
   model = BaseAttentive(..., attention_levels=3)                 # memory

Multi-Scale Aggregation
------------------------

.. code-block:: python

   model = BaseAttentive(
       ...,
       scales=[1, 2, 4],
       multi_scale_agg="average",
       final_agg="last",
   )

V2 Schema-based Configuration
-------------------------------

For programmatic / backend-neutral construction use ``BaseAttentiveSpec``:

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
       dropout_rate=0.1,
       head_type="point",
       backend_name="tensorflow",
   )

   # Override individual component registry keys
   spec_custom = BaseAttentiveSpec(
       static_input_dim=4, dynamic_input_dim=8, future_input_dim=6,
       components=BaseAttentiveComponentSpec(sequence_pooling="pool.last"),
   )

Using BaseAttentive as a Kernel
--------------------------------

Wrapper Pattern (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import keras
   from base_attentive import BaseAttentive

   class RobustForecastModel(keras.Model):
       def __init__(self, forecast_horizon=24, output_dim=1):
           super().__init__()
           self.kernel = BaseAttentive(
               static_input_dim=4, dynamic_input_dim=8, future_input_dim=6,
               output_dim=output_dim, forecast_horizon=forecast_horizon,
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
           base_forecast = self.kernel([static_x, dynamic_x, future_x],
                                       training=training)
           context  = self.context_pool(dynamic_x)
           residual = self.residual_head(context, training=training)
           gate     = keras.ops.expand_dims(self.gate(context), axis=1)
           return base_forecast + gate * residual

Direct Inheritance (Advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from base_attentive import BaseAttentive
   import keras

   class PhysicsAwareBaseAttentive(BaseAttentive):
       def __init__(self, physics_weight=0.1, **kwargs):
           super().__init__(**kwargs)
           self.physics_weight  = physics_weight
           self.constraint_head = keras.layers.Dense(self.output_dim)

       def call(self, inputs, training=False):
           base_forecast  = super().call(inputs, training=training)
           _, _, future_x = inputs
           correction     = self.constraint_head(keras.ops.mean(future_x, axis=1))
           correction     = keras.ops.expand_dims(correction, axis=1)
           return base_forecast + self.physics_weight * correction

       def get_config(self):
           config = super().get_config()
           config.update({"physics_weight": self.physics_weight})
           return config

Serialization and reconfiguration
-----------------------------------

.. code-block:: python

   config  = model.get_config()
   cloned  = BaseAttentive.from_config(config)

   # Variant without mutating the original
   transformer_model = model.reconfigure({"encoder_type": "transformer"})

Validation helpers
------------------

.. code-block:: python

   from base_attentive.validation import (
       validate_model_inputs,
       maybe_reduce_quantiles_bh,
       ensure_bh1,
   )

   static, dynamic, future = validate_model_inputs(
       [x_static, x_dynamic, x_future],
       static_input_dim=4,
       dynamic_input_dim=8,
       verbose=1,
   )

   reduced  = maybe_reduce_quantiles_bh(predictions)
   reshaped = ensure_bh1(predictions)

Accelerated inference
----------------------

.. code-block:: python

   from base_attentive import make_fast_predict_fn

   fast_predict = make_fast_predict_fn(
       model,
       warmup_inputs=[x_static, x_dynamic, x_future],
   )
   predictions = fast_predict([x_static, x_dynamic, x_future])

Development workflow
---------------------

.. code-block:: bash

   make test-fast
   make lint
   make format
   make build
