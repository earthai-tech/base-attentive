=======================
Configuration Guide
=======================

Parameter Reference
====================

Required Parameters
-------------------

.. code-block:: python

   from base_attentive import BaseAttentive

   model = BaseAttentive(
       static_input_dim=4,      # Number of static features
       dynamic_input_dim=8,     # Number of dynamic features
       future_input_dim=6,      # Number of future features
       output_dim=2,            # Number of output variables
       forecast_horizon=24,     # Forecast horizon (steps)
   )

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
     - Static feature dimension (0 = no static input)
   * - ``dynamic_input_dim``
     - int
     - >= 1
     - Dynamic (historical) feature dimension
   * - ``future_input_dim``
     - int
     - >= 0
     - Future covariate dimension (0 = no future input)
   * - ``output_dim``
     - int
     - >= 1
     - Number of output variables
   * - ``forecast_horizon``
     - int
     - >= 1
     - Forecast length in time steps

Architectural Parameters
------------------------

.. code-block:: python

   model = BaseAttentive(
       static_input_dim=4, dynamic_input_dim=8, future_input_dim=6,
       output_dim=2, forecast_horizon=24,
       embed_dim=32,
       hidden_units=64,
       lstm_units=64,
       attention_units=32,
       num_heads=4,
       num_encoder_layers=2,
       max_window_size=10,
       memory_size=100,
   )

.. list-table:: Architectural Parameters
   :header-rows: 1
   :widths: 22 10 10 15 43

   * - Parameter
     - Type
     - Default
     - Range
     - Description
   * - ``embed_dim``
     - int
     - 32
     - [8, 512]
     - Shared embedding dimension
   * - ``hidden_units``
     - int
     - 64
     - [16, 1024]
     - Dense hidden layer width
   * - ``lstm_units``
     - int
     - 64
     - [16, 1024]
     - LSTM hidden size (hybrid mode)
   * - ``attention_units``
     - int
     - 32
     - [16, 1024]
     - Attention projection dimension
   * - ``num_heads``
     - int
     - 4
     - [1, 16]
     - Multi-head attention heads (``embed_dim`` must be divisible by ``num_heads``)
   * - ``num_encoder_layers``
     - int
     - 2
     - [1, 12]
     - Stacked encoder layer count
   * - ``max_window_size``
     - int
     - 10
     - [1, oo)
     - Maximum dynamic time window size
   * - ``memory_size``
     - int
     - 100
     - [1, oo)
     - Memory bank size for memory-augmented attention

Temporal Aggregation Parameters
---------------------------------

.. code-block:: python

   model = BaseAttentive(
       ...,
       scales=[1, 2, 4],
       multi_scale_agg="last",
       final_agg="last",
   )

.. list-table:: Temporal Aggregation
   :header-rows: 1
   :widths: 22 12 12 54

   * - Parameter
     - Type
     - Default
     - Description
   * - ``scales``
     - list[int] / 'auto' / None
     - None
     - LSTM sub-sampling strides. ``None`` uses single scale ``[1]``. ``'auto'`` selects automatically.
   * - ``multi_scale_agg``
     - str
     - 'last'
     - Merge multi-scale outputs: ``'last'``, ``'average'``, ``'flatten'``, ``'sum'``, ``'concat'``
   * - ``final_agg``
     - str
     - 'last'
     - Final temporal aggregation: ``'last'``, ``'average'``, ``'flatten'``

Regularization Parameters
--------------------------

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

Feature Processing Parameters
-------------------------------

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
     - VSN projection size (defaults to ``embed_dim``)
   * - ``apply_dtw``
     - bool
     - True
     - Apply Dynamic Time Warping alignment

Configuration / Routing Parameters
-------------------------------------

.. list-table:: Configuration / Routing
   :header-rows: 1
   :widths: 22 12 12 54

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
     - Mode shortcut: ``'tft'``, ``'tft_like'``, ``'pihal'``, ``'pihal_like'``, or ``None``
   * - ``attention_levels``
     - str / list / int / None
     - None
     - Decoder attention stack control (see below)
   * - ``quantiles``
     - list[float] or None
     - None
     - Enables probabilistic output
   * - ``architecture_config``
     - dict or None
     - None
     - Structural overrides (highest precedence)
   * - ``verbose``
     - int
     - 0
     - Logging verbosity

Architecture Configuration
==========================

Use ``architecture_config`` for structural choices:

.. code-block:: python

   config = {
       "encoder_type": "hybrid",
       "decoder_attention_stack": ["cross", "hierarchical", "memory"],
       "feature_processing": "vsn",
   }

   model = BaseAttentive(
       static_input_dim=4, dynamic_input_dim=8, future_input_dim=6,
       output_dim=2, forecast_horizon=24,
       architecture_config=config,
   )

Attention Level Shortcuts
--------------------------

.. code-block:: python

   model = BaseAttentive(..., attention_levels=None)              # all three
   model = BaseAttentive(..., attention_levels="cross")           # string
   model = BaseAttentive(..., attention_levels=["cross", "memory"]) # list
   model = BaseAttentive(..., attention_levels=1)  # 1=cross, 2=hier, 3=memory

V2 Schema Configuration
========================

For programmatic, backend-neutral construction use ``BaseAttentiveSpec``:

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
           sequence_pooling="pool.last",
       ),
   )

Configuration Presets
=====================

Minimal Configuration
---------------------

.. code-block:: python

   MINIMAL = dict(embed_dim=8, hidden_units=16, lstm_units=16,
                  attention_units=16, num_heads=1, dropout_rate=0.1)

   model = BaseAttentive(
       static_input_dim=4, dynamic_input_dim=8, future_input_dim=6,
       output_dim=2, forecast_horizon=24, **MINIMAL,
   )

Standard Configuration
----------------------

.. code-block:: python

   STANDARD = dict(embed_dim=32, hidden_units=64, lstm_units=64,
                   attention_units=32, num_heads=4, dropout_rate=0.1,
                   use_residuals=True)

Large Configuration
-------------------

.. code-block:: python

   LARGE = dict(embed_dim=128, hidden_units=256, lstm_units=256,
                attention_units=128, num_heads=8, dropout_rate=0.3,
                use_batch_norm=True, use_residuals=True)

Hybrid Preset
-------------

.. code-block:: python

   HYBRID = dict(
       objective="hybrid",
       scales=[1, 2, 4],
       multi_scale_agg="last",
       embed_dim=32,
       num_heads=4,
       dropout_rate=0.1,
   )

Transformer Preset
------------------

.. code-block:: python

   TRANSFORMER = dict(
       objective="transformer",
       num_encoder_layers=4,
       embed_dim=64,
       num_heads=8,
       dropout_rate=0.15,
   )

Tuning Guidelines
=================

For Longer Sequences (T > 500)
--------------------------------

.. code-block:: python

   model = BaseAttentive(
       ..., objective="hybrid", scales=[1, 2, 4],
       embed_dim=32, dropout_rate=0.15,
   )

For Complex Patterns
---------------------

.. code-block:: python

   model = BaseAttentive(
       ..., embed_dim=64, num_heads=8,
       use_batch_norm=True, use_residuals=True, dropout_rate=0.2,
   )

For Fast Inference
------------------

.. code-block:: python

   model = BaseAttentive(
       ..., objective="hybrid", embed_dim=16,
       hidden_units=32, dropout_rate=0.1,
       attention_levels="cross",
   )

For Probabilistic Forecasts
-----------------------------

.. code-block:: python

   model = BaseAttentive(
       ..., quantiles=[0.1, 0.5, 0.9],
       dropout_rate=0.2, use_residuals=True,
   )

Configuration Management
=========================

Get Configuration
------------------

.. code-block:: python

   config = model.get_config()
   print(config)
   # {'static_input_dim': 4, ..., 'scales': None, 'mode': None, ...}

Create from Configuration
--------------------------

.. code-block:: python

   new_model = BaseAttentive.from_config(model.get_config())

Reconfigure Model
-----------------

.. code-block:: python

   model2 = model.reconfigure({"encoder_type": "transformer"})

Common Mistakes
===============

Mismatched input dimensions
----------------------------

.. code-block:: python

   # Wrong
   model = BaseAttentive(static_input_dim=4, ...)
   static = np.random.randn(32, 5)   # 5 features but model expects 4

   # Correct
   static = np.random.randn(32, 4)

num_heads must divide embed_dim
---------------------------------

.. code-block:: python

   # Wrong — 32 / 6 is not integer
   model = BaseAttentive(..., embed_dim=32, num_heads=6)

   # Correct
   model = BaseAttentive(..., embed_dim=32, num_heads=4)

See Also
========

- :doc:`quick_start` — Quick start guide
- :doc:`architecture_guide` — Architecture details
- :doc:`api_reference` — Full API reference
