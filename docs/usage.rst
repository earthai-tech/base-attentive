Usage
=====

Quick start
-----------

.. code-block:: python

   import tensorflow as tf
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
   x_static = tf.random.normal([batch_size, 4])
   x_dynamic = tf.random.normal([batch_size, 10, 8])
   x_future = tf.random.normal([batch_size, 24, 6])

   predictions = model([x_static, x_dynamic, x_future])

Architecture configuration
--------------------------

The model can be configured through ``architecture_config``:

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

Common options
--------------

- ``encoder_type``: ``"hybrid"`` or ``"transformer"``
- ``decoder_attention_stack``: attention layers such as
  ``["cross", "hierarchical", "memory"]``
- ``feature_processing``: ``"vsn"`` or ``"dense"``

Development workflow
--------------------

The repository ships with a top-level ``Makefile`` for common tasks:

.. code-block:: bash

   make test-fast
   make lint
   make format
   make build
