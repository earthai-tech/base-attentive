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

Input contract
--------------

``BaseAttentive`` expects up to three inputs in this order:

1. static features
2. dynamic historical features
3. known future covariates

The common shape contract is:

- static: ``(batch, static_input_dim)``
- dynamic: ``(batch, past_steps, dynamic_input_dim)``
- future: ``(batch, forecast_horizon, future_input_dim)``

If you pass a single tensor to the validation helpers, it is treated as the
static slot and the missing slots are normalized to ``None``.

Output shapes
-------------

Point-forecast output typically has shape:

.. code-block:: text

   (batch, forecast_horizon, output_dim)

Quantile output typically has shape:

.. code-block:: text

   (batch, forecast_horizon, num_quantiles, output_dim)

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

Using BaseAttentive as a Kernel
-------------------------------

``BaseAttentive`` does not need to be the final model you serve. In many
projects it is most useful as a forecasting kernel: it handles temporal
representation learning, encoder-decoder fusion, and multi-horizon prediction,
while you add domain-specific logic around it.

This pattern is especially helpful when you want to:

- add a correction head without rewriting the temporal stack
- attach domain constraints or rule-based penalties
- combine forecasting with auxiliary tasks such as anomaly detection
- keep a stable forecasting core while experimenting with downstream heads

Wrapper Pattern (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Composition is usually the safest approach. You keep ``BaseAttentive`` as an
internal kernel and build a larger model around it.

.. code-block:: python

   import keras
   from base_attentive import BaseAttentive

   class RobustForecastModel(keras.Model):
       def __init__(self, forecast_horizon=24, output_dim=1):
           super().__init__()
           self.kernel = BaseAttentive(
               static_input_dim=4,
               dynamic_input_dim=8,
               future_input_dim=6,
               output_dim=output_dim,
               forecast_horizon=forecast_horizon,
               quantiles=None,  # keep shapes simple for the correction head
           )
           self.context_pool = keras.layers.GlobalAveragePooling1D()
           self.residual_head = keras.Sequential(
               [
                   keras.layers.Dense(64, activation="relu"),
                   keras.layers.RepeatVector(forecast_horizon),
                   keras.layers.Dense(output_dim),
               ]
           )
           self.gate = keras.layers.Dense(output_dim, activation="sigmoid")

       def call(self, inputs, training=False):
           static_x, dynamic_x, future_x = inputs

           base_forecast = self.kernel(
               [static_x, dynamic_x, future_x],
               training=training,
           )

           context = self.context_pool(dynamic_x)
           residual = self.residual_head(context, training=training)
           gate = self.gate(context, training=training)
           gate = keras.ops.expand_dims(gate, axis=1)

           return base_forecast + gate * residual

The advantage of this pattern is that the forecasting kernel stays reusable:

- you can swap ``architecture_config`` without changing the outer model
- kernel serialization stays close to the base implementation
- extra robustness logic remains small, testable, and domain-specific

Direct Inheritance (Advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Direct subclassing is useful when your custom model still follows the same
three-input contract and you want one model class that extends the base
behavior.

.. code-block:: python

   import keras
   from base_attentive import BaseAttentive

   class PhysicsAwareBaseAttentive(BaseAttentive):
       def __init__(self, physics_weight=0.1, **kwargs):
           super().__init__(**kwargs)
           self.physics_weight = physics_weight
           self.constraint_head = keras.layers.Dense(self.output_dim)

       def call(self, inputs, training=False):
           base_forecast = super().call(inputs, training=training)
           _, _, future_x = inputs

           future_summary = keras.ops.mean(future_x, axis=1)
           correction = self.constraint_head(future_summary)
           correction = keras.ops.expand_dims(correction, axis=1)

           return base_forecast + self.physics_weight * correction

       def get_config(self):
           config = super().get_config()
           config.update({"physics_weight": self.physics_weight})
           return config

Use inheritance when you want the custom behavior to be part of the model
class itself. Use wrapping when you want clearer separation between the
forecasting kernel and the downstream application logic.

Practical Guidance
~~~~~~~~~~~~~~~~~~

- If you only need a post-processing or correction stage, prefer wrapping.
- If you need access to intermediate temporal features, inspect
  ``run_encoder_decoder_core`` and extend carefully.
- If you enable ``quantiles``, make sure any added head matches the forecast
  output shape.
- Keep the kernel responsible for forecasting and the outer model responsible
  for business rules, physics, safety logic, or task fusion.

Serialization and reconfiguration
---------------------------------

The model exposes a standard Keras-style configuration API:

.. code-block:: python

   config = model.get_config()
   cloned = BaseAttentive.from_config(config)

To experiment with architecture changes without mutating the original
instance, use ``reconfigure``:

.. code-block:: python

   transformer_model = model.reconfigure(
       {"encoder_type": "transformer"}
   )

Validation helpers
------------------

The ``base_attentive.validation`` module provides small backend-aware helpers
that are useful when preparing or post-processing model inputs:

- ``validate_model_inputs`` normalizes the three-slot input structure
- ``maybe_reduce_quantiles_bh`` reduces a quantile axis when present
- ``ensure_bh1`` reshapes outputs into ``(batch, horizon, 1)``

Accelerated inference
---------------------

With the TensorFlow backend, you can build a traced prediction callable for
repeated inference workloads:

.. code-block:: python

   from base_attentive import make_fast_predict_fn

   fast_predict = make_fast_predict_fn(
       model,
       warmup_inputs=[x_static, x_dynamic, x_future],
   )

   predictions = fast_predict([x_static, x_dynamic, x_future])

The helper wraps ``model(inputs, training=False)`` with ``tf.function``.
Use it when you want a reusable inference path without changing the model
class itself.

Development workflow
--------------------

The repository ships with a top-level ``Makefile`` for common tasks:

.. code-block:: bash

   make test-fast
   make lint
   make format
   make build
