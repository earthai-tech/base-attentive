Quick Start Guide
=================

Installation
------------

Install with the TensorFlow backend:

.. code-block:: bash

   pip install "base-attentive[tensorflow]"

Or from source for development:

.. code-block:: bash

   git clone https://github.com/earthai-tech/base-attentive.git
   cd base-attentive
   pip install -e ".[dev,tensorflow]"

Your First Model
----------------

This minimal example covers model creation, fitting, and prediction:

.. code-block:: python

   import numpy as np
   from base_attentive import BaseAttentive

   # 1. Create model instance
   model = BaseAttentive(
       static_input_dim=4,         # 4 static features
       dynamic_input_dim=8,        # 8 dynamic features
       future_input_dim=6,         # 6 future features
       output_dim=2,               # 2 output variables
       forecast_horizon=24,        # 24 time steps ahead
   )

   # 2. Prepare data
   batch_size = 32
   lookback   = 100

   static_features  = np.random.randn(batch_size, 4).astype('float32')
   dynamic_features = np.random.randn(batch_size, lookback, 8).astype('float32')
   future_features  = np.random.randn(batch_size, 24, 6).astype('float32')
   targets          = np.random.randn(batch_size, 24, 2).astype('float32')

   # 3. Compile and train
   model.compile(optimizer='adam', loss='mse', metrics=['mae'])
   model.fit(
       [static_features, dynamic_features, future_features],
       targets,
       epochs=10,
       batch_size=32,
       verbose=1,
   )

   # 4. Make predictions
   preds = model.predict([static_features, dynamic_features, future_features])
   print(f"Shape: {preds.shape}")  # (32, 24, 2)

Understanding Inputs
--------------------

**Static Features** ``(batch_size, static_dim)``

Time-invariant properties:

.. code-block:: python

   static = np.array([
       [40.7128, -74.0060, 10, 2020],   # NYC: lat, lon, elev, year
       [34.0522, -118.243,  285, 2019], # LA
   ], dtype='float32')
   # Shape: (2, 4)

**Dynamic Features** ``(batch_size, time_steps, dynamic_dim)``

Historical time series:

.. code-block:: python

   dynamic = np.random.randn(2, 100, 8).astype('float32')
   # Shape: (2, 100, 8)

**Future Features** ``(batch_size, forecast_horizon, future_dim)``

Known future values:

.. code-block:: python

   future = np.random.randn(2, 24, 6).astype('float32')
   # Shape: (2, 24, 6)

Output Formats
--------------

Point Forecast
~~~~~~~~~~~~~~

By default, returns single point predictions:

.. code-block:: python

   predictions = model([static, dynamic, future])
   print(predictions.shape)  # (32, 24, 2) — (batch, horizon, output_dim)

Probabilistic Forecasts with Quantiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Include ``quantiles`` for uncertainty estimates:

.. code-block:: python

   model_q = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       quantiles=[0.1, 0.5, 0.9],
   )

   preds = model_q([static, dynamic, future])
   print(preds.shape)  # (32, 24, 3, 2) — (batch, horizon, quantiles, output_dim)

   lower  = preds[:, :, 0, :]   # 10th percentile
   median = preds[:, :, 1, :]   # 50th percentile
   upper  = preds[:, :, 2, :]   # 90th percentile

Encoder Objective
-----------------

Use ``objective`` to choose the encoder design:

Hybrid (Default)
~~~~~~~~~~~~~~~~

Multi-scale LSTM with attention — suitable for longer sequences:

.. code-block:: python

   model = BaseAttentive(..., objective="hybrid")

Transformer
~~~~~~~~~~~

Pure self-attention — better parallelism on shorter sequences:

.. code-block:: python

   model = BaseAttentive(..., objective="transformer")

Operational Mode Shortcuts
--------------------------

The ``mode`` parameter applies a pre-configured combination of settings:

.. code-block:: python

   # TFT-like (Temporal Fusion Transformer style)
   model = BaseAttentive(
       static_input_dim=4, dynamic_input_dim=8, future_input_dim=6,
       output_dim=1, forecast_horizon=24,
       mode="tft",
   )

   # PIHALNet-like (Physics-Informed HAL style)
   model = BaseAttentive(
       static_input_dim=4, dynamic_input_dim=8, future_input_dim=6,
       output_dim=1, forecast_horizon=24,
       mode="pihal",
   )

Valid values: ``"tft"``, ``"tft_like"``, ``"pihal"``, ``"pihal_like"``,
or ``None`` (default — manual configuration).

Attention Levels
----------------

Use ``attention_levels`` to declare which decoder attention mechanisms to enable:

.. code-block:: python

   # All three (default when attention_levels=None)
   model = BaseAttentive(..., attention_levels=None)

   # Cross-attention only
   model = BaseAttentive(..., attention_levels="cross")

   # Cross + hierarchical
   model = BaseAttentive(..., attention_levels=["cross", "hierarchical"])

   # Integer shortcuts: 1=cross, 2=hierarchical, 3=memory
   model = BaseAttentive(..., attention_levels=1)

Multi-Scale Aggregation
-----------------------

Control how temporal features are aggregated across LSTM scales:

.. code-block:: python

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       scales=[1, 2, 4],           # 3 temporal resolutions
       multi_scale_agg="last",     # 'last', 'average', 'flatten', 'concat'
       final_agg="last",           # final sequence step aggregation
   )

Serialization
-------------

Save and Load Models
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   model.save('my_model.keras')

   from keras import models
   loaded = models.load_model('my_model.keras')
   preds  = loaded([static, dynamic, future])

Get / Restore Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   config    = model.get_config()
   new_model = BaseAttentive.from_config(config)

   # Create a variant without mutating the original
   bigger = model.reconfigure({"encoder_type": "transformer"})

Common Patterns
---------------

Training with Early Stopping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import keras

   model.compile(optimizer='adam', loss='mse', metrics=['mae'])
   history = model.fit(
       [static_features, dynamic_features, future_features],
       targets,
       validation_split=0.2,
       epochs=50,
       batch_size=32,
       callbacks=[
           keras.callbacks.EarlyStopping(
               monitor='val_loss', patience=5, restore_best_weights=True,
           )
       ],
   )

Confidence Interval Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   model_ci = BaseAttentive(
       static_input_dim=4, dynamic_input_dim=8, future_input_dim=6,
       output_dim=2, forecast_horizon=24,
       quantiles=[0.025, 0.5, 0.975],
   )
   preds    = model_ci([static, dynamic, future])
   lower_ci = preds[:, :, 0, :]
   point    = preds[:, :, 1, :]
   upper_ci = preds[:, :, 2, :]

   t = np.arange(24)
   plt.fill_between(t, lower_ci[0, :, 0], upper_ci[0, :, 0],
                    alpha=0.3, label='95% CI')
   plt.plot(t, point[0, :, 0], 'r-', label='Median')
   plt.legend()
   plt.show()

Using BaseAttentive as a Kernel
--------------------------------

Wrap ``BaseAttentive`` inside a larger model to add domain-specific logic:

.. code-block:: python

   import keras
   from base_attentive import BaseAttentive

   class RobustForecastModel(keras.Model):
       def __init__(self):
           super().__init__()
           self.kernel = BaseAttentive(
               static_input_dim=4, dynamic_input_dim=8, future_input_dim=6,
               output_dim=2, forecast_horizon=24,
           )
           self.residual_head = keras.layers.Dense(2)

       def call(self, inputs, training=False):
           _, dynamic_x, _ = inputs
           base_forecast = self.kernel(inputs, training=training)
           residual = self.residual_head(keras.ops.mean(dynamic_x, axis=1))
           return base_forecast + keras.ops.expand_dims(residual, axis=1)

For a fuller guide see :doc:`usage`. For ensemble and physics-guided patterns
see :doc:`applications`.

Backend Selection
-----------------

.. code-block:: bash

   export KERAS_BACKEND=tensorflow
   python your_script.py

.. code-block:: python

   import os
   os.environ['KERAS_BACKEND'] = 'tensorflow'
   from base_attentive import BaseAttentive

Supported backends: ``tensorflow`` (stable), ``jax`` (experimental),
``torch`` (experimental).

Faster TensorFlow Inference
---------------------------

.. code-block:: python

   from base_attentive import make_fast_predict_fn

   fast_predict = make_fast_predict_fn(
       model,
       warmup_inputs=[static_features, dynamic_features, future_features],
   )
   predictions = fast_predict([static_features, dynamic_features, future_features])

This is TensorFlow-specific and wraps inference with ``tf.function``.

Next Steps
----------

1. **Explore configurations**: See :doc:`configuration_guide`
2. **Understand architecture**: See :doc:`architecture_guide`
3. **API reference**: See :doc:`api_reference`
4. **Extended usage and kernel patterns**: See :doc:`usage`
5. **Application blueprints**: See :doc:`applications`

See Also
--------

- :ref:`genindex`
- :ref:`search`
