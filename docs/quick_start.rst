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

   import tensorflow as tf
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
   lookback = 100  # Historical time steps

   static_features = np.random.randn(batch_size, 4).astype('float32')
   dynamic_features = np.random.randn(batch_size, lookback, 8).astype('float32')
   future_features = np.random.randn(batch_size, 24, 6).astype('float32')

   targets = np.random.randn(batch_size, 24, 2).astype('float32')

   # 3. Compile model
   model.compile(
       optimizer='adam',
       loss='mse',
       metrics=['mae']
   )

   # 4. Train model
   model.fit(
       [static_features, dynamic_features, future_features],
       targets,
       epochs=10,
       batch_size=32,
       verbose=1
   )

   # 5. Make predictions
   predictions = model.predict([static_features, dynamic_features, future_features])
   print(f"Predictions shape: {predictions.shape}")  # (32, 24, 2)

Understanding Inputs
--------------------

**Static Features** (batch_size, static_dim)

These are time-invariant properties like:
- Geographic coordinates
- Site elevation
- Terrain characteristics
- Installation date

.. code-block:: python

   static = np.array([
       [40.7128, -74.0060, 10, 2020],  # NYC coordinates, elevation, year
       [34.0522, -118.2437, 285, 2019],  # LA coordinates, elevation, year
   ], dtype='float32')
   # Shape: (2, 4)  # batch=2, static_dim=4

**Dynamic Features** (batch_size, time_steps, dynamic_dim)

Historical time series like:
- Temperature
- Humidity
- Solar radiation
- Wind speed

.. code-block:: python

   dynamic = np.random.randn(2, 100, 8).astype('float32')
   # Shape: (2, 100, 8)  # batch=2, 100 past steps, 8 features

**Future Features** (batch_size, forecast_horizon, future_dim)

Known future values like:
- Weather forecast
- Calendar features
- Scheduled maintenance

.. code-block:: python

   future = np.random.randn(2, 24, 6).astype('float32')
   # Shape: (2, 24, 6)  # batch=2, 24-step horizon, 6 features

Output Formats
--------------

Point Forecast
~~~~~~~~~~~~~~

By default, returns point predictions:

.. code-block:: python

   predictions = model([static, dynamic, future])
   print(predictions.shape)  # (32, 24, 2)
   #                          (batch, horizon, output_dim)

Probabilistic Forecasts with Quantiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Include quantiles for uncertainty estimates:

.. code-block:: python

   model_quantile = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       quantiles=[0.1, 0.5, 0.9],  # Lower, median, upper
   )

   predictions = model_quantile([static, dynamic, future])
   print(predictions.shape)  # (32, 24, 3, 2)
   #                          (batch, horizon, quantiles, output_dim)

   lower = predictions[:, :, 0, :]    # 10th percentile
   median = predictions[:, :, 1, :]   # 50th percentile
   upper = predictions[:, :, 2, :]    # 90th percentile

Model Modes
-----------

Hybrid Mode (Default)
~~~~~~~~~~~~~~~~~~~~~

Combines an LSTM encoder with attention:

.. code-block:: python

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       architecture_config={
           "encoder_type": "hybrid",  # LSTM + Attention
           "decoder_attention_stack": ["cross", "hierarchical"]
       }
   )

Often used for:
- Long time series (1000+ steps)
- lower compute budgets
- resource-constrained environments

Transformer Mode
~~~~~~~~~~~~~~~~

Uses a self-attention encoder:

.. code-block:: python

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       architecture_config={
           "encoder_type": "transformer",  # Pure attention
           "decoder_attention_stack": ["cross", "attention"]
       }
   )

Often used for:
- Short to medium sequences (< 500 steps)
- full self-attention interactions
- settings where parallel evaluation is useful

Serialization
--------------

Save and Load Models
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save model
   model.save('my_model.keras')

   # Load model
   from keras import models
   loaded_model = models.load_model('my_model.keras')

   # Predictions still work
   predictions = loaded_model([static, dynamic, future])

Get/Restore Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get config dictionary
   config = model.get_config()
   print(config.keys())
   #  dict_keys(['static_input_dim', 'dynamic_input_dim', ..., 'embed_dim', ...])

   # Create new model from config
   new_model = BaseAttentive.from_config(config)

   # Models are equivalent
   print(model.get_config() == new_model.get_config())  # True

Common Patterns
---------------

Multi-Step Training
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Prepare batches
   X = [static_features, dynamic_features, future_features]
   y = targets

   # Train with validation split
   history = model.fit(
       X, y,
       validation_split=0.2,
       epochs=50,
       batch_size=32,
       callbacks=[
           tf.keras.callbacks.EarlyStopping(
               monitor='val_loss',
               patience=5,
               restore_best_weights=True
           )
       ]
   )

   # Evaluate on test set
   loss, mae = model.evaluate(X_test, y_test)
   print(f"Test Loss: {loss:.4f}, MAE: {mae:.4f}")

Prediction with Quantile Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Model with quantiles
   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=2,
       forecast_horizon=24,
       quantiles=[0.025, 0.5, 0.975],  # 95% confidence interval
   )

   # Get probabilistic forecast
   predictions = model([static, dynamic, future])

   # Extract components
   lower_ci = predictions[:, :, 0, :]   # 2.5th percentile
   point = predictions[:, :, 1, :]      # median
   upper_ci = predictions[:, :, 2, :]   # 97.5th percentile

   # Visualization
   import matplotlib.pyplot as plt

   t = np.arange(24)
   plt.fill_between(t, lower_ci[0, :, 0], upper_ci[0, :, 0], alpha=0.3)
   plt.plot(t, point[0, :, 0], 'r-', label='Median')
   plt.legend()
   plt.show()

Using BaseAttentive as a Kernel
-------------------------------

After you have a working forecast, a common next step is to keep
``BaseAttentive`` as the forecasting kernel and wrap it with a larger model.
This is a good fit when you want correction heads, anomaly-aware gating, or
domain-specific safety logic without rewriting the temporal stack.

.. code-block:: python

   import keras

   class RobustForecastModel(keras.Model):
       def __init__(self):
           super().__init__()
           self.kernel = BaseAttentive(
               static_input_dim=4,
               dynamic_input_dim=8,
               future_input_dim=6,
               output_dim=2,
               forecast_horizon=24,
           )
           self.residual_head = keras.layers.Dense(2)

       def call(self, inputs, training=False):
           _, dynamic_x, _ = inputs
           base_forecast = self.kernel(inputs, training=training)
           residual = self.residual_head(keras.ops.mean(dynamic_x, axis=1))
           residual = keras.ops.expand_dims(residual, axis=1)
           return base_forecast + residual

For a fuller wrapper and subclassing guide, see :doc:`usage`. For larger
production patterns such as ensembles, transfer learning, and
physics-guided systems, see :doc:`applications`.

Backend Selection
-----------------

.. code-block:: bash

   # Set via environment (before Python import)
   export KERAS_BACKEND=tensorflow
   python your_script.py

.. code-block:: python

   # Or set via environment variable in Python
   import os
   os.environ['KERAS_BACKEND'] = 'tensorflow'  # Before import
   from base_attentive import BaseAttentive

Supported backends:

- ``tensorflow``: current full model path
- ``jax``: experimental
- ``torch``: experimental

Faster TensorFlow Inference
---------------------------

When you are running the same forecast path repeatedly, you can create a
traced prediction function:

.. code-block:: python

   from base_attentive import make_fast_predict_fn

   fast_predict = make_fast_predict_fn(
       model,
       warmup_inputs=[static_features, dynamic_features, future_features],
   )

   predictions = fast_predict(
       [static_features, dynamic_features, future_features]
   )

This is TensorFlow-specific and is meant for inference. For training, try
``model.compile(..., jit_compile="auto")``.

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
