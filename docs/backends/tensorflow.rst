TensorFlow Backend
==================

TensorFlow is the default backend for BaseAttentive and the most thoroughly
tested path.

Installation
------------

.. code-block:: bash

   pip install base-attentive[tensorflow]

Or manually:

.. code-block:: bash

   pip install tensorflow>=2.15.0

Selecting TensorFlow
--------------------

.. code-block:: python

   import os
   os.environ["KERAS_BACKEND"] = "tensorflow"

   from base_attentive import BaseAttentive

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
   )

Quick Training Example
----------------------

.. code-block:: python

   import numpy as np
   from base_attentive import BaseAttentive

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
       embed_dim=32,
       num_heads=4,
   )

   model.compile(optimizer="adam", loss="mse")

   x_static  = np.random.randn(32, 4).astype("float32")
   x_dynamic = np.random.randn(32, 100, 8).astype("float32")
   x_future  = np.random.randn(32, 24, 6).astype("float32")
   y         = np.random.randn(32, 24, 1).astype("float32")

   model.fit([x_static, x_dynamic, x_future], y, epochs=3)

Accelerated Inference
---------------------

Wrap repeated inference with ``make_fast_predict_fn`` to compile it with
``tf.function``:

.. code-block:: python

   from base_attentive import BaseAttentive, make_fast_predict_fn
   import numpy as np

   model = BaseAttentive(
       static_input_dim=4, dynamic_input_dim=8, future_input_dim=6,
       output_dim=1, forecast_horizon=24,
   )

   x_static  = np.random.randn(32, 4).astype("float32")
   x_dynamic = np.random.randn(32, 100, 8).astype("float32")
   x_future  = np.random.randn(32, 24, 6).astype("float32")

   fast_predict = make_fast_predict_fn(
       model,
       warmup_inputs=[x_static, x_dynamic, x_future],
   )
   predictions = fast_predict([x_static, x_dynamic, x_future])

.. note::

   Keep batch and sequence shapes stable across calls for best results.
   For training, ``model.compile(..., jit_compile="auto")`` may also help.

Compatibility Check
-------------------

.. code-block:: python

   from base_attentive.backend import check_tensorflow_compatibility

   ok, msg = check_tensorflow_compatibility()
   print(msg)

Minimum required version: **TensorFlow 2.15.0**.

See Also
--------

- :doc:`index` — Backend overview and selection
- :doc:`../installation` — Full installation instructions
