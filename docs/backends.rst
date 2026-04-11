Backend Guide
=============

BaseAttentive uses a Keras 3 runtime layer. At present, the full model path is
available with TensorFlow, while JAX and Torch remain experimental.

Support status
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Backend
     - Status
     - Notes
   * - TensorFlow
     - Current full path
     - The full ``BaseAttentive`` model is the main target of the current examples and tests.
   * - JAX
     - Experimental
     - Runtime abstraction is available, but the full model path is still under evaluation.
   * - Torch
     - Experimental
     - Runtime abstraction is available, but the full model path is still under evaluation.

Selecting a backend
-------------------

Set the backend before importing ``base_attentive`` or ``keras``:

.. code-block:: bash

   export KERAS_BACKEND=tensorflow
   export BASE_ATTENTIVE_BACKEND=tensorflow

The package also exposes helper functions for runtime inspection:

.. code-block:: python

   from base_attentive import get_backend
   from base_attentive import get_available_backends
   from base_attentive import get_backend_capabilities
   from base_attentive import set_backend

   set_backend("tensorflow")
   backend = get_backend()
   print(backend.name)
   print(get_available_backends())
   print(get_backend_capabilities("jax"))

How backend resolution works
----------------------------

When you call ``get_backend()`` without an explicit name, BaseAttentive checks
backends in this order:

1. ``BASE_ATTENTIVE_BACKEND``
2. ``KERAS_BACKEND``
3. the backend previously set in the current Python process
4. ``tensorflow``

Current status
--------------

Use TensorFlow for training, testing, and model serialization at the moment.
JAX and Torch can be useful for backend experiments, but the full
``BaseAttentive`` model should still be treated as exploratory on those
runtimes.

Accelerated Inference on TensorFlow
-----------------------------------

If you are serving forecasts repeatedly, you can create a traced inference
callable with ``make_fast_predict_fn``:

.. code-block:: python

   from base_attentive import BaseAttentive, make_fast_predict_fn

   model = BaseAttentive(...)
   fast_predict = make_fast_predict_fn(
       model,
       warmup_inputs=[x_static, x_dynamic, x_future],
   )

   predictions = fast_predict([x_static, x_dynamic, x_future])

This helper wraps inference with ``tf.function`` and uses ``training=False``.
For best results, keep batch and sequence shapes relatively stable. For
training workloads, you can also try ``model.compile(..., jit_compile="auto")``.
