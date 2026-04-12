Backend Guide
=============

BaseAttentive uses Keras 3 as its neural network runtime. The full model
path is available with TensorFlow; JAX and Torch are experimental.

Support status
--------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Backend
     - Status
     - Notes
   * - TensorFlow
     - **Stable**
     - Full ``BaseAttentive`` model path; recommended for training and deployment.
   * - JAX
     - Experimental
     - Runtime abstraction available; full model path under evaluation.
   * - Torch (PyTorch)
     - Experimental
     - Runtime abstraction available; ``TorchDeviceManager`` for GPU/MPS management. Full model path under evaluation.

Selecting a backend
-------------------

Set the backend **before** importing ``base_attentive`` or ``keras``:

.. code-block:: bash

   export KERAS_BACKEND=tensorflow
   export BASE_ATTENTIVE_BACKEND=tensorflow

Or in Python:

.. code-block:: python

   import os
   os.environ["KERAS_BACKEND"] = "tensorflow"
   from base_attentive import BaseAttentive

The package also exposes helper functions for runtime inspection and control:

.. code-block:: python

   from base_attentive import (
       get_backend,
       get_available_backends,
       get_backend_capabilities,
       set_backend,
       normalize_backend_name,
   )

   set_backend("tensorflow")
   backend = get_backend()
   print(backend.name)
   print(get_available_backends())
   print(get_backend_capabilities("jax"))
   print(normalize_backend_name("tf"))   # -> "tensorflow"

How backend resolution works
----------------------------

When you call ``get_backend()`` without an explicit name, BaseAttentive
checks in this order:

1. ``BASE_ATTENTIVE_BACKEND`` environment variable
2. ``KERAS_BACKEND`` environment variable
3. The backend previously set in the current Python process
4. ``tensorflow`` (default)

V2 Backend Detection and Selection
-----------------------------------

New in v1.0.0 — richer detection and selection utilities:

.. code-block:: python

   from base_attentive import (
       detect_available_backends,
       select_best_backend,
       ensure_default_backend,
   )

   # Inspect all installed backends
   info = detect_available_backends()
   # Returns: {'tensorflow': {'available': True, 'version': '2.14.0'}, ...}
   for name, details in info.items():
       print(f"{name}: available={details.get('available')}, "
             f"version={details.get('version')}")

   # Select the best available backend automatically
   best = select_best_backend(require_supported=True)
   print(f"Best backend: {best}")

   # Ensure a default backend is ready (optionally attempt auto-install)
   name = ensure_default_backend(auto_install=False)

Version Compatibility Checks
-------------------------------

.. code-block:: python

   from base_attentive.backend import (
       check_tensorflow_compatibility,
       check_torch_compatibility,
       get_backend_version,
       version_at_least,
   )

   # Check if installed TF version is compatible
   ok, msg = check_tensorflow_compatibility()
   print(msg)   # e.g. "TensorFlow 2.14.0 is compatible"

   # Check PyTorch compatibility (requires >= 2.0.0)
   ok, msg = check_torch_compatibility()

   # Get installed version string
   ver = get_backend_version("tensorflow")  # e.g. "2.14.0"

   # Version comparison utility
   ok = version_at_least("2.13.0", "2.12.0")  # True

Current recommendations
-----------------------

- Use **TensorFlow** for training, testing, and model serialization.
- Use **JAX** or **Torch** for backend experiments and research; treat the
  full ``BaseAttentive`` execution path as exploratory on those runtimes.

Accelerated Inference on TensorFlow
------------------------------------

Wrap the model with ``make_fast_predict_fn`` for repeated inference:

.. code-block:: python

   from base_attentive import BaseAttentive, make_fast_predict_fn
   import numpy as np

   model = BaseAttentive(
       static_input_dim=4, dynamic_input_dim=8, future_input_dim=6,
       output_dim=1, forecast_horizon=24,
   )

   x_static  = np.random.randn(32, 4).astype('float32')
   x_dynamic = np.random.randn(32, 100, 8).astype('float32')
   x_future  = np.random.randn(32, 24, 6).astype('float32')

   fast_predict = make_fast_predict_fn(
       model,
       warmup_inputs=[x_static, x_dynamic, x_future],
   )
   predictions = fast_predict([x_static, x_dynamic, x_future])

This helper wraps inference with ``tf.function`` and uses ``training=False``.
For best results, keep batch and sequence shapes relatively stable. For
training, try ``model.compile(..., jit_compile="auto")``.

See Also
--------

- :doc:`torch_backend_guide` — PyTorch-specific device management
- :doc:`installation` — Backend installation instructions
- :doc:`api_reference` — Backend API reference
