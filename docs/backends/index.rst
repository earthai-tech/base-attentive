Backend Guide
=============

BaseAttentive uses `Keras 3 <https://keras.io/>`_ as its neural network
runtime.  Keras 3 supports three backends — TensorFlow, Torch, and JAX —
and BaseAttentive inherits that flexibility.

.. toctree::
   :maxdepth: 1
   :caption: Backends

   tensorflow
   torch
   jax

Support Status
--------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Backend
     - Status
     - Notes
   * - TensorFlow
     - Supported
     - Full ``BaseAttentive`` model path; tested on Linux, macOS, Windows.
   * - Torch
     - Supported
     - Full model path tested; includes CPU, CUDA, and Apple MPS devices.
   * - JAX
     - Supported
     - Runtime abstraction available; full model path tested on CPU/TPU.

Selecting a Backend
-------------------

Set the backend **before** importing ``base_attentive`` or ``keras``:

.. code-block:: bash

   export KERAS_BACKEND=tensorflow   # or torch / jax
   export BASE_ATTENTIVE_BACKEND=tensorflow

Or in Python:

.. code-block:: python

   import os
   os.environ["KERAS_BACKEND"] = "torch"   # set before any keras import
   from base_attentive import BaseAttentive

Runtime inspection and control:

.. code-block:: python

   from base_attentive import (
       get_backend,
       get_available_backends,
       get_backend_capabilities,
       set_backend,
       normalize_backend_name,
   )

   backend = get_backend()
   print(get_available_backends())
   print(get_backend_capabilities("jax"))
   print(normalize_backend_name("tf"))   # -> "tensorflow"

How Backend Resolution Works
----------------------------

When no explicit backend is specified, BaseAttentive checks in this order:

1. ``BASE_ATTENTIVE_BACKEND`` environment variable
2. ``KERAS_BACKEND`` environment variable
3. The backend previously set in the current Python process via ``set_backend()``
4. The best available backend detected automatically
5. ``tensorflow`` as the final fallback

Detection and Selection Utilities
----------------------------------

.. code-block:: python

   from base_attentive import (
       detect_available_backends,
       select_best_backend,
       ensure_default_backend,
   )

   # Inspect all installed backends
   info = detect_available_backends()
   for name, details in info.items():
       print(f"{name}: available={details.get('available')}, "
             f"version={details.get('version')}")

   # Select the best available backend
   best = select_best_backend(require_supported=True)
   print(f"Selected backend: {best}")

   # Ensure a default backend is ready
   name = ensure_default_backend(auto_install=False)

Version Compatibility
---------------------

.. code-block:: python

   from base_attentive.backend import (
       check_tensorflow_compatibility,
       check_torch_compatibility,
       get_backend_version,
       version_at_least,
   )

   ok, msg = check_tensorflow_compatibility()
   print(msg)   # e.g. "TensorFlow 2.15.0 is compatible"

   ok, msg = check_torch_compatibility()

   ver = get_backend_version("tensorflow")
   ok = version_at_least("2.13.0", "2.12.0")   # True

See Also
--------

- :doc:`tensorflow` — TensorFlow-specific notes
- :doc:`torch` — Torch device management and MPS
- :doc:`jax` — JAX accelerator setup
- :doc:`../installation` — Installation instructions
