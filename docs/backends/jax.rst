JAX Backend
===========

BaseAttentive supports `JAX <https://jax.readthedocs.io/>`_ through Keras 3.
JAX can run on CPU, GPU (CUDA), and TPU.

Installation
------------

.. code-block:: bash

   pip install base-attentive[jax]

Or manually:

.. code-block:: bash

   pip install "jax>=0.4.0" "jaxlib>=0.4.0"

For GPU (CUDA 12):

.. code-block:: bash

   pip install "jax[cuda12]"

For TPU:

.. code-block:: bash

   pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

Selecting the JAX Backend
--------------------------

.. code-block:: python

   import os
   os.environ["KERAS_BACKEND"] = "jax"

   from base_attentive import BaseAttentive

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
   )

Training Example
----------------

.. code-block:: python

   import os
   os.environ["KERAS_BACKEND"] = "jax"

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

Device Inspection
-----------------

.. code-block:: python

   import jax
   print(jax.devices())           # e.g. [CpuDevice(id=0)]
   print(jax.default_backend())   # "cpu", "gpu", or "tpu"

Compatibility Check
-------------------

.. code-block:: python

   from base_attentive.backend import get_backend_version, version_at_least

   ver = get_backend_version("jax")
   ok = version_at_least(ver, "0.4.0")
   print(f"JAX {ver} compatible: {ok}")

Minimum required versions: **jax 0.4.0**, **jaxlib 0.4.0**.

Troubleshooting
---------------

**JAX not found**

.. code-block:: bash

   pip install jax jaxlib

**GPU not detected in JAX**

Verify the CUDA build:

.. code-block:: python

   import jax
   print(jax.devices("gpu"))

If the list is empty, install the CUDA-enabled jaxlib:

.. code-block:: bash

   pip install "jax[cuda12]"

**XLA compilation warnings**

JAX traces and JIT-compiles operations on first call.  Warm-up
latency on the first ``model.predict`` call is expected.

See Also
--------

- :doc:`index` — Backend overview and selection
- :doc:`../installation` — Full installation instructions
