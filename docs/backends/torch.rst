Torch Backend
=============

BaseAttentive supports `Torch <https://pytorch.org/>`_ (PyTorch) through
Keras 3.  CPU, CUDA, and Apple MPS (Metal Performance Shaders) devices
are all covered.

.. note::

   When running on Apple Silicon with MPS, BaseAttentive automatically
   moves tensors to CPU inside operations where ``keras.ops`` would
   otherwise trigger a NumPy conversion that fails on MPS devices.

Installation
------------

.. code-block:: bash

   pip install base-attentive[torch]

Or manually:

.. code-block:: bash

   pip install "torch>=2.1.0"

For CUDA:

.. code-block:: bash

   # CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121

   # CUDA 11.8
   pip install torch --index-url https://download.pytorch.org/whl/cu118

CPU-only (lighter install):

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cpu

Apple Silicon (M-series, MPS):

.. code-block:: bash

   pip install torch   # standard pip wheel includes MPS support

Selecting the Torch Backend
---------------------------

.. code-block:: python

   import os
   os.environ["KERAS_BACKEND"] = "torch"

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

Training Example
----------------

.. code-block:: python

   import os
   os.environ["KERAS_BACKEND"] = "torch"

   import numpy as np
   from base_attentive import BaseAttentive

   model = BaseAttentive(
       static_input_dim=4,
       dynamic_input_dim=8,
       future_input_dim=6,
       output_dim=1,
       forecast_horizon=24,
   )

   model.compile(optimizer="adam", loss="mse")

   x_static  = np.random.randn(32, 4).astype("float32")
   x_dynamic = np.random.randn(32, 100, 8).astype("float32")
   x_future  = np.random.randn(32, 24, 6).astype("float32")
   y         = np.random.randn(32, 24, 1).astype("float32")

   model.fit([x_static, x_dynamic, x_future], y, epochs=3)

Device Management
-----------------

BaseAttentive exposes helpers for device inspection and control:

.. code-block:: python

   from base_attentive.backend import (
       get_torch_device,
       TorchDeviceManager,
       torch_is_available,
       get_torch_version,
   )

   if torch_is_available():
       print(f"Torch version: {get_torch_version()}")

   # Automatic device selection (CUDA > MPS > CPU)
   device = get_torch_device()
   print(device)   # e.g. "cuda:0", "mps", or "cpu"

   # Manual preference
   device = get_torch_device(prefer="cpu")
   device = get_torch_device(prefer="mps")

   # Full device manager
   manager = TorchDeviceManager(prefer="cuda")
   print(manager.device)
   print(manager.get_available_devices())   # {'cuda': True, 'cpu': True, 'mps': False}
   manager.clear_gpu_cache()

Verification Script
-------------------

.. code-block:: python

   import torch
   from base_attentive.backend import (
       get_torch_device, get_torch_version, TorchDeviceManager,
   )

   print(f"Torch version:  {get_torch_version()}")
   print(f"CUDA available: {torch.cuda.is_available()}")
   if hasattr(torch.backends, "mps"):
       print(f"MPS available:  {torch.backends.mps.is_available()}")
   print(f"Selected device: {get_torch_device()}")

   info = TorchDeviceManager().get_device_info()
   for key, value in info.items():
       print(f"  {key}: {value}")

Mixed Precision
---------------

.. code-block:: python

   import keras
   keras.mixed_precision.set_global_policy("mixed_float16")

Compatibility Check
-------------------

.. code-block:: python

   from base_attentive.backend import check_torch_compatibility
   ok, msg = check_torch_compatibility()
   print(msg)

Minimum required version: **Torch 2.1.0**.

Troubleshooting
---------------

**Torch not found**

.. code-block:: bash

   pip install torch

**CUDA not detected**

.. code-block:: python

   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.device_count())

Check that the NVIDIA driver and CUDA toolkit are installed, then install
the matching Torch build (see the CUDA installation commands above).

**Out-of-memory errors**

Reduce batch size or clear the cache:

.. code-block:: python

   from base_attentive.backend import TorchDeviceManager
   TorchDeviceManager().clear_gpu_cache()

Switching from TensorFlow
-------------------------

The API is the same across backends.  Only the environment variable
changes:

.. code-block:: python

   # Before (TensorFlow — default)
   from base_attentive import BaseAttentive

   # After (Torch)
   import os
   os.environ["KERAS_BACKEND"] = "torch"
   from base_attentive import BaseAttentive
   # The rest of the code is unchanged.

See Also
--------

- :doc:`index` — Backend overview and selection
- :doc:`../installation` — Full installation instructions
- :doc:`../configuration_guide` — Configuration reference
