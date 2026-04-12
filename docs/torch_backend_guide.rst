PyTorch Backend Guide
=====================

This guide explains how to use BaseAttentive with the PyTorch backend through
Keras 3.

.. note::

   The PyTorch (Torch) backend is **experimental** in BaseAttentive v1.0.0.
   The full ``BaseAttentive`` model path is still under evaluation on this
   runtime. For production use, prefer the TensorFlow backend.

Installation
============

Basic PyTorch Installation
---------------------------

Install PyTorch (version 2.0.0 or higher required):

.. code-block:: bash

   pip install torch

For CUDA support:

.. code-block:: bash

   # CUDA 11.8
   pip install torch torchvision torchaudio \
       --index-url https://download.pytorch.org/whl/cu118

   # CUDA 12.1
   pip install torch torchvision torchaudio \
       --index-url https://download.pytorch.org/whl/cu121

CPU-only (lightweight):

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cpu

Apple Silicon (M1/M2/M3) with MPS support:

.. code-block:: bash

   pip install torch torchvision torchaudio

Setting Up BaseAttentive with Torch
-------------------------------------

Once PyTorch is installed, set the backend before importing:

.. code-block:: python

   import os
   os.environ["KERAS_BACKEND"] = "torch"

   from base_attentive.backend import get_backend, set_backend
   from base_attentive import BaseAttentive

   backend = set_backend("torch")

Verifying Installation
----------------------

.. code-block:: python

   from base_attentive.backend import torch_is_available, get_torch_version

   if torch_is_available():
       print(f"PyTorch version: {get_torch_version()}")
   else:
       print("PyTorch not available")

GPU Verification Script
~~~~~~~~~~~~~~~~~~~~~~~~

Create ``check_torch_setup.py``:

.. code-block:: python

   #!/usr/bin/env python
   """Verify PyTorch GPU setup."""

   import torch
   from base_attentive.backend import (
       get_torch_device,
       get_torch_version,
       TorchDeviceManager,
   )

   print(f"PyTorch version: {get_torch_version()}")
   print(f"CUDA available: {torch.cuda.is_available()}")

   if torch.cuda.is_available():
       print(f"CUDA version: {torch.version.cuda}")
       print(f"GPU count: {torch.cuda.device_count()}")
       for i in range(torch.cuda.device_count()):
           print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

   if hasattr(torch.backends, 'mps'):
       print(f"MPS available: {torch.backends.mps.is_available()}")

   print(f"Selected device: {get_torch_device()}")

   manager = TorchDeviceManager()
   info    = manager.get_device_info()
   print("Device info:")
   for key, value in info.items():
       print(f"  {key}: {value}")

Run with:

.. code-block:: bash

   python check_torch_setup.py

Device Selection
================

Automatic Device Selection
----------------------------

BaseAttentive automatically selects the best available device:

.. code-block:: python

   from base_attentive.backend import get_torch_device

   device = get_torch_device()
   print(device)   # e.g. "cuda:0", "cpu", or "mps"

Manual Device Preference
-------------------------

.. code-block:: python

   from base_attentive.backend import get_torch_device

   device = get_torch_device(prefer="cuda", verbose=True)
   device = get_torch_device(prefer="cpu")
   device = get_torch_device(prefer="mps")   # Apple Silicon

Using TorchDeviceManager
-------------------------

For full device management control:

.. code-block:: python

   from base_attentive.backend import TorchDeviceManager

   manager = TorchDeviceManager(prefer="cuda")

   # Current device (lazy-loaded, cached)
   device = manager.device
   print(f"Using: {device}")

   # Available devices
   devices = manager.get_available_devices()
   print(devices)   # {'cuda': True, 'cpu': True, 'mps': False}

   # Detailed info
   info = manager.get_device_info()
   # Contains: torch_version, cuda_available, current_device,
   #           available_devices, cuda_device_count, cuda_devices, etc.

   # Change device explicitly
   manager.set_device("cpu")

   # Clear GPU cache
   manager.clear_gpu_cache()

Configuration
=============

Environment Variables
----------------------

.. code-block:: bash

   # PyTorch backend
   export BASE_ATTENTIVE_BACKEND=torch

   # Or Keras standard
   export KERAS_BACKEND=torch

Priority Order
~~~~~~~~~~~~~~

Backend selection (highest to lowest priority):

1. ``BASE_ATTENTIVE_BACKEND`` environment variable
2. ``KERAS_BACKEND`` environment variable
3. Previously set in-process backend via ``set_backend()``
4. Best available backend (auto-detected)
5. Default: TensorFlow

Usage Examples
==============

Basic Model Training with Torch Backend
-----------------------------------------

.. code-block:: python

   import os
   os.environ["KERAS_BACKEND"] = "torch"

   import numpy as np
   import keras
   from base_attentive.backend import set_backend
   from base_attentive import BaseAttentive

   set_backend("torch")

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

   x_static  = np.random.randn(32, 4).astype('float32')
   x_dynamic = np.random.randn(32, 100, 8).astype('float32')
   x_future  = np.random.randn(32, 24, 6).astype('float32')
   y         = np.random.randn(32, 24, 1).astype('float32')

   model.fit([x_static, x_dynamic, x_future], y, epochs=3)

Checking Backend Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from base_attentive.backend import (
       get_backend,
       get_backend_capabilities,
       detect_available_backends,
   )

   backend = get_backend()
   print(f"Current backend: {backend}")

   caps = get_backend_capabilities()
   print(f"Backend version: {caps.get('version')}")

   available = detect_available_backends()
   for name, info in available.items():
       print(f"{name}: {info}")

Troubleshooting
===============

PyTorch Not Found
------------------

**Problem:** ``ImportError: No module named 'torch'``

**Solution:**

.. code-block:: bash

   pip install torch

CUDA Not Available
-------------------

**Problem:** Device selection returns ``"cpu"`` instead of ``"cuda:0"``

Check availability:

.. code-block:: python

   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.device_count())

If CUDA is absent, verify:

1. NVIDIA GPU drivers installed: ``nvidia-smi``
2. CUDA toolkit installed and compatible
3. PyTorch CUDA build:

.. code-block:: bash

   pip install torch --index-url https://download.pytorch.org/whl/cu121

Device Memory Issues
---------------------

**Problem:** ``RuntimeError: CUDA out of memory``

Solutions:

1. Reduce batch size:

   .. code-block:: python

      model.fit([x_static, x_dynamic, x_future], y, batch_size=16)

2. Clear GPU cache:

   .. code-block:: python

      from base_attentive.backend import TorchDeviceManager
      TorchDeviceManager().clear_gpu_cache()

3. Use CPU:

   .. code-block:: python

      from base_attentive.backend import TorchDeviceManager
      manager = TorchDeviceManager(prefer="cpu")

MPS (Apple Silicon) Issues
---------------------------

**Problem:** MPS not available on macOS

.. code-block:: bash

   pip install torch>=2.0.0
   python -c "import torch; print(torch.backends.mps.is_available())"

Version Compatibility
---------------------

**Problem:** Version check failed for PyTorch

BaseAttentive requires PyTorch >= 2.0.0:

.. code-block:: python

   from base_attentive.backend import check_torch_compatibility
   ok, msg = check_torch_compatibility()
   print(msg)

Upgrade if needed:

.. code-block:: bash

   pip install --upgrade torch

Performance Best Practices
==========================

GPU Utilization
----------------

1. Use appropriate batch sizes for your GPU memory:

   .. code-block:: bash

      # Monitor GPU memory
      nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

2. Minimize device-to-host data transfers within training loops.

3. Use mixed precision if supported:

   .. code-block:: python

      import keras
      keras.mixed_precision.set_global_policy("mixed_float16")

Device Switching
----------------

Set the device once and reuse:

.. code-block:: python

   from base_attentive.backend import TorchDeviceManager

   manager = TorchDeviceManager(prefer="cuda")
   device  = manager.device

Multi-GPU Training
-------------------

For multi-GPU workloads, inspect available GPUs first:

.. code-block:: python

   from base_attentive.backend import TorchDeviceManager

   manager = TorchDeviceManager()
   info    = manager.get_device_info()

   if info.get("cuda_device_count", 0) > 1:
       print(f"Found {info['cuda_device_count']} GPUs")

Migration from TensorFlow Backend
==================================

Switching is minimal with Keras 3:

.. code-block:: python

   # Before: TensorFlow (default)
   from base_attentive import BaseAttentive

   # After: PyTorch backend
   import os
   os.environ["KERAS_BACKEND"] = "torch"
   from base_attentive.backend import set_backend
   set_backend("torch")

   from base_attentive import BaseAttentive
   # Rest of code is identical

API Reference
=============

Core Functions
--------------

.. function:: torch_is_available()

   Check if PyTorch is installed.

   :return: ``True`` if available
   :rtype: bool

.. function:: get_torch_version()

   Get installed PyTorch version string.

   :return: Version string (e.g. ``"2.1.0"``) or ``None``
   :rtype: str or None

.. function:: get_torch_device(prefer="cuda", verbose=False)

   Get the selected device string.

   :param prefer: Preferred device type (``"cuda"``, ``"cpu"``, ``"mps"``)
   :param verbose: Print debug info
   :return: Device string (e.g. ``"cuda:0"``, ``"cpu"``, ``"mps"``)
   :rtype: str

.. function:: check_torch_compatibility(torch_version=None)

   Check if PyTorch version is compatible (>= 2.0.0).

   :return: Tuple of ``(is_compatible, message)``
   :rtype: Tuple[bool, str]

TorchDeviceManager Class
------------------------

.. class:: TorchDeviceManager(prefer="cuda")

   Manages PyTorch device selection and configuration.

   .. attribute:: device

      Current selected device (lazy-loaded, cached).
      :type: str

   .. method:: set_device(device)

      Set device explicitly.
      :return: The set device string
      :rtype: str

   .. method:: get_available_devices()

      Get availability of each device type.
      :return: ``{'cuda': bool, 'cpu': bool, 'mps': bool}``
      :rtype: dict[str, bool]

   .. method:: get_device_info()

      Get comprehensive device information.
      :return: Dict with ``torch_version``, ``cuda_available``,
               ``current_device``, ``cuda_device_count``, etc.
      :rtype: dict

   .. method:: clear_gpu_cache()

      Clear the CUDA / MPS memory cache.

See Also
========

- :doc:`backends` — Backends overview
- :doc:`installation` — Installation instructions
- :doc:`configuration_guide` — Configuration guide
