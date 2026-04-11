PyTorch Backend Guide
=====================

This guide explains how to use BaseAttentive with the PyTorch backend through Keras 3.

Installation
============

Basic PyTorch Installation
---------------------------

Install PyTorch (requires version 2.0.0 or higher)::

    pip install torch

For CUDA support::

    # For CUDA 11.8
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # For CUDA 12.1
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

For CPU-only (lightweight)::

    pip install torch --index-url https://download.pytorch.org/whl/cpu

For Apple Silicon (M1/M2/M3) with MPS support::

    pip install torch torchvision torchaudio

Setting Up BaseAttentive with Torch
-------------------------------------

Once PyTorch is installed, BaseAttentive will automatically detect it::

    from base_attentive.backend import get_backend, set_backend

    # Automatically select best available backend (will prefer Torch if installed)
    backend = get_backend()
    
    # Or explicitly set to Torch
    backend = set_backend("torch")

Verifying Installation
----------------------

Check if PyTorch is properly configured::

    from base_attentive.backend import torch_is_available, get_torch_version
    
    if torch_is_available():
        print(f"PyTorch version: {get_torch_version()}")
    else:
        print("PyTorch not available")

GPU Verification Script
~~~~~~~~~~~~~~~~~~~~~~~~

Create a file ``check_torch_setup.py``::

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
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Get selected device
    print(f"\nSelected device: {get_torch_device()}")
    
    # Get device info
    manager = TorchDeviceManager()
    info = manager.get_device_info()
    print(f"\nDevice info:")
    for key, value in info.items():
        print(f"  {key}: {value}")

Run with::

    python check_torch_setup.py

Device Selection
================

Automatic Device Selection
----------------------------

BaseAttentive automatically selects the best available device::

    from base_attentive.backend import get_torch_device
    
    device = get_torch_device()
    print(device)  # e.g., "cuda:0", "cpu", or "mps"

Manual Device Preference
-------------------------

Specify preferred device type::

    from base_attentive.backend import get_torch_device
    
    # Prefer CUDA (default)
    device = get_torch_device(prefer="cuda", verbose=True)
    
    # Prefer CPU
    device = get_torch_device(prefer="cpu")
    
    # Prefer Apple Metal Performance Shaders
    device = get_torch_device(prefer="mps")

Using TorchDeviceManager
-------------------------

For more control over device management::

    from base_attentive.backend import TorchDeviceManager
    
    # Create a device manager
    manager = TorchDeviceManager(prefer="cuda")
    
    # Get current device
    device = manager.device
    print(f"Using device: {device}")
    
    # Get available devices
    devices = manager.get_available_devices()
    print(devices)  # {'cuda': True, 'cpu': True, 'mps': False}
    
    # Get detailed device information
    info = manager.get_device_info()
    print(info)
    # Output includes: torch_version, cuda_available, current_device,
    # available_devices, cuda_device_count, cuda_devices, etc.
    
    # Change device explicitly
    manager.set_device("cpu")
    print(f"Device changed to: {manager.device}")

Configuration
=============

Environment Variables
----------------------

Control backend selection via environment variables::

    # Use PyTorch backend explicitly
    export BASE_ATTENTIVE_BACKEND=torch
    
    # Or use Keras standard
    export KERAS_BACKEND=pytorch

Priority Order
~~~~~~~~~~~~~~

Backend selection priority (in order):

1. ``BASE_ATTENTIVE_BACKEND`` environment variable
2. ``KERAS_BACKEND`` environment variable  
3. Previously set in-process backend
4. Best available backend (auto-detected)
5. Default backend (TensorFlow, with auto-install if needed)

Usage Examples
==============

Basic Model Training
---------------------

::

    import keras
    from keras import layers
    from base_attentive.backend import set_backend
    
    # Set PyTorch backend
    set_backend("torch")
    
    # Build model
    model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    # Train
    model.fit(x_train, y_train, epochs=10, batch_size=32)

Vision Model with GPU
~~~~~~~~~~~~~~~~~~~~~~

::

    import keras
    from keras import layers, applications
    from base_attentive.backend import (
        set_backend,
        get_torch_device,
        TorchDeviceManager,
    )
    
    # Set PyTorch backend
    set_backend("torch")
    
    # Get device info for logging
    manager = TorchDeviceManager()
    info = manager.get_device_info()
    print(f"Training on: {info['current_device']}")
    
    # Build model
    base_model = applications.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    # Train
    model.fit(x_train, y_train, epochs=20, batch_size=32)

Checking Backend Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    from base_attentive.backend import (
        get_backend,
        get_backend_capabilities,
        detect_available_backends,
    )
    
    # Current backend
    backend = get_backend()
    print(f"Current backend: {backend}")
    
    # Backend capabilities
    caps = get_backend_capabilities()
    print(f"Backend version: {caps['version']}")
    print(f"Supported features: {caps['features']}")
    
    # Available backends
    available = detect_available_backends()
    for name, info in available.items():
        print(f"{name}: {info}")

Troubleshooting
===============

PyTorch Not Found
------------------

Problem: ``ImportError: No module named 'torch'``

Solution: Install PyTorch::

    pip install torch

CUDA Not Available
-------------------

Problem: Device selection returns "cpu" instead of "cuda:0"

Check GPU availability::

    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

If empty, verify:

1. NVIDIA GPU drivers installed: ``nvidia-smi``
2. CUDA toolkit installed and compatible
3. PyTorch built with CUDA support::

    pip install torch --index-url https://download.pytorch.org/whl/cu121

Device Memory Issues
---------------------

Problem: ``RuntimeError: CUDA out of memory``

Solutions:

1. Reduce batch size::

    model.fit(x_train, y_train, batch_size=16)  # Smaller batch

2. Clear GPU cache::

    from base_attentive.backend import TorchDeviceManager
    manager = TorchDeviceManager()
    manager.clear_gpu_cache()

3. Use CPU instead::

    from base_attentive.backend import TorchDeviceManager
    manager = TorchDeviceManager(prefer="cpu")

MPS (Apple Silicon) Issues
---------------------------

Problem: MPS device not available on macOS

Ensure PyTorch installed for Apple Silicon::

    pip install torch==2.0.0  # Or newer
    python -c "import torch; print(torch.backends.mps.is_available())"

Version Compatibility
---------------------

Problem: ``Version check failed for PyTorch``

BaseAttentive requires PyTorch >= 2.0.0::

    import torch
    from base_attentive.backend import check_torch_compatibility
    
    is_compatible, msg = check_torch_compatibility()
    print(msg)

Upgrade if needed::

    pip install --upgrade torch

Performance Best Practices
==========================

GPU Utilization
----------------

1. **Use appropriate batch sizes** for your GPU memory::

    # Monitor GPU memory with nvidia-smi
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

2. **Pin memory for faster transfers** (when possible in your framework)

3. **Use mixed precision training**::

    import tensorflow as tf
    
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)

Device Switching
----------------

Minimize device switching overhead::

    from base_attentive.backend import TorchDeviceManager
    
    # Set once and reuse
    manager = TorchDeviceManager(prefer="cuda")
    device = manager.device
    
    # Use device consistently in training loop
    for epoch in range(num_epochs):
        for batch in data_loader:
            # All computations on selected device

Multi-GPU Training
-------------------

For models spanning multiple GPUs (when supported)::

    import torch
    from base_attentive.backend import TorchDeviceManager
    
    manager = TorchDeviceManager()
    info = manager.get_device_info()
    
    if info.get("cuda_device_count", 0) > 1:
        print(f"Found {info['cuda_device_count']} GPUs")
        # Model-specific multi-GPU setup needed
        # (depends on your framework)

Migration from TensorFlow Backend
==================================

Switching from TensorFlow to PyTorch is straightforward::

    # Before: TensorFlow backend (default)
    # After: PyTorch backend
    
    from base_attentive.backend import set_backend
    
    # Set to PyTorch
    set_backend("torch")
    
    # Rest of code remains the same!
    # (Keras 3 provides unified interface)

Code changes minimal::

    # OLD: Using TensorFlow implicitly
    # (no backend specification)
    
    # NEW: Specify PyTorch backend
    from base_attentive.backend import set_backend
    set_backend("torch")
    
    # Everything else identical!

API Reference
=============

Core Functions
--------------

.. function:: torch_is_available()

    Check if PyTorch is installed and available.
    
    :return: True if PyTorch is available, False otherwise
    :rtype: bool

.. function:: get_torch_version()

    Get installed PyTorch version.
    
    :return: Version string (e.g., "2.0.0") or None if PyTorch not available
    :rtype: str or None

.. function:: get_torch_device(prefer="cuda", verbose=False)

    Get the selected device string.
    
    :param prefer: Preferred device type ('cuda', 'cpu', 'mps')
    :type prefer: str
    :param verbose: Print debug information
    :type verbose: bool
    :return: Device string (e.g., "cuda:0", "cpu", "mps")
    :rtype: str

.. function:: check_torch_compatibility(torch_version=None)

    Check if PyTorch version is compatible (>= 2.0.0).
    
    :param torch_version: PyTorch version string (detected if None)
    :type torch_version: str or None
    :return: Tuple of (is_compatible, message)
    :rtype: Tuple[bool, str]

TorchDeviceManager Class
------------------------

.. class:: TorchDeviceManager(prefer="cuda")

    Manages PyTorch device selection and configuration.
    
    .. method:: __init__(prefer="cuda")
    
        Initialize device manager.
        
        :param prefer: Preferred device type
        :type prefer: Literal["cuda", "cpu", "mps"]
    
    .. attribute:: device
    
        Current selected device (lazy-loaded, cached).
        
        :type: str
    
    .. method:: set_device(device)
    
        Set device explicitly.
        
        :param device: Device string or name
        :type device: str
        :return: The set device string
        :rtype: str
    
    .. method:: get_available_devices()
    
        Get availability of different device types.
        
        :return: Mapping of device types to availability
        :rtype: dict[str, bool]
    
    .. method:: get_device_info()
    
        Get detailed device information.
        
        :return: Comprehensive device info including GPU count, memory, etc.
        :rtype: dict

See Also
========

- :doc:`../index` - Main documentation
- :doc:`../backends` - Backends overview
- :doc:`../configuration_guide` - Configuration guide
