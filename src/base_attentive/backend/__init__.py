# SPDX-License-Identifier: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>
"""Backend runtime abstraction with intelligent selection.

This module provides a flexible backend system supporting TensorFlow, JAX,
and PyTorch through Keras 3 multi-backend runtimes.

Architecture Overview
---------------------
The backend package is structured for maximum modularity and clarity:

- ``base.py`` : Abstract Backend class (extensible for new backends)
- ``implementations.py`` : Concrete implementations (TensorFlow, JAX, Torch)
- ``detector.py`` : Intelligent backend selection and fallback logic
- ``version_check.py`` : Version compatibility utilities
- ``__init__.py`` : Public API (this file)

This modular design enables:
- Easy addition of new backends (extend Backend class)
- Clear separation of concerns (each module has single responsibility)
- Simple testing and debugging
- Better maintainability and code reuse

Features
--------
- Automatic backend detection based on installed packages
- Intelligent fallback when preferred backend unavailable
- Version compatibility checking (e.g., TensorFlow ≥ 2.10.0)
- Automatic backend installation if none available
- Environment variable configuration (BASE_ATTENTIVE_BACKEND, KERAS_BACKEND)
- Detailed capability reporting

Clean Architecture
-------------------
This package is the sole source of all backend functionality.
No legacy compatibility shims or file-based modules.

Usage Examples
--------------
Basic usage::

    from base_attentive.backend import get_backend, set_backend

    # Automatically select best available backend
    backend = get_backend()

    # Or specify a backend explicitly
    backend = set_backend("tensorflow")

    # Query backend capabilities
    from base_attentive.backend import get_backend_capabilities

    caps = get_backend_capabilities()
    print(caps["version"])  # e.g., "2.15.0"

Advanced usage::

    from base_attentive.backend import (
        detect_available_backends,
        select_best_backend,
        ensure_default_backend,
        check_tensorflow_compatibility,
    )

    # Detect all available backends
    backends = detect_available_backends()
    print(backends["tensorflow"]["version"])

    # Intelligently select the best backend
    best = select_best_backend(prefer="jax")

    # Ensure a backend is available (auto-install if needed)
    backend_name = ensure_default_backend(auto_install=True)

    # Check version compatibility
    is_compatible, msg = check_tensorflow_compatibility()

Environment Configuration
--------------------------
Control backend selection via environment variables::

    # Highest priority: BaseAttentive-specific
    export BASE_ATTENTIVE_BACKEND=tensorflow

    # Fallback: Keras standard
    export KERAS_BACKEND=jax

Selection Priority Order
------------------------
When no backend is explicitly specified, the system tries in order:

1. ``BASE_ATTENTIVE_BACKEND`` environment variable
2. ``KERAS_BACKEND`` environment variable
3. Previously set in-process backend
4. Best available backend (auto-detected)
5. Default backend (TensorFlow, with auto-install if needed)

Supported Backends
------------------
- **TensorFlow** (✅ Fully Supported)
  - Status: Production ready
  - Required: TensorFlow ≥ 2.10.0

- **JAX** (⚠️ Experimental)
  - Status: Limited feature parity
  - Required: Keras 3 + JAX

- **PyTorch** (⚠️ Experimental)
  - Status: Limited feature parity
  - Required: Keras 3 + PyTorch
  - Aliases: torch, pytorch
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Optional, Type

from .base import Backend
from .detector import (
    _has_module as _has_module,
)
from .detector import (
    _import_module as _import_module,
)
from .detector import (
    detect_available_backends,
    ensure_default_backend,
    get_available_backends,
    normalize_backend_name,
    select_best_backend,
)
from .implementations import (
    JaxBackend,
    PyTorchBackend,
    TensorFlowBackend,
    TorchBackend,
)
from .torch_utils import (
    TorchDeviceManager,
    get_torch_device,
    get_torch_version,
    torch_is_available,
)
from .version_check import (
    check_tensorflow_compatibility,
    check_torch_compatibility,
    get_backend_version,
    parse_version,
    version_at_least,
)

__all__ = [
    # Base classes
    "Backend",
    # Backend implementations
    "TensorFlowBackend",
    "JaxBackend",
    "TorchBackend",
    "PyTorchBackend",
    # Core API
    "get_backend",
    "set_backend",
    "get_available_backends",
    "get_backend_capabilities",
    "normalize_backend_name",
    # Detection and selection
    "detect_available_backends",
    "select_best_backend",
    "ensure_default_backend",
    # Version checking
    "get_backend_version",
    "check_tensorflow_compatibility",
    "check_torch_compatibility",
    "parse_version",
    "version_at_least",
    # Torch utilities
    "get_torch_device",
    "get_torch_version",
    "torch_is_available",
    "TorchDeviceManager",
]

# Registry of available backends
_BACKENDS: dict[str, Type[Backend]] = {
    "tensorflow": TensorFlowBackend,
    "jax": JaxBackend,
    "torch": TorchBackend,
    "pytorch": PyTorchBackend,
}

# Global backend instance
_CURRENT_BACKEND: Optional[Backend] = None


def get_backend(name: Optional[str] = None) -> Backend:
    """Get the current or requested backend runtime.

    Lookup order:
        1. Explicit ``name`` parameter
        2. ``BASE_ATTENTIVE_BACKEND`` environment variable
        3. ``KERAS_BACKEND`` environment variable
        4. Previously set in-process backend
        5. Best available backend (auto-detect)
        6. Default (tensorflow, auto-install if needed)

    Parameters
    ----------
    name : str, optional
        Backend name ('tensorflow', 'jax', 'torch', 'pytorch').
        If None, will auto-detect.

    Returns
    -------
    Backend
        The configured backend instance.

    Raises
    ------
    ValueError
        If the requested backend is not available.

    Examples
    --------
    >>> # Use default backend (auto-detected)
    >>> backend = get_backend()

    >>> # Explicitly request TensorFlow
    >>> backend = get_backend("tensorflow")
    """
    global _CURRENT_BACKEND

    if name is None:
        # Check environment variables
        env_name = os.environ.get("BASE_ATTENTIVE_BACKEND")
        if env_name is None:
            env_name = os.environ.get("KERAS_BACKEND")

        # Use current if already set
        if env_name is None and _CURRENT_BACKEND is not None:
            return _CURRENT_BACKEND

        # Auto-detect if not specified
        if env_name is None:
            name = select_best_backend(require_supported=True)
            if name is None:
                name = ensure_default_backend(auto_install=True)
        else:
            name = env_name

    normalized = normalize_backend_name(name)
    if normalized not in _BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available: {list(_BACKENDS.keys())}"
        )

    backend_cls = _BACKENDS[normalized]
    try:
        return backend_cls()
    except ImportError as exc:
        available = get_available_backends()
        raise ValueError(
            f"Backend '{normalized}' is not available. "
            f"Available backends: {available}. "
            f"Try: pip install {normalized}"
        ) from exc


def get_backend_capabilities(
    name: Optional[str] = None,
) -> dict[str, Any]:
    """Return a capability report for the current or requested backend.

    Parameters
    ----------
    name : str, optional
        Backend name. If None, uses current backend.

    Returns
    -------
    dict
        Capabilities including name, framework, version, support status.

    Examples
    --------
    >>> caps = get_backend_capabilities("tensorflow")
    >>> print(caps["supported"])  # True/False
    >>> print(caps["version"])  # "2.15.0"
    """
    if name is None:
        try:
            backend = get_backend()
            caps = backend.get_capabilities()
            caps.setdefault("name", getattr(backend, "name", "unknown"))
            caps.setdefault(
                "framework",
                getattr(backend, "framework", getattr(backend, "name", "unknown")),
            )
            caps.setdefault(
                "available",
                backend.is_available() if hasattr(backend, "is_available") else True,
            )
            caps.setdefault(
                "uses_keras_runtime",
                getattr(backend, "uses_keras_runtime", False),
            )
            caps.setdefault("experimental", getattr(backend, "experimental", False))
            caps.setdefault(
                "supports_base_attentive",
                getattr(backend, "supports_base_attentive", False),
            )
            caps.setdefault(
                "supports_base_attentive_v2",
                getattr(backend, "supports_base_attentive_v2", False),
            )
            caps.setdefault("blockers", list(getattr(backend, "blockers", ())))
            caps.setdefault("v2_blockers", list(getattr(backend, "v2_blockers", ())))
            caps.setdefault(
                "version",
                get_backend_version(getattr(backend, "name", "tensorflow")),
            )
            return caps
        except Exception:
            name = "tensorflow"

    normalized = normalize_backend_name(name)
    if normalized not in _BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available: {list(_BACKENDS.keys())}"
        )

    try:
        backend = _BACKENDS[normalized](load_runtime=False)
        caps = backend.get_capabilities()
        caps.setdefault("name", getattr(backend, "name", normalized))
        caps.setdefault(
            "framework",
            getattr(_BACKENDS[normalized], "framework", normalized),
        )
        caps.setdefault(
            "available",
            backend.is_available() if hasattr(backend, "is_available") else True,
        )
        caps.setdefault(
            "uses_keras_runtime",
            getattr(backend, "uses_keras_runtime", False),
        )
        caps.setdefault("experimental", getattr(backend, "experimental", False))
        caps.setdefault(
            "supports_base_attentive",
            getattr(backend, "supports_base_attentive", False),
        )
        caps.setdefault(
            "supports_base_attentive_v2",
            getattr(backend, "supports_base_attentive_v2", False),
        )
        caps.setdefault("blockers", list(getattr(backend, "blockers", ())))
        caps.setdefault("v2_blockers", list(getattr(backend, "v2_blockers", ())))
        caps["version"] = get_backend_version(normalized)
        return caps
    except Exception as e:
        return {
            "name": normalized,
            "framework": getattr(_BACKENDS[normalized], "framework", normalized),
            "available": False,
            "uses_keras_runtime": getattr(
                _BACKENDS[normalized], "uses_keras_runtime", False
            ),
            "experimental": getattr(_BACKENDS[normalized], "experimental", False),
            "supports_base_attentive": getattr(
                _BACKENDS[normalized], "supports_base_attentive", False
            ),
            "supports_base_attentive_v2": getattr(
                _BACKENDS[normalized], "supports_base_attentive_v2", False
            ),
            "blockers": list(getattr(_BACKENDS[normalized], "blockers", ())),
            "v2_blockers": list(getattr(_BACKENDS[normalized], "v2_blockers", ())),
            "version": get_backend_version(normalized),
            "error": str(e),
        }


def set_backend(name: str) -> Backend:
    """Set the default backend.

    Notes
    -----
    For Keras 3 multi-backend runtimes, the backend should ideally be set
    before importing ``keras``. If Keras is already loaded with a different
    runtime, a restart is recommended.

    Parameters
    ----------
    name : str
        Backend name ('tensorflow', 'jax', 'torch', 'pytorch').

    Returns
    -------
    Backend
        The configured backend instance.

    Examples
    --------
    >>> backend = set_backend("tensorflow")
    >>> backend = set_backend("jax")
    """
    global _CURRENT_BACKEND

    normalized = normalize_backend_name(name)

    # Check version compatibility for TensorFlow
    if normalized == "tensorflow":
        is_compatible, msg = check_tensorflow_compatibility()
        if not is_compatible:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # Warn if Keras already loaded
    from .base import _read_loaded_keras_backend

    loaded_backend = _read_loaded_keras_backend()
    if loaded_backend and loaded_backend != normalized:
        warnings.warn(
            "Keras is already loaded with backend "
            f"'{loaded_backend}'. Restart Python after switching to "
            f"'{normalized}' for the change to take full effect.",
            RuntimeWarning,
            stacklevel=2,
        )

    _CURRENT_BACKEND = get_backend(normalized)
    os.environ["BASE_ATTENTIVE_BACKEND"] = normalized
    os.environ["KERAS_BACKEND"] = normalized
    return _CURRENT_BACKEND


# Initialize on import with intelligent selection
def _auto_initialize():
    """Auto-initialize backend on module import."""
    try:
        configured_backend = os.environ.get("BASE_ATTENTIVE_BACKEND")
        if configured_backend:
            normalized = normalize_backend_name(configured_backend)
            os.environ["BASE_ATTENTIVE_BACKEND"] = normalized
            os.environ.setdefault("KERAS_BACKEND", normalized)
            return

        keras_backend = os.environ.get("KERAS_BACKEND")
        if keras_backend:
            normalized = normalize_backend_name(keras_backend)
            os.environ["BASE_ATTENTIVE_BACKEND"] = normalized
            os.environ["KERAS_BACKEND"] = normalized
            return

        # Try to auto-select best backend
        best = select_best_backend(require_supported=True)
        if best:
            normalized = normalize_backend_name(best)
            os.environ["BASE_ATTENTIVE_BACKEND"] = normalized
            os.environ["KERAS_BACKEND"] = normalized
            return

        # Fall back to any available
        available = select_best_backend(require_supported=False)
        if available:
            normalized = normalize_backend_name(available)
            os.environ["BASE_ATTENTIVE_BACKEND"] = normalized
            os.environ["KERAS_BACKEND"] = normalized
            return

        # Will trigger auto-install on first get_backend() call
    except Exception:
        pass


_auto_initialize()
