# SPDX-License-Identifier: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>
"""Base Backend class definition."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any, Optional

__all__ = ["Backend"]


def _get_backend_helper(name: str):
    """Return helper overrides from ``base_attentive.backend`` when present."""
    backend_module = sys.modules.get("base_attentive.backend")
    helper = getattr(backend_module, name, None) if backend_module else None
    current = globals().get(name)
    if callable(helper) and helper is not current:
        return helper
    return None


def _has_module(module_name: str) -> bool:
    """Return whether a module appears importable without importing it."""
    helper = _get_backend_helper("_has_module")
    if helper is not None:
        return helper(module_name)
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _import_module(module_name: str):
    """Import a module by name."""
    helper = _get_backend_helper("_import_module")
    if helper is not None:
        return helper(module_name)
    return importlib.import_module(module_name)


def _read_loaded_keras_backend() -> Optional[str]:
    """Return the already-loaded Keras runtime backend, if available."""
    if "keras" not in sys.modules:
        return None

    try:
        keras = sys.modules["keras"]
        backend_ns = getattr(keras, "backend", None)
        backend_fn = getattr(backend_ns, "backend", None)
        if callable(backend_fn):
            from .detector import normalize_backend_name
            return normalize_backend_name(backend_fn())
    except Exception:
        return None
    return None


class Backend:
    """Base class for runtime backend descriptors."""

    name: str = "base"
    framework: str = "unknown"
    required_modules: tuple[str, ...] = ()
    uses_keras_runtime: bool = False
    experimental: bool = False
    supports_base_attentive: bool = False
    blockers: tuple[str, ...] = ()

    Tensor: Any = None
    Layer: Any = None
    Model: Any = None
    Sequential: Any = None
    Dense: Any = None
    LSTM: Any = None
    MultiHeadAttention: Any = None
    LayerNormalization: Any = None
    Dropout: Any = None
    BatchNormalization: Any = None

    def __init__(self, load_runtime: bool = True):
        self._verify_installation()
        if load_runtime:
            self._initialize_imports()

    def _verify_installation(self):
        """Verify that the required framework is installed."""
        for module_name in self.required_modules:
            if not _has_module(module_name):
                raise ImportError(
                    f"Backend '{self.name}' requires '{module_name}'."
                )
        return True

    def _initialize_imports(self):
        """Load framework-specific handles."""

    def is_available(self) -> bool:
        """Check whether the backend can be imported."""
        try:
            self._verify_installation()
            return True
        except ImportError:
            return False

    def get_capabilities(self) -> dict[str, Any]:
        """Return a capability summary for the backend."""
        return {
            "name": self.name,
            "framework": self.framework,
            "available": self.is_available(),
            "uses_keras_runtime": self.uses_keras_runtime,
            "experimental": self.experimental,
            "supports_base_attentive": self.supports_base_attentive,
            "blockers": list(self.blockers),
            "loaded_keras_backend": _read_loaded_keras_backend(),
        }
