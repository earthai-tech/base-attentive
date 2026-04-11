# SPDX-License-Identifier: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>
"""Intelligent backend detection and fallback logic."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import subprocess
import sys
from typing import Optional, Type

from .base import Backend
from .implementations import JaxBackend, PyTorchBackend, TensorFlowBackend, TorchBackend
from .version_check import check_tensorflow_compatibility, get_backend_version

__all__ = [
    "normalize_backend_name",
    "detect_available_backends",
    "select_best_backend",
    "ensure_default_backend",
]

_logger = logging.getLogger(__name__)

_BACKEND_ALIASES = {
    "tf": "tensorflow",
    "tensorflow": "tensorflow",
    "jax": "jax",
    "torch": "torch",
    "pytorch": "torch",
}

# Registry of available backends
_BACKENDS: dict[str, Type[Backend]] = {
    "tensorflow": TensorFlowBackend,
    "jax": JaxBackend,
    "torch": TorchBackend,
    "pytorch": PyTorchBackend,
}

_BACKEND_PREFERENCES = ["tensorflow", "jax", "torch"]


def normalize_backend_name(name: Optional[str]) -> str:
    """Normalize user-facing backend aliases to canonical names.
    
    Examples
    --------
    >>> normalize_backend_name("tf")
    'tensorflow'
    >>> normalize_backend_name("pytorch")
    'torch'
    """
    if name is None:
        return "tensorflow"

    normalized = str(name).strip().lower()
    if not normalized:
        return "tensorflow"
    if normalized == "keras":
        return normalize_backend_name(
            os.environ.get("KERAS_BACKEND", "tensorflow")
        )
    return _BACKEND_ALIASES.get(normalized, normalized)


def _has_module(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _import_module(module_name: str):
    """Import a module by name."""
    return importlib.import_module(module_name)


def detect_available_backends() -> dict[str, dict]:
    """Detect all available backends with their details.
    
    Returns
    -------
    dict
        Mapping of backend names to their details (available, version, supported).
    """
    backends_info = {}
    
    for backend_name, backend_cls in _BACKENDS.items():
        try:
            backend = backend_cls(load_runtime=False)
            is_available = backend.is_available()
            version = get_backend_version(backend_name)
            
            backends_info[backend_name] = {
                "available": is_available,
                "version": version,
                "supported": backend.supports_base_attentive,
                "experimental": backend.experimental,
                "class": backend_cls,
            }
        except Exception as e:
            backends_info[backend_name] = {
                "available": False,
                "version": None,
                "supported": False,
                "experimental": False,
                "error": str(e),
            }
    
    return backends_info


def select_best_backend(
    prefer: Optional[str] = None,
    require_supported: bool = True,
) -> Optional[str]:
    """Select the best available backend using intelligent fallback.
    
    Strategy:
    1. If env var BASE_ATTENTIVE_BACKEND is set, use it
    2. If prefer is specified and available, use it
    3. Check for supported backends in order: tensorflow > jax > torch
    4. Fall back to any available backend if require_supported=False
    
    Parameters
    ----------
    prefer : str, optional
        Preferred backend. If not available, will fall back.
    require_supported : bool, default=True
        If True, only select backends marked as supported.
        
    Returns
    -------
    str or None
        The selected backend name, or None if no backend available.
    """
    # Check environment variable first
    env_backend = os.environ.get("BASE_ATTENTIVE_BACKEND")
    if env_backend:
        normalized = normalize_backend_name(env_backend)
        if normalized in _BACKENDS:
            return normalized
    
    backends_info = detect_available_backends()
    
    # Filter candidates
    def is_candidate(name: str) -> bool:
        info = backends_info.get(name, {})
        if not info.get("available"):
            return False
        if require_supported and not info.get("supported"):
            return False
        return True
    
    # Try preferred backend first
    if prefer:
        normalized = normalize_backend_name(prefer)
        if is_candidate(normalized):
            return normalized
    
    # Try in preference order
    for backend_name in _BACKEND_PREFERENCES:
        if is_candidate(backend_name):
            return backend_name
    
    # Fall back to any available backend
    if not require_supported:
        for backend_name in backends_info:
            if backends_info[backend_name].get("available"):
                return backend_name
    
    return None


def ensure_default_backend(
    auto_install: bool = True,
    install_tensorflow: bool = True,
) -> str:
    """Ensure a default backend is available, installing if necessary.
    
    Parameters
    ----------
    auto_install : bool, default=True
        Whether to automatically install a backend if none available.
    install_tensorflow : bool, default=True
        If auto_install=True, prefer TensorFlow as default.
        
    Returns
    -------
    str
        The selected backend name.
        
    Raises
    ------
    RuntimeError
        If no backend available and auto_install=False.
    """
    # Try to select best available backend
    backend = select_best_backend(require_supported=True)
    if backend:
        _logger.info(f"Using available backend: {backend}")
        return backend
    
    # Try any available backend
    backend = select_best_backend(require_supported=False)
    if backend:
        _logger.warning(
            f"No 'supported' backend available. "
            f"Using experimental backend: {backend}"
        )
        return backend
    
    # No backend available
    if not auto_install:
        raise RuntimeError(
            "No compatible backend installed (tensorflow, jax, or torch). "
            "Please install one of them or set auto_install=True."
        )
    
    # Auto-install default backend
    install_package = "tensorflow[and-cuda]" if install_tensorflow else "jax"
    _logger.info(f"No backend found. Installing {install_package}...")
    
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", install_package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _logger.info(f"Successfully installed {install_package}")
        return "tensorflow" if install_tensorflow else "jax"
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to install {install_package}: {e}"
        ) from e


def get_available_backends() -> list[str]:
    """Get the installed backends that can be imported."""
    available = []
    for name, backend_cls in _BACKENDS.items():
        try:
            backend = backend_cls(load_runtime=False)
            if backend.is_available():
                available.append(name)
        except ImportError:
            pass
    return available
