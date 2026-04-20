# SPDX-License-Identifier: Apache-2.0
# Author: LKouadio <etanoyau@gmail.com>
"""Lazy backend runtime abstraction for Base-Attentive.

This package exposes backend selection, capability inspection, and helper
utilities without importing all backend implementations eagerly.
"""

from __future__ import annotations

import importlib
import os
import warnings
from typing import Any

__all__ = [
    "Backend",
    "TensorFlowBackend",
    "JaxBackend",
    "TorchBackend",
    "PyTorchBackend",
    "get_backend",
    "set_backend",
    "get_available_backends",
    "get_backend_capabilities",
    "normalize_backend_name",
    "detect_available_backends",
    "select_best_backend",
    "ensure_default_backend",
    "get_backend_version",
    "check_tensorflow_compatibility",
    "check_torch_compatibility",
    "parse_version",
    "version_at_least",
    "get_torch_device",
    "get_torch_version",
    "torch_is_available",
    "TorchDeviceManager",
    "_has_module",
    "_import_module",
]

_CURRENT_BACKEND = None

# Eagerly bind torch_utils symbols so they are always real objects in
# this module's namespace regardless of test-ordering or sys.modules state.
try:
    from base_attentive.backend.torch_utils import (  # noqa: E402
        TorchDeviceManager,
        check_torch_compatibility,
        get_torch_device,
        get_torch_version,
        torch_is_available,
    )
except Exception:
    pass


def _module(name: str):
    return importlib.import_module(name)


def _backend_classes() -> dict[str, type]:
    implementations = _module(
        "base_attentive.backend.implementations"
    )
    return {
        "tensorflow": implementations.TensorFlowBackend,
        "jax": implementations.JaxBackend,
        "torch": implementations.TorchBackend,
        "pytorch": implementations.PyTorchBackend,
    }


def normalize_backend_name(name: str | None) -> str:
    detector = _module("base_attentive.backend.detector")
    return detector.normalize_backend_name(name)


# detector wrappers -------------------------------------------------------
def detect_available_backends():
    detector = _module("base_attentive.backend.detector")
    return detector.detect_available_backends()


def select_best_backend(
    prefer: str | None = None,
    require_supported: bool = True,
):
    detector = _module("base_attentive.backend.detector")
    return detector.select_best_backend(
        prefer=prefer,
        require_supported=require_supported,
    )


def ensure_default_backend(
    auto_install: bool = False,
    install_tensorflow: bool = True,
) -> str:
    detector = _module("base_attentive.backend.detector")
    return detector.ensure_default_backend(
        auto_install=auto_install,
        install_tensorflow=install_tensorflow,
    )


def get_available_backends():
    detector = _module("base_attentive.backend.detector")
    return detector.get_available_backends()


# version wrappers --------------------------------------------------------
def get_backend_version(name: str):
    version_check = _module(
        "base_attentive.backend.version_check"
    )
    return version_check.get_backend_version(name)


def check_tensorflow_compatibility():
    version_check = _module(
        "base_attentive.backend.version_check"
    )
    return version_check.check_tensorflow_compatibility()


def check_torch_compatibility():
    version_check = _module(
        "base_attentive.backend.version_check"
    )
    return version_check.check_torch_compatibility()


def parse_version(version: str):
    version_check = _module(
        "base_attentive.backend.version_check"
    )
    return version_check.parse_version(version)


def version_at_least(version: str, minimum: str):
    version_check = _module(
        "base_attentive.backend.version_check"
    )
    return version_check.version_at_least(version, minimum)


# core API ----------------------------------------------------------------
def get_backend(name: str | None = None):
    global _CURRENT_BACKEND

    requested_name = name

    if name is None:
        env_name = os.environ.get("BASE_ATTENTIVE_BACKEND")
        if env_name is None:
            env_name = os.environ.get("KERAS_BACKEND")

        if env_name is None and _CURRENT_BACKEND is not None:
            return _CURRENT_BACKEND

        if env_name is None or not str(env_name).strip():
            raise RuntimeError(
                "BaseAttentive backend is not configured. Set "
                "BASE_ATTENTIVE_BACKEND to one of: tensorflow, torch, jax, or auto."
            )
        name = env_name

    normalized = normalize_backend_name(name)
    backends = _backend_classes()
    detector = _module("base_attentive.backend.detector")
    auto_install = os.environ.get(
        "BASE_ATTENTIVE_AUTO_INSTALL", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}

    if normalized == "auto":
        normalized = detector.ensure_default_backend(
            auto_install=auto_install,
            install_tensorflow=True,
        )

    if normalized not in backends:
        raise ValueError(
            f"Unknown backend: {name}. Available: {list(backends.keys())}"
        )

    backend_cls = backends[normalized]
    try:
        backend = backend_cls()
    except ImportError as exc:
        if auto_install:
            detector.install_backend_runtime(normalized)
            backend = backend_cls()
        else:
            available = detector.get_available_backends()
            install_cmd = detector.backend_install_command(normalized)
            raise ValueError(
                f"Backend '{normalized}' is not available. "
                f"Available backends: {available}. "
                f"Install it with: {install_cmd}. "
                "Or set BASE_ATTENTIVE_AUTO_INSTALL=1."
            ) from exc

    if requested_name is None:
        _CURRENT_BACKEND = backend
    return backend


def get_backend_capabilities(
    name: str | None = None,
) -> dict[str, Any]:
    backends = _backend_classes()

    if name is None:
        try:
            backend = get_backend()
            caps = backend.get_capabilities()
            caps.setdefault(
                "name", getattr(backend, "name", "unknown")
            )
            caps.setdefault(
                "framework",
                getattr(
                    backend,
                    "framework",
                    getattr(backend, "name", "unknown"),
                ),
            )
            caps.setdefault(
                "available",
                backend.is_available()
                if hasattr(backend, "is_available")
                else True,
            )
            caps.setdefault(
                "uses_keras_runtime",
                getattr(backend, "uses_keras_runtime", False),
            )
            caps.setdefault(
                "experimental",
                getattr(backend, "experimental", False),
            )
            caps.setdefault(
                "supports_base_attentive",
                getattr(
                    backend, "supports_base_attentive", False
                ),
            )
            caps.setdefault(
                "supports_base_attentive_v2",
                getattr(
                    backend,
                    "supports_base_attentive_v2",
                    False,
                ),
            )
            caps.setdefault(
                "blockers",
                list(getattr(backend, "blockers", ())),
            )
            caps.setdefault(
                "v2_blockers",
                list(getattr(backend, "v2_blockers", ())),
            )
            caps.setdefault(
                "version",
                get_backend_version(
                    getattr(backend, "name", "tensorflow")
                ),
            )
            return caps
        except Exception:
            name = os.environ.get(
                "BASE_ATTENTIVE_BACKEND"
            ) or os.environ.get(
                "KERAS_BACKEND",
                "tensorflow",
            )

    normalized = normalize_backend_name(name)
    if normalized not in backends:
        raise ValueError(
            f"Unknown backend: {name}. Available: {list(backends.keys())}"
        )

    backend_cls = backends[normalized]
    try:
        backend = backend_cls(load_runtime=False)
        caps = backend.get_capabilities()
        caps.setdefault(
            "name", getattr(backend, "name", normalized)
        )
        caps.setdefault(
            "framework",
            getattr(backend_cls, "framework", normalized),
        )
        caps.setdefault(
            "available",
            backend.is_available()
            if hasattr(backend, "is_available")
            else True,
        )
        caps.setdefault(
            "uses_keras_runtime",
            getattr(backend, "uses_keras_runtime", False),
        )
        caps.setdefault(
            "experimental",
            getattr(backend, "experimental", False),
        )
        caps.setdefault(
            "supports_base_attentive",
            getattr(
                backend, "supports_base_attentive", False
            ),
        )
        caps.setdefault(
            "supports_base_attentive_v2",
            getattr(
                backend, "supports_base_attentive_v2", False
            ),
        )
        caps.setdefault(
            "blockers", list(getattr(backend, "blockers", ()))
        )
        caps.setdefault(
            "v2_blockers",
            list(getattr(backend, "v2_blockers", ())),
        )
        caps["version"] = get_backend_version(normalized)
        return caps
    except Exception as exc:
        return {
            "name": normalized,
            "framework": getattr(
                backend_cls, "framework", normalized
            ),
            "available": False,
            "uses_keras_runtime": getattr(
                backend_cls,
                "uses_keras_runtime",
                False,
            ),
            "experimental": getattr(
                backend_cls, "experimental", False
            ),
            "supports_base_attentive": getattr(
                backend_cls,
                "supports_base_attentive",
                False,
            ),
            "supports_base_attentive_v2": getattr(
                backend_cls,
                "supports_base_attentive_v2",
                False,
            ),
            "blockers": list(
                getattr(backend_cls, "blockers", ())
            ),
            "v2_blockers": list(
                getattr(backend_cls, "v2_blockers", ())
            ),
            "version": get_backend_version(normalized),
            "error": str(exc),
        }


def set_backend(name: str):
    global _CURRENT_BACKEND

    normalized = normalize_backend_name(name)

    if normalized == "tensorflow":
        is_compatible, msg = check_tensorflow_compatibility()
        if not is_compatible:
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    base = _module("base_attentive.backend.base")
    loaded_backend = base._read_loaded_keras_backend()
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


def _auto_initialize():
    env_name = os.environ.get("BASE_ATTENTIVE_BACKEND")
    if env_name is None:
        env_name = os.environ.get("KERAS_BACKEND")
    if env_name is None or not str(env_name).strip():
        raise RuntimeError(
            "BaseAttentive backend is not configured. Set BASE_ATTENTIVE_BACKEND first."
        )
    if normalize_backend_name(env_name) == "auto":
        chosen = ensure_default_backend(
            auto_install=os.environ.get("BASE_ATTENTIVE_AUTO_INSTALL", "0").strip().lower() in {"1", "true", "yes", "on"},
            install_tensorflow=True,
        )
        return set_backend(chosen)
    return set_backend(env_name)


# lazy attribute surface --------------------------------------------------
_LAZY_ATTRS = {
    "_BACKENDS": ("base_attentive.backend.detector", "_BACKENDS"),
    "Backend": ("base_attentive.backend.base", "Backend"),
    "TensorFlowBackend": (
        "base_attentive.backend.implementations",
        "TensorFlowBackend",
    ),
    "JaxBackend": (
        "base_attentive.backend.implementations",
        "JaxBackend",
    ),
    "TorchBackend": (
        "base_attentive.backend.implementations",
        "TorchBackend",
    ),
    "PyTorchBackend": (
        "base_attentive.backend.implementations",
        "PyTorchBackend",
    ),
    "TorchDeviceManager": (
        "base_attentive.backend.torch_utils",
        "TorchDeviceManager",
    ),
    "get_torch_device": (
        "base_attentive.backend.torch_utils",
        "get_torch_device",
    ),
    "get_torch_version": (
        "base_attentive.backend.torch_utils",
        "get_torch_version",
    ),
    "torch_is_available": (
        "base_attentive.backend.torch_utils",
        "torch_is_available",
    ),
    "_has_module": (
        "base_attentive.backend.detector",
        "_has_module",
    ),
    "_import_module": (
        "base_attentive.backend.detector",
        "_import_module",
    ),
}


def __getattr__(name: str):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    module_name, attr_name = target
    value = getattr(_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
