"""Keras compatibility helpers for multi-backend runtimes.

This module is intentionally backend-neutral. It prefers the standalone
``keras`` package used by Keras 3 multi-backend runtimes and only falls
back to ``tensorflow.keras`` when needed for the TensorFlow backend or
legacy environments.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "import_keras_attr",
    "standalone_keras",
]


_KERAS_IMPORT_ERROR = (
    "Module {module_name!r} could not be imported from standalone "
    "keras or tensorflow.keras. Ensure that Keras 3 is installed and, "
    "for the TensorFlow backend, that TensorFlow is available."
)


def import_keras_attr(module_name: str) -> Any:
    """Import an attribute from ``keras`` with a TensorFlow fallback.

    Parameters
    ----------
    module_name : str
        Attribute to load from the Keras root module, such as
        ``"activations"`` or ``"layers"``.

    Returns
    -------
    Any
        Requested Keras attribute.

    Raises
    ------
    ImportError
        Raised when neither ``keras`` nor ``tensorflow.keras`` exposes
        the requested attribute.
    """
    errors: list[BaseException] = []

    try:
        keras = importlib.import_module("keras")
        return getattr(keras, module_name)
    except (
        ImportError,
        AttributeError,
        ModuleNotFoundError,
    ) as exc:
        errors.append(exc)

    try:
        tf_keras = importlib.import_module("tensorflow.keras")
        return getattr(tf_keras, module_name)
    except (
        ImportError,
        AttributeError,
        ModuleNotFoundError,
    ) as exc:
        errors.append(exc)

    raise ImportError(
        _KERAS_IMPORT_ERROR.format(module_name=module_name)
    ) from (errors[-1] if errors else None)


# Backward-compatible alias kept for older imports.
standalone_keras = import_keras_attr
