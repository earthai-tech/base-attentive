"""Keras compatibility helpers for multi-backend runtimes.

This module is intentionally backend-neutral. It prefers the standalone
``keras`` package used by Keras 3 multi-backend runtimes and only falls
back to ``tensorflow.keras`` when TensorFlow is *already loaded*.
That avoids accidentally importing TensorFlow in lightweight paths on
platforms where TensorFlow import can be expensive or unstable.
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

__all__ = [
    "import_keras_attr",
    "standalone_keras",
]


_KERAS_IMPORT_ERROR = (
    "Module {module_name!r} could not be imported from standalone "
    "keras or an already-loaded tensorflow.keras namespace. Ensure that "
    "Keras 3 is installed and, for legacy TensorFlow fallback use, that "
    "TensorFlow is already loaded."
)


def _configured_backend() -> str:
    configured = os.environ.get("BASE_ATTENTIVE_BACKEND")
    if configured:
        return str(configured).strip().lower()
    configured = os.environ.get("KERAS_BACKEND")
    if configured:
        return str(configured).strip().lower()
    return "tensorflow"


def _allow_tf_fallback() -> bool:
    # Only use tensorflow.keras when TensorFlow is already loaded, or the
    # process is explicitly on the TensorFlow backend and TensorFlow has
    # already been imported by the caller. This keeps helper lookups from
    # causing surprise TensorFlow imports.
    if (
        "tensorflow" in sys.modules
        or "tensorflow.keras" in sys.modules
    ):
        return True
    return False


def import_keras_attr(module_name: str) -> Any:
    """Import an attribute from ``keras`` with a guarded TensorFlow fallback."""
    errors: list[BaseException] = []

    keras = None
    if sys.modules.get("keras") is None:
        try:
            keras = importlib.import_module("keras")
        except (ImportError, ModuleNotFoundError) as exc:
            errors.append(exc)
    else:
        keras = sys.modules["keras"]

    if keras is not None:
        try:
            return getattr(keras, module_name)
        except AttributeError as exc:
            errors.append(exc)

    if _allow_tf_fallback():
        tf_keras = sys.modules.get("tensorflow.keras")
        if tf_keras is None:
            try:
                tf_keras = importlib.import_module(
                    "tensorflow.keras"
                )
            except (
                ImportError,
                ModuleNotFoundError,
                AttributeError,
            ) as exc:
                errors.append(exc)
                tf_keras = None
        if tf_keras is not None:
            try:
                return getattr(tf_keras, module_name)
            except AttributeError as exc:
                errors.append(exc)

    raise ImportError(
        _KERAS_IMPORT_ERROR.format(module_name=module_name)
    ) from (errors[-1] if errors else None)


standalone_keras = import_keras_attr
