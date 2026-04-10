"""Dependency utilities for base-attentive."""

from __future__ import annotations

import functools
import importlib
import sys
from typing import Any, Callable, TypeVar

_T = TypeVar("_T")

__all__ = [
    "ensure_pkg",
]


def ensure_pkg(
    name: str,
    extra: str = "",
    error: str = "raise",
    **kwargs,
) -> Callable[[_T], _T]:
    """
    Decorator to ensure a Python package is installed before function execution.

    Parameters
    ----------
    name : str
        The name of the package to import.
    extra : str, optional
        Additional message to show if package is missing.
    error : str, optional
        Error handling strategy: 'raise' (default), 'warn', or 'ignore'.
    **kwargs
        Additional arguments (ignored for compatibility).

    Returns
    -------
    Callable
        A decorator that ensures the package is available.
    """

    def decorator(func: _T) -> _T:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Try to import the package
            try:
                importlib.import_module(name)
            except ImportError as e:
                msg = f"Package '{name}' is required but not installed."
                if extra:
                    msg += f" {extra}"

                if error == "raise":
                    raise ImportError(msg) from e
                elif error == "warn":
                    import warnings
                    warnings.warn(msg, UserWarning, stacklevel=2)
                # else: ignore

            # Call the original function
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
