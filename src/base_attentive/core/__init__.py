"""Core BaseAttentive package exports."""

from __future__ import annotations

import warnings

__all__: list[str] = []

try:
    from base_attentive.core.base_attentive import (
        BaseAttentive as BaseAttentive,
    )
except Exception as exc:
    warnings.warn(
        f"Failed to import BaseAttentive from base_attentive.core: {exc}",
        RuntimeWarning,
        stacklevel=2,
    )
else:
    __all__.append("BaseAttentive")
