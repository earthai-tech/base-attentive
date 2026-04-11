"""Experimental models and migration paths."""

from __future__ import annotations

import warnings

__all__: list[str] = []

try:
    from .base_attentive_v2 import BaseAttentiveV2
except Exception as exc:
    warnings.warn(
        f"Failed to import BaseAttentiveV2 from base_attentive.experimental: {exc}",
        RuntimeWarning,
        stacklevel=2,
    )
else:
    __all__.append("BaseAttentiveV2")
