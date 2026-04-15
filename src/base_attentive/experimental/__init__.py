"""Experimental models and migration paths."""

from __future__ import annotations

import importlib

__all__ = ["BaseAttentiveV2"]


def __getattr__(name: str):
    if name != "BaseAttentiveV2":
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    module = importlib.import_module(
        "base_attentive.experimental.base_attentive_v2"
    )
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
