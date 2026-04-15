"""Core BaseAttentive package exports."""

from __future__ import annotations

import importlib

__all__ = ["BaseAttentive", "BaseAttentiveLegacy"]

_LAZY_EXPORTS = {
    "BaseAttentive": (
        "base_attentive.core.base_attentive",
        "BaseAttentive",
    ),
    "BaseAttentiveLegacy": (
        "base_attentive.core.base_attentive_legacy",
        "BaseAttentive",
    ),
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    module_name, export_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, export_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
