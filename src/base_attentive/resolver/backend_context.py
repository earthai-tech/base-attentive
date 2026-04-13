"""Backend context objects for the V2 resolver."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from .. import KERAS_BACKEND, KERAS_DEPS
from ..backend import normalize_backend_name
from ..registry import (
    BackendCapabilityReport,
    get_backend_capability_report,
)


def _resolve_optional_attr(namespace: Any, name: str) -> Any:
    try:
        return getattr(namespace, name)
    except (AttributeError, ImportError):
        return None


@dataclass(frozen=True)
class BackendContext:
    """Runtime context used when resolving V2 components."""

    name: str
    capability_report: BackendCapabilityReport
    keras_deps: Any = KERAS_DEPS

    @property
    def ops(self) -> Any:
        return self.keras_deps

    @property
    def layers(self) -> Any:
        return SimpleNamespace(
            Dense=self.keras_deps.Dense,
            Dropout=_resolve_optional_attr(
                self.keras_deps, "Dropout"
            ),
            Layer=self.keras_deps.Layer,
            Model=self.keras_deps.Model,
            MultiHeadAttention=_resolve_optional_attr(
                self.keras_deps,
                "MultiHeadAttention",
            ),
            LayerNormalization=_resolve_optional_attr(
                self.keras_deps,
                "LayerNormalization",
            ),
        )

    @classmethod
    def current(cls, name: str | None = None) -> "BackendContext":
        normalized_name = normalize_backend_name(
            name or KERAS_BACKEND
        )
        capability_report = get_backend_capability_report(
            normalized_name
        )
        return cls(
            name=normalized_name,
            capability_report=capability_report,
        )


__all__ = ["BackendContext"]
