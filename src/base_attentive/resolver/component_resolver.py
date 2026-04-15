"""Component resolution helpers for resolver-driven models."""

from __future__ import annotations

from typing import Any

from ..registry import (
    ComponentRegistration,
    ComponentRegistry,
    ModelRegistry,
)
from .backend_context import BackendContext
from .registrars import ensure_backend_registrations


def resolve_component(
    key: str,
    *,
    backend_context: BackendContext,
    registry: ComponentRegistry | None = None,
    model_registry: ModelRegistry | None = None,
    allow_generic: bool = True,
) -> ComponentRegistration:
    """Resolve a component registration for the requested backend."""
    active_registry, _ = ensure_backend_registrations(
        backend_context=backend_context,
        component_registry=registry,
        model_registry=model_registry,
    )
    return active_registry.resolve(
        key,
        backend=backend_context.name,
        allow_generic=allow_generic,
    )


def build_component(
    key: str,
    *,
    backend_context: BackendContext,
    registry: ComponentRegistry | None = None,
    model_registry: ModelRegistry | None = None,
    allow_generic: bool = True,
    spec: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Resolve and build a component for the requested backend."""
    registration = resolve_component(
        key,
        backend_context=backend_context,
        registry=registry,
        model_registry=model_registry,
        allow_generic=allow_generic,
    )
    return registration.builder(
        context=backend_context,
        spec=spec,
        component_key=key,
        **kwargs,
    )


__all__ = [
    "resolve_component",
    "build_component",
]
