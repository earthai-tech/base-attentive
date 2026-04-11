"""Component resolution helpers for V2."""

from __future__ import annotations

from typing import Any

from ..registry import (
    DEFAULT_COMPONENT_REGISTRY,
    ComponentRegistration,
    ComponentRegistry,
)
from .backend_context import BackendContext


def resolve_component(
    key: str,
    *,
    backend_context: BackendContext,
    registry: ComponentRegistry | None = None,
    allow_generic: bool = True,
) -> ComponentRegistration:
    """Resolve a component registration for the requested backend."""
    active_registry = registry or DEFAULT_COMPONENT_REGISTRY
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
    allow_generic: bool = True,
    **kwargs: Any,
) -> Any:
    """Resolve and build a component for the requested backend."""
    registration = resolve_component(
        key,
        backend_context=backend_context,
        registry=registry,
        allow_generic=allow_generic,
    )
    return registration.builder(
        context=backend_context,
        **kwargs,
    )


__all__ = [
    "resolve_component",
    "build_component",
]
