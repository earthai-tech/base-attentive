"""Model assembly helpers for resolver-driven models."""

from __future__ import annotations

from typing import Any

from ..config import BaseAttentiveSpec
from ..registry import (
    DEFAULT_COMPONENT_REGISTRY,
    DEFAULT_MODEL_REGISTRY,
    ComponentRegistry,
    ModelRegistry,
)
from .backend_context import BackendContext
from .registrars import ensure_backend_registrations


def assemble_model(
    key: str,
    *,
    spec: BaseAttentiveSpec,
    backend_context: BackendContext,
    component_registry: ComponentRegistry | None = None,
    model_registry: ModelRegistry | None = None,
    **kwargs: Any,
) -> Any:
    """Resolve and assemble a model for the requested backend."""
    active_component_registry, active_model_registry = (
        ensure_backend_registrations(
            backend_context=backend_context,
            component_registry=component_registry,
            model_registry=model_registry,
        )
    )

    registration = active_model_registry.resolve(
        key,
        backend=backend_context.name,
        allow_generic=True,
    )
    return registration.builder(
        spec=spec,
        backend_context=backend_context,
        component_registry=active_component_registry,
        **kwargs,
    )


__all__ = ["assemble_model"]
