"""Model assembly helpers for V2."""

from __future__ import annotations

from typing import Any

from ..config import BaseAttentiveSpec
from ..implementations.generic import ensure_generic_v2_registered
from ..registry import (
    DEFAULT_COMPONENT_REGISTRY,
    DEFAULT_MODEL_REGISTRY,
    ComponentRegistry,
    ModelRegistry,
)
from .backend_context import BackendContext


def assemble_model(
    key: str,
    *,
    spec: BaseAttentiveSpec,
    backend_context: BackendContext,
    component_registry: ComponentRegistry | None = None,
    model_registry: ModelRegistry | None = None,
    **kwargs: Any,
) -> Any:
    """Resolve and assemble a V2 model."""
    ensure_generic_v2_registered(
        component_registry=component_registry,
        model_registry=model_registry,
    )
    active_model_registry = model_registry or DEFAULT_MODEL_REGISTRY
    registration = active_model_registry.resolve(
        key,
        backend=backend_context.name,
        allow_generic=True,
    )
    return registration.builder(
        spec=spec,
        backend_context=backend_context,
        component_registry=component_registry or DEFAULT_COMPONENT_REGISTRY,
        **kwargs,
    )


__all__ = ["assemble_model"]
