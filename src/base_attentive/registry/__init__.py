"""Registries for V2 components and model assemblers."""

from .capabilities import BackendCapabilityReport, get_backend_capability_report
from .component_registry import (
    DEFAULT_COMPONENT_REGISTRY,
    ComponentRegistration,
    ComponentRegistry,
)
from .model_registry import (
    DEFAULT_MODEL_REGISTRY,
    ModelRegistration,
    ModelRegistry,
)

__all__ = [
    "BackendCapabilityReport",
    "get_backend_capability_report",
    "ComponentRegistration",
    "ComponentRegistry",
    "DEFAULT_COMPONENT_REGISTRY",
    "ModelRegistration",
    "ModelRegistry",
    "DEFAULT_MODEL_REGISTRY",
]
