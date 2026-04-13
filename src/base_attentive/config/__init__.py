"""Backend-neutral configuration helpers for V2."""

from .defaults import (
    DEFAULT_BASE_ATTENTIVE_COMPONENTS,
    DEFAULT_BASE_ATTENTIVE_V2_CONFIG,
)
from .normalize import (
    normalize_base_attentive_spec,
    normalize_component_spec,
)
from .schema import BaseAttentiveComponentSpec, BaseAttentiveSpec
from .validate import validate_base_attentive_spec

__all__ = [
    "BaseAttentiveComponentSpec",
    "BaseAttentiveSpec",
    "DEFAULT_BASE_ATTENTIVE_COMPONENTS",
    "DEFAULT_BASE_ATTENTIVE_V2_CONFIG",
    "normalize_base_attentive_spec",
    "normalize_component_spec",
    "validate_base_attentive_spec",
]
