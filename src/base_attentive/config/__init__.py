"""Configuration helpers for BaseAttentive models."""

from .architecture_helpers import (
    configure_architecture,
    resolve_attn_levels,
    resolve_fusion_mode,
)
from .defaults import (
    DEFAULT_BASE_ATTENTIVE_ARCHITECTURE,
    DEFAULT_BASE_ATTENTIVE_COMPONENTS,
    DEFAULT_BASE_ATTENTIVE_RUNTIME,
    DEFAULT_BASE_ATTENTIVE_V2_CONFIG,
    DEFAULT_LEGACY_ARCHITECTURE,
)
from .legacy_adapter import (
    legacy_base_attentive_to_spec,
    normalize_legacy_architecture_spec,
    normalize_legacy_runtime_spec,
)
from .normalize import (
    normalize_architecture_spec,
    normalize_base_attentive_spec,
    normalize_component_spec,
    normalize_runtime_spec,
    serialize_base_attentive_spec,
)
from .schema import (
    BaseAttentiveArchitectureSpec,
    BaseAttentiveComponentSpec,
    BaseAttentiveRuntimeSpec,
    BaseAttentiveSpec,
)
from .validate import validate_base_attentive_spec

__all__ = [
    "BaseAttentiveArchitectureSpec",
    "configure_architecture",
    "BaseAttentiveComponentSpec",
    "BaseAttentiveRuntimeSpec",
    "BaseAttentiveSpec",
    "DEFAULT_BASE_ATTENTIVE_ARCHITECTURE",
    "DEFAULT_BASE_ATTENTIVE_COMPONENTS",
    "DEFAULT_BASE_ATTENTIVE_RUNTIME",
    "DEFAULT_BASE_ATTENTIVE_V2_CONFIG",
    "DEFAULT_LEGACY_ARCHITECTURE",
    "legacy_base_attentive_to_spec",
    "resolve_attn_levels",
    "resolve_fusion_mode",
    "normalize_architecture_spec",
    "normalize_base_attentive_spec",
    "normalize_component_spec",
    "serialize_base_attentive_spec",
    "normalize_legacy_architecture_spec",
    "normalize_legacy_runtime_spec",
    "normalize_runtime_spec",
    "validate_base_attentive_spec",
]
