"""Normalization helpers for V2 specs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from .. import KERAS_BACKEND
from ..backend import normalize_backend_name
from .defaults import (
    DEFAULT_BASE_ATTENTIVE_COMPONENTS,
    DEFAULT_BASE_ATTENTIVE_V2_CONFIG,
)
from .schema import BaseAttentiveComponentSpec, BaseAttentiveSpec
from .validate import validate_base_attentive_spec


def _coerce_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"{name} must be a mapping or dataclass instance.")


def normalize_component_spec(
    components: BaseAttentiveComponentSpec | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> BaseAttentiveComponentSpec:
    """Normalize logical component selections for V2."""
    data = dict(DEFAULT_BASE_ATTENTIVE_COMPONENTS)
    data.update(_coerce_mapping(components, name="components"))
    data.update(overrides)
    return BaseAttentiveComponentSpec(**data)


def normalize_base_attentive_spec(
    spec: BaseAttentiveSpec | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> BaseAttentiveSpec:
    """Normalize user input into a validated ``BaseAttentiveSpec``."""
    data = dict(DEFAULT_BASE_ATTENTIVE_V2_CONFIG)
    data.update(_coerce_mapping(spec, name="spec"))
    data.update(overrides)

    component_data = data.pop("components", None)
    if component_data is None:
        component_data = data.pop("component_keys", None)
    if component_data is None:
        component_data = data.pop("architecture", None)

    extras = dict(data.pop("extras", {}) or {})
    if spec is not None and hasattr(spec, "extras"):
        extras = {**getattr(spec, "extras"), **extras}

    backend_name = normalize_backend_name(
        data.get("backend_name") or KERAS_BACKEND
    )
    data["backend_name"] = backend_name
    data["components"] = normalize_component_spec(component_data)
    if "quantiles" in data and data["quantiles"] is not None:
        data["quantiles"] = tuple(data["quantiles"])

    known_fields = BaseAttentiveSpec.__dataclass_fields__.keys()
    extra_fields = {
        key: value for key, value in list(data.items()) if key not in known_fields
    }
    for key in extra_fields:
        data.pop(key)
    data["extras"] = {**extras, **extra_fields}

    normalized = BaseAttentiveSpec(**data)
    return validate_base_attentive_spec(normalized)


__all__ = [
    "normalize_base_attentive_spec",
    "normalize_component_spec",
]
