"""Normalization helpers for BaseAttentive specs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from .. import KERAS_BACKEND
from ..backend import normalize_backend_name
from .defaults import (
    DEFAULT_BASE_ATTENTIVE_ARCHITECTURE,
    DEFAULT_BASE_ATTENTIVE_COMPONENTS,
    DEFAULT_BASE_ATTENTIVE_RUNTIME,
    DEFAULT_BASE_ATTENTIVE_V2_CONFIG,
)
from .schema import (
    BaseAttentiveArchitectureSpec,
    BaseAttentiveComponentSpec,
    BaseAttentiveRuntimeSpec,
    BaseAttentiveSpec,
)
from .validate import validate_base_attentive_spec


def _coerce_mapping(
    value: Any, *, name: str
) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(
        f"{name} must be a mapping or dataclass instance."
    )


def _serialize_value(value: Any) -> Any:
    """Convert nested dataclass payloads to serialization-safe values."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {
            str(key): _serialize_value(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_serialize_value(item) for item in value]
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    return value


def serialize_base_attentive_spec(
    spec: BaseAttentiveSpec | Mapping[str, Any],
) -> dict[str, Any]:
    """Return a normalized, serialization-safe spec mapping.

    The output is suitable for Keras config round-trips and avoids
    leaking legacy alias placements such as
    ``architecture.sequence_pooling`` back into saved configs.
    """
    normalized = normalize_base_attentive_spec(spec)
    return _serialize_value(asdict(normalized))


def normalize_architecture_spec(
    architecture: (
        BaseAttentiveArchitectureSpec
        | Mapping[str, Any]
        | None
    ) = None,
    **overrides: Any,
) -> BaseAttentiveArchitectureSpec:
    """Normalize logical architecture selections."""
    data = dict(DEFAULT_BASE_ATTENTIVE_ARCHITECTURE)
    data.update(
        _coerce_mapping(architecture, name="architecture")
    )
    data.update(overrides)

    if "decoder_attention_stack" in data:
        values = data["decoder_attention_stack"]
        if values is None:
            values = ()
        elif isinstance(values, str):
            values = (values,)
        else:
            values = tuple(values)
        data["decoder_attention_stack"] = values

    return BaseAttentiveArchitectureSpec(**data)


def normalize_runtime_spec(
    runtime: BaseAttentiveRuntimeSpec
    | Mapping[str, Any]
    | None = None,
    **overrides: Any,
) -> BaseAttentiveRuntimeSpec:
    """Normalize runtime options shared across model variants."""
    data = dict(DEFAULT_BASE_ATTENTIVE_RUNTIME)
    data.update(_coerce_mapping(runtime, name="runtime"))
    data.update(overrides)

    scales = data.get("scales", ())
    if scales is None:
        data["scales"] = ()
    elif isinstance(scales, str):
        data["scales"] = scales.lower()
    else:
        data["scales"] = tuple(scales)

    return BaseAttentiveRuntimeSpec(**data)


def normalize_component_spec(
    components: BaseAttentiveComponentSpec
    | Mapping[str, Any]
    | None = None,
    **overrides: Any,
) -> BaseAttentiveComponentSpec:
    """Normalize logical component selections."""
    data = dict(DEFAULT_BASE_ATTENTIVE_COMPONENTS)
    data.update(
        _coerce_mapping(components, name="components")
    )
    data.update(overrides)
    return BaseAttentiveComponentSpec(**data)


def normalize_base_attentive_spec(
    spec: BaseAttentiveSpec | Mapping[str, Any] | None = None,
    **overrides: Any,
) -> BaseAttentiveSpec:
    """Normalize user input into a validated spec."""
    data = dict(DEFAULT_BASE_ATTENTIVE_V2_CONFIG)
    data.update(_coerce_mapping(spec, name="spec"))
    data.update(overrides)

    component_data = data.pop("components", None)
    if component_data is None:
        component_data = data.pop("component_keys", None)
    if component_data is None:
        component_data = data.pop("component_overrides", None)

    architecture_data = data.pop("architecture", None)
    runtime_data = data.pop("runtime", None)

    architecture_mapping = _coerce_mapping(
        architecture_data,
        name="architecture",
    )
    migrated_component_keys = {}
    for alias in ("sequence_pooling", "fusion"):
        if alias in architecture_mapping:
            migrated_component_keys[alias] = (
                architecture_mapping.pop(alias)
            )

    extras = dict(data.pop("extras", {}) or {})
    if spec is not None and hasattr(spec, "extras"):
        extras = {**getattr(spec, "extras"), **extras}

    if "num_heads" in data and "attention_heads" not in data:
        data["attention_heads"] = data.pop("num_heads")

    backend_name = normalize_backend_name(
        data.get("backend_name") or KERAS_BACKEND
    )
    data["backend_name"] = backend_name
    data["components"] = normalize_component_spec(
        component_data,
        **migrated_component_keys,
    )
    data["architecture"] = normalize_architecture_spec(
        architecture_mapping
    )
    data["runtime"] = normalize_runtime_spec(runtime_data)

    if "quantiles" in data and data["quantiles"] is not None:
        data["quantiles"] = tuple(data["quantiles"])

    known_fields = (
        BaseAttentiveSpec.__dataclass_fields__.keys()
    )
    extra_fields = {
        key: value
        for key, value in list(data.items())
        if key not in known_fields
    }
    for key in extra_fields:
        data.pop(key)
    data["extras"] = {**extras, **extra_fields}

    normalized = BaseAttentiveSpec(**data)
    return validate_base_attentive_spec(normalized)


__all__ = [
    "normalize_architecture_spec",
    "normalize_base_attentive_spec",
    "normalize_component_spec",
    "normalize_runtime_spec",
    "serialize_base_attentive_spec",
]
