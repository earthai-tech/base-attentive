"""Registration loaders for backend-specific resolver builders."""

from __future__ import annotations

import importlib
import inspect
from typing import Any
from weakref import WeakKeyDictionary

from ..backend import normalize_backend_name
from ..registry import (
    DEFAULT_COMPONENT_REGISTRY,
    DEFAULT_MODEL_REGISTRY,
    ComponentRegistry,
    ModelRegistry,
)
from .backend_context import BackendContext

_COMPONENT_REGISTRARS = {
    "tensorflow": (
        "base_attentive.implementations.tensorflow",
        "ensure_tensorflow_v2_registered",
    ),
    "jax": (
        "base_attentive.implementations.jax",
        "ensure_jax_v2_registered",
    ),
    "torch": (
        "base_attentive.implementations.torch",
        "ensure_torch_v2_registered",
    ),
}

_GENERIC_REGISTRAR = (
    "base_attentive.implementations.generic",
    "ensure_generic_v2_registered",
)

_LOADED_COMPONENT_REGISTRARS: WeakKeyDictionary[
    ComponentRegistry,
    set[str],
] = WeakKeyDictionary()

_LOADED_MODEL_REGISTRARS: WeakKeyDictionary[
    ModelRegistry,
    set[str],
] = WeakKeyDictionary()


def _loaded_backends(
    state: WeakKeyDictionary[Any, set[str]],
    registry: Any,
) -> set[str]:
    loaded = state.get(registry)
    if loaded is None:
        loaded = set()
        state[registry] = loaded
    return loaded


def _import_registrar(
    module_name: str,
    attr_name: str,
):
    module = importlib.import_module(module_name)
    registrar = getattr(module, attr_name, None)
    if registrar is None:
        raise ImportError(
            f"Registrar {attr_name!r} was not found in "
            f"module {module_name!r}."
        )
    return registrar


def _call_registrar(
    registrar,
    *,
    component_registry: ComponentRegistry,
    model_registry: ModelRegistry,
) -> Any:
    parameters = inspect.signature(registrar).parameters
    kwargs: dict[str, Any] = {}

    if "registry" in parameters:
        kwargs["registry"] = component_registry
    if "component_registry" in parameters:
        kwargs["component_registry"] = component_registry
    if "model_registry" in parameters:
        kwargs["model_registry"] = model_registry

    return registrar(**kwargs)


def _ensure_generic_registrations(
    *,
    component_registry: ComponentRegistry,
    model_registry: ModelRegistry,
) -> None:
    registrar = _import_registrar(*_GENERIC_REGISTRAR)
    _call_registrar(
        registrar,
        component_registry=component_registry,
        model_registry=model_registry,
    )


def _ensure_backend_component_registrations(
    backend_name: str,
    *,
    component_registry: ComponentRegistry,
    model_registry: ModelRegistry,
) -> None:
    registrar_spec = _COMPONENT_REGISTRARS.get(backend_name)
    if registrar_spec is None:
        return

    loaded = _loaded_backends(
        _LOADED_COMPONENT_REGISTRARS,
        component_registry,
    )
    if backend_name in loaded:
        return

    registrar = _import_registrar(*registrar_spec)
    _call_registrar(
        registrar,
        component_registry=component_registry,
        model_registry=model_registry,
    )
    loaded.add(backend_name)


def ensure_backend_registrations(
    *,
    backend_context: BackendContext,
    component_registry: ComponentRegistry | None = None,
    model_registry: ModelRegistry | None = None,
) -> tuple[ComponentRegistry, ModelRegistry]:
    """Ensure generic and backend-specific builders are loaded."""
    active_component_registry = (
        component_registry or DEFAULT_COMPONENT_REGISTRY
    )
    active_model_registry = (
        model_registry or DEFAULT_MODEL_REGISTRY
    )

    _ensure_generic_registrations(
        component_registry=active_component_registry,
        model_registry=active_model_registry,
    )

    normalized_backend = normalize_backend_name(
        backend_context.name
    )
    if (
        normalized_backend != "generic"
        and backend_context.capability_report.available
    ):
        try:
            _ensure_backend_component_registrations(
                normalized_backend,
                component_registry=active_component_registry,
                model_registry=active_model_registry,
            )
        except ImportError:
            pass

    _loaded_backends(
        _LOADED_MODEL_REGISTRARS,
        active_model_registry,
    )

    return active_component_registry, active_model_registry


__all__ = ["ensure_backend_registrations"]
