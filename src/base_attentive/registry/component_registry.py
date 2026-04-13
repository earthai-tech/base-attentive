"""Component registry for V2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..backend import normalize_backend_name


@dataclass(frozen=True)
class ComponentRegistration:
    """Registered component builder metadata."""

    key: str
    backend: str
    builder: Callable[..., Any]
    description: str = ""
    experimental: bool = False


class ComponentRegistry:
    """Registry of backend-specific component builders."""

    def __init__(self):
        self._registrations: dict[
            str, dict[str, ComponentRegistration]
        ] = {}

    def register(
        self,
        key: str,
        builder: Callable[..., Any],
        *,
        backend: str = "generic",
        description: str = "",
        experimental: bool = False,
        replace: bool = False,
    ) -> ComponentRegistration:
        normalized_backend = (
            "generic"
            if backend == "generic"
            else normalize_backend_name(backend)
        )
        by_backend = self._registrations.setdefault(key, {})
        if normalized_backend in by_backend and not replace:
            raise KeyError(
                f"Component {key!r} is already registered for backend "
                f"{normalized_backend!r}."
            )

        registration = ComponentRegistration(
            key=key,
            backend=normalized_backend,
            builder=builder,
            description=description,
            experimental=experimental,
        )
        by_backend[normalized_backend] = registration
        return registration

    def has(self, key: str, *, backend: str | None = None) -> bool:
        if key not in self._registrations:
            return False
        if backend is None:
            return True
        normalized_backend = (
            "generic"
            if backend == "generic"
            else normalize_backend_name(backend)
        )
        return normalized_backend in self._registrations[key]

    def resolve(
        self,
        key: str,
        *,
        backend: str,
        allow_generic: bool = True,
    ) -> ComponentRegistration:
        normalized_backend = normalize_backend_name(backend)
        by_backend = self._registrations.get(key)
        if not by_backend:
            raise KeyError(f"Unknown component key: {key!r}.")

        registration = by_backend.get(normalized_backend)
        if registration is not None:
            return registration

        if allow_generic:
            registration = by_backend.get("generic")
            if registration is not None:
                return registration

        available = ", ".join(sorted(by_backend))
        raise KeyError(
            f"Component {key!r} is not registered for backend "
            f"{normalized_backend!r}. Available: {available}."
        )

    def list_keys(self) -> list[str]:
        return sorted(self._registrations)

    def clone(self) -> "ComponentRegistry":
        cloned = ComponentRegistry()
        cloned._registrations = {
            key: dict(registrations)
            for key, registrations in self._registrations.items()
        }
        return cloned


DEFAULT_COMPONENT_REGISTRY = ComponentRegistry()

__all__ = [
    "ComponentRegistration",
    "ComponentRegistry",
    "DEFAULT_COMPONENT_REGISTRY",
]
