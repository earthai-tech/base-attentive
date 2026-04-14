"""Capability reporting helpers for V2."""

from __future__ import annotations

from dataclasses import dataclass

from ..backend import (
    get_backend_capabilities,
    normalize_backend_name,
)


@dataclass(frozen=True)
class BackendCapabilityReport:
    """Normalized backend capability summary."""

    name: str
    framework: str
    available: bool
    uses_keras_runtime: bool
    experimental: bool
    supports_base_attentive: bool
    supports_base_attentive_v2: bool
    blockers: tuple[str, ...] = ()
    v2_blockers: tuple[str, ...] = ()
    version: str | None = None
    error: str | None = None


def get_backend_capability_report(
    name: str | None = None,
) -> BackendCapabilityReport:
    """Return a normalized capability report."""
    caps = get_backend_capabilities(name)
    normalized_name = normalize_backend_name(caps["name"])
    return BackendCapabilityReport(
        name=normalized_name,
        framework=caps.get("framework", normalized_name),
        available=bool(caps.get("available", False)),
        uses_keras_runtime=bool(
            caps.get("uses_keras_runtime", False)
        ),
        experimental=bool(caps.get("experimental", False)),
        supports_base_attentive=bool(
            caps.get("supports_base_attentive", False)
        ),
        supports_base_attentive_v2=bool(
            caps.get("supports_base_attentive_v2", False)
        ),
        blockers=tuple(caps.get("blockers", ())),
        v2_blockers=tuple(caps.get("v2_blockers", ())),
        version=caps.get("version"),
        error=caps.get("error"),
    )


__all__ = [
    "BackendCapabilityReport",
    "get_backend_capability_report",
]
