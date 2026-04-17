"""Backend runtime requirements and installation guidance."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BACKEND_INSTALL_SPECS",
    "backend_packages",
    "backend_install_command",
    "backend_description",
]

BACKEND_INSTALL_SPECS: dict[str, dict[str, Any]] = {
    "tensorflow": {
        "packages": ("tensorflow", "keras"),
        "install": "pip install tensorflow keras",
        "description": "TensorFlow-backed Keras runtime",
    },
    "torch": {
        "packages": ("torch", "keras"),
        "install": "pip install torch keras",
        "description": "Keras-on-Torch runtime",
    },
    "jax": {
        "packages": ("jax", "jaxlib", "keras"),
        "install": "pip install jax jaxlib keras",
        "description": "Keras-on-JAX runtime",
    },
}


def backend_packages(name: str | None) -> tuple[str, ...]:
    spec = BACKEND_INSTALL_SPECS.get(str(name or "").strip().lower(), {})
    return tuple(spec.get("packages", ()))


def backend_install_command(name: str | None) -> str:
    spec = BACKEND_INSTALL_SPECS.get(str(name or "").strip().lower(), {})
    return str(spec.get("install", "pip install keras"))


def backend_description(name: str | None) -> str:
    spec = BACKEND_INSTALL_SPECS.get(str(name or "").strip().lower(), {})
    return str(spec.get("description", "Keras runtime"))
