"""Assembly objects returned by V2 model resolvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backend_context import BackendContext


@dataclass
class BaseAttentiveV2Assembly:
    """Resolved V2 model components."""

    backend_context: BackendContext
    static_projection: Any | None
    dynamic_projection: Any
    future_projection: Any | None
    dynamic_encoder: Any | None
    future_encoder: Any | None
    sequence_pool: Any
    fusion: Any
    hidden_projection: Any
    output_head: Any
    dropout: Any | None = None


__all__ = ["BaseAttentiveV2Assembly"]
