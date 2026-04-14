"""Assembly objects returned by resolver-driven model builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .backend_context import BackendContext


@dataclass
class BaseAttentiveV2Assembly:
    """Resolved V2 model components.

    The assembly keeps the original V2 field names for compatibility,
    while also exposing migrated component names used by the
    legacy-to-resolver rewrite.
    """

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

    # First migration wave aliases.
    static_processor: Any | None = None
    dynamic_processor: Any | None = None
    future_processor: Any | None = None
    encoder_positional_encoding: Any | None = None
    future_positional_encoding: Any | None = None

    # Decoder migration wave.
    dynamic_window: Any | None = None
    decoder_input_projection: Any | None = None
    decoder_cross_attention: Any | None = None
    decoder_cross_postprocess: Any | None = None
    decoder_hierarchical_attention: Any | None = None
    decoder_memory_attention: Any | None = None
    decoder_fusion: Any | None = None
    residual_projection: Any | None = None
    decoder_residual_add: Any | None = None
    decoder_residual_norm: Any | None = None
    final_residual_add: Any | None = None
    final_residual_norm: Any | None = None
    final_pool: Any | None = None

    # Output migration wave.
    multi_horizon_head: Any | None = None
    quantile_distribution_head: Any | None = None


__all__ = ["BaseAttentiveV2Assembly"]
