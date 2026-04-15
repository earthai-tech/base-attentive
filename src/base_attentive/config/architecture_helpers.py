"""Architecture helper utilities for BaseAttentive.

These helpers are pure-Python normalization utilities used by tests and by
legacy configuration flows.  They are intentionally lightweight and should not
pull in any backend runtime.
"""

from __future__ import annotations

import warnings
from typing import Any

from ..utils.generic_utils import select_mode

__all__ = [
    "configure_architecture",
    "resolve_attn_levels",
    "resolve_fusion_mode",
]


_DEFAULT_ATTENTION = ["cross", "hierarchical", "memory"]


def resolve_attn_levels(
    attention_levels: str
    | int
    | list[str | int]
    | tuple[str | int, ...]
    | None,
) -> list[str]:
    """Resolve user attention selections into canonical decoder stack names."""
    if attention_levels is None:
        return list(_DEFAULT_ATTENTION)

    aliases = {
        "cross": "cross",
        "cross_att": "cross",
        "cross_attention": "cross",
        "hier": "hierarchical",
        "hier_att": "hierarchical",
        "hierarchical": "hierarchical",
        "hierarchical_attention": "hierarchical",
        "memory": "memory",
        "memo_aug": "memory",
        "memo_aug_att": "memory",
        "memory_augmented_attention": "memory",
        "1": "cross",
        "2": "hierarchical",
        "3": "memory",
    }
    int_aliases = {1: "cross", 2: "hierarchical", 3: "memory"}

    if isinstance(attention_levels, int):
        if attention_levels not in int_aliases:
            raise ValueError(
                f"Invalid integer for attention level: {attention_levels}"
            )
        return [int_aliases[attention_levels]]

    if isinstance(attention_levels, str):
        normalized = attention_levels.strip().lower()
        if normalized in {"*", "all", "use_all", "auto"}:
            return list(_DEFAULT_ATTENTION)
        if normalized not in aliases:
            raise ValueError(
                f"Invalid attention type: {attention_levels}"
            )
        return [aliases[normalized]]

    if isinstance(attention_levels, (list, tuple)):
        resolved: list[str] = []
        for value in attention_levels:
            if isinstance(value, int):
                if value not in int_aliases:
                    raise ValueError(
                        f"Invalid integer for attention level: {value}"
                    )
                resolved.append(int_aliases[value])
                continue
            if not isinstance(value, str):
                raise TypeError(
                    f"Invalid type for attention level: {type(value).__name__}"
                )
            normalized = value.strip().lower()
            if normalized not in aliases:
                raise ValueError(
                    f"Invalid attention type: {value}"
                )
            resolved.append(aliases[normalized])
        return resolved

    raise TypeError(
        f"Invalid type for attention level: {type(attention_levels).__name__}"
    )


def resolve_fusion_mode(mode: str | None) -> str:
    """Normalize fusion-mode aliases to canonical values."""
    if mode is None:
        return "integrated"

    normalized = str(mode).strip().lower()
    if normalized == "integrated":
        return "integrated"
    if normalized in {"disjoint", "independent", "isolated"}:
        return "disjoint"
    return "integrated"


def configure_architecture(
    *,
    objective: str | None = None,
    use_vsn: bool = True,
    attention_levels: str
    | int
    | list[str | int]
    | tuple[str | int, ...]
    | None = None,
    architecture_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the legacy-style architecture configuration mapping."""
    final_config: dict[str, Any] = {
        "encoder_type": "hybrid",
        "decoder_attention_stack": list(_DEFAULT_ATTENTION),
        "feature_processing": "vsn",
    }

    final_config["encoder_type"] = select_mode(
        objective,
        default="hybrid",
        canonical=["hybrid", "transformer"],
    )

    if not use_vsn:
        final_config["feature_processing"] = "dense"

    # Keep behavior aligned with the legacy helper but use the stricter
    # pure-Python resolver expected by the coverage tests.
    final_config["decoder_attention_stack"] = (
        resolve_attn_levels(attention_levels)
    )

    if architecture_config:
        user_config = architecture_config.copy()
        if "objective" in user_config:
            warnings.warn(
                "The 'objective' key-role in `architecture_config` is"
                " deprecated and will be rename in a future version."
                " Please use 'encoder_type' instead.",
                FutureWarning,
                stacklevel=2,
            )
            user_config["encoder_type"] = user_config.pop(
                "objective"
            )
        final_config.update(user_config)

    if (
        not use_vsn
        and final_config.get("feature_processing") == "vsn"
    ):
        final_config["feature_processing"] = "dense"

    # Normalize in case user_config set raw attention values.
    if "decoder_attention_stack" in final_config:
        final_config["decoder_attention_stack"] = (
            resolve_attn_levels(
                final_config["decoder_attention_stack"]
            )
        )

    return final_config
