"""Component utility functions."""

from __future__ import annotations

from typing import Any, Optional

__all__ = ["resolve_attention_levels"]


def resolve_attention_levels(
    architecture_config: Optional[Any] = None,
) -> Any:
    """
    Resolve attention levels from architecture configuration.

    Parameters
    ----------
    architecture_config : dict, list, str, optional
        Architecture configuration dictionary or an attention-level selector.

    Returns
    -------
    Any
        Either a resolved architecture dict or a decoder attention stack list.
    """
    default_attention = {
        "decoder_attention_stack": [
            "cross",
            "hierarchical",
            "memory",
        ],
        "attention_heads": 4,
        "attention_dim": 64,
    }

    if architecture_config is None:
        return default_attention

    if isinstance(architecture_config, dict):
        return {**default_attention, **architecture_config}

    aliases = {
        "cross_attention": "cross",
        "hier_att": "hierarchical",
        "hierarchical_attention": "hierarchical",
        "memo_aug_att": "memory",
        "memory_augmented_attention": "memory",
    }

    if isinstance(architecture_config, str):
        normalized = architecture_config.strip().lower()
        if normalized in {"*", "all", "use_all", "auto"}:
            return list(
                default_attention["decoder_attention_stack"]
            )

        resolved = aliases.get(normalized, normalized)
        if (
            resolved
            not in default_attention[
                "decoder_attention_stack"
            ]
        ):
            raise ValueError(
                f"Unknown attention level: {architecture_config}"
            )
        return [resolved]

    if isinstance(architecture_config, (list, tuple)):
        resolved_stack = []
        for level in architecture_config:
            if not isinstance(level, str):
                raise TypeError(
                    "Attention levels must contain only string values."
                )
            normalized = aliases.get(
                level.strip().lower(), level.strip().lower()
            )
            if (
                normalized
                not in default_attention[
                    "decoder_attention_stack"
                ]
            ):
                raise ValueError(
                    f"Unknown attention level: {level}"
                )
            resolved_stack.append(normalized)
        return resolved_stack

    raise TypeError(
        "architecture_config must be a dict, string, or sequence of strings."
    )
