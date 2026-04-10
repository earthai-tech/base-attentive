"""Component utility functions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = ["resolve_attention_levels"]


def resolve_attention_levels(
    architecture_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Resolve attention levels from architecture configuration.

    Parameters
    ----------
    architecture_config : dict, optional
        Architecture configuration dictionary.

    Returns
    -------
    dict
        Resolved attention levels configuration.
    """
    if architecture_config is None:
        architecture_config = {}

    # Default attention configuration
    default_attention = {
        "decoder_attention_stack": ["cross", "hierarchical", "memory"],
        "attention_heads": 4,
        "attention_dim": 64,
    }

    # Merge with user config
    resolved = {**default_attention, **architecture_config}
    return resolved
