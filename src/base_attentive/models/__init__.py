"""Model utilities and component helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional

__all__ = [
    "resolve_attention_levels",
    "set_default_params",
]


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
        "decoder_attention_stack": [
            "cross",
            "hierarchical",
            "memory",
        ],
        "attention_heads": 4,
        "attention_dim": 64,
    }

    # Merge with user config
    resolved = {**default_attention, **architecture_config}
    return resolved


def set_default_params(
    model_params: Dict[str, Any],
    **defaults: Any,
) -> Dict[str, Any]:
    """
    Set default parameters for model, only overwriting if not already set.

    Parameters
    ----------
    model_params : dict
        Model parameters dictionary.
    **defaults
        Default parameter values to apply.

    Returns
    -------
    dict
        Model parameters with defaults applied.
    """
    result = dict(defaults)  # Start with defaults
    result.update(model_params)  # Override with user params
    return result
