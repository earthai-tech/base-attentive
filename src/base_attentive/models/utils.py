"""Model-specific utility functions."""

from __future__ import annotations

from typing import Any, Dict

__all__ = ["set_default_params"]


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
