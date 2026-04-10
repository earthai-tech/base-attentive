"""Generic utility functions for base-attentive."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = ["select_mode"]


def select_mode(
    data: Any,
    mode: str = "auto",
    **options,
) -> Any:
    """
    Select or process data based on specified mode.

    Parameters
    ----------
    data : Any
        Input data to process.
    mode : str, optional
        Selection mode. Default is 'auto'.
    **options
        Additional mode-specific options.

    Returns
    -------
    Any
        Processed data or original data if mode is 'auto'.
    """
    if mode == "auto":
        return data

    # Add more mode handling as needed
    return data
