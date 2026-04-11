"""Generic utility functions for base-attentive."""

from __future__ import annotations

from typing import Any

__all__ = ["select_mode"]


def select_mode(
    data: Any,
    mode: str = "auto",
    default: Any = None,
    canonical: list[str] | tuple[str, ...] | None = None,
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
    if data is None:
        return default

    if mode != "auto":
        return data

    if not isinstance(data, str):
        return data

    normalized = data.strip().lower().replace("-", "_")

    if canonical:
        canonical_map = {
            value.strip().lower().replace("-", "_"): value for value in canonical
        }
        if normalized in canonical_map:
            return canonical_map[normalized]

    aliases = {
        "tft": "tft_like",
        "tft_like": "tft_like",
        "pihal": "pihal_like",
        "pihal_like": "pihal_like",
    }
    return aliases.get(normalized, data)
