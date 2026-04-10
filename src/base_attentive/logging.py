"""Logging utilities for base-attentive."""

from __future__ import annotations

import logging
from typing import Optional

# Module-level logger
_LOGGER = None


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get or create a logger for the module.

    Parameters
    ----------
    name : str
        Logger name (usually __name__).
    level : int, optional
        Logging level. Default is logging.INFO.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


__all__ = ["get_logger"]
