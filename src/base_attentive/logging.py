"""Logging utilities for base-attentive."""

from __future__ import annotations

import logging
from typing import Optional

# Module-level logger
_LOGGER = None


class OncePerMessageFilter(logging.Filter):
    """Filter to log each unique message only once."""

    def __init__(self):
        """Initialize the filter with a set of seen messages."""
        super().__init__()
        self.seen = set()

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter to allow each unique message only once.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to filter.

        Returns
        -------
        bool
            True if message hasn't been seen before, False otherwise.
        """
        msg = record.getMessage()
        if msg not in self.seen:
            self.seen.add(msg)
            return True
        return False


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


__all__ = ["get_logger", "OncePerMessageFilter"]
