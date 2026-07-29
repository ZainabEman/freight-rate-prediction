"""Central logging configuration.

Phase-3 audit finding C-10 / TD-16: the original codebase used bare ``print()``
calls in orchestrators and had no logging anywhere else, so pipeline stages
produced no diagnostic trail.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Install a single stdout handler on the root logger.

    Idempotent: repeated calls (e.g. from several CLI entry points in one
    process) do not stack duplicate handlers.

    Args:
        level: Root log level.
    """
    global _configured
    if _configured:
        logging.getLogger().setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring logging on first use.

    Args:
        name: Logger name, conventionally ``__name__``.

    Returns:
        Configured :class:`logging.Logger`.
    """
    configure_logging()
    return logging.getLogger(name)
