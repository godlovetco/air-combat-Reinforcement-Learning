"""Logging setup for the UCAV AI Pilot.

Gives the product a proper log: lifecycle and error messages go to the
console and to a rotating file under ``~/.ucav_pilot/logs/`` so users can
attach a log when reporting an issue.  The high-rate per-frame status line
stays on stdout; the log captures the events that matter for support.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

LOGGER_NAME = "ucav"
DEFAULT_LOG_DIR = os.path.join(os.path.expanduser("~"), ".ucav_pilot", "logs")
DEFAULT_LOG_FILE = os.path.join(DEFAULT_LOG_DIR, "ucav_pilot.log")


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def setup_logging(level: str = "INFO", log_file: Optional[str] = DEFAULT_LOG_FILE) -> logging.Logger:
    """Configure and return the ``ucav`` logger (idempotent).

    ``level`` is a name like ``DEBUG``/``INFO``/``WARNING``.  ``log_file`` is
    the rotating log path, or ``None``/``""`` to log to the console only (e.g.
    when the home directory is read-only).
    """
    logger = logging.getLogger(LOGGER_NAME)
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    logger.setLevel(numeric)

    # Reconfigure cleanly if called twice (tests, re-entry).
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)

    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fileh = RotatingFileHandler(
                log_file, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
            )
            fileh.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"
            ))
            logger.addHandler(fileh)
        except OSError as exc:
            logger.warning("could not open log file %s (%s); console only", log_file, exc)

    logger.propagate = False
    return logger
