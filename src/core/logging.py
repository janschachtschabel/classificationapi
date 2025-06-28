"""Logging configuration for the application."""

import sys
from typing import Any

from loguru import logger

from .config import settings

# Track if logging has been setup
_logging_setup = False


def setup_logging() -> None:
    """Configure logging for the application."""
    global _logging_setup

    if _logging_setup:
        return

    # Remove default logger
    logger.remove()

    # Configure log format based on settings
    if settings.log_format == "json":
        log_format = (
            "{"
            "time: {time:YYYY-MM-DD HH:mm:ss.SSS}, "
            "level: {level}, "
            "module: {module}, "
            "function: {function}, "
            "line: {line}, "
            "message: {message}"
            "}"
        )
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    # Add console handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.log_level,
        colorize=settings.log_format == "text",
        serialize=settings.log_format == "json",
    )

    # Add file handler (always enabled for debugging)
    logger.add(
        "logs/app.log",
        format=log_format,
        level=settings.log_level,
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        serialize=settings.log_format == "json",
    )

    _logging_setup = True


def get_logger(name: str) -> Any:
    """Get a logger instance with the given name."""
    return logger.bind(name=name)


# Setup logging when module is imported
setup_logging()
