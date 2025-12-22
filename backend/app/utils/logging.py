import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from app.core.constants import LOG_LEVELS

_LOG_CONFIGURED = False


def _get_log_level() -> int:
    """Return numeric log level from LOG_LEVEL env with INFO fallback."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return LOG_LEVELS.get(level_name, logging.INFO)


def _build_formatter() -> logging.Formatter:
    """Build a formatter honoring LOG_FORMAT env (json or text)."""
    format_type = os.getenv("LOG_FORMAT", "json").lower()
    datefmt = "%Y-%m-%dT%H:%M:%S%z"

    if format_type == "json":
        pattern = (
            '{"timestamp":"%(asctime)s","level":"%(levelname)s",'
            '"logger":"%(name)s","message":"%(message)s"}'
        )
    else:
        pattern = "%({})s [%(levelname)s] %(name)s - %(message)s".format("asctime")

    return logging.Formatter(pattern, datefmt=datefmt)


def _add_handlers_if_missing(root_logger: logging.Logger, formatter: logging.Formatter) -> None:
    """Attach console and rotating file handlers if none exist."""
    if root_logger.handlers:
        return

    logs_dir = Path(os.getenv("LOG_DIR", "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(logs_dir / "app.log", maxBytes=10_000_000, backupCount=5)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def setup_logging() -> logging.Logger:
    """Configure the root logger once, honoring LOG_LEVEL and LOG_FORMAT."""
    global _LOG_CONFIGURED

    root_logger = logging.getLogger()
    level = _get_log_level()
    formatter = _build_formatter()

    root_logger.setLevel(level)
    _add_handlers_if_missing(root_logger, formatter)

    for handler in root_logger.handlers:
        if handler.level == logging.NOTSET:
            handler.setLevel(level)
        if handler.formatter is None:
            handler.setFormatter(formatter)

    _LOG_CONFIGURED = True
    return root_logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger instance after ensuring logging is configured."""
    if not _LOG_CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
