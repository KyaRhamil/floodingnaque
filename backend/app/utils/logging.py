"""Enhanced Structured Logging Module.

Provides structured JSON logging with request context injection and distributed tracing support.
"""

import logging
import os
import json
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from contextvars import ContextVar
from functools import wraps

from app.core.constants import LOG_LEVELS

_LOG_CONFIGURED = False

# Context variable for request-scoped data
request_context: ContextVar[Dict[str, Any]] = ContextVar('request_context', default={})


class StructuredFormatter(logging.Formatter):
    """JSON formatter with request context injection."""
    
    def __init__(self, include_extras: bool = True):
        super().__init__()
        self.include_extras = include_extras
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Inject request context
        ctx = request_context.get()
        if ctx:
            if ctx.get('request_id'):
                log_data['request_id'] = ctx.get('request_id')
            if ctx.get('trace_id'):
                log_data['trace_id'] = ctx.get('trace_id')
            if ctx.get('span_id'):
                log_data['span_id'] = ctx.get('span_id')
            if ctx.get('user_id'):
                log_data['user_id'] = ctx.get('user_id')
        
        # Include extra fields
        if self.include_extras:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in (
                    'name', 'msg', 'args', 'created', 'filename', 'funcName',
                    'levelname', 'levelno', 'lineno', 'module', 'msecs',
                    'pathname', 'process', 'processName', 'relativeCreated',
                    'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                    'message', 'taskName'
                ):
                    extra_fields[key] = value
            if extra_fields:
                log_data['extra'] = extra_fields
        
        # Include exception info
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Include location for warnings and above
        if record.levelno >= logging.WARNING:
            log_data['location'] = {
                'file': record.filename,
                'line': record.lineno,
                'function': record.funcName
            }
        
        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable formatter with color support."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'
    }
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()
    
    def format(self, record: logging.LogRecord) -> str:
        ctx = request_context.get()
        request_id = ctx.get('request_id', '-')[:8] if ctx and ctx.get('request_id') else '-'
        
        level = record.levelname
        if self.use_colors:
            level = f"{self.COLORS.get(level, '')}{level}{self.COLORS['RESET']}"
        
        timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        msg = f"{timestamp} [{level}] [{request_id}] {record.name}: {record.getMessage()}"
        
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        
        return msg


def _get_log_level() -> int:
    """Return numeric log level from LOG_LEVEL env with INFO fallback."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return LOG_LEVELS.get(level_name, logging.INFO)


def _build_formatter() -> logging.Formatter:
    """Build a formatter honoring LOG_FORMAT env (json or text)."""
    format_type = os.getenv("LOG_FORMAT", "json").lower()
    
    if format_type == "json":
        return StructuredFormatter()
    else:
        use_colors = os.getenv("LOG_COLORS", "true").lower() == "true"
        return TextFormatter(use_colors=use_colors)


def _add_handlers_if_missing(root_logger: logging.Logger, formatter: logging.Formatter) -> None:
    """Attach console and rotating file handlers if none exist."""
    if root_logger.handlers:
        return

    logs_dir = Path(os.getenv("LOG_DIR", "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)

    # File handler always uses JSON for structured parsing
    file_handler = RotatingFileHandler(
        logs_dir / "app.log", 
        maxBytes=10_000_000, 
        backupCount=5
    )
    file_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(file_handler)

    # Console handler uses configured format
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


def set_request_context(
    request_id: str = None,
    trace_id: str = None,
    span_id: str = None,
    user_id: str = None,
    **extra
) -> None:
    """Set request context for log correlation."""
    ctx = {
        'request_id': request_id,
        'trace_id': trace_id,
        'span_id': span_id,
        'user_id': user_id,
        **extra
    }
    request_context.set({k: v for k, v in ctx.items() if v is not None})


def clear_request_context() -> None:
    """Clear request context."""
    request_context.set({})


def get_request_context() -> Dict[str, Any]:
    """Get current request context."""
    return request_context.get()


def log_with_context(logger: logging.Logger, level: int, message: str, **extra) -> None:
    """Log message with additional context fields."""
    logger.log(level, message, extra=extra)


class LogContext:
    """Context manager for adding temporary log context."""
    
    def __init__(self, **context):
        self.context = context
        self.previous_context = {}
    
    def __enter__(self):
        self.previous_context = request_context.get().copy()
        current = self.previous_context.copy()
        current.update(self.context)
        request_context.set(current)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        request_context.set(self.previous_context)
        return False


def log_function_call(logger: logging.Logger = None):
    """Decorator to log function entry and exit with timing."""
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Entering {func_name}", extra={'function': func_name})
            
            import time
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                logger.debug(
                    f"Exiting {func_name} (took {elapsed:.2f}ms)",
                    extra={'function': func_name, 'duration_ms': elapsed}
                )
                return result
            except Exception as e:
                elapsed = (time.perf_counter() - start) * 1000
                logger.error(
                    f"Error in {func_name}: {e}",
                    extra={'function': func_name, 'duration_ms': elapsed},
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator
