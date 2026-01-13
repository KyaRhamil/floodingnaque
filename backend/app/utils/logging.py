"""Enhanced Structured Logging Module.

Provides structured JSON logging with request context injection and distributed tracing support.
Compatible with ELK Stack, Datadog, Splunk, and other log aggregation systems.

Features:
- Structured JSON output with ECS (Elastic Common Schema) compatible fields
- Automatic correlation ID injection from request context
- Service metadata injection for multi-service environments
- Log level filtering and sampling
- Human-readable format option for development
"""

import logging
import os
import json
import sys
import socket
import platform
import random
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
from contextvars import ContextVar
from functools import wraps

from app.core.constants import LOG_LEVELS

_LOG_CONFIGURED = False

# Log sampling configuration
_SAMPLING_ENABLED = os.getenv('LOG_SAMPLING_ENABLED', 'False').lower() == 'true'
_SAMPLING_RATE = float(os.getenv('LOG_SAMPLING_RATE', '0.1'))
_SAMPLING_EXCLUDE_ERRORS = os.getenv('LOG_SAMPLING_EXCLUDE_ERRORS', 'True').lower() == 'true'

# Service metadata (populated at startup)
_SERVICE_METADATA = {
    'name': os.getenv('SERVICE_NAME', 'floodingnaque-api'),
    'version': os.getenv('APP_VERSION', '2.0.0'),
    'environment': os.getenv('APP_ENV', 'development'),
    'hostname': socket.gethostname(),
    'pid': os.getpid(),
    'platform': platform.system().lower(),
}

# Context variable for request-scoped data
request_context: ContextVar[Dict[str, Any]] = ContextVar('request_context', default={})


class SamplingFilter(logging.Filter):
    """
    Filter that samples log records based on configured rate.
    
    Useful for high-volume production environments to reduce log storage costs
    while maintaining observability. Always allows ERROR and CRITICAL logs.
    
    Example:
        - SAMPLING_RATE=0.1 means 10% of logs are kept
        - SAMPLING_RATE=1.0 means all logs are kept (no sampling)
    """
    
    def __init__(self, sample_rate: float = 0.1, exclude_errors: bool = True):
        """
        Initialize the sampling filter.
        
        Args:
            sample_rate: Probability of keeping a log (0.0 to 1.0)
            exclude_errors: If True, always keep ERROR and CRITICAL logs
        """
        super().__init__()
        self.sample_rate = max(0.0, min(1.0, sample_rate))  # Clamp to [0, 1]
        self.exclude_errors = exclude_errors
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Determine if log record should be emitted."""
        # Always emit errors and critical logs
        if self.exclude_errors and record.levelno >= logging.ERROR:
            return True
        
        # Sample based on rate
        return random.random() < self.sample_rate


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for log aggregation systems.
    
    Produces logs compatible with:
    - ELK Stack (Elastic Common Schema)
    - Datadog
    - Splunk
    - AWS CloudWatch
    - Google Cloud Logging
    
    Features:
    - Automatic correlation ID injection
    - Service metadata injection
    - Exception formatting with full stack traces
    - Custom field support via 'extra' parameter
    """
    
    # Fields to exclude from extras (internal logging fields)
    EXCLUDED_FIELDS = {
        'name', 'msg', 'args', 'created', 'filename', 'funcName',
        'levelname', 'levelno', 'lineno', 'module', 'msecs',
        'pathname', 'process', 'processName', 'relativeCreated',
        'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
        'message', 'taskName', 'color_message'
    }
    
    def __init__(
        self, 
        include_extras: bool = True,
        include_service_metadata: bool = True,
        include_location: bool = True,
        ecs_compatible: bool = True
    ):
        """
        Initialize the structured formatter.
        
        Args:
            include_extras: Include extra fields from log record
            include_service_metadata: Include service name, version, etc.
            include_location: Include source file location for errors
            ecs_compatible: Use Elastic Common Schema field names
        """
        super().__init__()
        self.include_extras = include_extras
        self.include_service_metadata = include_service_metadata
        self.include_location = include_location
        self.ecs_compatible = ecs_compatible
    
    def format(self, record: logging.LogRecord) -> str:
        # Base log structure
        log_data = self._build_base_log(record)
        
        # Add correlation context
        self._inject_correlation_context(log_data)
        
        # Add service metadata
        if self.include_service_metadata:
            self._inject_service_metadata(log_data)
        
        # Add source location for warnings and above
        if self.include_location and record.levelno >= logging.WARNING:
            self._inject_location(log_data, record)
        
        # Add extra fields
        if self.include_extras:
            self._inject_extras(log_data, record)
        
        # Add exception info
        if record.exc_info:
            self._inject_exception(log_data, record)
        
        return json.dumps(log_data, default=self._json_serializer, ensure_ascii=False)
    
    def _build_base_log(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Build the base log structure."""
        timestamp = datetime.now(timezone.utc).isoformat()
        
        if self.ecs_compatible:
            return {
                '@timestamp': timestamp,
                'log': {
                    'level': record.levelname.lower(),
                    'logger': record.name,
                },
                'message': record.getMessage(),
            }
        else:
            return {
                'timestamp': timestamp,
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
            }
    
    def _inject_correlation_context(self, log_data: Dict[str, Any]) -> None:
        """Inject correlation IDs from request context."""
        ctx = request_context.get()
        if not ctx:
            return
        
        # Try to get correlation context from the new module
        try:
            from app.utils.correlation import get_correlation_context
            corr_ctx = get_correlation_context()
            if corr_ctx:
                if self.ecs_compatible:
                    log_data['trace'] = {
                        'id': corr_ctx.trace_id,
                    }
                    log_data['span'] = {
                        'id': corr_ctx.span_id,
                    }
                    if corr_ctx.parent_span_id:
                        log_data['parent'] = {'id': corr_ctx.parent_span_id}
                    log_data['correlation_id'] = corr_ctx.correlation_id
                else:
                    log_data['trace_id'] = corr_ctx.trace_id
                    log_data['span_id'] = corr_ctx.span_id
                    log_data['correlation_id'] = corr_ctx.correlation_id
                    if corr_ctx.parent_span_id:
                        log_data['parent_span_id'] = corr_ctx.parent_span_id
                
                # Add user context if available
                if corr_ctx.user_id:
                    if self.ecs_compatible:
                        log_data['user'] = {'id': corr_ctx.user_id}
                    else:
                        log_data['user_id'] = corr_ctx.user_id
                
                if corr_ctx.session_id:
                    log_data['session_id'] = corr_ctx.session_id
                if corr_ctx.tenant_id:
                    log_data['tenant_id'] = corr_ctx.tenant_id
                    
                return
        except ImportError:
            pass
        
        # Fallback to legacy context
        correlation_fields = {
            'request_id': 'request_id',
            'trace_id': 'trace_id',
            'span_id': 'span_id',
            'user_id': 'user_id',
            'correlation_id': 'correlation_id',
        }
        
        for ctx_key, log_key in correlation_fields.items():
            if ctx.get(ctx_key):
                if self.ecs_compatible and log_key == 'trace_id':
                    if 'trace' not in log_data:
                        log_data['trace'] = {}
                    log_data['trace']['id'] = ctx[ctx_key]
                elif self.ecs_compatible and log_key == 'span_id':
                    if 'span' not in log_data:
                        log_data['span'] = {}
                    log_data['span']['id'] = ctx[ctx_key]
                elif self.ecs_compatible and log_key == 'user_id':
                    if 'user' not in log_data:
                        log_data['user'] = {}
                    log_data['user']['id'] = ctx[ctx_key]
                else:
                    log_data[log_key] = ctx[ctx_key]
    
    def _inject_service_metadata(self, log_data: Dict[str, Any]) -> None:
        """Inject service metadata."""
        if self.ecs_compatible:
            log_data['service'] = {
                'name': _SERVICE_METADATA['name'],
                'version': _SERVICE_METADATA['version'],
                'environment': _SERVICE_METADATA['environment'],
            }
            log_data['host'] = {
                'hostname': _SERVICE_METADATA['hostname'],
                'os': {'platform': _SERVICE_METADATA['platform']},
            }
            log_data['process'] = {
                'pid': _SERVICE_METADATA['pid'],
            }
        else:
            log_data['service'] = _SERVICE_METADATA['name']
            log_data['version'] = _SERVICE_METADATA['version']
            log_data['environment'] = _SERVICE_METADATA['environment']
            log_data['hostname'] = _SERVICE_METADATA['hostname']
            log_data['pid'] = _SERVICE_METADATA['pid']
    
    def _inject_location(self, log_data: Dict[str, Any], record: logging.LogRecord) -> None:
        """Inject source code location."""
        if self.ecs_compatible:
            log_data['log']['origin'] = {
                'file': {
                    'name': record.filename,
                    'line': record.lineno,
                },
                'function': record.funcName,
            }
        else:
            log_data['location'] = {
                'file': record.filename,
                'line': record.lineno,
                'function': record.funcName,
            }
    
    def _inject_extras(self, log_data: Dict[str, Any], record: logging.LogRecord) -> None:
        """Inject extra fields from log record."""
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in self.EXCLUDED_FIELDS:
                extra_fields[key] = value
        
        if extra_fields:
            # Flatten common patterns
            if 'http' in extra_fields or 'duration_ms' in extra_fields:
                if self.ecs_compatible and 'http' not in log_data:
                    log_data['http'] = {}
                if 'duration_ms' in extra_fields:
                    if self.ecs_compatible:
                        log_data['event'] = log_data.get('event', {})
                        log_data['event']['duration'] = int(extra_fields.pop('duration_ms') * 1_000_000)  # Convert to nanoseconds
                    else:
                        log_data['duration_ms'] = extra_fields.pop('duration_ms')
            
            # Add remaining extras
            if extra_fields:
                log_data['extra'] = extra_fields
    
    def _inject_exception(self, log_data: Dict[str, Any], record: logging.LogRecord) -> None:
        """Inject exception information."""
        exc_type, exc_value, exc_tb = record.exc_info
        
        if self.ecs_compatible:
            log_data['error'] = {
                'type': exc_type.__name__ if exc_type else None,
                'message': str(exc_value) if exc_value else None,
                'stack_trace': self.formatException(record.exc_info) if record.exc_info else None,
            }
        else:
            log_data['exception'] = {
                'type': exc_type.__name__ if exc_type else None,
                'message': str(exc_value) if exc_value else None,
                'traceback': self.formatException(record.exc_info) if record.exc_info else None,
            }
    
    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Custom JSON serializer for non-serializable objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        if hasattr(obj, '__dict__'):
            return str(obj)
        return str(obj)


class TextFormatter(logging.Formatter):
    """
    Human-readable formatter with color support and correlation IDs.
    
    Ideal for local development and debugging.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',
        'DIM': '\033[2m',
        'BOLD': '\033[1m',
    }
    
    def __init__(self, use_colors: bool = True, include_correlation: bool = True):
        """
        Initialize the text formatter.
        
        Args:
            use_colors: Use ANSI color codes (auto-disabled if not TTY)
            include_correlation: Include correlation ID in output
        """
        super().__init__()
        self.use_colors = use_colors and sys.stdout.isatty()
        self.include_correlation = include_correlation
    
    def format(self, record: logging.LogRecord) -> str:
        # Get correlation info
        correlation_str = self._get_correlation_string()
        
        # Format level with colors
        level = record.levelname
        if self.use_colors:
            level_color = self.COLORS.get(level, '')
            reset = self.COLORS['RESET']
            dim = self.COLORS['DIM']
            level = f"{level_color}{level:8}{reset}"
            correlation_str = f"{dim}{correlation_str}{reset}" if correlation_str else ""
            logger_name = f"{dim}{record.name}{reset}"
        else:
            level = f"{level:8}"
            logger_name = record.name
        
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        # Build message parts
        parts = [timestamp, level]
        if correlation_str:
            parts.append(f"[{correlation_str}]")
        parts.append(f"{logger_name}:")
        parts.append(record.getMessage())
        
        msg = ' '.join(parts)
        
        # Add extra fields if present
        extra_fields = self._get_extra_fields(record)
        if extra_fields:
            extra_str = ' '.join(f"{k}={v}" for k, v in extra_fields.items())
            if self.use_colors:
                msg += f" {self.COLORS['DIM']}| {extra_str}{self.COLORS['RESET']}"
            else:
                msg += f" | {extra_str}"
        
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        
        return msg
    
    def _get_correlation_string(self) -> str:
        """Get correlation ID string for log prefix."""
        try:
            from app.utils.correlation import get_correlation_context
            ctx = get_correlation_context()
            if ctx:
                # Show abbreviated IDs for readability
                return f"{ctx.request_id[:8]}|{ctx.trace_id[:8]}"
        except ImportError:
            pass
        
        # Fallback to legacy context
        ctx = request_context.get()
        if ctx:
            request_id = ctx.get('request_id', '')[:8]
            trace_id = ctx.get('trace_id', '')[:8]
            if request_id or trace_id:
                return f"{request_id or '-'}|{trace_id or '-'}"
        
        return ""
    
    def _get_extra_fields(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Extract extra fields for inline display."""
        excluded = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName',
            'levelname', 'levelno', 'lineno', 'module', 'msecs',
            'pathname', 'process', 'processName', 'relativeCreated',
            'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
            'message', 'taskName', 'color_message'
        }
        
        extras = {}
        for key, value in record.__dict__.items():
            if key not in excluded:
                # Truncate long values
                str_value = str(value)
                if len(str_value) > 50:
                    str_value = str_value[:47] + "..."
                extras[key] = str_value
        
        return extras


def _get_log_level() -> int:
    """Return numeric log level from LOG_LEVEL env with INFO fallback."""
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return LOG_LEVELS.get(level_name, logging.INFO)


def _build_formatter(format_type: str = None) -> logging.Formatter:
    """
    Build a formatter based on configuration.
    
    Args:
        format_type: 'json', 'ecs', or 'text'. Defaults to LOG_FORMAT env.
        
    Returns:
        Configured formatter instance
    """
    if format_type is None:
        format_type = os.getenv("LOG_FORMAT", "json").lower()
    
    if format_type == "ecs":
        return StructuredFormatter(ecs_compatible=True)
    elif format_type == "json":
        return StructuredFormatter(ecs_compatible=False)
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
    # Use timed rotation for better log management
    file_handler = TimedRotatingFileHandler(
        logs_dir / "app.log", 
        when='midnight',
        interval=1,
        backupCount=30,  # Keep 30 days of logs
        encoding='utf-8'
    )
    file_handler.setFormatter(StructuredFormatter(ecs_compatible=True))
    root_logger.addHandler(file_handler)
    
    # Error-only file handler for quick error analysis
    error_handler = RotatingFileHandler(
        logs_dir / "error.log",
        maxBytes=50_000_000,  # 50MB
        backupCount=10,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(StructuredFormatter(ecs_compatible=True))
    root_logger.addHandler(error_handler)

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
    
    # Add sampling filter if enabled (applied to non-error logs)
    if _SAMPLING_ENABLED and _SAMPLING_RATE < 1.0:
        sampling_filter = SamplingFilter(
            sample_rate=_SAMPLING_RATE,
            exclude_errors=_SAMPLING_EXCLUDE_ERRORS
        )
        for handler in root_logger.handlers:
            # Only add sampling to file handlers, not console
            if isinstance(handler, (RotatingFileHandler, TimedRotatingFileHandler)):
                handler.addFilter(sampling_filter)
        logger.info(
            f"Log sampling enabled: rate={_SAMPLING_RATE:.1%}, exclude_errors={_SAMPLING_EXCLUDE_ERRORS}"
        )

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
