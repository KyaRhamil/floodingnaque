"""Utility functions and helpers."""

from app.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    openweathermap_breaker,
    weatherstack_breaker,
    meteostat_breaker,
    retry_with_backoff
)
from app.utils.validation import validate_coordinates, validate_weather_data
from app.utils.cache import (
    cached,
    cache_get,
    cache_set,
    cache_delete,
    get_cache_stats,
    is_cache_enabled
)
from app.utils.metrics import (
    init_prometheus_metrics,
    get_metrics,
    record_prediction,
    record_external_api_call,
    record_alert_sent
)
from app.utils.sentry import (
    init_sentry,
    capture_exception,
    capture_message,
    add_breadcrumb,
    set_user_context,
    set_tag,
    is_sentry_enabled
)
from app.utils.api_responses import (
    api_success,
    api_error,
    api_error_from_exception,
    api_created,
    api_accepted,
    api_no_content,
    api_paginated
)
from app.utils.api_errors import (
    AppException,
    ValidationError,
    NotFoundError,
    UnauthorizedError,
    ForbiddenError,
    ConflictError,
    RateLimitExceededError,
    InternalServerError,
    ServiceUnavailableError,
    BadRequestError,
    ModelError,
    ExternalServiceError,
    # Backward compatibility aliases
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    ExternalAPIError,
    DatabaseError,
    ConfigurationError
)
from app.utils.logging import (
    setup_logging,
    get_logger,
    set_request_context,
    clear_request_context,
    get_request_context,
    log_with_context,
    LogContext,
    log_function_call
)
from app.utils.tracing import (
    TraceContext,
    Span,
    get_current_trace,
    set_current_trace,
    clear_current_trace,
    trace_operation,
    trace_operation_async,
    SpanContext,
    inject_trace_headers
)

__all__ = [
    # Circuit breaker
    'CircuitBreaker',
    'CircuitOpenError',
    'CircuitState',
    'openweathermap_breaker',
    'weatherstack_breaker',
    'meteostat_breaker',
    'retry_with_backoff',
    # Validation
    'validate_coordinates',
    'validate_weather_data',
    # Caching
    'cached',
    'cache_get',
    'cache_set',
    'cache_delete',
    'get_cache_stats',
    'is_cache_enabled',
    # Metrics
    'init_prometheus_metrics',
    'get_metrics',
    'record_prediction',
    'record_external_api_call',
    'record_alert_sent',
    # Sentry
    'init_sentry',
    'capture_exception',
    'capture_message',
    'add_breadcrumb',
    'set_user_context',
    'set_tag',
    'is_sentry_enabled',
    # API Responses
    'api_success',
    'api_error',
    'api_error_from_exception',
    'api_created',
    'api_accepted',
    'api_no_content',
    'api_paginated',
    # API Errors
    'AppException',
    'ValidationError',
    'NotFoundError',
    'UnauthorizedError',
    'ForbiddenError',
    'ConflictError',
    'RateLimitExceededError',
    'InternalServerError',
    'ServiceUnavailableError',
    'BadRequestError',
    'ModelError',
    'ExternalServiceError',
    'AuthenticationError',
    'AuthorizationError',
    'RateLimitError',
    'ExternalAPIError',
    'DatabaseError',
    'ConfigurationError',
    # Logging
    'setup_logging',
    'get_logger',
    'set_request_context',
    'clear_request_context',
    'get_request_context',
    'log_with_context',
    'LogContext',
    'log_function_call',
    # Tracing
    'TraceContext',
    'Span',
    'get_current_trace',
    'set_current_trace',
    'clear_current_trace',
    'trace_operation',
    'trace_operation_async',
    'SpanContext',
    'inject_trace_headers'
]
