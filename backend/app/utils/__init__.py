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
    'is_sentry_enabled'
]
