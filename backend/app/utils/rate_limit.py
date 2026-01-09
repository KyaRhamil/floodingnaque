"""
Rate Limit Utilities.

Re-exports rate limiting components from the middleware for convenience.
This module provides a simpler import path for route handlers.
"""

from app.api.middleware.rate_limit import (
    limiter,
    get_limiter,
    init_rate_limiter,
    get_endpoint_limit,
    get_current_rate_limit_info,
    rate_limit_standard,
    rate_limit_strict,
    rate_limit_relaxed,
    rate_limit_by_ip_only,
    rate_limit_authenticated_only,
    rate_limit_with_burst,
    check_burst_allowance,
    get_burst_stats,
    ENDPOINT_LIMITS,
    RATE_LIMIT_ENABLED,
    RATE_LIMIT_STORAGE,
    BURST_ENABLED,
    BURST_MULTIPLIER,
    BURST_WINDOW_SECONDS,
)

__all__ = [
    'limiter',
    'get_limiter',
    'init_rate_limiter',
    'get_endpoint_limit',
    'get_current_rate_limit_info',
    'rate_limit_standard',
    'rate_limit_strict',
    'rate_limit_relaxed',
    'rate_limit_by_ip_only',
    'rate_limit_authenticated_only',
    'rate_limit_with_burst',
    'check_burst_allowance',
    'get_burst_stats',
    'ENDPOINT_LIMITS',
    'RATE_LIMIT_ENABLED',
    'RATE_LIMIT_STORAGE',
    'BURST_ENABLED',
    'BURST_MULTIPLIER',
    'BURST_WINDOW_SECONDS',
]
