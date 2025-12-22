"""
Rate Limiting Middleware.

Provides rate limiting for API endpoints to prevent abuse and ensure fair usage.
Supports multiple backends (memory, Redis) and API key-based limits.
"""

from app.utils.logging import get_logger
from app.utils.rate_limit_tiers import get_rate_limit_for_key, get_anonymous_limits
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import request, g, has_request_context
import os

logger = get_logger(__name__)

# Check if rate limiting is enabled
RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'True').lower() == 'true'

# Get storage URI from environment
# Supports: memory://, redis://host:port, memcached://host:port
RATE_LIMIT_STORAGE = os.getenv('RATE_LIMIT_STORAGE_URL', 'memory://')

# Default limits from environment
DEFAULT_LIMIT = os.getenv('RATE_LIMIT_DEFAULT', '100')
WINDOW_SECONDS = os.getenv('RATE_LIMIT_WINDOW_SECONDS', '3600')


def get_rate_limit_key():
    """
    Get rate limit key - uses API key hash if authenticated, otherwise IP address.
    
    This provides:
    - Per-API-key limits for authenticated users (more generous)
    - Per-IP limits for anonymous users (more restrictive)
    """
    # Check if authenticated via API key
    api_key_hash = getattr(g, 'api_key_hash', None)
    if api_key_hash:
        return f"api_key:{api_key_hash}"
    
    # Fall back to IP address for anonymous users
    return get_remote_address()


def get_rate_limit_key_ip_only():
    """Get rate limit key based only on IP address."""
    return get_remote_address()


# Create limiter instance with flexible key function
limiter = Limiter(
    key_func=get_rate_limit_key,
    default_limits=[f"{DEFAULT_LIMIT} per {WINDOW_SECONDS} seconds"],
    storage_uri=RATE_LIMIT_STORAGE,
    enabled=RATE_LIMIT_ENABLED,
    strategy='fixed-window',  # Options: fixed-window, fixed-window-elastic-expiry, moving-window
    headers_enabled=True,  # Add X-RateLimit-* headers
    header_name_mapping={
        "LIMIT": "X-RateLimit-Limit",
        "REMAINING": "X-RateLimit-Remaining",
        "RESET": "X-RateLimit-Reset"
    }
)


def get_limiter():
    """Get the limiter instance."""
    return limiter


def init_rate_limiter(app):
    """
    Initialize rate limiter with Flask app.
    
    Args:
        app: Flask application instance
    """
    limiter.init_app(app)
    
    storage_type = 'Redis' if 'redis' in RATE_LIMIT_STORAGE else 'Memory'
    
    if RATE_LIMIT_ENABLED:
        logger.info(
            f"Rate limiting enabled: {DEFAULT_LIMIT} requests per {WINDOW_SECONDS}s "
            f"(storage: {storage_type})"
        )
    else:
        logger.info("Rate limiting is disabled")
    
    return limiter


# Predefined rate limit decorators for common use cases
# These now support both IP and API key-based limiting

def rate_limit_standard():
    """Standard rate limit for general endpoints: 100 per hour, 20 per minute."""
    return limiter.limit("100 per hour;20 per minute")


def rate_limit_strict():
    """Strict rate limit for sensitive endpoints: 30 per hour, 5 per minute."""
    return limiter.limit("30 per hour;5 per minute")


def rate_limit_auth():
    """
    Very strict rate limit for authentication/login endpoints.
    
    Provides protection against brute force attacks:
    - 5 attempts per minute
    - 20 attempts per hour
    - 100 attempts per day
    
    Uses IP-only limiting to prevent credential stuffing attacks.
    """
    return limiter.limit(
        "5 per minute;20 per hour;100 per day",
        key_func=get_rate_limit_key_ip_only
    )


def rate_limit_password_reset():
    """
    Very strict rate limit for password reset endpoints.
    
    Prevents abuse of password reset functionality:
    - 3 attempts per minute
    - 10 attempts per hour
    - 50 attempts per day
    """
    return limiter.limit(
        "3 per minute;10 per hour;50 per day",
        key_func=get_rate_limit_key_ip_only
    )


def rate_limit_relaxed():
    """Relaxed rate limit for public endpoints: 200 per hour, 50 per minute."""
    return limiter.limit("200 per hour;50 per minute")


def rate_limit_by_ip_only(limit_string):
    """Rate limit based on IP only, ignoring API key."""
    return limiter.limit(limit_string, key_func=get_rate_limit_key_ip_only)


def rate_limit_authenticated_only(limit_string):
    """
    Higher rate limit for authenticated users only.
    
    Unauthenticated users get the default stricter limit.
    """
    def dynamic_limit():
        if getattr(g, 'authenticated', False):
            return limit_string
        # More restrictive for anonymous
        return "30 per hour;5 per minute"
    
    return limiter.limit(dynamic_limit)


# Specific limits for different endpoint types
# Authenticated users get 2x the limit
ENDPOINT_LIMITS = {
    'predict': "60 per hour;10 per minute",      # ML predictions (resource intensive)
    'predict_auth': "120 per hour;20 per minute", # Authenticated prediction limit
    'ingest': "30 per hour;5 per minute",         # Data ingestion (external APIs)
    'ingest_auth': "60 per hour;10 per minute",   # Authenticated ingest limit
    'data': "120 per hour;30 per minute",         # Data retrieval
    'data_auth': "240 per hour;60 per minute",    # Authenticated data limit
    'status': "300 per hour;60 per minute",       # Health checks (relaxed)
    'docs': "200 per hour;40 per minute",         # Documentation
    # Auth-specific limits (very strict for security)
    'auth_login': "5 per minute;20 per hour;100 per day",
    'auth_register': "3 per minute;10 per hour;30 per day",
    'auth_reset': "3 per minute;10 per hour;50 per day",
    'auth_token': "10 per minute;30 per hour",
}


def get_endpoint_limit(endpoint_name, *, as_callable: bool = True):
    """
    Get the rate limit for an endpoint.

    Returns a callable (default) so Flask-Limiter evaluates within a request
    context. When a plain string is needed (e.g., for introspection), set
    as_callable=False.
    """

    def _resolve_limit():
        # If no request context (module import), fall back to default limit
        if not has_request_context():
            return ENDPOINT_LIMITS.get(
                f"{endpoint_name}_auth", f"{DEFAULT_LIMIT} per {WINDOW_SECONDS} seconds"
            )

        api_key_hash = getattr(g, 'api_key_hash', None)

        if api_key_hash:
            try:
                return get_rate_limit_for_key(api_key_hash, 'per_minute')
            except Exception:
                return ENDPOINT_LIMITS.get(
                    f"{endpoint_name}_auth", f"{DEFAULT_LIMIT} per {WINDOW_SECONDS} seconds"
                )

        return get_anonymous_limits()

    return _resolve_limit if as_callable else _resolve_limit()


def get_current_rate_limit_info():
    """
    Get current rate limit information for the request.
    
    Returns:
        dict: Rate limit information
    """
    try:
        from flask import g
        return {
            'key_type': 'api_key' if getattr(g, 'api_key_hash', None) else 'ip',
            'authenticated': getattr(g, 'authenticated', False),
            'storage': 'redis' if 'redis' in RATE_LIMIT_STORAGE else 'memory',
            'enabled': RATE_LIMIT_ENABLED
        }
    except RuntimeError:
        # Outside of request context
        return {
            'key_type': 'unknown',
            'authenticated': False,
            'storage': 'redis' if 'redis' in RATE_LIMIT_STORAGE else 'memory',
            'enabled': RATE_LIMIT_ENABLED
        }
