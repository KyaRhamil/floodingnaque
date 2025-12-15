"""
Rate Limiting Middleware.

Provides rate limiting for API endpoints to prevent abuse and ensure fair usage.
Uses Flask-Limiter with configurable storage backend.
"""

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import logging

logger = logging.getLogger(__name__)

# Check if rate limiting is enabled
RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'True').lower() == 'true'

# Get storage URI from environment (default to memory for development)
RATE_LIMIT_STORAGE = os.getenv('RATE_LIMIT_STORAGE_URL', 'memory://')

# Default limits from environment
DEFAULT_LIMIT = os.getenv('RATE_LIMIT_DEFAULT', '100')
WINDOW_SECONDS = os.getenv('RATE_LIMIT_WINDOW_SECONDS', '3600')

# Create limiter instance
# Note: Will be initialized with app in init_app()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{DEFAULT_LIMIT} per {WINDOW_SECONDS} seconds"],
    storage_uri=RATE_LIMIT_STORAGE,
    enabled=RATE_LIMIT_ENABLED
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
    
    if RATE_LIMIT_ENABLED:
        logger.info(f"Rate limiting enabled: {DEFAULT_LIMIT} requests per {WINDOW_SECONDS}s")
    else:
        logger.info("Rate limiting is disabled")
    
    return limiter


# Predefined rate limit decorators for common use cases
def rate_limit_standard():
    """Standard rate limit for general endpoints: 100 per hour, 20 per minute."""
    return limiter.limit("100 per hour;20 per minute")


def rate_limit_strict():
    """Strict rate limit for sensitive endpoints: 30 per hour, 5 per minute."""
    return limiter.limit("30 per hour;5 per minute")


def rate_limit_relaxed():
    """Relaxed rate limit for public endpoints: 200 per hour, 50 per minute."""
    return limiter.limit("200 per hour;50 per minute")


# Specific limits for different endpoint types
ENDPOINT_LIMITS = {
    'predict': "60 per hour;10 per minute",    # ML predictions (resource intensive)
    'ingest': "30 per hour;5 per minute",       # Data ingestion (rate limited due to external APIs)
    'data': "120 per hour;30 per minute",       # Data retrieval
    'status': "300 per hour;60 per minute",     # Health checks (relaxed)
    'docs': "200 per hour;40 per minute"        # Documentation
}


def get_endpoint_limit(endpoint_name):
    """
    Get the rate limit string for a specific endpoint.
    
    Args:
        endpoint_name: Name of the endpoint
    
    Returns:
        str: Rate limit string for the endpoint
    """
    return ENDPOINT_LIMITS.get(endpoint_name, f"{DEFAULT_LIMIT} per {WINDOW_SECONDS} seconds")
