"""
Request Body Size Validation Middleware.

Provides per-endpoint request body size limits to complement the global
MAX_CONTENT_LENGTH setting in Flask. This allows different size limits
for different endpoint types.

Security benefits:
- Prevents denial of service via large payloads
- Allows stricter limits on sensitive endpoints
- Enables larger limits for file upload endpoints when needed
"""

import logging
import os
from functools import wraps

from flask import jsonify, request

logger = logging.getLogger(__name__)

# Default body size limits per endpoint type (in bytes)
# These can be overridden via environment variables
DEFAULT_LIMITS = {
    "default": 1 * 1024 * 1024,  # 1 MB (default)
    "auth": 10 * 1024,  # 10 KB (auth payloads should be small)
    "predict": 100 * 1024,  # 100 KB (prediction requests)
    "ingest": 500 * 1024,  # 500 KB (data ingestion)
    "webhook": 1 * 1024 * 1024,  # 1 MB (webhooks from external services)
    "batch": 5 * 1024 * 1024,  # 5 MB (batch operations)
    "upload": 10 * 1024 * 1024,  # 10 MB (file uploads)
}


def get_body_size_limit(endpoint_type: str = "default") -> int:
    """
    Get the body size limit for an endpoint type.

    First checks for environment variable override, then falls back to defaults.

    Args:
        endpoint_type: The type of endpoint (auth, predict, ingest, etc.)

    Returns:
        int: Maximum allowed body size in bytes
    """
    # Check for environment variable override
    env_var = f"MAX_BODY_SIZE_{endpoint_type.upper()}_KB"
    env_value = os.getenv(env_var)

    if env_value:
        try:
            return int(env_value) * 1024  # Convert KB to bytes
        except ValueError:
            logger.warning(f"Invalid value for {env_var}: {env_value}")

    # Fall back to defaults
    return DEFAULT_LIMITS.get(endpoint_type, DEFAULT_LIMITS["default"])


def limit_body_size(endpoint_type: str = "default", custom_limit_bytes: int = None):
    """
    Decorator to limit request body size for specific endpoints.

    Args:
        endpoint_type: The type of endpoint for predefined limits
        custom_limit_bytes: Optional custom limit in bytes (overrides endpoint_type)

    Usage:
        @app.route('/api/auth/login', methods=['POST'])
        @limit_body_size('auth')
        def login():
            ...

        @app.route('/api/upload', methods=['POST'])
        @limit_body_size(custom_limit_bytes=20*1024*1024)  # 20 MB
        def upload():
            ...
    """

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Determine the limit
            if custom_limit_bytes is not None:
                max_size = custom_limit_bytes
            else:
                max_size = get_body_size_limit(endpoint_type)

            # Check Content-Length header first (if available)
            content_length = request.content_length
            if content_length is not None and content_length > max_size:
                logger.warning(
                    f"Request body too large: {content_length} bytes "
                    f"(limit: {max_size} bytes) for endpoint {request.path}"
                )
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Request Entity Too Large",
                            "message": f"Request body exceeds maximum allowed size of {max_size // 1024} KB",
                            "max_size_kb": max_size // 1024,
                        }
                    ),
                    413,
                )

            # Also check actual data length for chunked transfers
            if request.data and len(request.data) > max_size:
                logger.warning(
                    f"Request body too large: {len(request.data)} bytes "
                    f"(limit: {max_size} bytes) for endpoint {request.path}"
                )
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Request Entity Too Large",
                            "message": f"Request body exceeds maximum allowed size of {max_size // 1024} KB",
                            "max_size_kb": max_size // 1024,
                        }
                    ),
                    413,
                )

            return f(*args, **kwargs)

        return decorated_function

    return decorator


def validate_json_body_size(max_size_kb: int = 100):
    """
    Decorator specifically for JSON endpoints.

    Args:
        max_size_kb: Maximum size in kilobytes
    """
    return limit_body_size(custom_limit_bytes=max_size_kb * 1024)


class BodySizeLimits:
    """
    Constants for common body size limits (in bytes).

    Usage:
        @limit_body_size(custom_limit_bytes=BodySizeLimits.SMALL)
    """

    TINY = 10 * 1024  # 10 KB - auth, simple forms
    SMALL = 100 * 1024  # 100 KB - typical API requests
    MEDIUM = 1 * 1024 * 1024  # 1 MB - standard uploads
    LARGE = 5 * 1024 * 1024  # 5 MB - batch operations
    XLARGE = 10 * 1024 * 1024  # 10 MB - file uploads
    MAX = 50 * 1024 * 1024  # 50 MB - large file uploads
