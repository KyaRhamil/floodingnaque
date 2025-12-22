"""Middleware package for Floodingnaque API.

Contains:
- auth: API key authentication middleware
- rate_limit: Rate limiting middleware  
- security: Security headers middleware
- logging: Request/response logging middleware
- body_size: Per-endpoint request body size validation
"""

from app.api.middleware.auth import require_api_key, optional_api_key
from app.api.middleware.rate_limit import (
    limiter, 
    get_limiter, 
    init_rate_limiter, 
    get_endpoint_limit,
    rate_limit_auth,
    rate_limit_password_reset,
    rate_limit_standard,
    rate_limit_strict,
    rate_limit_relaxed
)
from app.api.middleware.security import setup_security_headers, add_security_headers, get_cors_origins
from app.api.middleware.logging import setup_request_logging, request_logger, add_request_id
from app.api.middleware.body_size import limit_body_size, validate_json_body_size, BodySizeLimits

__all__ = [
    'require_api_key',
    'optional_api_key', 
    'limiter',
    'get_limiter',
    'init_rate_limiter',
    'get_endpoint_limit',
    'rate_limit_auth',
    'rate_limit_password_reset',
    'rate_limit_standard',
    'rate_limit_strict',
    'rate_limit_relaxed',
    'setup_security_headers',
    'add_security_headers',
    'get_cors_origins',
    'setup_request_logging',
    'request_logger',
    'add_request_id',
    'limit_body_size',
    'validate_json_body_size',
    'BodySizeLimits',
]
