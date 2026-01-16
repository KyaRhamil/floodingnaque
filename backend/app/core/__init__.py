"""
Core functionality package for Floodingnaque API.

Contains:
- config: Configuration management
- exceptions: Custom exception classes
- security: Security utilities
- constants: Application constants
"""

from app.core.config import Config, get_config, load_env
from app.core.constants import API_NAME, API_VERSION, DEFAULT_LATITUDE, DEFAULT_LONGITUDE, RISK_LEVELS
from app.core.exceptions import (
    AppException,
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    DatabaseError,
    ExternalAPIError,
    ModelError,
    NotFoundError,
    RateLimitError,
    ValidationError,
)
from app.core.security import (
    generate_api_key,
    generate_secret_key,
    get_secure_headers,
    hash_api_key,
    sanitize_input,
    verify_api_key,
)

__all__ = [
    # Config
    "load_env",
    "get_config",
    "Config",
    # Exceptions
    "AppException",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "RateLimitError",
    "ExternalAPIError",
    "DatabaseError",
    "ModelError",
    "ConfigurationError",
    # Security
    "generate_secret_key",
    "generate_api_key",
    "hash_api_key",
    "verify_api_key",
    "sanitize_input",
    "get_secure_headers",
    # Constants
    "API_VERSION",
    "API_NAME",
    "DEFAULT_LATITUDE",
    "DEFAULT_LONGITUDE",
    "RISK_LEVELS",
]
