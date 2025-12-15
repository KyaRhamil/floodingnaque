"""
Core functionality package for Floodingnaque API.

Contains:
- config: Configuration management
- exceptions: Custom exception classes
- security: Security utilities
- constants: Application constants
"""

from app.core.config import load_env, get_config, Config
from app.core.exceptions import (
    AppException,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    RateLimitError,
    ExternalAPIError,
    DatabaseError,
    ModelError,
    ConfigurationError
)
from app.core.security import (
    generate_secret_key,
    generate_api_key,
    hash_api_key,
    verify_api_key,
    sanitize_input,
    get_secure_headers
)
from app.core.constants import (
    API_VERSION,
    API_NAME,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    RISK_LEVELS
)

__all__ = [
    # Config
    'load_env',
    'get_config',
    'Config',
    # Exceptions
    'AppException',
    'ValidationError',
    'AuthenticationError',
    'AuthorizationError',
    'NotFoundError',
    'RateLimitError',
    'ExternalAPIError',
    'DatabaseError',
    'ModelError',
    'ConfigurationError',
    # Security
    'generate_secret_key',
    'generate_api_key',
    'hash_api_key',
    'verify_api_key',
    'sanitize_input',
    'get_secure_headers',
    # Constants
    'API_VERSION',
    'API_NAME',
    'DEFAULT_LATITUDE',
    'DEFAULT_LONGITUDE',
    'RISK_LEVELS'
]
