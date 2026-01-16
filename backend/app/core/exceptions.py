"""Custom Exception Classes for Floodingnaque API.

This module re-exports all exceptions from app.utils.api_errors for backward compatibility.
All exceptions follow RFC 7807 Problem Details format.

Usage:
    from app.core.exceptions import ValidationError, NotFoundError

    raise ValidationError("Invalid input")
    raise NotFoundError("Resource not found", resource_type="model")
"""

# Re-export all exceptions from the unified api_errors module
from app.utils.api_errors import (  # Aliases for backward compatibility
    AppException,
    AuthenticationError,
    AuthorizationError,
    BadRequestError,
    ConfigurationError,
    ConflictError,
    DatabaseError,
    ExternalAPIError,
    ExternalServiceError,
    ForbiddenError,
    InternalServerError,
    ModelError,
    NotFoundError,
    RateLimitError,
    RateLimitExceededError,
    ServiceUnavailableError,
    UnauthorizedError,
    ValidationError,
)

# Re-export response helpers with trace correlation
from app.utils.api_responses import (
    api_accepted,
    api_created,
    api_error,
    api_error_from_exception,
    api_success,
)

__all__ = [
    # Base exception
    "AppException",
    # Client errors (4xx)
    "ValidationError",
    "NotFoundError",
    "UnauthorizedError",
    "AuthenticationError",
    "ForbiddenError",
    "AuthorizationError",
    "ConflictError",
    "RateLimitExceededError",
    "RateLimitError",
    "BadRequestError",
    # Server errors (5xx)
    "InternalServerError",
    "ServiceUnavailableError",
    "ModelError",
    "ExternalServiceError",
    "ExternalAPIError",
    "DatabaseError",
    "ConfigurationError",
    # Response helpers
    "api_error",
    "api_error_from_exception",
    "api_success",
    "api_created",
    "api_accepted",
]
