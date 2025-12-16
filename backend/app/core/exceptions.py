"""
Custom Exception Classes for Floodingnaque API.

Provides structured error handling with consistent HTTP status codes and messages.
"""

from typing import Optional, Dict, Any, Tuple
from flask import jsonify, Response


class AppException(Exception):
    """
    Base application exception.
    
    All custom exceptions should inherit from this class.
    Provides a consistent interface for error handling.
    """
    
    def __init__(self, message: str, status_code: int = 500, payload: dict = None):
        """
        Initialize exception.
        
        Args:
            message: Human-readable error message
            status_code: HTTP status code (default: 500)
            payload: Additional error details (optional)
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.payload = payload or {}
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary for JSON response."""
        error_dict = {
            'error': self.__class__.__name__,
            'message': self.message,
            'status_code': self.status_code
        }
        if self.payload:
            error_dict['details'] = self.payload
        return error_dict


class ValidationError(AppException):
    """
    Raised when input validation fails.
    
    HTTP Status: 400 Bad Request
    """
    
    def __init__(self, message: str, field: str = None, payload: dict = None):
        payload = payload or {}
        if field:
            payload['field'] = field
        super().__init__(message, 400, payload)


class AuthenticationError(AppException):
    """
    Raised when authentication fails.
    
    HTTP Status: 401 Unauthorized
    """
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, 401)


class AuthorizationError(AppException):
    """
    Raised when user lacks permission for the requested action.
    
    HTTP Status: 403 Forbidden
    """
    
    def __init__(self, message: str = "Permission denied"):
        super().__init__(message, 403)


class NotFoundError(AppException):
    """
    Raised when a requested resource is not found.
    
    HTTP Status: 404 Not Found
    """
    
    def __init__(self, message: str = "Resource not found", resource_type: str = None):
        payload = {}
        if resource_type:
            payload['resource_type'] = resource_type
        super().__init__(message, 404, payload)


class RateLimitError(AppException):
    """
    Raised when rate limit is exceeded.
    
    HTTP Status: 429 Too Many Requests
    """
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        payload = {}
        if retry_after:
            payload['retry_after'] = retry_after
        super().__init__(message, 429, payload)


class ExternalAPIError(AppException):
    """
    Raised when an external API call fails.
    
    HTTP Status: 502 Bad Gateway
    """
    
    def __init__(self, message: str, api_name: str = None, original_error: str = None):
        payload = {}
        if api_name:
            payload['api'] = api_name
        if original_error:
            payload['original_error'] = original_error
        super().__init__(message, 502, payload)


class DatabaseError(AppException):
    """
    Raised when a database operation fails.
    
    HTTP Status: 500 Internal Server Error
    """
    
    def __init__(self, message: str = "Database operation failed"):
        super().__init__(message, 500)


class ModelError(AppException):
    """
    Raised when ML model operations fail.
    
    HTTP Status: 500 Internal Server Error
    """
    
    def __init__(self, message: str, model_version: str = None):
        payload = {}
        if model_version:
            payload['model_version'] = model_version
        super().__init__(message, 500, payload)


class ConfigurationError(AppException):
    """
    Raised when application configuration is invalid.
    
    HTTP Status: 500 Internal Server Error
    """
    
    def __init__(self, message: str, config_key: str = None):
        payload = {}
        if config_key:
            payload['config_key'] = config_key
        super().__init__(message, 500, payload)


# =============================================================================
# Unified API Error Response Helper
# =============================================================================

def api_error(
    error_type: str,
    message: str,
    status_code: int,
    request_id: str,
    details: Optional[Dict[str, Any]] = None
) -> Tuple[Response, int]:
    """
    Create a standardized API error response.
    
    This helper ensures consistent error response format across all endpoints.
    
    Args:
        error_type: Type/category of the error (e.g., 'ValidationError', 'NotFound')
        message: Human-readable error message
        status_code: HTTP status code
        request_id: Request tracking ID
        details: Optional additional error details
    
    Returns:
        Tuple of (Flask JSON response, status code)
    
    Example:
        return api_error(
            error_type='ValidationError',
            message='Invalid temperature value',
            status_code=400,
            request_id=g.request_id,
            details={'field': 'temperature', 'received': -500}
        )
    """
    response_body: Dict[str, Any] = {
        'error': error_type,
        'message': message,
        'request_id': request_id
    }
    
    if details:
        response_body['details'] = details
    
    return jsonify(response_body), status_code


def api_error_from_exception(
    exception: AppException,
    request_id: str
) -> Tuple[Response, int]:
    """
    Create a standardized API error response from an AppException.
    
    Args:
        exception: The AppException instance
        request_id: Request tracking ID
    
    Returns:
        Tuple of (Flask JSON response, status code)
    """
    error_dict = exception.to_dict()
    error_dict['request_id'] = request_id
    return jsonify(error_dict), exception.status_code
