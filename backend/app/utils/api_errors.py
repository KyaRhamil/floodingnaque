"""
API Error Classes.

Custom exception classes for API error handling.
"""

class AppException(Exception):
    """Base application exception class."""
    
    def __init__(self, message: str, status_code: int = 400, error_code: str = None):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or self.__class__.__name__
        super().__init__(self.message)
    
    def to_dict(self) -> dict:
        """Convert exception to dictionary representation."""
        return {
            'error': self.error_code,
            'message': self.message
        }


class ValidationError(AppException):
    """Validation error exception."""
    
    def __init__(self, message: str):
        super().__init__(message, 400, 'ValidationError')


class NotFoundError(AppException):
    """Resource not found exception."""
    
    def __init__(self, message: str):
        super().__init__(message, 404, 'NotFoundError')


class UnauthorizedError(AppException):
    """Unauthorized access exception."""
    
    def __init__(self, message: str):
        super().__init__(message, 401, 'UnauthorizedError')


class ForbiddenError(AppException):
    """Forbidden access exception."""
    
    def __init__(self, message: str):
        super().__init__(message, 403, 'ForbiddenError')


class ConflictError(AppException):
    """Resource conflict exception."""
    
    def __init__(self, message: str):
        super().__init__(message, 409, 'ConflictError')


class RateLimitExceededError(AppException):
    """Rate limit exceeded exception."""
    
    def __init__(self, message: str):
        super().__init__(message, 429, 'RateLimitExceededError')


class InternalServerError(AppException):
    """Internal server error exception."""
    
    def __init__(self, message: str):
        super().__init__(message, 500, 'InternalServerError')


class ServiceUnavailableError(AppException):
    """Service unavailable exception."""
    
    def __init__(self, message: str):
        super().__init__(message, 503, 'ServiceUnavailableError')