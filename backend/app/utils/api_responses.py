"""API Response Utilities.

Standardized response formatting functions for consistent API responses.
Follows RFC 7807 Problem Details format for errors.
"""

import html
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Tuple

from flask import Response, g, jsonify

if TYPE_CHECKING:
    from app.utils.api_errors import AppException

logger = logging.getLogger(__name__)

# Patterns that might indicate sensitive information in error messages
_SENSITIVE_PATTERNS = [
    r"password",
    r"secret",
    r"token",
    r"key",
    r"credential",
    r"auth",
    r"/[a-zA-Z]:/",  # Windows paths
    r"/home/",  # Unix home paths
    r"/usr/",  # Unix system paths
    r"\\\\[a-zA-Z]",  # UNC paths
    r"postgresql://",  # Database connection strings
    r"mysql://",
    r"mongodb://",
    r"redis://",
    r"amqp://",
]


def _sanitize_error_message(message: str) -> str:
    """Sanitize error message to prevent information disclosure."""
    if not message:
        return message

    # HTML escape to prevent XSS
    sanitized = html.escape(str(message))

    # Check for sensitive patterns
    message_lower = sanitized.lower()
    for pattern in _SENSITIVE_PATTERNS:
        if re.search(pattern, message_lower, re.IGNORECASE):
            # If sensitive pattern detected, return generic message
            logger.debug(f"Sanitized potentially sensitive error message")
            return "An error occurred. Please contact support with your request ID."

    # Truncate overly long messages that might contain stack traces
    if len(sanitized) > 500:
        return sanitized[:500] + "..."

    return sanitized


def _get_request_context() -> Dict[str, str]:
    """Get request and trace IDs from Flask g context."""
    return {"request_id": getattr(g, "request_id", None), "trace_id": getattr(g, "trace_id", None)}


def api_success(
    data: Any = None, message: str = None, status_code: int = 200, request_id: str = None, meta: Dict[str, Any] = None
) -> tuple:
    """
    Create a standardized success response.

    Args:
        data: Response data payload
        message: Optional success message
        status_code: HTTP status code (default: 200)
        request_id: Request identifier for tracing (auto-detected if None)
        meta: Optional metadata (pagination, etc.)

    Returns:
        Tuple of (response_dict, status_code)
    """
    ctx = _get_request_context()

    response = {
        "success": True,
        "request_id": request_id or ctx.get("request_id"),
    }

    if ctx.get("trace_id"):
        response["trace_id"] = ctx["trace_id"]

    if data is not None:
        response["data"] = data

    if message:
        response["message"] = message

    if meta:
        response["meta"] = meta

    return jsonify(response), status_code


def api_error(
    error_code: str,
    message: str,
    status_code: int = 400,
    request_id: str = None,
    details: Dict[str, Any] = None,
    errors: list = None,
    help_url: str = None,
) -> tuple:
    """
    Create a standardized RFC 7807 error response.

    Args:
        error_code: Error code identifier
        message: Human-readable error message
        status_code: HTTP status code (default: 400)
        request_id: Request identifier for tracing (auto-detected if None)
        details: Optional additional error details
        errors: Optional list of field-level errors
        help_url: Optional URL for more information

    Returns:
        Tuple of (response_dict, status_code)
    """
    ctx = _get_request_context()
    req_id = request_id or ctx.get("request_id")
    trace_id = ctx.get("trace_id")

    # Sanitize error message to prevent information disclosure
    safe_message = _sanitize_error_message(message)

    response = {
        "success": False,
        "error": {
            "type": f'/errors/{error_code.lower().replace("error", "").replace("_", "-")}',
            "title": _get_error_title(error_code),
            "status": status_code,
            "detail": safe_message,
            "code": error_code,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    }

    if req_id:
        response["error"]["request_id"] = req_id

    if trace_id:
        response["error"]["trace_id"] = trace_id

    if details:
        response["error"]["details"] = details

    if errors:
        response["error"]["errors"] = errors

    if help_url:
        response["error"]["help_url"] = help_url

    logger.warning(
        f"API Error [{req_id}]: {error_code}",
        extra={"error_code": error_code, "status_code": status_code, "request_id": req_id},
    )

    return jsonify(response), status_code


def api_error_from_exception(
    exception: "AppException", request_id: str = None, include_debug: bool = False
) -> Tuple[Response, int]:
    """
    Create a standardized RFC 7807 error response from an AppException.

    Args:
        exception: The AppException instance
        request_id: Request tracking ID (auto-detected if None)
        include_debug: Whether to include debug details

    Returns:
        Tuple of (Flask JSON response, status code)
    """
    ctx = _get_request_context()

    response = exception.to_dict(include_debug=include_debug)
    response["error"]["request_id"] = request_id or ctx.get("request_id")

    if ctx.get("trace_id"):
        response["error"]["trace_id"] = ctx["trace_id"]

    logger.warning(
        f"API Exception [{response['error'].get('request_id')}]: {exception.error_code}",
        extra={
            "error_code": exception.error_code,
            "status_code": exception.status_code,
            "request_id": response["error"].get("request_id"),
        },
    )

    resp = jsonify(response)

    # Add Retry-After header if applicable
    if hasattr(exception, "retry_after") and exception.retry_after:
        resp.headers["Retry-After"] = str(exception.retry_after)

    return resp, exception.status_code


def _get_error_title(error_code: str) -> str:
    """Get human-readable title for error code."""
    titles = {
        "VALIDATION_ERROR": "Validation Failed",
        "ValidationError": "Validation Failed",
        "NOT_FOUND": "Resource Not Found",
        "NotFoundError": "Resource Not Found",
        "UNAUTHORIZED": "Authentication Required",
        "UnauthorizedError": "Authentication Required",
        "FORBIDDEN": "Access Denied",
        "ForbiddenError": "Access Denied",
        "CONFLICT": "Resource Conflict",
        "ConflictError": "Resource Conflict",
        "RATE_LIMIT_EXCEEDED": "Rate Limit Exceeded",
        "RateLimitExceededError": "Rate Limit Exceeded",
        "BAD_REQUEST": "Bad Request",
        "BadRequestError": "Bad Request",
        "INTERNAL_ERROR": "Internal Server Error",
        "InternalServerError": "Internal Server Error",
        "SERVICE_UNAVAILABLE": "Service Unavailable",
        "ServiceUnavailableError": "Service Unavailable",
        "MODEL_ERROR": "Model Processing Error",
        "ModelError": "Model Processing Error",
        "EXTERNAL_SERVICE_ERROR": "External Service Error",
        "ExternalServiceError": "External Service Error",
        "DATABASE_ERROR": "Database Error",
        "DatabaseError": "Database Error",
    }
    return titles.get(error_code, "Error")


def api_created(
    data: Any = None, message: str = "Resource created successfully", location: str = None, request_id: str = None
) -> tuple:
    """
    Create a standardized response for created resources (201 Created).

    Args:
        data: Response data payload
        message: Success message
        location: URI of created resource
        request_id: Request identifier for tracing (auto-detected if None)

    Returns:
        Tuple of (response_dict, 201, headers)
    """
    ctx = _get_request_context()

    response = {"success": True, "message": message, "request_id": request_id or ctx.get("request_id")}

    if ctx.get("trace_id"):
        response["trace_id"] = ctx["trace_id"]

    if data is not None:
        response["data"] = data

    headers = {}
    if location:
        headers["Location"] = location

    return jsonify(response), 201, headers


def api_accepted(data: Any = None, message: str = "Request accepted for processing", request_id: str = None) -> tuple:
    """
    Create a standardized response for accepted requests (202 Accepted).

    Args:
        data: Response data payload
        message: Success message
        request_id: Request identifier for tracing (auto-detected if None)

    Returns:
        Tuple of (response_dict, 202)
    """
    ctx = _get_request_context()

    response = {"success": True, "message": message, "request_id": request_id or ctx.get("request_id")}

    if ctx.get("trace_id"):
        response["trace_id"] = ctx["trace_id"]

    if data is not None:
        response["data"] = data

    return jsonify(response), 202


def api_no_content() -> tuple:
    """
    Create a 204 No Content response.

    Returns:
        Tuple of ('', 204)
    """
    return "", 204


def api_paginated(data: list, page: int, page_size: int, total: int, request_id: str = None) -> tuple:
    """
    Create a standardized paginated response.

    Args:
        data: List of items for current page
        page: Current page number (1-indexed)
        page_size: Number of items per page
        total: Total number of items
        request_id: Request identifier for tracing

    Returns:
        Tuple of (response_dict, 200)
    """
    total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0

    meta = {
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total_items": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
        }
    }

    return api_success(data=data, request_id=request_id, meta=meta)
