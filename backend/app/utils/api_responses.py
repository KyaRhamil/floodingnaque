"""
API Response Utilities.

Standardized response formatting functions for consistent API responses.
"""
from flask import jsonify
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def api_success(data: Any = None, message: str = None, status_code: int = 200, request_id: str = None) -> tuple:
    """
    Create a standardized success response.
    
    Args:
        data: Response data payload
        message: Optional success message
        status_code: HTTP status code (default: 200)
        request_id: Request identifier for tracing
        
    Returns:
        Tuple of (response_dict, status_code)
    """
    response = {
        'success': True,
        'request_id': request_id
    }
    
    if data is not None:
        response['data'] = data
    
    if message:
        response['message'] = message
    
    return jsonify(response), status_code


def api_error(error_code: str, message: str, status_code: int = 400, request_id: str = None, 
              details: Dict[str, Any] = None) -> tuple:
    """
    Create a standardized error response.
    
    Args:
        error_code: Error code identifier
        message: Human-readable error message
        status_code: HTTP status code (default: 400)
        request_id: Request identifier for tracing
        details: Optional additional error details
        
    Returns:
        Tuple of (response_dict, status_code)
    """
    response = {
        'success': False,
        'error': error_code,
        'message': message,
        'request_id': request_id
    }
    
    if details:
        response['details'] = details
    
    logger.debug(f"API Error [{request_id}]: {error_code} - {message}")
    
    return jsonify(response), status_code


def api_created(data: Any = None, message: str = "Resource created successfully", 
                location: str = None, request_id: str = None) -> tuple:
    """
    Create a standardized response for created resources (201 Created).
    
    Args:
        data: Response data payload
        message: Success message
        location: URI of created resource
        request_id: Request identifier for tracing
        
    Returns:
        Tuple of (response_dict, 201)
    """
    response = {
        'success': True,
        'message': message,
        'request_id': request_id
    }
    
    if data is not None:
        response['data'] = data
    
    headers = {}
    if location:
        headers['Location'] = location
    
    return jsonify(response), 201, headers


def api_accepted(data: Any = None, message: str = "Request accepted for processing", 
                 request_id: str = None) -> tuple:
    """
    Create a standardized response for accepted requests (202 Accepted).
    
    Args:
        data: Response data payload
        message: Success message
        request_id: Request identifier for tracing
        
    Returns:
        Tuple of (response_dict, 202)
    """
    response = {
        'success': True,
        'message': message,
        'request_id': request_id
    }
    
    if data is not None:
        response['data'] = data
    
    return jsonify(response), 202