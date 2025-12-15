"""
API Key Authentication Middleware.

Provides decorator-based authentication for protecting API endpoints.
"""

from functools import wraps
from flask import request, jsonify, current_app
import os
import logging

logger = logging.getLogger(__name__)


def get_valid_api_keys():
    """
    Get valid API keys from environment variables.
    
    Returns:
        set: Set of valid API keys (empty set if none configured)
    """
    keys_str = os.getenv('VALID_API_KEYS', '')
    if not keys_str:
        return set()
    return set(key.strip() for key in keys_str.split(',') if key.strip())


def require_api_key(f):
    """
    Decorator that requires a valid API key for endpoint access.
    
    The API key should be provided in the X-API-Key header.
    If VALID_API_KEYS environment variable is not set, authentication is bypassed
    (useful for development).
    
    Usage:
        @app.route('/protected')
        @require_api_key
        def protected_endpoint():
            return jsonify({'message': 'Access granted'})
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        valid_keys = get_valid_api_keys()
        
        # If no API keys are configured, bypass authentication (development mode)
        if not valid_keys:
            logger.debug("No API keys configured - authentication bypassed")
            return f(*args, **kwargs)
        
        # Get API key from header
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            logger.warning(f"Missing API key for {request.method} {request.path}")
            return jsonify({
                'error': 'API key required',
                'message': 'Please provide a valid API key in the X-API-Key header'
            }), 401
        
        if api_key not in valid_keys:
            logger.warning(f"Invalid API key attempt for {request.method} {request.path}")
            return jsonify({
                'error': 'Invalid API key',
                'message': 'The provided API key is not valid'
            }), 401
        
        return f(*args, **kwargs)
    return decorated


def optional_api_key(f):
    """
    Decorator that accepts but doesn't require an API key.
    
    Useful for endpoints that provide enhanced features for authenticated users
    but still work for anonymous users.
    
    Sets request.is_authenticated to True if valid key provided.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        valid_keys = get_valid_api_keys()
        api_key = request.headers.get('X-API-Key')
        
        # Set authentication status on request
        request.is_authenticated = False
        
        if api_key and valid_keys and api_key in valid_keys:
            request.is_authenticated = True
        
        return f(*args, **kwargs)
    return decorated
