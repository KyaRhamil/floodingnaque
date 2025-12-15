"""
API Key Authentication Middleware.

Provides decorator-based authentication for protecting API endpoints.
Implements timing-safe comparison to prevent timing attacks.
"""

from functools import wraps
from flask import request, jsonify, current_app, g
import os
import hmac
import hashlib
import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Cache for hashed API keys (computed once at startup)
_hashed_api_keys: Optional[Set[str]] = None


def _hash_api_key(api_key: str) -> str:
    """Hash an API key using SHA-256 for secure storage/comparison."""
    return hashlib.sha256(api_key.encode('utf-8')).hexdigest()


def _timing_safe_compare(a: str, b: str) -> bool:
    """
    Compare two strings in constant time to prevent timing attacks.
    
    Uses hmac.compare_digest which is designed to prevent timing analysis.
    """
    return hmac.compare_digest(a.encode('utf-8'), b.encode('utf-8'))


def get_valid_api_keys() -> Set[str]:
    """
    Get valid API keys from environment variables.
    
    Returns:
        set: Set of valid API keys (empty set if none configured)
    """
    keys_str = os.getenv('VALID_API_KEYS', '')
    if not keys_str:
        return set()
    return set(key.strip() for key in keys_str.split(',') if key.strip())


def get_hashed_api_keys() -> Set[str]:
    """
    Get pre-hashed API keys for secure comparison.
    
    Hashes are computed once and cached for performance.
    """
    global _hashed_api_keys
    if _hashed_api_keys is None:
        valid_keys = get_valid_api_keys()
        _hashed_api_keys = {_hash_api_key(key) for key in valid_keys}
    return _hashed_api_keys


def invalidate_api_key_cache():
    """Invalidate the API key cache (call when keys are updated)."""
    global _hashed_api_keys
    _hashed_api_keys = None


def validate_api_key(api_key: str) -> bool:
    """
    Validate an API key using timing-safe comparison.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not api_key:
        return False
    
    hashed_input = _hash_api_key(api_key)
    hashed_keys = get_hashed_api_keys()
    
    # Use timing-safe comparison for each key
    # We iterate all keys to maintain constant time regardless of match position
    valid = False
    for hashed_key in hashed_keys:
        if _timing_safe_compare(hashed_input, hashed_key):
            valid = True
            # Don't break early - continue to maintain constant time
    
    return valid


def require_api_key(f):
    """
    Decorator that requires a valid API key for endpoint access.
    
    The API key should be provided in the X-API-Key header.
    Authentication bypass requires explicit AUTH_BYPASS_ENABLED=true in development.
    
    Security features:
    - Timing-safe comparison prevents timing attacks
    - API keys are hashed before comparison
    - No information leakage about valid keys
    
    Usage:
        @app.route('/protected')
        @require_api_key
        def protected_endpoint():
            return jsonify({'message': 'Access granted'})
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        valid_keys = get_valid_api_keys()
        is_debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
        auth_bypass = os.getenv('AUTH_BYPASS_ENABLED', 'False').lower() == 'true'
        
        # Require explicit AUTH_BYPASS_ENABLED=true to skip auth in development
        if not valid_keys:
            if is_debug and auth_bypass:
                logger.warning(
                    "AUTH BYPASS: No API keys configured and AUTH_BYPASS_ENABLED=true. "
                    "This should NEVER happen in production!"
                )
                g.authenticated = False
                g.api_key_id = None
                return f(*args, **kwargs)
            elif is_debug:
                logger.warning(
                    f"No API keys configured for {request.method} {request.path}. "
                    "Set VALID_API_KEYS or AUTH_BYPASS_ENABLED=true for development."
                )
                return jsonify({
                    'error': 'Authentication not configured',
                    'message': 'API keys not configured. Set VALID_API_KEYS in .env'
                }), 500
            else:
                logger.error("SECURITY: No API keys configured in production!")
                return jsonify({
                    'error': 'Service unavailable',
                    'message': 'Authentication service is not properly configured'
                }), 503
        
        # Get API key from header
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            logger.warning(
                f"Missing API key for {request.method} {request.path} "
                f"from {request.remote_addr}"
            )
            return jsonify({
                'error': 'API key required',
                'message': 'Please provide a valid API key in the X-API-Key header'
            }), 401
        
        # Validate using timing-safe comparison
        if not validate_api_key(api_key):
            logger.warning(
                f"Invalid API key attempt for {request.method} {request.path} "
                f"from {request.remote_addr}"
            )
            return jsonify({
                'error': 'Invalid API key',
                'message': 'The provided API key is not valid'
            }), 401
        
        # Set authentication context
        g.authenticated = True
        g.api_key_hash = _hash_api_key(api_key)[:8]  # Store truncated hash for logging
        
        return f(*args, **kwargs)
    return decorated


def optional_api_key(f):
    """
    Decorator that accepts but doesn't require an API key.
    
    Useful for endpoints that provide enhanced features for authenticated users
    but still work for anonymous users.
    
    Sets g.authenticated to True if valid key provided (timing-safe).
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        # Set authentication status using Flask's g object
        g.authenticated = False
        g.api_key_hash = None
        
        if api_key and validate_api_key(api_key):
            g.authenticated = True
            g.api_key_hash = _hash_api_key(api_key)[:8]
        
        return f(*args, **kwargs)
    return decorated


def get_auth_context() -> dict:
    """
    Get current authentication context.
    
    Returns:
        dict with 'authenticated' (bool) and 'api_key_hash' (str or None)
    """
    return {
        'authenticated': getattr(g, 'authenticated', False),
        'api_key_hash': getattr(g, 'api_key_hash', None)
    }
