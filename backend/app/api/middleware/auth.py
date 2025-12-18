"""
API Key Authentication Middleware.

Provides decorator-based authentication for protecting API endpoints.
Implements bcrypt hashing and timing-safe comparison to prevent attacks.

Security Features:
- bcrypt for API key hashing (resistant to rainbow table attacks)
- Timing-safe comparison prevents timing attacks
- No information leakage about valid keys
"""

from functools import wraps
from flask import request, jsonify, current_app, g
import os
import hmac
import hashlib
import logging
from typing import Optional, Set, Dict
from app.core.config import is_debug_mode
from app.core.constants import MIN_API_KEY_LENGTH

try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logging.warning("bcrypt not available, falling back to SHA-256 (less secure)")

logger = logging.getLogger(__name__)

# Cache for hashed API keys (computed once at startup)
_hashed_api_keys: Optional[Dict[str, bytes]] = None  # Maps key_id to bcrypt hash
_legacy_hashed_keys: Optional[Set[str]] = None  # Fallback SHA-256 hashes


def _hash_api_key_bcrypt(api_key: str) -> bytes:
    """
    Hash an API key using bcrypt for secure storage.
    
    bcrypt includes salt and is resistant to rainbow table attacks.
    Cost factor 12 provides good security/performance balance.
    """
    if not BCRYPT_AVAILABLE:
        raise RuntimeError("bcrypt is not installed")
    return bcrypt.hashpw(api_key.encode('utf-8'), bcrypt.gensalt(rounds=12))


def _verify_api_key_bcrypt(api_key: str, hashed: bytes) -> bool:
    """
    Verify an API key against a bcrypt hash.
    
    bcrypt.checkpw is timing-safe by design.
    """
    if not BCRYPT_AVAILABLE:
        return False
    try:
        return bcrypt.checkpw(api_key.encode('utf-8'), hashed)
    except (ValueError, TypeError):
        return False


def _hash_api_key_sha256(api_key: str) -> str:
    """Legacy: Hash an API key using SHA-256 (fallback when bcrypt unavailable)."""
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
    
    Validates that keys meet minimum length requirement (MIN_API_KEY_LENGTH).
    Keys shorter than the minimum are rejected with a warning.
    
    Returns:
        set: Set of valid API keys (empty set if none configured)
    """
    keys_str = os.getenv('VALID_API_KEYS', '')
    if not keys_str:
        return set()
    
    valid_keys = set()
    for key in keys_str.split(','):
        key = key.strip()
        if not key:
            continue
        if len(key) < MIN_API_KEY_LENGTH:
            logger.warning(
                f"API key rejected: length {len(key)} is below minimum {MIN_API_KEY_LENGTH} characters. "
                f"Generate secure keys with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
            continue
        valid_keys.add(key)
    
    return valid_keys


def get_hashed_api_keys() -> Dict[str, bytes]:
    """
    Get pre-hashed API keys for secure comparison.
    
    Uses bcrypt if available, falls back to SHA-256.
    Hashes are computed once and cached for performance.
    
    Returns:
        Dict mapping key identifiers to bcrypt hashes
    """
    global _hashed_api_keys, _legacy_hashed_keys
    
    if _hashed_api_keys is None:
        valid_keys = get_valid_api_keys()
        
        if BCRYPT_AVAILABLE:
            # Use bcrypt for secure hashing
            _hashed_api_keys = {}
            for i, key in enumerate(valid_keys):
                key_id = f"key_{i}"
                _hashed_api_keys[key_id] = _hash_api_key_bcrypt(key)
            logger.info(f"Initialized {len(_hashed_api_keys)} API keys with bcrypt hashing")
        else:
            # Fallback to SHA-256 (less secure)
            _hashed_api_keys = {}
            _legacy_hashed_keys = {_hash_api_key_sha256(key) for key in valid_keys}
            logger.warning("Using SHA-256 for API keys (bcrypt not available)")
    
    return _hashed_api_keys


def invalidate_api_key_cache():
    """Invalidate the API key cache (call when keys are updated)."""
    global _hashed_api_keys, _legacy_hashed_keys
    _hashed_api_keys = None
    _legacy_hashed_keys = None


def validate_api_key(api_key: str) -> bool:
    """
    Validate an API key using bcrypt (preferred) or SHA-256 fallback.
    
    bcrypt.checkpw is inherently timing-safe.
    For SHA-256 fallback, uses hmac.compare_digest.
    
    Args:
        api_key: The API key to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not api_key:
        return False
    
    # Ensure keys are initialized
    get_hashed_api_keys()
    
    if BCRYPT_AVAILABLE and _hashed_api_keys:
        # Verify against all bcrypt hashes (timing-safe)
        # We check all keys to maintain constant time regardless of match position
        valid = False
        for key_id, hashed in _hashed_api_keys.items():
            if _verify_api_key_bcrypt(api_key, hashed):
                valid = True
                # Don't break early - continue to maintain constant time
        return valid
    elif _legacy_hashed_keys:
        # Fallback to SHA-256 comparison
        hashed_input = _hash_api_key_sha256(api_key)
        valid = False
        for hashed_key in _legacy_hashed_keys:
            if _timing_safe_compare(hashed_input, hashed_key):
                valid = True
        return valid
    
    return False


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
        is_debug = is_debug_mode()  # Use centralized check
        auth_bypass = os.getenv('AUTH_BYPASS_ENABLED', 'False').lower() == 'true'
        
        # Require explicit AUTH_BYPASS_ENABLED=true to skip auth in development
        if not valid_keys:
            if is_debug and auth_bypass:
                logger.warning(
                    "AUTH BYPASS: No API keys configured and AUTH_BYPASS_ENABLED=true. "
                    "This should NEVER happen in production!"
                )
                # Mark as bypass mode, NOT authenticated
                # This prevents privilege escalation by clearly distinguishing bypass from auth
                g.authenticated = False
                g.bypass_mode = True  # Explicit bypass flag for audit
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
        g.api_key_hash = _hash_api_key_sha256(api_key)[:8]  # Store truncated hash for logging
        
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
            g.api_key_hash = _hash_api_key_sha256(api_key)[:8]
        
        return f(*args, **kwargs)
    return decorated


def get_auth_context() -> dict:
    """
    Get current authentication context.
    
    Returns:
        dict with:
        - 'authenticated' (bool): True if properly authenticated with valid key
        - 'bypass_mode' (bool): True if auth was bypassed (development only)
        - 'api_key_hash' (str or None): Truncated hash for logging
    """
    return {
        'authenticated': getattr(g, 'authenticated', False),
        'bypass_mode': getattr(g, 'bypass_mode', False),
        'api_key_hash': getattr(g, 'api_key_hash', None)
    }


def is_using_bcrypt() -> bool:
    """Check if bcrypt is being used for API key hashing."""
    return BCRYPT_AVAILABLE
