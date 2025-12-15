"""
Security Utilities.

Provides security-related utilities for the Floodingnaque API.
"""

import secrets
import hashlib
import hmac
import os
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def generate_secret_key(length: int = 32) -> str:
    """
    Generate a cryptographically secure secret key.
    
    Args:
        length: Length of the key in bytes (default: 32)
    
    Returns:
        str: Hex-encoded secret key
    """
    return secrets.token_hex(length)


def generate_api_key() -> str:
    """
    Generate a new API key.
    
    Returns:
        str: A unique API key
    """
    return f"floodingnaque_{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    """
    Hash an API key for secure storage.
    
    Args:
        api_key: The API key to hash
    
    Returns:
        str: SHA-256 hash of the API key
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """
    Verify an API key against its hash.
    
    Args:
        api_key: The API key to verify
        hashed_key: The stored hash to compare against
    
    Returns:
        bool: True if the API key matches the hash
    """
    return hmac.compare_digest(hash_api_key(api_key), hashed_key)


def is_secure_password(password: str, min_length: int = 12) -> tuple:
    """
    Check if a password meets security requirements.
    
    Args:
        password: The password to check
        min_length: Minimum required length (default: 12)
    
    Returns:
        tuple: (is_valid: bool, errors: list)
    """
    errors = []
    
    if len(password) < min_length:
        errors.append(f"Password must be at least {min_length} characters long")
    
    if not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")
    
    if not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter")
    
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one digit")
    
    if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
        errors.append("Password must contain at least one special character")
    
    return (len(errors) == 0, errors)


def sanitize_input(value: str, max_length: int = 1000) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        value: The input string to sanitize
        max_length: Maximum allowed length
    
    Returns:
        str: Sanitized string
    """
    if not value:
        return ""
    
    # Truncate to max length
    value = value[:max_length]
    
    # Remove null bytes
    value = value.replace('\x00', '')
    
    # Basic HTML escape (for display purposes)
    value = value.replace('&', '&amp;')
    value = value.replace('<', '&lt;')
    value = value.replace('>', '&gt;')
    value = value.replace('"', '&quot;')
    value = value.replace("'", '&#x27;')
    
    return value


def get_secure_headers() -> dict:
    """
    Get a dictionary of recommended security headers.
    
    Returns:
        dict: Security headers for HTTP responses
    """
    return {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
        'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0'
    }


def validate_origin(origin: str, allowed_origins: list) -> bool:
    """
    Validate if an origin is in the allowed list.
    
    Args:
        origin: The origin to validate
        allowed_origins: List of allowed origins
    
    Returns:
        bool: True if origin is allowed
    """
    if not origin or not allowed_origins:
        return False
    
    # Normalize origin
    origin = origin.rstrip('/')
    
    for allowed in allowed_origins:
        allowed = allowed.rstrip('/')
        if origin == allowed:
            return True
        # Support wildcard subdomains
        if allowed.startswith('*.'):
            domain = allowed[2:]
            if origin.endswith(domain) or origin.endswith('.' + domain):
                return True
    
    return False


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    pass


class SecurityViolation(Exception):
    """Exception raised for security violations."""
    pass
