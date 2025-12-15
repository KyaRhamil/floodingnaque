"""
Security Headers Middleware.

Adds security headers to all responses to protect against common web vulnerabilities.
"""

from flask import Flask
import os
import logging

logger = logging.getLogger(__name__)


def add_security_headers(response):
    """
    Add security headers to response.
    
    Headers added:
    - X-Content-Type-Options: Prevents MIME type sniffing
    - X-Frame-Options: Prevents clickjacking
    - X-XSS-Protection: Enables browser XSS filter
    - Strict-Transport-Security: Enforces HTTPS
    - Content-Security-Policy: Restricts resource loading
    - Referrer-Policy: Controls referrer information
    - Permissions-Policy: Controls browser features
    
    Args:
        response: Flask response object
    
    Returns:
        Modified response with security headers
    """
    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # Prevent clickjacking - deny all framing
    response.headers['X-Frame-Options'] = 'DENY'
    
    # Enable browser XSS filter
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Enforce HTTPS (1 year, include subdomains)
    # Only add in production to avoid issues during development
    if os.getenv('ENABLE_HTTPS', 'False').lower() == 'true':
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
    
    # Content Security Policy - adjust based on your needs
    # This is a restrictive policy; you may need to relax it for your frontend
    csp_policy = os.getenv('CSP_POLICY', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'")
    response.headers['Content-Security-Policy'] = csp_policy
    
    # Referrer Policy - send referrer only for same-origin requests
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    # Permissions Policy - disable unnecessary browser features
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    
    # Cache control for API responses
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    
    return response


def setup_security_headers(app: Flask):
    """
    Setup security headers middleware for Flask app.
    
    Args:
        app: Flask application instance
    """
    app.after_request(add_security_headers)
    logger.info("Security headers middleware enabled")


def get_cors_origins():
    """
    Get allowed CORS origins from environment.
    
    Returns:
        list: List of allowed origins
    """
    origins_str = os.getenv('CORS_ORIGINS', '')
    
    if not origins_str:
        # Default to allowing localhost for development
        if os.getenv('FLASK_DEBUG', 'False').lower() == 'true':
            return ['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:5000']
        else:
            # In production, must explicitly set CORS_ORIGINS
            return []
    
    return [origin.strip() for origin in origins_str.split(',') if origin.strip()]


def setup_cors(app: Flask, cors_instance):
    """
    Configure CORS with security settings.
    
    Args:
        app: Flask application instance
        cors_instance: Flask-CORS instance
    """
    origins = get_cors_origins()
    
    if origins:
        cors_instance.init_app(
            app,
            origins=origins,
            methods=['GET', 'POST', 'OPTIONS'],
            allow_headers=['Content-Type', 'Authorization', 'X-API-Key', 'X-Request-ID'],
            expose_headers=['X-Request-ID'],
            supports_credentials=True,
            max_age=600  # Cache preflight for 10 minutes
        )
        logger.info(f"CORS configured for origins: {origins}")
    else:
        logger.warning("No CORS origins configured - all origins will be blocked in production")
