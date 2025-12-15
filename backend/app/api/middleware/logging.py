"""
Request Logging Middleware.

Provides request/response logging for API monitoring and debugging.
"""

from flask import Flask, request, g
import logging
import time
import uuid
from functools import wraps

logger = logging.getLogger(__name__)


def add_request_id():
    """
    Add request ID to Flask request context.
    
    Returns:
        str: The request ID (either from header or newly generated)
    """
    request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    request.request_id = request_id
    return request_id


def log_request():
    """Log incoming request details."""
    request_id = getattr(request, 'request_id', 'unknown')
    logger.info(
        f"Request {request_id}: {request.method} {request.path}",
        extra={
            'request_id': request_id,
            'method': request.method,
            'path': request.path,
            'remote_addr': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', 'unknown')
        }
    )


def log_response(response):
    """
    Log response details.
    
    Args:
        response: Flask response object
        
    Returns:
        Response object with logging completed
    """
    request_id = getattr(request, 'request_id', 'unknown')
    duration = getattr(g, 'request_start_time', None)
    
    if duration:
        elapsed = time.time() - duration
        logger.info(
            f"Response {request_id}: {response.status_code} ({elapsed:.3f}s)",
            extra={
                'request_id': request_id,
                'status_code': response.status_code,
                'duration_seconds': elapsed
            }
        )
    else:
        logger.info(
            f"Response {request_id}: {response.status_code}",
            extra={
                'request_id': request_id,
                'status_code': response.status_code
            }
        )
    
    return response


def setup_request_logging(app: Flask):
    """
    Setup request logging middleware for Flask app.
    
    Adds before_request and after_request handlers for comprehensive logging.
    
    Args:
        app: Flask application instance
    """
    
    @app.before_request
    def before_request():
        """Execute before each request."""
        g.request_start_time = time.time()
        add_request_id()
        log_request()
    
    @app.after_request
    def after_request(response):
        """Execute after each request."""
        return log_response(response)
    
    logger.info("Request logging middleware enabled")


def request_logger(f):
    """
    Decorator for logging individual route handlers.
    
    Usage:
        @app.route('/endpoint')
        @request_logger
        def endpoint():
            return jsonify({'message': 'success'})
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        request_id = getattr(request, 'request_id', str(uuid.uuid4()))
        start_time = time.time()
        
        logger.debug(f"Entering handler for {request.path} [{request_id}]")
        
        try:
            result = f(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.debug(f"Handler completed for {request.path} [{request_id}] ({elapsed:.3f}s)")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Handler error for {request.path} [{request_id}] ({elapsed:.3f}s): {str(e)}")
            raise
    
    return decorated
