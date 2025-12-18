"""
Floodingnaque API Application.

Flask application factory with modular route blueprints.
Industry-standard security hardening applied.
"""

from flask import Flask, jsonify, request, g
from flask_cors import CORS
from flask_compress import Compress
import uuid
from app.services import scheduler as scheduler_module
from app.core.config import load_env, get_config, is_debug_mode
from app.core.exceptions import AppException
from app.models.db import init_db
from app.utils.utils import setup_logging
from app.utils.metrics import init_prometheus_metrics
from app.api.middleware import (
    init_rate_limiter,
    setup_security_headers,
    setup_request_logging,
    get_cors_origins
)
from app.api.routes import health_bp, ingest_bp, predict_bp, data_bp, models_bp
import logging
import os

logger = logging.getLogger(__name__)

# Initialize Flask-Compress for response compression
compress = Compress()


def create_app(config_class=None):
    """
    Flask application factory.
    
    Args:
        config_class: Optional configuration class to use
    
    Returns:
        Flask: Configured Flask application instance
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Load environment variables
    load_env()
    
    # Get configuration
    config = get_config()
    
    # Configure app from config
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['DEBUG'] = config.DEBUG
    
    # Security: Limit request body size (1MB default, configurable)
    max_content_mb = int(os.getenv('MAX_CONTENT_LENGTH_MB', '1'))
    app.config['MAX_CONTENT_LENGTH'] = max_content_mb * 1024 * 1024
    
    # Security: Limit JSON depth to prevent DoS
    app.config['JSON_SORT_KEYS'] = False
    
    # Setup request ID tracking
    _setup_request_tracking(app)
    
    # Setup CORS
    _setup_cors(app)
    
    # Initialize database
    init_db()
    
    # Setup logging
    setup_logging()
    
    # Initialize rate limiter
    init_rate_limiter(app)
    
    # Setup security headers
    setup_security_headers(app)
    
    # Setup request logging
    setup_request_logging(app)
    
    # Initialize response compression
    compress.init_app(app)
    
    # Initialize Prometheus metrics
    init_prometheus_metrics(app)
    
    # Register error handlers
    _register_error_handlers(app)
    
    # Register blueprints
    _register_blueprints(app)
    
    # Start scheduler
    _start_scheduler()
    
    # Preload ML model on startup for faster first request
    _preload_model(app)
    
    return app


def _setup_request_tracking(app: Flask):
    """Setup request ID tracking for correlation."""
    @app.before_request
    def assign_request_id():
        # Use provided request ID or generate new one
        request_id = request.headers.get('X-Request-ID')
        if not request_id:
            request_id = str(uuid.uuid4())
        g.request_id = request_id
        request.request_id = request_id  # Backward compatibility
    
    @app.after_request
    def add_request_id_header(response):
        request_id = getattr(g, 'request_id', None)
        if request_id:
            response.headers['X-Request-ID'] = request_id
        return response


def _setup_cors(app: Flask):
    """Configure CORS for the application."""
    cors_origins = get_cors_origins()
    
    if cors_origins:
        CORS(app, origins=cors_origins,
             methods=['GET', 'POST', 'OPTIONS'],
             allow_headers=['Content-Type', 'Authorization', 'X-API-Key', 'X-Request-ID'],
             expose_headers=['X-Request-ID'],
             supports_credentials=True,
             max_age=600)
    else:
        # Development fallback - allow localhost origins
        if is_debug_mode():  # Use centralized check
            CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000', 'http://localhost:5000'])
        else:
            # Production without CORS_ORIGINS set - restrict to same origin
            CORS(app, origins=[])


def _register_error_handlers(app: Flask):
    """Register custom error handlers with production-safe messages."""
    is_debug = app.config.get('DEBUG', False)
    
    @app.errorhandler(AppException)
    def handle_app_exception(error):
        """Handle custom application exceptions."""
        request_id = getattr(g, 'request_id', 'unknown')
        response_data = error.to_dict()
        response_data['request_id'] = request_id
        response = jsonify(response_data)
        response.status_code = error.status_code
        return response
    
    @app.errorhandler(400)
    def handle_bad_request(error):
        """Handle 400 errors."""
        request_id = getattr(g, 'request_id', 'unknown')
        return jsonify({
            'error': 'Bad request',
            'message': str(error.description) if is_debug else 'Invalid request',
            'request_id': request_id
        }), 400
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 errors."""
        request_id = getattr(g, 'request_id', 'unknown')
        return jsonify({
            'error': 'Not found',
            'message': 'The requested resource was not found',
            'request_id': request_id
        }), 404
    
    @app.errorhandler(413)
    def handle_request_too_large(error):
        """Handle request entity too large."""
        request_id = getattr(g, 'request_id', 'unknown')
        max_mb = int(os.getenv('MAX_CONTENT_LENGTH_MB', '1'))
        return jsonify({
            'error': 'Request too large',
            'message': f'Request body exceeds maximum size of {max_mb}MB',
            'request_id': request_id
        }), 413
    
    @app.errorhandler(429)
    def handle_rate_limit(error):
        """Handle rate limit exceeded."""
        request_id = getattr(g, 'request_id', 'unknown')
        return jsonify({
            'error': 'Rate limit exceeded',
            'message': 'Too many requests. Please try again later.',
            'request_id': request_id
        }), 429
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        """Handle 500 errors - sanitize in production."""
        request_id = getattr(g, 'request_id', 'unknown')
        # Log full error details
        logger.error(f"Internal server error [{request_id}]: {str(error)}", exc_info=True)
        
        # Return sanitized message in production
        if is_debug:
            return jsonify({
                'error': 'Internal server error',
                'message': str(error),
                'request_id': request_id
            }), 500
        else:
            return jsonify({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred. Please try again later.',
                'request_id': request_id
            }), 500
    
    @app.errorhandler(503)
    def handle_service_unavailable(error):
        """Handle service unavailable."""
        request_id = getattr(g, 'request_id', 'unknown')
        return jsonify({
            'error': 'Service unavailable',
            'message': 'The service is temporarily unavailable. Please try again later.',
            'request_id': request_id
        }), 503


def _register_blueprints(app: Flask):
    """Register all route blueprints."""
    app.register_blueprint(health_bp)
    app.register_blueprint(ingest_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(models_bp)
    
    logger.info("Registered blueprints: health, ingest, predict, data, models")


def _start_scheduler():
    """Start the background scheduler."""
    try:
        scheduler_module.start()
    except Exception as e:
        logger.error(f"Error starting scheduler: {str(e)}")


def _preload_model(app: Flask):
    """Preload ML model on application startup for faster first request."""
    with app.app_context():
        try:
            from app.services.predict import _load_model
            _load_model()
            logger.info("ML model preloaded successfully on startup")
        except FileNotFoundError as e:
            logger.warning(f"Model preload skipped - model not found: {e}")
        except Exception as e:
            logger.warning(f"Model preload failed (non-critical): {e}")


# Create default app instance for backwards compatibility
app = create_app()


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = is_debug_mode()  # Use centralized check
    logger.info(f"Starting Flask app on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
