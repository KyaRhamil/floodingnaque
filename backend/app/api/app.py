"""
Floodingnaque API Application.

Flask application factory with modular route blueprints.
Industry-standard security hardening applied.
"""

import os
import uuid
import logging
from flask import Flask, jsonify, request, g
from flask_cors import CORS
from flask_compress import Compress

from app.services import scheduler as scheduler_module
from app.core.config import load_env, get_config, is_debug_mode
from app.core.exceptions import AppException
from app.models.db import init_db
from app.utils.utils import setup_logging
from app.utils.metrics import init_prometheus_metrics
from app.utils.sentry import init_sentry
from app.api.middleware import (
    init_rate_limiter,
    setup_security_headers,
    setup_request_logging,
    get_cors_origins
)
from app.utils.startup_health import validate_startup_health
from app.api.middleware.request_logger import setup_request_logging_middleware
from app.api.routes.health import health_bp
from app.api.routes.health_k8s import health_k8s_bp
from app.api.routes.ingest import ingest_bp
from app.api.routes.predict import predict_bp
from app.api.routes.data import data_bp
from app.api.routes.models import models_bp
from app.api.routes.webhooks import webhooks_bp
from app.api.routes.batch import batch_bp
from app.api.routes.export import export_bp
from app.api.routes.celery import celery_bp
from app.api.routes.rate_limits import rate_limits_bp
from app.api.routes.tides import tides_bp
from app.api.routes.graphql import graphql_bp, init_graphql_route
from app.api.swagger_config import init_swagger

# Initialize module-level logger
logger = logging.getLogger(__name__)

# Initialize extensions (without app binding)
compress = Compress()
cors = CORS()


def create_app(config_override: dict = None) -> Flask:
    """
    Flask application factory.
    
    Creates and configures the Flask application with all middleware,
    blueprints, and extensions.
    
    Args:
        config_override: Optional dictionary of configuration overrides
    
    Returns:
        Flask: Configured Flask application instance
    """
    # Load environment variables first
    load_env()
    
    # Setup logging
    setup_logging()
    
    # Create Flask app
    app = Flask(__name__)
    
    # Get configuration
    config = get_config()
    
    # Apply Flask configuration
    app.config.update(
        SECRET_KEY=config.SECRET_KEY,
        DEBUG=config.DEBUG,
        JSON_SORT_KEYS=False,
        MAX_CONTENT_LENGTH=int(os.getenv('MAX_CONTENT_LENGTH_MB', 1)) * 1024 * 1024,
        JSONIFY_PRETTYPRINT_REGULAR=config.DEBUG,
    )
    
    # Apply any configuration overrides (useful for testing)
    if config_override:
        app.config.update(config_override)
    
    # ==========================================
    # Request ID Middleware
    # ==========================================
    
    @app.before_request
    def add_request_id():
        """Add a unique request ID to each request."""
        g.request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    
    @app.after_request
    def add_request_id_header(response):
        """Add request ID to response headers."""
        if hasattr(g, 'request_id'):
            response.headers['X-Request-ID'] = g.request_id
        return response
    
    # ==========================================
    # Initialize Extensions
    # ==========================================
    
    # Compression (gzip responses)
    compress.init_app(app)
    logger.info("Response compression enabled")
    
    # CORS (Cross-Origin Resource Sharing)
    cors_origins = get_cors_origins()
    if cors_origins:
        cors.init_app(
            app,
            origins=cors_origins,
            methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            allow_headers=['Content-Type', 'Authorization', 'X-API-Key', 'X-Request-ID'],
            expose_headers=['X-Request-ID', 'X-RateLimit-Limit', 'X-RateLimit-Remaining'],
            supports_credentials=True,
            max_age=600
        )
        logger.info(f"CORS configured for: {cors_origins}")
    else:
        cors.init_app(app)
        if not config.DEBUG:
            logger.warning("No CORS origins configured - restricting cross-origin requests")
    
    # Rate limiting
    init_rate_limiter(app)
    logger.info(f"Rate limiting {'enabled' if config.RATE_LIMIT_ENABLED else 'disabled'}")
    
    # ==========================================
    # Initialize Security
    # ==========================================
    
    # Add security headers
    setup_security_headers(app)
    
    # Request logging middleware
    setup_request_logging(app)
    setup_request_logging_middleware(app)
    
    # Initialize Sentry for error tracking (if configured)
    sentry_dsn = os.getenv('SENTRY_DSN')
    if sentry_dsn:
        init_sentry(app)
        logger.info("Sentry error tracking initialized")
    
    # Initialize Prometheus metrics
    init_prometheus_metrics(app)
    
    # ==========================================
    # Initialize Database
    # ==========================================
    
    init_db()
    logger.info("Database initialized")
    
    # ==========================================
    # Register Blueprints
    # ==========================================
    
    # Core routes (no prefix)
    app.register_blueprint(health_bp)
    app.register_blueprint(health_k8s_bp)
    
    # API routes
    app.register_blueprint(ingest_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(data_bp)
    app.register_blueprint(models_bp)
    app.register_blueprint(webhooks_bp)
    app.register_blueprint(batch_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(celery_bp)
    app.register_blueprint(rate_limits_bp)
    app.register_blueprint(tides_bp)
    app.register_blueprint(graphql_bp)
    
    logger.info("All blueprints registered")
    
    # ==========================================
    # Initialize GraphQL Route
    # ==========================================
    
    init_graphql_route(app)
    
    # ==========================================
    # Swagger/OpenAPI Documentation
    # ==========================================
    
    if os.getenv('FEATURE_API_DOCS_ENABLED', 'True').lower() == 'true':
        init_swagger(app)
        logger.info("Swagger documentation enabled at /apidocs")
    
    # ==========================================
    # Error Handlers
    # ==========================================
    
    @app.errorhandler(AppException)
    def handle_app_exception(error):
        """Handle custom application exceptions."""
        response = error.to_dict()
        response['request_id'] = getattr(g, 'request_id', 'unknown')
        return jsonify(response), error.status_code
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request errors."""
        return jsonify({
            'success': False,
            'error': 'Bad Request',
            'message': str(error.description) if hasattr(error, 'description') else 'Invalid request',
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        """Handle 401 Unauthorized errors."""
        return jsonify({
            'success': False,
            'error': 'Unauthorized',
            'message': 'Authentication required',
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        """Handle 403 Forbidden errors."""
        return jsonify({
            'success': False,
            'error': 'Forbidden',
            'message': 'Access denied',
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 403
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors."""
        return jsonify({
            'success': False,
            'error': 'Not Found',
            'message': 'The requested resource was not found',
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 Method Not Allowed errors."""
        return jsonify({
            'success': False,
            'error': 'Method Not Allowed',
            'message': 'The HTTP method is not allowed for this endpoint',
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 405
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        """Handle 429 Rate Limit Exceeded errors."""
        return jsonify({
            'success': False,
            'error': 'Too Many Requests',
            'message': 'Rate limit exceeded. Please slow down your requests.',
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 429
    
    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle 500 Internal Server Error."""
        logger.error(f"Internal server error: {error}")
        return jsonify({
            'success': False,
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred' if not config.DEBUG else str(error),
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 500
    
    @app.errorhandler(503)
    def service_unavailable(error):
        """Handle 503 Service Unavailable errors."""
        return jsonify({
            'success': False,
            'error': 'Service Unavailable',
            'message': 'The service is temporarily unavailable',
            'request_id': getattr(g, 'request_id', 'unknown')
        }), 503
    
    # ==========================================
    # Startup Tasks
    # ==========================================
    
    with app.app_context():
        # Validate configuration
        warnings = config.validate()
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
        
        # ==========================================
        # Startup Health Validation
        # ==========================================
        # Perform comprehensive health checks during startup
        # In production, raise on critical failures to fail fast
        env = os.getenv('APP_ENV', 'development')
        is_production = env in ('production', 'prod', 'staging', 'stage')
        startup_check_enabled = os.getenv('STARTUP_HEALTH_CHECK', 'True').lower() == 'true'
        
        if startup_check_enabled:
            try:
                health_report = validate_startup_health(
                    check_env=True,
                    check_model=True,
                    check_database_conn=True,
                    check_redis_conn=True,
                    raise_on_failure=is_production,  # Fail fast in production
                    log_results=True,
                )
                
                if not health_report.is_healthy:
                    logger.error(f"Startup health validation failed: {health_report.summary}")
                elif health_report.has_warnings:
                    logger.warning(f"Startup completed with warnings: {health_report.summary}")
                else:
                    logger.info("Startup health validation passed")
                    
            except RuntimeError as e:
                # Critical failure in production - re-raise to prevent startup
                logger.critical(f"CRITICAL: Application startup blocked - {e}")
                raise
            except Exception as e:
                # Non-critical health check failure - log and continue
                logger.warning(f"Startup health check encountered an error: {e}")
        
        # Log startup info
        logger.info(f"Floodingnaque API initialized - Environment: {env}")
        logger.info(f"Debug mode: {config.DEBUG}")
        
        # Start scheduler if enabled
        scheduler_enabled = os.getenv('SCHEDULER_ENABLED', 'True').lower() == 'true'
        if scheduler_enabled:
            try:
                scheduler_module.start()
                logger.info("Background scheduler started")
            except Exception as e:
                logger.error(f"Failed to start scheduler: {e}")
    
    return app


# Create a default app instance for Gunicorn and testing
app = None


def get_app():
    """Get or create the Flask app instance."""
    global app
    if app is None:
        app = create_app()
    return app
