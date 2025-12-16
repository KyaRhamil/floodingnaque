"""
Health and Status Routes.

Provides endpoints for health checks and API status monitoring.
Includes dependency health checks for database connections.
"""

from flask import Blueprint, jsonify, request
from app.services.predict import get_current_model_info
from app.services.scheduler import scheduler
from app.api.middleware.rate_limit import limiter, get_endpoint_limit
from app.models.db import engine, get_db_session
from sqlalchemy import text
import logging
import time

logger = logging.getLogger(__name__)

health_bp = Blueprint('health', __name__)


def check_database_health() -> dict:
    """
    Check database connection health.
    
    Returns:
        dict: Database health status with latency
    """
    try:
        start = time.time()
        with get_db_session() as session:
            # Execute a simple query to test connection
            session.execute(text('SELECT 1'))
        latency_ms = (time.time() - start) * 1000
        
        return {
            'status': 'healthy',
            'connected': True,
            'latency_ms': round(latency_ms, 2)
        }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return {
            'status': 'unhealthy',
            'connected': False,
            'error': str(e)
        }


def check_external_api_health() -> dict:
    """
    Check circuit breaker status for external APIs.
    
    Returns:
        dict: External API circuit breaker status
    """
    try:
        from app.utils.circuit_breaker import openweathermap_breaker, weatherstack_breaker
        
        return {
            'openweathermap': {
                'circuit_state': openweathermap_breaker.state.value,
                'failures': openweathermap_breaker._failures
            },
            'weatherstack': {
                'circuit_state': weatherstack_breaker.state.value,
                'failures': weatherstack_breaker._failures
            }
        }
    except ImportError:
        return {'status': 'circuit_breaker_not_available'}


@health_bp.route('/', methods=['GET'])
def root():
    """Root endpoint - API information."""
    return jsonify({
        'name': 'Floodingnaque API',
        'version': '1.0.0',
        'description': 'Flood prediction API with weather data ingestion',
        'endpoints': {
            'status': '/status',
            'health': '/health',
            'ingest': '/ingest',
            'data': '/data',
            'predict': '/predict',
            'docs': '/api/docs'
        },
        'documentation': f'{request.url_root}api/docs'
    }), 200


@health_bp.route('/status', methods=['GET'])
@limiter.limit(get_endpoint_limit('status'))
def status():
    """Health check endpoint with database connectivity."""
    model_info = get_current_model_info()
    model_status = 'loaded' if model_info else 'not found'
    
    # Check database health
    db_health = check_database_health()
    
    response = {
        'status': 'running',
        'database': db_health['status'],
        'database_latency_ms': db_health.get('latency_ms'),
        'model': model_status
    }
    
    if model_info and model_info.get('metadata'):
        response['model_version'] = model_info['metadata'].get('version')
        response['model_accuracy'] = model_info['metadata'].get('metrics', {}).get('accuracy')
    
    # Set appropriate HTTP status based on health
    http_status = 200 if db_health['connected'] else 503
    
    return jsonify(response), http_status


@health_bp.route('/health', methods=['GET'])
@limiter.limit(get_endpoint_limit('status'))
def health():
    """Detailed health check endpoint with all dependency status."""
    model_info = get_current_model_info()
    model_available = model_info is not None
    
    # Check database health
    db_health = check_database_health()
    
    # Check external API circuit breakers
    external_apis = check_external_api_health()
    
    # Determine overall health status
    is_healthy = db_health['connected'] and model_available
    
    response = {
        'status': 'healthy' if is_healthy else 'degraded',
        'checks': {
            'database': db_health,
            'model_available': model_available,
            'scheduler_running': scheduler.running if hasattr(scheduler, 'running') else False,
            'external_apis': external_apis
        }
    }
    
    if model_info:
        response['model'] = {
            'loaded': True,
            'type': model_info.get('model_type'),
            'path': model_info.get('model_path'),
            'features': model_info.get('features', [])
        }
        if model_info.get('metadata'):
            metadata = model_info['metadata']
            response['model']['version'] = metadata.get('version')
            response['model']['created_at'] = metadata.get('created_at')
            response['model']['metrics'] = {
                'accuracy': metadata.get('metrics', {}).get('accuracy'),
                'precision': metadata.get('metrics', {}).get('precision'),
                'recall': metadata.get('metrics', {}).get('recall'),
                'f1_score': metadata.get('metrics', {}).get('f1_score')
            }
    else:
        response['model'] = {'loaded': False}
    
    # Set appropriate HTTP status
    http_status = 200 if is_healthy else 503
    
    return jsonify(response), http_status
