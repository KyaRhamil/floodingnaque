"""
Health and Status Routes.

Provides endpoints for health checks and API status monitoring.
"""

from flask import Blueprint, jsonify, request
from app.services.predict import get_current_model_info
from app.services.scheduler import scheduler
from app.api.middleware.rate_limit import limiter, get_endpoint_limit
import logging

logger = logging.getLogger(__name__)

health_bp = Blueprint('health', __name__)


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
    """Health check endpoint."""
    model_info = get_current_model_info()
    model_status = 'loaded' if model_info else 'not found'
    
    response = {
        'status': 'running',
        'database': 'connected',
        'model': model_status
    }
    
    if model_info and model_info.get('metadata'):
        response['model_version'] = model_info['metadata'].get('version')
        response['model_accuracy'] = model_info['metadata'].get('metrics', {}).get('accuracy')
    
    return jsonify(response)


@health_bp.route('/health', methods=['GET'])
@limiter.limit(get_endpoint_limit('status'))
def health():
    """Detailed health check endpoint."""
    model_info = get_current_model_info()
    model_available = model_info is not None
    
    response = {
        'status': 'healthy',
        'database': 'connected',
        'model_available': model_available,
        'scheduler_running': scheduler.running if hasattr(scheduler, 'running') else False
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
    
    return jsonify(response)
