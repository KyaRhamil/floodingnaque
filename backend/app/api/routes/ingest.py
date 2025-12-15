"""
Data Ingestion Routes.

Provides endpoints for ingesting weather data from external APIs.
Includes input validation and security measures.
"""

from flask import Blueprint, jsonify, request, g
from werkzeug.exceptions import BadRequest
from app.services.ingest import ingest_data
from app.utils.validation import InputValidator, ValidationError as InputValidationError
from app.api.middleware.auth import require_api_key
from app.api.middleware.rate_limit import limiter, get_endpoint_limit
from app.api.schemas.weather import parse_json_safely
import logging

logger = logging.getLogger(__name__)

ingest_bp = Blueprint('ingest', __name__)


@ingest_bp.route('/ingest', methods=['GET', 'POST'])
@limiter.limit(get_endpoint_limit('ingest'))
@require_api_key
def ingest():
    """Ingest weather data from external APIs."""
    # Handle GET requests - show usage information
    if request.method == 'GET':
        return jsonify({
            'endpoint': '/ingest',
            'method': 'POST',
            'description': 'Ingest weather data from external APIs (OpenWeatherMap and Weatherstack)',
            'usage': {
                'curl_example': 'curl -X POST http://127.0.0.1:5000/ingest -H "Content-Type: application/json" -d \'{"lat": 14.6, "lon": 120.98}\'',
                'powershell_example': '$body = @{lat=14.6; lon=120.98} | ConvertTo-Json; Invoke-RestMethod -Uri http://127.0.0.1:5000/ingest -Method POST -ContentType "application/json" -Body $body',
                'request_body': {
                    'lat': 'float (optional, -90 to 90) - Latitude',
                    'lon': 'float (optional, -180 to 180) - Longitude'
                },
                'note': 'If lat/lon are not provided, defaults to New York City (40.7128, -74.0060)'
            },
            'response_example': {
                'message': 'Data ingested successfully',
                'data': {
                    'temperature': 298.15,
                    'humidity': 65.0,
                    'precipitation': 0.0,
                    'timestamp': '2025-12-11T03:00:00'
                },
                'request_id': 'uuid-string'
            },
            'alternative_endpoints': {
                'api_docs': '/api/docs - Full API documentation',
                'status': '/status - Health check',
                'health': '/health - Detailed health check'
            }
        }), 200
    
    # Handle POST requests - actual ingestion
    request_id = getattr(g, 'request_id', 'unknown')
    
    try:
        # Handle JSON parsing with better error handling
        try:
            request_data = request.get_json(force=True, silent=True)
        except BadRequest as e:
            logger.error(f"BadRequest parsing JSON [{request_id}]: {str(e)}")
            return jsonify({
                'error': 'Invalid JSON format',
                'message': 'Please check your request body.',
                'request_id': request_id
            }), 400
        
        if request_data is None:
            # Try to parse manually if get_json failed
            if request.data:
                request_data = parse_json_safely(request.data)
                if request_data is None:
                    logger.error(f"All JSON parsing attempts failed [{request_id}]")
                    return jsonify({
                        'error': 'Invalid JSON format',
                        'message': 'Please ensure your JSON is properly formatted.',
                        'request_id': request_id
                    }), 400
            else:
                request_data = {}
        
        lat = request_data.get('lat')
        lon = request_data.get('lon')
        
        # Validate coordinates if provided using InputValidator
        if lat is not None or lon is not None:
            try:
                lat, lon = InputValidator.validate_coordinates(lat, lon)
            except InputValidationError as e:
                logger.warning(f"Coordinate validation failed [{request_id}]: {str(e)}")
                return jsonify({
                    'error': 'Validation error',
                    'message': str(e),
                    'request_id': request_id
                }), 400
        
        data = ingest_data(lat=lat, lon=lon)
        
        return jsonify({
            'message': 'Data ingested successfully',
            'data': {
                'temperature': data.get('temperature'),
                'humidity': data.get('humidity'),
                'precipitation': data.get('precipitation'),
                'timestamp': data.get('timestamp').isoformat() if data.get('timestamp') else None
            },
            'request_id': request_id
        }), 200
        
    except InputValidationError as e:
        logger.error(f"Validation error in ingest [{request_id}]: {str(e)}")
        return jsonify({
            'error': 'Validation error',
            'message': str(e),
            'request_id': request_id
        }), 400
    except ValueError as e:
        logger.error(f"Validation error in ingest [{request_id}]: {str(e)}")
        return jsonify({
            'error': 'Validation error',
            'message': str(e),
            'request_id': request_id
        }), 400
    except BadRequest as e:
        logger.error(f"BadRequest error in ingest [{request_id}]: {str(e)}")
        return jsonify({
            'error': 'Invalid request',
            'message': 'The request could not be processed',
            'request_id': request_id
        }), 400
    except Exception as e:
        logger.error(f"Error in ingest endpoint [{request_id}]: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Ingestion failed',
            'message': 'An error occurred during data ingestion',
            'request_id': request_id
        }), 500
