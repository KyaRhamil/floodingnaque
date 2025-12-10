from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
from scheduler import scheduler
from ingest import ingest_data
from predict import predict_flood, get_current_model_info, list_available_models, get_latest_model_version
from risk_classifier import RISK_LEVELS, get_risk_thresholds
from config import load_env
from db import init_db, WeatherData, get_db_session
from utils import setup_logging, validate_coordinates
import logging
import os
import json
import codecs
from datetime import datetime, timedelta
from functools import wraps
import uuid

def parse_json_safely(data_bytes):
    """
    Safely parse JSON from request data, handling double-escaped strings from PowerShell curl.
    
    Args:
        data_bytes: Raw bytes from request.data
    
    Returns:
        dict: Parsed JSON data or None if parsing fails
    """
    if not data_bytes:
        return {}
    
    try:
        # Decode bytes to string
        raw_str = data_bytes.decode('utf-8')
        
        # Try direct JSON parse first
        try:
            return json.loads(raw_str)
        except json.JSONDecodeError:
            pass
        
        # Handle double-escaped JSON (common with PowerShell curl)
        # Check if string contains escaped quotes like: {\"lat\": 14.6}
        if '\\"' in raw_str or '\\\\' in raw_str:
            # Try unescaping: replace \\\" with \"
            unescaped = raw_str.replace('\\"', '"').replace('\\\\', '\\')
            try:
                return json.loads(unescaped)
            except json.JSONDecodeError:
                pass
        
        # Try using codecs to decode escaped sequences
        try:
            decoded = codecs.decode(raw_str, 'unicode_escape')
            return json.loads(decoded)
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
        
        # Last resort: try removing all backslashes before quotes (aggressive fix)
        if raw_str.count('\\') > raw_str.count('"'):
            cleaned = raw_str.replace('\\"', '"').replace('\\n', '\n').replace('\\r', '\r').replace('\\t', '\t')
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        
        return None
    except Exception as e:
        # Logging will be handled by the calling function
        return None

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Load environment variables
load_env()

# Initialize database
init_db()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Request ID tracking
def add_request_id():
    """Add request ID to Flask request context."""
    request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
    request.request_id = request_id
    return request_id

@app.before_request
def before_request():
    """Execute before each request."""
    request_id = add_request_id()
    logger.info(f"Request {request_id}: {request.method} {request.path}")

# Start scheduler (with error handling)
try:
    scheduler.start()
    logger.info("Scheduler started successfully")
except Exception as e:
    logger.error(f"Error starting scheduler: {str(e)}")

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - API information."""
    return jsonify({
        'name': 'Flooding Naque API',
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

@app.route('/status', methods=['GET'])
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

@app.route('/ingest', methods=['GET', 'POST'])
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
    try:
        # Get location from request if provided
        # Handle JSON parsing with better error handling
        try:
            request_data = request.get_json(force=True, silent=True)
        except BadRequest as e:
            logger.error(f"BadRequest parsing JSON: {str(e)}")
            return jsonify({'error': 'Invalid JSON format. Please check your request body.'}), 400
        
        if request_data is None:
            # Try to parse manually if get_json failed
            if request.data:
                request_data = parse_json_safely(request.data)
                if request_data is None:
                    logger.error(f"All JSON parsing attempts failed for data: {request.data.decode('utf-8', errors='replace')[:200]}")
                    return jsonify({'error': 'Invalid JSON format. Please ensure your JSON is properly formatted.'}), 400
            else:
                request_data = {}
        
        lat = request_data.get('lat')
        lon = request_data.get('lon')
        
        # Validate coordinates if provided
        if lat is not None or lon is not None:
            validate_coordinates(lat, lon)
        
        data = ingest_data(lat=lat, lon=lon)
        
        request_id = getattr(request, 'request_id', 'unknown')
        return jsonify({
            'message': 'Data ingested successfully',
            'data': data,
            'request_id': request_id
        }), 200
    except ValueError as e:
        request_id = getattr(request, 'request_id', 'unknown')
        logger.error(f"Validation error in ingest [{request_id}]: {str(e)}")
        return jsonify({'error': str(e), 'request_id': request_id}), 400
    except BadRequest as e:
        request_id = getattr(request, 'request_id', 'unknown')
        logger.error(f"BadRequest error in ingest [{request_id}]: {str(e)}")
        return jsonify({'error': f'Invalid request: {str(e)}', 'request_id': request_id}), 400
    except Exception as e:
        request_id = getattr(request, 'request_id', 'unknown')
        logger.error(f"Error in ingest endpoint [{request_id}]: {str(e)}")
        return jsonify({'error': str(e), 'request_id': request_id}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Predict flood risk based on weather data."""
    try:
        # Handle JSON parsing with better error handling
        try:
            input_data = request.get_json(force=True, silent=True)
        except BadRequest as e:
            logger.error(f"BadRequest parsing JSON in predict: {str(e)}")
            return jsonify({'error': 'Invalid JSON format. Please check your request body.'}), 400
        
        if input_data is None:
            # Try to parse manually if get_json failed
            if request.data:
                input_data = parse_json_safely(request.data)
                if input_data is None:
                    logger.error(f"All JSON parsing attempts failed in predict")
                    return jsonify({'error': 'Invalid JSON format. Please ensure your JSON is properly formatted.'}), 400
            else:
                return jsonify({'error': 'No input data provided'}), 400
        
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Check if model version is specified (extract from input_data but don't pass to predict)
        model_version = input_data.get('model_version') if isinstance(input_data, dict) else None
        return_proba = request.args.get('return_proba', 'false').lower() == 'true'
        return_risk_level = request.args.get('risk_level', 'true').lower() == 'true'  # Default to true for 3-level classification
        
        # Create a copy of input_data without model_version for prediction
        prediction_data = input_data.copy() if isinstance(input_data, dict) else input_data
        if isinstance(prediction_data, dict) and 'model_version' in prediction_data:
            prediction_data = {k: v for k, v in prediction_data.items() if k != 'model_version'}
        
        prediction = predict_flood(
            prediction_data, 
            model_version=model_version, 
            return_proba=return_proba or return_risk_level,  # Need proba for risk level
            return_risk_level=return_risk_level
        )
        request_id = getattr(request, 'request_id', 'unknown')
        
        # Handle dict response (with probabilities and risk level) or int response
        if isinstance(prediction, dict):
            response = {
                'prediction': prediction['prediction'],
                'flood_risk': 'high' if prediction['prediction'] == 1 else 'low',  # Backward compatibility
                'model_version': prediction.get('model_version'),
                'request_id': request_id
            }
            # Add probability if available
            if 'probability' in prediction:
                response['probability'] = prediction['probability']
            # Add risk level classification (3-level: Safe/Alert/Critical)
            if 'risk_label' in prediction:
                response['risk_level'] = prediction.get('risk_level')
                response['risk_label'] = prediction.get('risk_label')
                response['risk_color'] = prediction.get('risk_color')
                response['risk_description'] = prediction.get('risk_description')
                response['confidence'] = prediction.get('confidence')
        else:
            # Simple int response - convert to dict with basic info
            response = {
                'prediction': prediction,
                'flood_risk': 'high' if prediction == 1 else 'low',
                'request_id': request_id
            }
        
        return jsonify(response), 200
    except ValueError as e:
        request_id = getattr(request, 'request_id', 'unknown')
        logger.error(f"Validation error in predict [{request_id}]: {str(e)}")
        return jsonify({'error': str(e), 'request_id': request_id}), 400
    except FileNotFoundError as e:
        request_id = getattr(request, 'request_id', 'unknown')
        logger.error(f"Model not found [{request_id}]: {str(e)}")
        return jsonify({'error': str(e), 'request_id': request_id}), 404
    except Exception as e:
        request_id = getattr(request, 'request_id', 'unknown')
        logger.error(f"Error in predict endpoint [{request_id}]: {str(e)}")
        return jsonify({'error': str(e), 'request_id': request_id}), 500

@app.route('/health', methods=['GET'])
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

@app.route('/data', methods=['GET'])
def get_weather_data():
    """Retrieve historical weather data."""
    try:
        # Get query parameters
        limit = request.args.get('limit', default=100, type=int)
        offset = request.args.get('offset', default=0, type=int)
        start_date = request.args.get('start_date', type=str)
        end_date = request.args.get('end_date', type=str)
        
        # Validate limit
        if limit < 1 or limit > 1000:
            return jsonify({'error': 'Limit must be between 1 and 1000'}), 400
        
        with get_db_session() as session:
            query = session.query(WeatherData)
            
            # Filter by date range if provided
            if start_date:
                try:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    query = query.filter(WeatherData.timestamp >= start_dt)
                except ValueError:
                    return jsonify({'error': 'Invalid start_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)'}), 400
            
            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    query = query.filter(WeatherData.timestamp <= end_dt)
                except ValueError:
                    return jsonify({'error': 'Invalid end_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)'}), 400
            
            # Get total count
            total = query.count()
            
            # Apply pagination
            query = query.order_by(WeatherData.timestamp.desc())
            query = query.offset(offset).limit(limit)
            
            # Fetch results
            results = query.all()
            
            # Convert to dict
            data = [{
                'id': r.id,
                'temperature': r.temperature,
                'humidity': r.humidity,
                'precipitation': r.precipitation,
                'timestamp': r.timestamp.isoformat() if r.timestamp else None
            } for r in results]
        
        request_id = getattr(request, 'request_id', 'unknown')
        return jsonify({
            'data': data,
            'total': total,
            'limit': limit,
            'offset': offset,
            'count': len(data),
            'request_id': request_id
        }), 200
    except Exception as e:
        request_id = getattr(request, 'request_id', 'unknown')
        logger.error(f"Error retrieving weather data [{request_id}]: {str(e)}")
        return jsonify({'error': str(e), 'request_id': request_id}), 500

@app.route('/api/version', methods=['GET'])
def api_version():
    """API version endpoint."""
    return jsonify({
        'version': '1.0.0',
        'name': 'Flooding Naque API',
        'base_url': request.url_root.rstrip('/')
    }), 200

@app.route('/api/models', methods=['GET'])
def list_models():
    """List all available model versions."""
    try:
        models = list_available_models()
        current_info = get_current_model_info()
        current_version = current_info.get('metadata', {}).get('version') if current_info else None
        
        # Format response
        formatted_models = []
        for model in models:
            formatted_model = {
                'version': model['version'],
                'path': model['path'],
                'is_current': model['version'] == current_version
            }
            if model.get('metadata'):
                metadata = model['metadata']
                formatted_model['created_at'] = metadata.get('created_at')
                formatted_model['metrics'] = {
                    'accuracy': metadata.get('metrics', {}).get('accuracy'),
                    'precision': metadata.get('metrics', {}).get('precision'),
                    'recall': metadata.get('metrics', {}).get('recall'),
                    'f1_score': metadata.get('metrics', {}).get('f1_score')
                }
            formatted_models.append(formatted_model)
        
        request_id = getattr(request, 'request_id', 'unknown')
        return jsonify({
            'models': formatted_models,
            'current_version': current_version,
            'total_versions': len(formatted_models),
            'request_id': request_id
        }), 200
    except Exception as e:
        request_id = getattr(request, 'request_id', 'unknown')
        logger.error(f"Error listing models [{request_id}]: {str(e)}")
        return jsonify({'error': str(e), 'request_id': request_id}), 500

@app.route('/api/docs', methods=['GET'])
def api_docs():
    """API documentation endpoint."""
    docs = {
        'endpoints': {
            'GET /status': {
                'description': 'Basic health check endpoint',
                'response': {
                    'status': 'running',
                    'database': 'connected',
                    'model': 'loaded | not found'
                }
            },
            'GET /health': {
                'description': 'Detailed health check endpoint',
                'response': {
                    'status': 'healthy',
                    'database': 'connected',
                    'model_available': 'boolean',
                    'scheduler_running': 'boolean'
                }
            },
            'POST /ingest': {
                'description': 'Ingest weather data from external APIs',
                'request_body': {
                    'lat': 'float (optional, -90 to 90)',
                    'lon': 'float (optional, -180 to 180)'
                },
                'response': {
                    'message': 'Data ingested successfully',
                    'data': {
                        'temperature': 'float',
                        'humidity': 'float',
                        'precipitation': 'float',
                        'timestamp': 'ISO datetime string'
                    }
                }
            },
            'GET /data': {
                'description': 'Retrieve historical weather data',
                'query_parameters': {
                    'limit': 'int (1-1000, default: 100)',
                    'offset': 'int (default: 0)',
                    'start_date': 'ISO datetime string (optional)',
                    'end_date': 'ISO datetime string (optional)'
                },
                'response': {
                    'data': 'array of weather records',
                    'total': 'int',
                    'limit': 'int',
                    'offset': 'int',
                    'count': 'int'
                }
            },
            'POST /predict': {
                'description': 'Predict flood risk based on weather data with 3-level classification (Safe/Alert/Critical)',
                'request_body': {
                    'temperature': 'float (required)',
                    'humidity': 'float (required)',
                    'precipitation': 'float (required)',
                    'model_version': 'int (optional) - Specific model version to use'
                },
                'query_parameters': {
                    'return_proba': 'boolean (default: false) - Include prediction probabilities',
                    'risk_level': 'boolean (default: true) - Include 3-level risk classification'
                },
                'response': {
                    'prediction': '0 or 1 (binary)',
                    'flood_risk': 'low | high (binary, backward compatible)',
                    'risk_level': '0 (Safe) | 1 (Alert) | 2 (Critical)',
                    'risk_label': 'Safe | Alert | Critical',
                    'risk_color': 'Hex color code',
                    'risk_description': 'Human-readable description',
                    'confidence': 'float (0-1)',
                    'probability': 'object with no_flood and flood probabilities (if return_proba=true)'
                }
            }
        },
        'version': '1.0.0',
        'base_url': request.url_root.rstrip('/')
    }
    return jsonify(docs), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    logger.info(f"Starting Flask app on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
