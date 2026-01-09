"""
Flood Prediction Routes.

Provides endpoints for flood risk prediction using ML models.
Includes input validation, security measures, and response caching.
"""

from flask import Blueprint, jsonify, request, g
from werkzeug.exceptions import BadRequest
from app.services.predict import predict_flood
from app.api.middleware.auth import require_api_key
from app.api.middleware.rate_limit import limiter, get_endpoint_limit, rate_limit_with_burst
from app.api.schemas.weather import parse_json_safely
from app.utils.validation import InputValidator, validate_request_size
from app.utils.cache import (
    get_cached_prediction, 
    cache_prediction_result, 
    make_weather_hash,
    is_cache_enabled,
)
from app.core.exceptions import ValidationError, api_error
from app.core.constants import HTTP_OK, HTTP_BAD_REQUEST, HTTP_NOT_FOUND, HTTP_INTERNAL_ERROR
import logging
import os

logger = logging.getLogger(__name__)

predict_bp = Blueprint('predict', __name__)

# Prediction cache TTL (seconds)
PREDICTION_CACHE_TTL = int(os.getenv('PREDICTION_CACHE_TTL', '300'))  # 5 minutes default
PREDICTION_CACHE_ENABLED = os.getenv('PREDICTION_CACHE_ENABLED', 'True').lower() == 'true'


@predict_bp.route('/predict', methods=['POST'])
@rate_limit_with_burst("60 per hour")
@validate_request_size(endpoint_name='predict')  # 10KB limit for prediction payloads
@require_api_key
def predict():
    """Predict flood risk based on weather data with optional caching."""
    request_id = getattr(g, 'request_id', 'unknown')
    
    try:
        # Handle JSON parsing with better error handling
        try:
            input_data = request.get_json(force=True, silent=True)
        except BadRequest as e:
            logger.error(f"BadRequest parsing JSON in predict [{request_id}]: {str(e)}")
            return api_error('InvalidJSON', 'Please check your request body.', HTTP_BAD_REQUEST, request_id)
        
        if input_data is None:
            # Try to parse manually if get_json failed
            if request.data:
                input_data = parse_json_safely(request.data)
                if input_data is None:
                    logger.error(f"All JSON parsing attempts failed in predict [{request_id}]")
                    return api_error('InvalidJSON', 'Please ensure your JSON is properly formatted.', HTTP_BAD_REQUEST, request_id)
            else:
                return api_error('NoInput', 'No input data provided', HTTP_BAD_REQUEST, request_id)
        
        if not input_data:
            return api_error('NoInput', 'No input data provided', HTTP_BAD_REQUEST, request_id)
        
        # Validate and sanitize input using InputValidator
        try:
            validated_data = InputValidator.validate_prediction_input(input_data)
        except ValidationError as e:
            logger.warning(f"Input validation failed [{request_id}]: {str(e)}")
            return api_error('ValidationError', str(e), HTTP_BAD_REQUEST, request_id)
        
        # Check prediction cache (if enabled)
        cache_hit = False
        weather_hash = None
        if PREDICTION_CACHE_ENABLED and is_cache_enabled():
            weather_hash = make_weather_hash(validated_data)
            cached_result = get_cached_prediction(weather_hash)
            if cached_result:
                logger.debug(f"Prediction cache HIT [{request_id}]: {weather_hash}")
                cached_result['request_id'] = request_id
                cached_result['cache_hit'] = True
                return jsonify(cached_result), HTTP_OK
        
        # Extract model version (validated separately)
        model_version = validated_data.pop('model_version', None)
        return_proba = request.args.get('return_proba', 'false').lower() == 'true'
        return_risk_level = request.args.get('risk_level', 'true').lower() == 'true'
        
        prediction = predict_flood(
            validated_data,  # Use validated data
            model_version=model_version, 
            return_proba=return_proba or return_risk_level,
            return_risk_level=return_risk_level
        )
        
        # Handle dict response (with probabilities and risk level) or int response
        if isinstance(prediction, dict):
            response = {
                'prediction': prediction['prediction'],
                'flood_risk': 'high' if prediction['prediction'] == 1 else 'low',
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
        
        # Cache the prediction result
        if PREDICTION_CACHE_ENABLED and weather_hash and is_cache_enabled():
            cache_prediction_result(weather_hash, response, PREDICTION_CACHE_TTL)
            logger.debug(f"Prediction cached [{request_id}]: {weather_hash}")
        
        response['cache_hit'] = False
        return jsonify(response), HTTP_OK
        
    except ValidationError as e:
        logger.error(f"Validation error in predict [{request_id}]: {str(e)}")
        return api_error('ValidationError', str(e), HTTP_BAD_REQUEST, request_id)
    except ValueError as e:
        logger.error(f"Validation error in predict [{request_id}]: {str(e)}")
        return api_error('ValidationError', str(e), HTTP_BAD_REQUEST, request_id)
    except FileNotFoundError as e:
        logger.error(f"Model not found [{request_id}]: {str(e)}")
        return api_error('ModelNotFound', str(e), HTTP_NOT_FOUND, request_id)
    except Exception as e:
        logger.error(f"Error in predict endpoint [{request_id}]: {str(e)}", exc_info=True)
        return api_error('PredictionFailed', 'An error occurred during prediction', HTTP_INTERNAL_ERROR, request_id)
