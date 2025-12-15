"""
Flood Prediction Routes.

Provides endpoints for flood risk prediction using ML models.
Includes input validation and security measures.
"""

from flask import Blueprint, jsonify, request, g
from werkzeug.exceptions import BadRequest
from app.services.predict import predict_flood
from app.api.middleware.auth import require_api_key
from app.api.middleware.rate_limit import limiter, get_endpoint_limit
from app.api.schemas.weather import parse_json_safely
from app.utils.validation import InputValidator, ValidationError as InputValidationError
import logging

logger = logging.getLogger(__name__)

predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/predict', methods=['POST'])
@limiter.limit(get_endpoint_limit('predict'))
@require_api_key
def predict():
    """Predict flood risk based on weather data."""
    request_id = getattr(g, 'request_id', 'unknown')
    
    try:
        # Handle JSON parsing with better error handling
        try:
            input_data = request.get_json(force=True, silent=True)
        except BadRequest as e:
            logger.error(f"BadRequest parsing JSON in predict [{request_id}]: {str(e)}")
            return jsonify({
                'error': 'Invalid JSON format',
                'message': 'Please check your request body.',
                'request_id': request_id
            }), 400
        
        if input_data is None:
            # Try to parse manually if get_json failed
            if request.data:
                input_data = parse_json_safely(request.data)
                if input_data is None:
                    logger.error(f"All JSON parsing attempts failed in predict [{request_id}]")
                    return jsonify({
                        'error': 'Invalid JSON format',
                        'message': 'Please ensure your JSON is properly formatted.',
                        'request_id': request_id
                    }), 400
            else:
                return jsonify({
                    'error': 'No input data',
                    'message': 'No input data provided',
                    'request_id': request_id
                }), 400
        
        if not input_data:
            return jsonify({
                'error': 'No input data',
                'message': 'No input data provided',
                'request_id': request_id
            }), 400
        
        # Validate and sanitize input using InputValidator
        try:
            validated_data = InputValidator.validate_prediction_input(input_data)
        except InputValidationError as e:
            logger.warning(f"Input validation failed [{request_id}]: {str(e)}")
            return jsonify({
                'error': 'Validation error',
                'message': str(e),
                'request_id': request_id
            }), 400
        
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
        
        return jsonify(response), 200
        
    except InputValidationError as e:
        logger.error(f"Validation error in predict [{request_id}]: {str(e)}")
        return jsonify({
            'error': 'Validation error',
            'message': str(e),
            'request_id': request_id
        }), 400
    except ValueError as e:
        logger.error(f"Validation error in predict [{request_id}]: {str(e)}")
        return jsonify({
            'error': 'Validation error',
            'message': str(e),
            'request_id': request_id
        }), 400
    except FileNotFoundError as e:
        logger.error(f"Model not found [{request_id}]: {str(e)}")
        return jsonify({
            'error': 'Model not found',
            'message': str(e),
            'request_id': request_id
        }), 404
    except Exception as e:
        logger.error(f"Error in predict endpoint [{request_id}]: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Prediction failed',
            'message': 'An error occurred during prediction',
            'request_id': request_id
        }), 500
