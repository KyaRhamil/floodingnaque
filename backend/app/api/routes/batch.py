"""
Batch Prediction API.

Endpoint for processing multiple flood predictions in a single request.
"""

from flask import Blueprint, request, jsonify
from app.services.predict import predict_flood
from app.api.middleware.auth import require_api_key
from app.api.middleware.rate_limit import limiter
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

batch_bp = Blueprint('batch', __name__, url_prefix='/batch')


@batch_bp.route('/predict', methods=['POST'])
@require_api_key
@limiter.limit("10 per minute")
def batch_predict():
    """
    Process multiple flood predictions in a single request.
    
    Request Body:
    {
        "predictions": [
            {
                "temperature": 298.15,
                "humidity": 65,
                "precipitation": 5.0,
                "wind_speed": 10.5,
                "location": "Paranaque City"
            },
            {
                "temperature": 300.15,
                "humidity": 70,
                "precipitation": 10.0
            }
        ]
    }
    
    Returns:
        200: Batch predictions completed
        400: Invalid request data
        413: Too many predictions (max 100)
    """
    try:
        data = request.get_json()
        
        if not data or 'predictions' not in data:
            return jsonify({
                'error': 'Missing required field: predictions'
            }), 400
        
        predictions_input = data['predictions']
        
        # Validate input
        if not isinstance(predictions_input, list):
            return jsonify({
                'error': 'predictions must be a list'
            }), 400
        
        if len(predictions_input) == 0:
            return jsonify({
                'error': 'predictions list cannot be empty'
            }), 400
        
        # Limit batch size
        max_batch_size = 100
        if len(predictions_input) > max_batch_size:
            return jsonify({
                'error': f'Too many predictions. Maximum batch size is {max_batch_size}'
            }), 413
        
        # Process each prediction
        results = []
        errors = []
        
        for idx, pred_input in enumerate(predictions_input):
            try:
                # Validate required fields
                if 'temperature' not in pred_input:
                    errors.append({
                        'index': idx,
                        'error': 'Missing required field: temperature'
                    })
                    continue
                
                if 'humidity' not in pred_input:
                    errors.append({
                        'index': idx,
                        'error': 'Missing required field: humidity'
                    })
                    continue
                
                if 'precipitation' not in pred_input:
                    errors.append({
                        'index': idx,
                        'error': 'Missing required field: precipitation'
                    })
                    continue
                
                # Extract parameters
                temperature = float(pred_input['temperature'])
                humidity = float(pred_input['humidity'])
                precipitation = float(pred_input['precipitation'])
                wind_speed = float(pred_input.get('wind_speed', 0))
                location = pred_input.get('location', 'Paranaque City')
                
                # Make prediction
                input_dict = {
                    'temperature': temperature,
                    'humidity': humidity,
                    'precipitation': precipitation,
                    'wind_speed': wind_speed
                }
                prediction_result = predict_flood(input_data=input_dict)
                
                # Add to results
                results.append({
                    'index': idx,
                    'input': {
                        'temperature': temperature,
                        'humidity': humidity,
                        'precipitation': precipitation,
                        'wind_speed': wind_speed,
                        'location': location
                    },
                    'prediction': prediction_result['prediction'],
                    'risk_level': prediction_result['risk_level'],
                    'confidence': prediction_result.get('confidence'),
                    'model_version': prediction_result.get('model_version')
                })
                
            except ValueError as e:
                errors.append({
                    'index': idx,
                    'error': f'Invalid data type: {str(e)}'
                })
            except Exception as e:
                logger.error(f"Error processing prediction {idx}: {e}")
                errors.append({
                    'index': idx,
                    'error': 'Internal processing error'
                })
        
        # Prepare response
        response = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_requested': len(predictions_input),
            'successful': len(results),
            'failed': len(errors),
            'results': results
        }
        
        if errors:
            response['errors'] = errors
        
        logger.info(f"Batch prediction completed: {len(results)}/{len(predictions_input)} successful")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500
