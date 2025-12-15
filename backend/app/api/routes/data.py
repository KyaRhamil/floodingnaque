"""
Weather Data Routes.

Provides endpoints for retrieving historical weather data.
"""

from flask import Blueprint, jsonify, request
from datetime import datetime
from app.models.db import WeatherData, get_db_session
from app.api.middleware.rate_limit import limiter, get_endpoint_limit
import logging

logger = logging.getLogger(__name__)

data_bp = Blueprint('data', __name__)


@data_bp.route('/data', methods=['GET'])
@limiter.limit(get_endpoint_limit('data'))
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
