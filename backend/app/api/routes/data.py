"""
Weather Data Routes.

Provides endpoints for retrieving historical weather data.
"""

from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, g
from sqlalchemy import desc
from app.models.weather import WeatherData
from app.models.predictions import FloodPrediction
from app.utils.database import get_db_session
from app.utils.api_responses import api_success, api_error
from app.utils.api_errors import AppException, ValidationError, NotFoundError
from app.utils.rate_limit import limiter, get_endpoint_limit
from app.utils.logging import get_logger
from app.services.meteostat import MeteostatService
from app.utils.config import get_config
from app.utils.cache import cached
from app.utils.api_constants import HTTP_OK, HTTP_BAD_REQUEST, HTTP_INTERNAL_ERROR, HTTP_SERVICE_UNAVAILABLE
import logging
import os

logger = logging.getLogger(__name__)

data_bp = Blueprint('data', __name__)


@data_bp.route('/data', methods=['GET'])
@limiter.limit(get_endpoint_limit('data'))
def get_weather_data():
    """Retrieve historical weather data."""
    request_id = getattr(g, 'request_id', 'unknown')
    
    try:
        # Get query parameters
        limit = request.args.get('limit', default=100, type=int)
        offset = request.args.get('offset', default=0, type=int)
        start_date = request.args.get('start_date', type=str)
        end_date = request.args.get('end_date', type=str)
        
        # Validate limit
        if limit < 1 or limit > 1000:
            return api_error('ValidationError', 'Limit must be between 1 and 1000', HTTP_BAD_REQUEST, request_id)
        
        with get_db_session() as session:
            query = session.query(WeatherData)
            
            # Filter by date range if provided
            if start_date:
                try:
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    query = query.filter(WeatherData.timestamp >= start_dt)
                except ValueError:
                    return api_error('ValidationError', 'Invalid start_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)', HTTP_BAD_REQUEST, request_id)
            
            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    query = query.filter(WeatherData.timestamp <= end_dt)
                except ValueError:
                    return api_error('ValidationError', 'Invalid end_date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)', HTTP_BAD_REQUEST, request_id)
            
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
        
        return jsonify({
            'data': data,
            'total': total,
            'limit': limit,
            'offset': offset,
            'count': len(data),
            'request_id': request_id
        }), HTTP_OK
    except Exception as e:
        logger.error(f"Error retrieving weather data [{request_id}]: {str(e)}")
        return api_error('DataRetrievalFailed', 'An error occurred while retrieving weather data', HTTP_INTERNAL_ERROR, request_id)


# ============================================================================
# Meteostat Historical Data Endpoints
# ============================================================================

def _get_meteostat_service():
    """Lazy load meteostat service."""
    try:
        from app.services.meteostat_service import get_meteostat_service
        return get_meteostat_service()
    except ImportError:
        logger.warning("Meteostat is not installed")
        return None


@data_bp.route('/meteostat/stations', methods=['GET'])
@limiter.limit(get_endpoint_limit('data'))
def get_nearby_stations():
    """
    Get nearby weather stations from Meteostat.
    
    Query Parameters:
        lat (float): Latitude (default: configured default)
        lon (float): Longitude (default: configured default)
        limit (int): Maximum number of stations (default: 5)
    
    Returns:
        List of nearby weather stations
    """
    request_id = getattr(g, 'request_id', 'unknown')
    
    try:
        service = _get_meteostat_service()
        if not service:
            return api_error('ServiceUnavailable', 'Meteostat service is not available', HTTP_SERVICE_UNAVAILABLE, request_id)
        
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        limit = request.args.get('limit', default=5, type=int)
        
        if limit < 1 or limit > 20:
            return api_error('ValidationError', 'Limit must be between 1 and 20', HTTP_BAD_REQUEST, request_id)
        
        stations = service.find_nearby_stations(lat, lon, limit=limit)
        
        return jsonify({
            'stations': stations,
            'count': len(stations),
            'request_id': request_id
        }), HTTP_OK
        
    except Exception as e:
        logger.error(f"Error fetching stations [{request_id}]: {str(e)}")
        return api_error('StationFetchFailed', 'Failed to fetch nearby stations', HTTP_INTERNAL_ERROR, request_id)


@data_bp.route('/weather/hourly', methods=['GET'])
@limiter.limit(get_endpoint_limit('data'))
@cached('weather_hourly', ttl=300)  # Cache for 5 minutes
def get_hourly_weather():
    """
    Get hourly weather observations from Meteostat.
    
    Query Parameters:
        lat (float): Latitude (default: configured default)
        lon (float): Longitude (default: configured default)
        start_date (str): Start date in ISO format (default: 7 days ago)
        end_date (str): End date in ISO format (default: now)
    
    Returns:
        Hourly weather observations
    """
    request_id = getattr(g, 'request_id', 'unknown')
    
    try:
        service = _get_meteostat_service()
        if not service:
            return api_error('ServiceUnavailable', 'Meteostat service is not available', HTTP_SERVICE_UNAVAILABLE, request_id)
        
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        start_date = request.args.get('start_date', type=str)
        end_date = request.args.get('end_date', type=str)
        
        # Parse dates
        end = datetime.now()
        start = end - timedelta(days=7)
        
        if start_date:
            try:
                start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                return api_error('ValidationError', 'Invalid start_date format. Use ISO format', HTTP_BAD_REQUEST, request_id)
        
        if end_date:
            try:
                end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                return api_error('ValidationError', 'Invalid end_date format. Use ISO format', HTTP_BAD_REQUEST, request_id)
        
        # Limit date range to 30 days for performance
        if (end - start).days > 30:
            return api_error('ValidationError', 'Date range cannot exceed 30 days for hourly data', HTTP_BAD_REQUEST, request_id)
        
        observations = service.get_hourly_data(lat, lon, start, end)
        
        # Convert to JSON-serializable format
        data = []
        for obs in observations:
            data.append({
                'timestamp': obs.timestamp.isoformat() if obs.timestamp else None,
                'temperature': obs.temperature,
                'humidity': obs.humidity,
                'precipitation': obs.precipitation,
                'wind_speed': obs.wind_speed,
                'pressure': obs.pressure,
                'source': obs.source
            })
        
        return jsonify({
            'data': data,
            'count': len(data),
            'start_date': start.isoformat(),
            'end_date': end.isoformat(),
            'request_id': request_id
        }), HTTP_OK
        
    except Exception as e:
        logger.error(f"Error fetching hourly data [{request_id}]: {str(e)}")
        return api_error('HourlyDataFetchFailed', 'Failed to fetch hourly weather data', HTTP_INTERNAL_ERROR, request_id)


@data_bp.route('/meteostat/daily', methods=['GET'])
@limiter.limit(get_endpoint_limit('data'))
def get_meteostat_daily():
    """
    Get daily weather data from Meteostat.
    
    Query Parameters:
        lat (float): Latitude (default: configured default)
        lon (float): Longitude (default: configured default)
        start_date (str): Start date in ISO format (default: 30 days ago)
        end_date (str): End date in ISO format (default: now)
    
    Returns:
        Daily weather data including min/max temperatures, precipitation
    """
    request_id = getattr(g, 'request_id', 'unknown')
    
    try:
        service = _get_meteostat_service()
        if not service:
            return api_error('ServiceUnavailable', 'Meteostat service is not available', HTTP_SERVICE_UNAVAILABLE, request_id)
        
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        start_date = request.args.get('start_date', type=str)
        end_date = request.args.get('end_date', type=str)
        
        # Parse dates
        end = datetime.now()
        start = end - timedelta(days=30)
        
        if start_date:
            try:
                start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                return api_error('ValidationError', 'Invalid start_date format. Use ISO format', HTTP_BAD_REQUEST, request_id)
        
        if end_date:
            try:
                end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                return api_error('ValidationError', 'Invalid end_date format. Use ISO format', HTTP_BAD_REQUEST, request_id)
        
        # Limit date range to 365 days for daily data
        if (end - start).days > 365:
            return api_error('ValidationError', 'Date range cannot exceed 365 days for daily data', HTTP_BAD_REQUEST, request_id)
        
        data = service.get_daily_data(lat, lon, start, end)
        
        return jsonify({
            'data': data,
            'count': len(data),
            'start_date': start.isoformat(),
            'end_date': end.isoformat(),
            'request_id': request_id
        }), HTTP_OK
        
    except Exception as e:
        logger.error(f"Error fetching daily data [{request_id}]: {str(e)}")
        return api_error('DailyDataFetchFailed', 'Failed to fetch daily weather data', HTTP_INTERNAL_ERROR, request_id)


@data_bp.route('/meteostat/current', methods=['GET'])
@limiter.limit(get_endpoint_limit('data'))
def get_meteostat_current():
    """
    Get the latest weather observation from Meteostat.
    
    Note: Meteostat data may be delayed by a few hours compared to real-time APIs.
    This is useful as a fallback when real-time APIs are unavailable.
    
    Query Parameters:
        lat (float): Latitude (default: configured default)
        lon (float): Longitude (default: configured default)
    
    Returns:
        Latest weather observation from the nearest station
    """
    request_id = getattr(g, 'request_id', 'unknown')
    
    try:
        service = _get_meteostat_service()
        if not service:
            return api_error('ServiceUnavailable', 'Meteostat service is not available', HTTP_SERVICE_UNAVAILABLE, request_id)
        
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        
        data = service.get_weather_for_prediction(lat, lon)
        
        if not data:
            return api_error('NoDataAvailable', 'No recent weather data available from Meteostat', HTTP_BAD_REQUEST, request_id)
        
        return jsonify({
            'data': data,
            'note': 'Meteostat data may be delayed by a few hours',
            'request_id': request_id
        }), HTTP_OK
        
    except Exception as e:
        logger.error(f"Error fetching current data [{request_id}]: {str(e)}")
        return api_error('CurrentDataFetchFailed', 'Failed to fetch current weather data', HTTP_INTERNAL_ERROR, request_id)


@data_bp.route('/meteostat/status', methods=['GET'])
@limiter.limit(get_endpoint_limit('data'))
def get_meteostat_status():
    """
    Get Meteostat service status and configuration.
    
    Returns:
        Service status, enabled state, and configuration
    """
    request_id = getattr(g, 'request_id', 'unknown')
    
    try:
        service = _get_meteostat_service()
        
        status = {
            'available': service is not None,
            'enabled': os.getenv('METEOSTAT_ENABLED', 'True').lower() == 'true',
            'as_fallback': os.getenv('METEOSTAT_AS_FALLBACK', 'True').lower() == 'true',
            'default_location': {
                'latitude': float(os.getenv('DEFAULT_LATITUDE', '14.4793')),
                'longitude': float(os.getenv('DEFAULT_LONGITUDE', '121.0198'))
            },
            'request_id': request_id
        }
        
        if service:
            # Check if we can connect by fetching a station
            try:
                stations = service.find_nearby_stations(limit=1)
                status['connection_status'] = 'connected' if stations else 'no_stations_found'
            except Exception:
                status['connection_status'] = 'error'
        else:
            status['connection_status'] = 'service_unavailable'
        
        return jsonify(status), HTTP_OK
        
    except Exception as e:
        logger.error(f"Error fetching meteostat status [{request_id}]: {str(e)}")
        return api_error('StatusCheckFailed', 'Failed to check Meteostat status', HTTP_INTERNAL_ERROR, request_id)
