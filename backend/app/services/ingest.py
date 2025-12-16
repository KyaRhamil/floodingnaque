import requests
import os
import re
from datetime import datetime
from app.models.db import WeatherData, get_db_session
from app.utils.circuit_breaker import (
    openweathermap_breaker,
    weatherstack_breaker,
    retry_with_backoff,
    CircuitOpenError
)
import logging

logger = logging.getLogger(__name__)

# Regex pattern for redacting API keys in URLs/logs
_API_KEY_PATTERNS = [
    (re.compile(r'(appid=)[^&]+'), r'\1[REDACTED]'),
    (re.compile(r'(access_key=)[^&]+'), r'\1[REDACTED]'),
    (re.compile(r'(api_key=)[^&]+'), r'\1[REDACTED]'),
    (re.compile(r'(key=)[^&]+'), r'\1[REDACTED]'),
]


def _redact_api_keys(text: str) -> str:
    """Redact API keys from URLs and log messages."""
    result = text
    for pattern, replacement in _API_KEY_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def _safe_log_url(url: str) -> str:
    """Return a URL safe for logging (with API keys redacted)."""
    return _redact_api_keys(url)

def ingest_data(lat=None, lon=None):
    """
    Ingest weather data from external APIs.
    
    Args:
        lat: Latitude (default: 40.7128 - New York City)
        lon: Longitude (default: -74.0060 - New York City)
    
    Returns:
        dict: Weather data dictionary
    """
    # API keys from environment variables
    owm_api_key = os.getenv('OWM_API_KEY')
    # Note: METEOSTAT_API_KEY can also be used for Weatherstack API
    weatherstack_api_key = os.getenv('METEOSTAT_API_KEY') or os.getenv('WEATHERSTACK_API_KEY')

    # Default location (New York City)
    if lat is None:
        lat = 40.7128
    if lon is None:
        lon = -74.0060

    # Validate API keys
    if not owm_api_key:
        raise ValueError("OWM_API_KEY environment variable is not set")
    
    data = {}
    
    try:
        # Fetch from OpenWeatherMap with circuit breaker protection
        # Note: OWM requires appid in query string, but we redact in logs
        owm_url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={owm_api_key}'
        logger.debug(f"Fetching weather data from: {_safe_log_url(owm_url)}")
        
        @retry_with_backoff(max_retries=2, base_delay=1.0, exceptions=(requests.exceptions.RequestException,))
        def fetch_owm():
            response = requests.get(owm_url, timeout=10)
            response.raise_for_status()
            return response.json()
        
        owm_data = openweathermap_breaker.call(fetch_owm)
        
        if 'main' not in owm_data:
            raise ValueError("Invalid response from OpenWeatherMap API")
        
        data['temperature'] = owm_data['main'].get('temp', 0)
        data['humidity'] = owm_data['main'].get('humidity', 0)
        
        logger.info(f"Successfully fetched data from OpenWeatherMap for lat={lat}, lon={lon}")
    except CircuitOpenError as e:
        logger.error(f"OpenWeatherMap circuit breaker is open: {str(e)}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from OpenWeatherMap: {str(e)}")
        raise
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing OpenWeatherMap response: {str(e)}")
        raise

    # Fetch precipitation data
    # Priority: Weatherstack API > OpenWeatherMap rain data
    precipitation = 0
    
    # Try Weatherstack API first if API key is provided (with circuit breaker)
    if weatherstack_api_key:
        try:
            # Check if circuit breaker is open before attempting
            if weatherstack_breaker.is_open:
                logger.warning("Weatherstack circuit breaker is open, skipping to fallback")
            else:
                # Weatherstack API endpoint for current weather
                # Note: HTTPS requires paid plan, but we use it for security
                # Free tier users should upgrade or the request will fail gracefully
                weatherstack_url = f'https://api.weatherstack.com/current?access_key={weatherstack_api_key}&query={lat},{lon}&units=m'
                logger.debug(f"Fetching precipitation from: {_safe_log_url(weatherstack_url)}")
                
                @retry_with_backoff(max_retries=2, base_delay=1.0, exceptions=(requests.exceptions.RequestException,))
                def fetch_weatherstack():
                    response = requests.get(weatherstack_url, timeout=10)
                    response.raise_for_status()
                    return response.json()
                
                weatherstack_data = weatherstack_breaker.call(fetch_weatherstack)
                
                # Check for errors in Weatherstack response
                if 'error' in weatherstack_data:
                    logger.warning(f"Weatherstack API error: {weatherstack_data.get('error', {}).get('info', 'Unknown error')}")
                elif 'current' in weatherstack_data:
                    # Weatherstack provides precipitation in 'precip' field (mm)
                    precip_value = weatherstack_data['current'].get('precip', 0)
                    if precip_value is not None:
                        precipitation = float(precip_value)
                        logger.info(f"Got precipitation from Weatherstack: {precipitation} mm")
        except CircuitOpenError as e:
            logger.warning(f"Weatherstack circuit breaker open (using fallback): {str(e)}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching data from Weatherstack API (continuing with OpenWeatherMap): {str(e)}")
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error parsing Weatherstack response (continuing with OpenWeatherMap): {str(e)}")
    
    # Fallback to OpenWeatherMap rain data if Weatherstack didn't provide precipitation
    if precipitation == 0:
        try:
            if 'rain' in owm_data and '3h' in owm_data['rain']:
                precipitation = owm_data['rain']['3h'] / 3.0  # Convert 3h to hourly rate
                logger.info(f"Got precipitation from OpenWeatherMap: {precipitation} mm/h")
            elif 'rain' in owm_data and '1h' in owm_data['rain']:
                precipitation = owm_data['rain']['1h']
                logger.info(f"Got precipitation from OpenWeatherMap: {precipitation} mm/h")
        except (KeyError, TypeError) as e:
            logger.debug(f"No rain data in OpenWeatherMap response: {str(e)}")
    
    data['precipitation'] = precipitation
    data['timestamp'] = datetime.now()

    # Save to DB
    try:
        with get_db_session() as session:
            weather_data = WeatherData(**data)
            session.add(weather_data)
        logger.info("Successfully saved weather data to database")
    except Exception as e:
        logger.error(f"Error saving data to database: {str(e)}")
        raise

    return data
