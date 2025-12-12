import requests
import os
from datetime import datetime
from app.models.db import WeatherData, get_db_session
import logging

logger = logging.getLogger(__name__)

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
        # Fetch from OpenWeatherMap
        owm_url = f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={owm_api_key}'
        owm_response = requests.get(owm_url, timeout=10)
        owm_response.raise_for_status()
        owm_data = owm_response.json()
        
        if 'main' not in owm_data:
            raise ValueError("Invalid response from OpenWeatherMap API")
        
        data['temperature'] = owm_data['main'].get('temp', 0)
        data['humidity'] = owm_data['main'].get('humidity', 0)
        
        logger.info(f"Successfully fetched data from OpenWeatherMap for lat={lat}, lon={lon}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data from OpenWeatherMap: {str(e)}")
        raise
    except (KeyError, ValueError) as e:
        logger.error(f"Error parsing OpenWeatherMap response: {str(e)}")
        raise

    # Fetch precipitation data
    # Priority: Weatherstack API > OpenWeatherMap rain data
    precipitation = 0
    
    # Try Weatherstack API first if API key is provided
    if weatherstack_api_key:
        try:
            # Weatherstack API endpoint for current weather
            weatherstack_url = f'http://api.weatherstack.com/current?access_key={weatherstack_api_key}&query={lat},{lon}&units=m'
            weatherstack_response = requests.get(weatherstack_url, timeout=10)
            weatherstack_response.raise_for_status()
            weatherstack_data = weatherstack_response.json()
            
            # Check for errors in Weatherstack response
            if 'error' in weatherstack_data:
                logger.warning(f"Weatherstack API error: {weatherstack_data.get('error', {}).get('info', 'Unknown error')}")
            elif 'current' in weatherstack_data:
                # Weatherstack provides precipitation in 'precip' field (mm)
                precip_value = weatherstack_data['current'].get('precip', 0)
                if precip_value is not None:
                    precipitation = float(precip_value)
                    logger.info(f"Got precipitation from Weatherstack: {precipitation} mm")
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
