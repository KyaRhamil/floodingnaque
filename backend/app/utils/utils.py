import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging():
    """Setup logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Ensure logs directory exists
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    handler = RotatingFileHandler(
        os.path.join(logs_dir, 'app.log'), 
        maxBytes=10000000,  # 10MB
        backupCount=5
    )
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def error_handler(func):
    """Decorator for error handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def validate_coordinates(lat, lon):
    """Validate latitude and longitude coordinates."""
    if lat is not None:
        if not isinstance(lat, (int, float)) or lat < -90 or lat > 90:
            raise ValueError("Latitude must be between -90 and 90")
    if lon is not None:
        if not isinstance(lon, (int, float)) or lon < -180 or lon > 180:
            raise ValueError("Longitude must be between -180 and 180")
    return True
