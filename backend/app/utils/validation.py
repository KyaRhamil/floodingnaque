"""
Enhanced Input Validation and Sanitization Module
Provides comprehensive validation for all API inputs.
"""

import re
import bleach
import validators
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom validation error."""
    pass


class InputValidator:
    """Comprehensive input validation for flood prediction system."""
    
    # Weather data constraints
    TEMPERATURE_MIN_KELVIN = 173.15  # -100°C
    TEMPERATURE_MAX_KELVIN = 333.15  # +60°C
    HUMIDITY_MIN = 0.0
    HUMIDITY_MAX = 100.0
    PRECIPITATION_MIN = 0.0
    PRECIPITATION_MAX = 500.0  # mm (extreme but possible)
    WIND_SPEED_MIN = 0.0
    WIND_SPEED_MAX = 150.0  # m/s (extreme)
    PRESSURE_MIN = 870.0  # hPa (record low)
    PRESSURE_MAX = 1085.0  # hPa (record high)
    
    # Location constraints
    LATITUDE_MIN = -90.0
    LATITUDE_MAX = 90.0
    LONGITUDE_MIN = -180.0
    LONGITUDE_MAX = 180.0
    
    # String length limits
    MAX_STRING_LENGTH = 500
    MAX_TEXT_LENGTH = 5000
    
    @staticmethod
    def validate_float(value: Any, field_name: str, min_val: Optional[float] = None, 
                      max_val: Optional[float] = None, required: bool = True) -> Optional[float]:
        """
        Validate a float value with range checking.
        
        Args:
            value: Value to validate
            field_name: Name of the field (for error messages)
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            required: Whether the field is required
            
        Returns:
            Validated float or None if not required and not provided
            
        Raises:
            ValidationError: If validation fails
        """
        if value is None:
            if required:
                raise ValidationError(f"{field_name} is required")
            return None
        
        try:
            float_val = float(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be a valid number")
        
        if min_val is not None and float_val < min_val:
            raise ValidationError(
                f"{field_name} must be >= {min_val}, got {float_val}"
            )
        
        if max_val is not None and float_val > max_val:
            raise ValidationError(
                f"{field_name} must be <= {max_val}, got {float_val}"
            )
        
        return float_val
    
    @staticmethod
    def validate_integer(value: Any, field_name: str, min_val: Optional[int] = None,
                        max_val: Optional[int] = None, required: bool = True) -> Optional[int]:
        """Validate an integer value with range checking."""
        if value is None:
            if required:
                raise ValidationError(f"{field_name} is required")
            return None
        
        try:
            int_val = int(value)
        except (ValueError, TypeError):
            raise ValidationError(f"{field_name} must be a valid integer")
        
        if min_val is not None and int_val < min_val:
            raise ValidationError(f"{field_name} must be >= {min_val}")
        
        if max_val is not None and int_val > max_val:
            raise ValidationError(f"{field_name} must be <= {max_val}")
        
        return int_val
    
    @staticmethod
    def validate_string(value: Any, field_name: str, max_length: Optional[int] = None,
                       pattern: Optional[str] = None, required: bool = True,
                       sanitize: bool = True) -> Optional[str]:
        """Validate and sanitize string input."""
        if value is None:
            if required:
                raise ValidationError(f"{field_name} is required")
            return None
        
        if not isinstance(value, str):
            value = str(value)
        
        # Sanitize HTML/scripts if requested
        if sanitize:
            value = bleach.clean(value, tags=[], strip=True)
        
        # Check length
        if max_length and len(value) > max_length:
            raise ValidationError(
                f"{field_name} exceeds maximum length of {max_length} characters"
            )
        
        # Check pattern
        if pattern and not re.match(pattern, value):
            raise ValidationError(f"{field_name} format is invalid")
        
        return value.strip()
    
    @classmethod
    def validate_temperature(cls, temp: Any, required: bool = True) -> Optional[float]:
        """Validate temperature in Kelvin."""
        return cls.validate_float(
            temp, "temperature", 
            cls.TEMPERATURE_MIN_KELVIN, 
            cls.TEMPERATURE_MAX_KELVIN,
            required
        )
    
    @classmethod
    def validate_humidity(cls, humidity: Any, required: bool = True) -> Optional[float]:
        """Validate humidity percentage."""
        return cls.validate_float(
            humidity, "humidity",
            cls.HUMIDITY_MIN,
            cls.HUMIDITY_MAX,
            required
        )
    
    @classmethod
    def validate_precipitation(cls, precip: Any, required: bool = True) -> Optional[float]:
        """Validate precipitation in mm."""
        return cls.validate_float(
            precip, "precipitation",
            cls.PRECIPITATION_MIN,
            cls.PRECIPITATION_MAX,
            required
        )
    
    @classmethod
    def validate_wind_speed(cls, wind: Any, required: bool = False) -> Optional[float]:
        """Validate wind speed in m/s."""
        return cls.validate_float(
            wind, "wind_speed",
            cls.WIND_SPEED_MIN,
            cls.WIND_SPEED_MAX,
            required
        )
    
    @classmethod
    def validate_pressure(cls, pressure: Any, required: bool = False) -> Optional[float]:
        """Validate atmospheric pressure in hPa."""
        return cls.validate_float(
            pressure, "pressure",
            cls.PRESSURE_MIN,
            cls.PRESSURE_MAX,
            required
        )
    
    @classmethod
    def validate_latitude(cls, lat: Any, required: bool = True) -> Optional[float]:
        """Validate latitude."""
        return cls.validate_float(
            lat, "latitude",
            cls.LATITUDE_MIN,
            cls.LATITUDE_MAX,
            required
        )
    
    @classmethod
    def validate_longitude(cls, lon: Any, required: bool = True) -> Optional[float]:
        """Validate longitude."""
        return cls.validate_float(
            lon, "longitude",
            cls.LONGITUDE_MIN,
            cls.LONGITUDE_MAX,
            required
        )
    
    @classmethod
    def validate_coordinates(cls, lat: Any, lon: Any) -> tuple:
        """Validate geographic coordinates."""
        validated_lat = cls.validate_latitude(lat)
        validated_lon = cls.validate_longitude(lon)
        return (validated_lat, validated_lon)
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address format."""
        return validators.email(email) is True
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        return validators.url(url) is True
    
    @staticmethod
    def validate_datetime(dt_str: str, fmt: str = '%Y-%m-%dT%H:%M:%S') -> datetime:
        """Validate and parse datetime string."""
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            # Try ISO format
            try:
                return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            except ValueError:
                raise ValidationError(f"Invalid datetime format: {dt_str}")
    
    @classmethod
    def validate_weather_data(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete weather data input.
        
        Args:
            data: Dictionary with weather parameters
            
        Returns:
            Validated and sanitized dictionary
            
        Raises:
            ValidationError: If validation fails
        """
        validated = {}
        
        # Required fields
        validated['temperature'] = cls.validate_temperature(data.get('temperature'))
        validated['humidity'] = cls.validate_humidity(data.get('humidity'))
        validated['precipitation'] = cls.validate_precipitation(data.get('precipitation'))
        
        # Optional fields
        if 'wind_speed' in data:
            validated['wind_speed'] = cls.validate_wind_speed(data.get('wind_speed'), required=False)
        
        if 'pressure' in data:
            validated['pressure'] = cls.validate_pressure(data.get('pressure'), required=False)
        
        if 'location_lat' in data:
            validated['location_lat'] = cls.validate_latitude(data.get('location_lat'), required=False)
        
        if 'location_lon' in data:
            validated['location_lon'] = cls.validate_longitude(data.get('location_lon'), required=False)
        
        if 'source' in data:
            validated['source'] = cls.validate_string(
                data.get('source'),
                'source',
                max_length=50,
                required=False
            )
        
        return validated
    
    @classmethod
    def validate_prediction_input(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate prediction request input.
        
        Args:
            data: Dictionary with prediction parameters
            
        Returns:
            Validated dictionary
            
        Raises:
            ValidationError: If validation fails
        """
        validated = cls.validate_weather_data(data)
        
        # Optional model version
        if 'model_version' in data:
            validated['model_version'] = cls.validate_integer(
                data.get('model_version'),
                'model_version',
                min_val=1,
                required=False
            )
        
        return validated
    
    @staticmethod
    def sanitize_sql_input(value: str) -> str:
        """Sanitize input to prevent SQL injection (additional layer)."""
        # Remove potentially dangerous characters
        dangerous_chars = ['--', ';', '/*', '*/', 'xp_', 'sp_', 'DROP', 'DELETE', 'INSERT', 'UPDATE']
        cleaned = value
        for char in dangerous_chars:
            cleaned = cleaned.replace(char, '')
        return cleaned
    
    @staticmethod
    def validate_pagination(limit: Any, offset: Any, max_limit: int = 1000) -> tuple:
        """Validate pagination parameters."""
        try:
            limit = int(limit) if limit is not None else 100
            offset = int(offset) if offset is not None else 0
        except (ValueError, TypeError):
            raise ValidationError("Invalid pagination parameters")
        
        if limit < 1 or limit > max_limit:
            raise ValidationError(f"Limit must be between 1 and {max_limit}")
        
        if offset < 0:
            raise ValidationError("Offset must be >= 0")
        
        return (limit, offset)


# Convenience functions
def validate_weather_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate weather data - convenience wrapper."""
    return InputValidator.validate_weather_data(data)


def validate_prediction_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate prediction input - convenience wrapper."""
    return InputValidator.validate_prediction_input(data)


def validate_coordinates(lat: Any, lon: Any) -> tuple:
    """Validate coordinates - convenience wrapper."""
    return InputValidator.validate_coordinates(lat, lon)
