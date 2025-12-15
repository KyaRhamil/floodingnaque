"""
Weather Data Schemas.

Provides schema definitions and parsing utilities for weather data.
"""

import json
import codecs
from dataclasses import dataclass
from typing import Optional


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
    except Exception:
        # Logging will be handled by the calling function
        return None


@dataclass
class WeatherDataSchema:
    """Schema for weather data records."""
    id: Optional[int] = None
    temperature: float = 0.0
    humidity: float = 0.0
    precipitation: float = 0.0
    timestamp: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'temperature': self.temperature,
            'humidity': self.humidity,
            'precipitation': self.precipitation,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'WeatherDataSchema':
        """Create instance from dictionary."""
        return cls(
            id=data.get('id'),
            temperature=data.get('temperature', 0.0),
            humidity=data.get('humidity', 0.0),
            precipitation=data.get('precipitation', 0.0),
            timestamp=data.get('timestamp')
        )


@dataclass
class IngestRequestSchema:
    """Schema for ingest endpoint request."""
    lat: Optional[float] = None
    lon: Optional[float] = None
    
    def validate(self) -> list:
        """
        Validate the request data.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if self.lat is not None:
            if not isinstance(self.lat, (int, float)):
                errors.append("lat must be a number")
            elif self.lat < -90 or self.lat > 90:
                errors.append("lat must be between -90 and 90")
        
        if self.lon is not None:
            if not isinstance(self.lon, (int, float)):
                errors.append("lon must be a number")
            elif self.lon < -180 or self.lon > 180:
                errors.append("lon must be between -180 and 180")
        
        return errors
    
    @classmethod
    def from_dict(cls, data: dict) -> 'IngestRequestSchema':
        """Create instance from dictionary."""
        return cls(
            lat=data.get('lat'),
            lon=data.get('lon')
        )
