"""
Schemas package for Floodingnaque API.

Contains request/response schemas and validation utilities:
- weather: Weather data schemas and JSON parsing utilities
- prediction: Prediction request/response schemas
"""

from app.api.schemas.prediction import PredictRequestSchema, PredictResponseSchema
from app.api.schemas.weather import IngestRequestSchema, WeatherDataSchema, parse_json_safely

__all__ = [
    "parse_json_safely",
    "WeatherDataSchema",
    "IngestRequestSchema",
    "PredictRequestSchema",
    "PredictResponseSchema",
]
