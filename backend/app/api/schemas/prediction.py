"""
Prediction Schemas.

Provides schema definitions for flood prediction requests and responses.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class PredictRequestSchema:
    """Schema for predict endpoint request."""
    temperature: float
    humidity: float
    precipitation: float
    model_version: Optional[int] = None
    
    def validate(self) -> list:
        """
        Validate the request data.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not isinstance(self.temperature, (int, float)):
            errors.append("temperature must be a number")
        
        if not isinstance(self.humidity, (int, float)):
            errors.append("humidity must be a number")
        elif self.humidity < 0 or self.humidity > 100:
            errors.append("humidity must be between 0 and 100")
        
        if not isinstance(self.precipitation, (int, float)):
            errors.append("precipitation must be a number")
        elif self.precipitation < 0:
            errors.append("precipitation cannot be negative")
        
        return errors
    
    def to_prediction_dict(self) -> dict:
        """Convert to dictionary for prediction (without model_version)."""
        return {
            'temperature': self.temperature,
            'humidity': self.humidity,
            'precipitation': self.precipitation
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PredictRequestSchema':
        """Create instance from dictionary."""
        return cls(
            temperature=data.get('temperature', 0.0),
            humidity=data.get('humidity', 0.0),
            precipitation=data.get('precipitation', 0.0),
            model_version=data.get('model_version')
        )


@dataclass
class PredictResponseSchema:
    """Schema for predict endpoint response."""
    prediction: int
    flood_risk: str
    request_id: str
    model_version: Optional[str] = None
    probability: Optional[Dict[str, float]] = None
    risk_level: Optional[int] = None
    risk_label: Optional[str] = None
    risk_color: Optional[str] = None
    risk_description: Optional[str] = None
    confidence: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON response."""
        response = {
            'prediction': self.prediction,
            'flood_risk': self.flood_risk,
            'request_id': self.request_id
        }
        
        if self.model_version is not None:
            response['model_version'] = self.model_version
        
        if self.probability is not None:
            response['probability'] = self.probability
        
        if self.risk_label is not None:
            response['risk_level'] = self.risk_level
            response['risk_label'] = self.risk_label
            response['risk_color'] = self.risk_color
            response['risk_description'] = self.risk_description
            response['confidence'] = self.confidence
        
        return response
    
    @classmethod
    def from_prediction(cls, prediction: Any, request_id: str) -> 'PredictResponseSchema':
        """Create response schema from prediction result."""
        if isinstance(prediction, dict):
            return cls(
                prediction=prediction['prediction'],
                flood_risk='high' if prediction['prediction'] == 1 else 'low',
                request_id=request_id,
                model_version=prediction.get('model_version'),
                probability=prediction.get('probability'),
                risk_level=prediction.get('risk_level'),
                risk_label=prediction.get('risk_label'),
                risk_color=prediction.get('risk_color'),
                risk_description=prediction.get('risk_description'),
                confidence=prediction.get('confidence')
            )
        else:
            return cls(
                prediction=prediction,
                flood_risk='high' if prediction == 1 else 'low',
                request_id=request_id
            )


@dataclass
class RiskLevelSchema:
    """Schema for risk level classification."""
    level: int
    label: str
    color: str
    description: str
    threshold_min: float
    threshold_max: float
    
    @classmethod
    def safe(cls) -> 'RiskLevelSchema':
        """Create Safe risk level."""
        return cls(
            level=0,
            label='Safe',
            color='#28a745',
            description='No significant flood risk detected',
            threshold_min=0.0,
            threshold_max=0.35
        )
    
    @classmethod
    def alert(cls) -> 'RiskLevelSchema':
        """Create Alert risk level."""
        return cls(
            level=1,
            label='Alert',
            color='#ffc107',
            description='Moderate flood risk - stay alert',
            threshold_min=0.35,
            threshold_max=0.65
        )
    
    @classmethod
    def critical(cls) -> 'RiskLevelSchema':
        """Create Critical risk level."""
        return cls(
            level=2,
            label='Critical',
            color='#dc3545',
            description='High flood risk - take immediate precautions',
            threshold_min=0.65,
            threshold_max=1.0
        )
