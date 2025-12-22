"""
GraphQL Schema Definition.

Defines GraphQL types, queries, and mutations for the Floodingnaque API.
"""

import os
from datetime import datetime
from pathlib import Path
import graphene
from graphene import ObjectType, Schema, String, Int, Float, Boolean, List, Field, DateTime
from graphene.types.scalars import Scalar
from graphene.types.json import JSONString as JSON
from dotenv import load_dotenv
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Load environment variables before checking GRAPHQL_ENABLED
_backend_dir = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(_backend_dir / '.env')
load_dotenv(_backend_dir / '.env.production', override=True)

# Check if GraphQL is enabled
GRAPHQL_ENABLED = os.getenv('GRAPHQL_ENABLED', 'false').lower() == 'true'

# Custom scalar for handling various data types
class GenericScalar(Scalar):
    """Custom scalar for handling generic JSON data."""
    
    @staticmethod
    def serialize(value):
        return value
    
    @staticmethod
    def parse_literal(node):
        return node.value
    
    @staticmethod
    def parse_value(value):
        return value


# GraphQL Types
class WeatherDataType(ObjectType):
    """Weather data type for GraphQL."""
    id = String()
    timestamp = DateTime()
    latitude = Float()
    longitude = Float()
    temperature = Float()
    humidity = Float()
    precipitation = Float()
    wind_speed = Float()
    pressure = Float()
    source = String()


class FloodPredictionType(ObjectType):
    """Flood prediction type for GraphQL."""
    id = String()
    timestamp = DateTime()
    latitude = Float()
    longitude = Float()
    flood_risk = Float()
    confidence = Float()
    model_version = String()
    features = GenericScalar()


class HealthStatusType(ObjectType):
    """Health status type for GraphQL."""
    status = String()
    timestamp = DateTime()
    checks = GenericScalar()
    model_available = Boolean()
    database_connected = Boolean()


class RateLimitInfoType(ObjectType):
    """Rate limit information type for GraphQL."""
    authenticated = Boolean()
    tier = String()
    limits = GenericScalar()
    requests_remaining = Int()


class TaskStatusType(ObjectType):
    """Background task status type for GraphQL."""
    task_id = String()
    status = String()
    progress = Int()
    result = GenericScalar()
    error = String()
    created_at = DateTime()


# Query Resolvers
class Query(ObjectType):
    """Root query type for GraphQL."""
    
    # Health queries
    health = Field(HealthStatusType)
    health_status = Field(String)
    
    # Weather data queries
    weather_data = List(
        WeatherDataType,
        latitude=Float(required=True),
        longitude=Float(required=True),
        start_date=DateTime(),
        end_date=DateTime(),
        limit=Int(default_value=100)
    )
    
    # Prediction queries
    flood_prediction = Field(
        FloodPredictionType,
        latitude=Float(required=True),
        longitude=Float(required=True),
        features=GenericScalar()
    )
    
    predictions = List(
        FloodPredictionType,
        latitude=Float(),
        longitude=Float(),
        start_date=DateTime(),
        end_date=DateTime(),
        limit=Int(default_value=100)
    )
    
    # Rate limit queries
    rate_limit_status = Field(RateLimitInfoType)
    
    # Task queries
    task_status = Field(
        TaskStatusType,
        task_id=String(required=True)
    )
    
    # System info queries
    system_info = GenericScalar()
    
    def resolve_health(self, info):
        """Get system health status."""
        try:
            from app.api.routes.health import check_database_health, check_model_health
            from app.utils.cache import get_cache_stats
            
            db_status = check_database_health()
            model_status = check_model_health()
            cache_stats = get_cache_stats()
            
            return HealthStatusType(
                status='healthy' if db_status.get('connected') and model_status == 'healthy' else 'unhealthy',
                timestamp=datetime.utcnow(),
                checks={
                    'database': db_status,
                    'model': model_status,
                    'cache': cache_stats
                },
                model_available=model_status == 'healthy',
                database_connected=db_status.get('connected', False)
            )
        except Exception as e:
            logger.error(f"GraphQL health query failed: {str(e)}")
            return HealthStatusType(
                status='unhealthy',
                timestamp=datetime.utcnow(),
                checks={'error': str(e)},
                model_available=False,
                database_connected=False
            )
    
    def resolve_health_status(self, info):
        """Simple health status string."""
        try:
            from app.api.routes.health import check_database_health, check_model_health
            
            db_status = check_database_health()
            model_status = check_model_health()
            
            return 'healthy' if db_status.get('connected') and model_status == 'healthy' else 'unhealthy'
        except Exception:
            return 'unhealthy'
    
    def resolve_weather_data(self, info, latitude, longitude, start_date=None, end_date=None, limit=100):
        """Get weather data for a location."""
        try:
            from app.models.db import get_db_session, WeatherData
            from sqlalchemy import and_, desc
            
            with get_db_session() as session:
                query = session.query(WeatherData).filter(
                    WeatherData.latitude == latitude,
                    WeatherData.longitude == longitude
                )
                
                if start_date:
                    query = query.filter(WeatherData.timestamp >= start_date)
                if end_date:
                    query = query.filter(WeatherData.timestamp <= end_date)
                
                results = query.order_by(desc(WeatherData.timestamp)).limit(limit).all()
                
                return [
                    WeatherDataType(
                        id=str(record.id),
                        timestamp=record.timestamp,
                        latitude=record.latitude,
                        longitude=record.longitude,
                        temperature=record.temperature,
                        humidity=record.humidity,
                        precipitation=record.precipitation,
                        wind_speed=record.wind_speed,
                        pressure=record.pressure,
                        source=record.source
                    )
                    for record in results
                ]
        except Exception as e:
            logger.error(f"GraphQL weather data query failed: {str(e)}")
            return []
    
    def resolve_flood_prediction(self, info, latitude, longitude, features=None):
        """Get flood prediction for a location."""
        try:
            from app.services.predict import predict_flood
            
            # Use provided features or generate defaults
            if features is None:
                features = {
                    'temperature': 20.0,
                    'humidity': 70.0,
                    'precipitation': 5.0,
                    'wind_speed': 10.0,
                    'pressure': 1013.25
                }
            
            prediction = predict_flood(latitude, longitude, features)
            
            return FloodPredictionType(
                id=str(prediction.get('id', 'unknown')),
                timestamp=datetime.utcnow(),
                latitude=latitude,
                longitude=longitude,
                flood_risk=prediction.get('flood_risk', 0.0),
                confidence=prediction.get('confidence', 0.0),
                model_version=prediction.get('model_version', 'unknown'),
                features=features
            )
        except Exception as e:
            logger.error(f"GraphQL flood prediction query failed: {str(e)}")
            return FloodPredictionType(
                id='error',
                timestamp=datetime.utcnow(),
                latitude=latitude,
                longitude=longitude,
                flood_risk=0.0,
                confidence=0.0,
                model_version='error',
                features={'error': str(e)}
            )
    
    def resolve_predictions(self, info, latitude=None, longitude=None, start_date=None, end_date=None, limit=100):
        """Get historical flood predictions."""
        try:
            from app.models.db import get_db_session, Prediction as FloodPrediction
            from sqlalchemy import and_, desc
            
            with get_db_session() as session:
                query = session.query(FloodPrediction)
                
                if latitude is not None:
                    query = query.filter(FloodPrediction.latitude == latitude)
                if longitude is not None:
                    query = query.filter(FloodPrediction.longitude == longitude)
                if start_date:
                    query = query.filter(FloodPrediction.timestamp >= start_date)
                if end_date:
                    query = query.filter(FloodPrediction.timestamp <= end_date)
                
                results = query.order_by(desc(FloodPrediction.timestamp)).limit(limit).all()
                
                return [
                    FloodPredictionType(
                        id=str(record.id),
                        timestamp=record.timestamp,
                        latitude=record.latitude,
                        longitude=record.longitude,
                        flood_risk=record.flood_risk,
                        confidence=record.confidence,
                        model_version=record.model_version,
                        features=record.features or {}
                    )
                    for record in results
                ]
        except Exception as e:
            logger.error(f"GraphQL predictions query failed: {str(e)}")
            return []
    
    def resolve_rate_limit_status(self, info):
        """Get current rate limiting status."""
        try:
            from flask import g
            from app.utils.rate_limit_tiers import check_rate_limit_status
            
            api_key_hash = getattr(g, 'api_key_hash', None)
            status = check_rate_limit_status(api_key_hash)
            
            return RateLimitInfoType(
                authenticated=status['authenticated'],
                tier=status['tier'],
                limits=status['limits'],
                requests_remaining=None  # Would need to get from Flask-Limiter
            )
        except Exception as e:
            logger.error(f"GraphQL rate limit status query failed: {str(e)}")
            return RateLimitInfoType(
                authenticated=False,
                tier='error',
                limits={'error': str(e)},
                requests_remaining=0
            )
    
    def resolve_task_status(self, info, task_id):
        """Get status of a background task."""
        try:
            from app.services.tasks import get_task_status
            
            status = get_task_status(task_id)
            
            return TaskStatusType(
                task_id=status['task_id'],
                status=status['status'],
                progress=status.get('progress', 0),
                result=status.get('result'),
                error=None,
                created_at=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"GraphQL task status query failed: {str(e)}")
            return TaskStatusType(
                task_id=task_id,
                status='error',
                progress=0,
                result=None,
                error=str(e),
                created_at=datetime.utcnow()
            )
    
    def resolve_system_info(self, info):
        """Get system information."""
        try:
            import sys
            import platform
            
            return {
                'python_version': sys.version.split()[0],
                'platform': platform.system(),
                'platform_version': platform.version()[:50] if platform.version() else 'unknown',
                'graphql_enabled': GRAPHQL_ENABLED,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"GraphQL system info query failed: {str(e)}")
            return {'error': str(e)}


# Mutation Resolvers
class Mutation(ObjectType):
    """Root mutation type for GraphQL."""
    
    # Task mutations
    trigger_model_retraining = Field(
        TaskStatusType,
        model_id=String()
    )
    
    trigger_data_processing = Field(
        TaskStatusType,
        data_batch=List(GenericScalar)
    )
    
    # Data mutations (placeholder for future data management)
    create_webhook = Field(
        GenericScalar,
        url=String(required=True),
        events=List(String)
    )
    
    def resolve_trigger_model_retraining(self, info, model_id=None):
        """Trigger model retraining task."""
        try:
            if not GRAPHQL_ENABLED:
                raise Exception("GraphQL is disabled")
            
            from app.services.tasks import trigger_model_retraining
            
            result = trigger_model_retraining(model_id)
            
            return TaskStatusType(
                task_id=result['task_id'],
                status='queued',
                progress=0,
                result=result,
                error=None,
                created_at=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"GraphQL model retraining mutation failed: {str(e)}")
            return TaskStatusType(
                task_id='error',
                status='error',
                progress=0,
                result=None,
                error=str(e),
                created_at=datetime.utcnow()
            )
    
    def resolve_trigger_data_processing(self, info, data_batch):
        """Trigger data processing task."""
        try:
            if not GRAPHQL_ENABLED:
                raise Exception("GraphQL is disabled")
            
            from app.services.tasks import trigger_data_processing
            
            result = trigger_data_processing(data_batch)
            
            return TaskStatusType(
                task_id=result['task_id'],
                status='queued',
                progress=0,
                result=result,
                error=None,
                created_at=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"GraphQL data processing mutation failed: {str(e)}")
            return TaskStatusType(
                task_id='error',
                status='error',
                progress=0,
                result=None,
                error=str(e),
                created_at=datetime.utcnow()
            )
    
    def resolve_create_webhook(self, info, url, events=None):
        """Create a webhook (placeholder)."""
        try:
            if not GRAPHQL_ENABLED:
                raise Exception("GraphQL is disabled")
            
            # Placeholder implementation
            return {
                'message': 'Webhook creation not implemented in GraphQL yet',
                'url': url,
                'events': events or [],
                'status': 'placeholder'
            }
        except Exception as e:
            logger.error(f"GraphQL webhook creation mutation failed: {str(e)}")
            return {'error': str(e)}


# Create GraphQL schema
schema = Schema(query=Query, mutation=Mutation)

def get_graphql_schema():
    """Get the GraphQL schema if enabled."""
    if GRAPHQL_ENABLED:
        return schema
    else:
        return None
