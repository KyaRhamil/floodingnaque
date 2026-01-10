"""Business logic services (weather, prediction, alerts, scheduling).

This module provides comprehensive service layer functionality:
- Weather data ingestion and processing
- Flood prediction and risk classification
- Alert generation and delivery
- Background task scheduling
- External API integrations (Meteostat, WorldTides, Google Earth Engine)
"""

# Core prediction and ingestion services
from app.services.ingest import ingest_data
from app.services.predict import predict_flood, get_current_model_info, ModelLoader

# Alert system
from app.services.alerts import AlertSystem, get_alert_system, send_flood_alert

# Meteostat weather service
from app.services.meteostat_service import (
    MeteostatService,
    get_meteostat_service,
    get_historical_weather,
    get_meteostat_weather_for_ingest,
    save_meteostat_data_to_db
)

# WorldTides service for coastal flood prediction
from app.services.worldtides_service import (
    WorldTidesService,
    TideData,
    TideExtreme,
)

# Google Weather/Earth Engine service
from app.services.google_weather_service import (
    GoogleWeatherService,
    SatellitePrecipitation,
    WeatherReanalysis,
)

# Celery background tasks
from app.services.celery_app import celery_app

# Scheduler for periodic tasks
from app.services import scheduler

# System evaluation for thesis validation
from app.services.evaluation import SystemEvaluator

# Risk classification
from app.services.risk_classifier import (
    classify_risk_level,
    get_risk_thresholds,
    format_alert_message,
    RISK_LEVELS,
    RISK_LEVEL_COLORS,
    RISK_LEVEL_DESCRIPTIONS,
)

__all__ = [
    # Core services
    'ingest_data',
    'predict_flood',
    'get_current_model_info',
    'ModelLoader',
    # Alert system
    'AlertSystem',
    'get_alert_system',
    'send_flood_alert',
    # Meteostat service
    'MeteostatService',
    'get_meteostat_service',
    'get_historical_weather',
    'get_meteostat_weather_for_ingest',
    'save_meteostat_data_to_db',
    # WorldTides service
    'WorldTidesService',
    'TideData',
    'TideExtreme',
    # Google Weather service
    'GoogleWeatherService',
    'SatellitePrecipitation',
    'WeatherReanalysis',
    # Celery
    'celery_app',
    # Scheduler
    'scheduler',
    # Evaluation
    'SystemEvaluator',
    # Risk classification
    'classify_risk_level',
    'get_risk_thresholds',
    'format_alert_message',
    'RISK_LEVELS',
    'RISK_LEVEL_COLORS',
    'RISK_LEVEL_DESCRIPTIONS',
]
