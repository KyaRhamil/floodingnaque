"""Business logic services (weather, prediction, alerts)."""

from app.services.ingest import ingest_data
from app.services.predict import predict_flood, get_current_model_info, ModelLoader
from app.services.alerts import AlertSystem, get_alert_system, send_flood_alert
from app.services.meteostat_service import (
    MeteostatService,
    get_meteostat_service,
    get_historical_weather,
    get_meteostat_weather_for_ingest,
    save_meteostat_data_to_db
)

__all__ = [
    'ingest_data',
    'predict_flood',
    'get_current_model_info',
    'ModelLoader',
    'AlertSystem',
    'get_alert_system',
    'send_flood_alert',
    'MeteostatService',
    'get_meteostat_service',
    'get_historical_weather',
    'get_meteostat_weather_for_ingest',
    'save_meteostat_data_to_db'
]
