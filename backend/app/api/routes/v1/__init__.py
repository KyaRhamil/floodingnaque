"""
API Version 1 Routes.

Imports all v1 endpoints for registration.
"""

from app.api.routes.health import health_bp
from app.api.routes.ingest import ingest_bp
from app.api.routes.predict import predict_bp
from app.api.routes.data import data_bp
from app.api.routes.models import models_bp

__all__ = ['health_bp', 'ingest_bp', 'predict_bp', 'data_bp', 'models_bp']
