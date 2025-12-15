"""
Routes package for Floodingnaque API.

Contains modular route definitions:
- health: /status, /health, / (root)
- ingest: /ingest
- predict: /predict
- data: /data
- models: /api/models, /api/docs, /api/version
"""

from app.api.routes.health import health_bp
from app.api.routes.ingest import ingest_bp
from app.api.routes.predict import predict_bp
from app.api.routes.data import data_bp
from app.api.routes.models import models_bp

__all__ = [
    'health_bp',
    'ingest_bp',
    'predict_bp',
    'data_bp',
    'models_bp'
]
