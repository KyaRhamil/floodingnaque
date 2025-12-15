"""
API package for Floodingnaque.

Contains:
- routes: Modular API route blueprints
- middleware: Request processing middleware
- schemas: Request/response data schemas
"""

from app.api.routes import health_bp, ingest_bp, predict_bp, data_bp, models_bp

__all__ = [
    'health_bp',
    'ingest_bp',
    'predict_bp',
    'data_bp',
    'models_bp'
]
