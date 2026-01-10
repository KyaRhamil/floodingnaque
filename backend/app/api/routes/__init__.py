"""
Routes package for Floodingnaque API.

Contains modular route definitions:
- health: /status, /health, / (root)
- health_k8s: /health/live, /health/ready (Kubernetes probes)
- ingest: /ingest
- predict: /predict
- data: /data, /meteostat/stations, /meteostat/hourly, /meteostat/daily, /meteostat/current, /meteostat/status
- models: /api/models, /api/docs, /api/version
- batch: /batch/predict
- export: /export/weather, /export/predictions
- webhooks: /webhooks/register, /webhooks/list
- celery: /tasks/retrain, /tasks/process-data, /tasks
- rate_limits: /rate-limits/status, /rate-limits/tiers
- tides: /tides/current, /tides/extremes, /tides/prediction
- graphql: /graphql/info, /graphql/schema
- security_txt: /.well-known/security.txt, /security.txt
- csp_report: /csp-report
- performance: /api/performance/dashboard, /api/performance/response-times
"""

from app.api.routes.health import health_bp
from app.api.routes.health_k8s import health_k8s_bp
from app.api.routes.ingest import ingest_bp
from app.api.routes.predict import predict_bp
from app.api.routes.data import data_bp
from app.api.routes.models import models_bp
from app.api.routes.batch import batch_bp
from app.api.routes.export import export_bp
from app.api.routes.webhooks import webhooks_bp
from app.api.routes.celery import celery_bp
from app.api.routes.rate_limits import rate_limits_bp
from app.api.routes.tides import tides_bp
from app.api.routes.graphql import graphql_bp
from app.api.routes.security_txt import security_txt_bp
from app.api.routes.csp_report import csp_report_bp
from app.api.routes.performance import performance_bp

__all__ = [
    # Core routes
    'health_bp',
    'health_k8s_bp',
    'ingest_bp',
    'predict_bp',
    'data_bp',
    'models_bp',
    # Extended routes
    'batch_bp',
    'export_bp',
    'webhooks_bp',
    'celery_bp',
    'rate_limits_bp',
    'tides_bp',
    'graphql_bp',
    'security_txt_bp',
    'csp_report_bp',
    'performance_bp',
]
