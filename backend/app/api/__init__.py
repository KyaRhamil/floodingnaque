"""API package for Floodingnaque.

Contains:
- routes: Modular API route blueprints (18 total)
- middleware: Request processing middleware
- schemas: Request/response data schemas
- graphql: GraphQL schema and resolvers
- sse: Server-Sent Events for real-time alerts
"""

# Import all route blueprints from the routes package
from app.api.routes import (
    health_bp,
    health_k8s_bp,
    ingest_bp,
    predict_bp,
    data_bp,
    models_bp,
    batch_bp,
    export_bp,
    webhooks_bp,
    celery_bp,
    rate_limits_bp,
    tides_bp,
    graphql_bp,
    security_txt_bp,
    csp_report_bp,
    performance_bp,
    sse_bp,
    upload_bp,
)

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
    # SSE for real-time alerts
    'sse_bp',
    # File uploads
    'upload_bp',
]
