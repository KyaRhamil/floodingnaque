"""API Version 1 Routes - Prepared for Future API Versioning.

This module re-exports all route blueprints for API v1.
Currently, v1 uses the same routes as the base API.

STATUS: Intentionally kept for forward compatibility.
REVIEWED: 2026-01 - Retained during codebase audit.

Usage Example (when versioning is implemented):
    from app.api.routes.v1 import health_bp
    app.register_blueprint(health_bp, url_prefix='/api/v1')

Note: API versioning via URL prefix (/api/v1, /api/v2) will be
implemented when breaking API changes are needed. This module
provides the foundation for that without requiring code changes
to the route implementations themselves.

See Also:
    - backend/docs/BACKEND_ARCHITECTURE.md for API design decisions
"""

# Re-export all blueprints from the main routes package
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
)

__all__ = [
    'health_bp',
    'health_k8s_bp',
    'ingest_bp',
    'predict_bp',
    'data_bp',
    'models_bp',
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
