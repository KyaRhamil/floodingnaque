"""
Swagger/OpenAPI Configuration.

Auto-generated interactive API documentation.
"""

from flasgger import Swagger

SWAGGER_TEMPLATE = {
    "swagger": "2.0",
    "info": {
        "title": "Floodingnaque API",
        "description": "Flood prediction and monitoring system for Paranaque City",
        "version": "2.0.0",
        "contact": {
            "name": "Floodingnaque Team",
            "url": "https://github.com/floodingnaque"
        },
        "license": {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        }
    },
    "host": "localhost:5000",
    "basePath": "/",
    "schemes": ["http", "https"],
    "securityDefinitions": {
        "APIKeyHeader": {
            "type": "apiKey",
            "name": "X-API-Key",
            "in": "header",
            "description": "API key for authentication"
        }
    },
    "security": [
        {"APIKeyHeader": []}
    ],
    "tags": [
        {
            "name": "Health",
            "description": "Health check and system status endpoints"
        },
        {
            "name": "Prediction",
            "description": "Flood prediction endpoints"
        },
        {
            "name": "Data",
            "description": "Historical data management"
        },
        {
            "name": "Webhooks",
            "description": "Webhook management for alerts"
        },
        {
            "name": "Export",
            "description": "Data export endpoints"
        },
        {
            "name": "Batch",
            "description": "Batch processing endpoints"
        }
    ]
}

SWAGGER_CONFIG = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs"
}


def init_swagger(app):
    """Initialize Swagger documentation."""
    swagger = Swagger(app, template=SWAGGER_TEMPLATE, config=SWAGGER_CONFIG)
    return swagger
