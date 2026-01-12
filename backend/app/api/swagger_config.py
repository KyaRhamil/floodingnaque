"""Swagger/OpenAPI 3.0 Configuration.

Auto-generated interactive API documentation.
Upgraded to OpenAPI 3.0 specification for better frontend codegen support.
"""

import os
import json
from flask import jsonify, Response
from flasgger import Swagger

# OpenAPI 3.0.3 Template
SWAGGER_TEMPLATE = {
    "openapi": "3.0.3",
    "info": {
        "title": "Floodingnaque API",
        "description": """Flood prediction and monitoring system for Paranaque City.

## Features
- Real-time flood predictions using ML models
- Historical weather data management
- Alert system with SSE for real-time notifications
- Dashboard analytics and statistics
- User authentication with JWT

## API Versioning
All API endpoints are prefixed with `/api/v1/`.

## Authentication
Most endpoints require authentication via:
- JWT Bearer Token in Authorization header
- API Key in X-API-Key header
""",
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
    "servers": [
        {
            "url": "http://localhost:5000",
            "description": "Local development server"
        },
        {
            "url": "https://api.floodingnaque.com",
            "description": "Production server"
        }
    ],
    "components": {
        "securitySchemes": {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT Bearer token authentication"
            },
            "APIKeyHeader": {
                "type": "apiKey",
                "name": "X-API-Key",
                "in": "header",
                "description": "API key for authentication"
            }
        },
        "schemas": {
            "Error": {
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "example": False
                    },
                    "error": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "title": {"type": "string"},
                            "status": {"type": "integer"},
                            "detail": {"type": "string"},
                            "code": {"type": "string"},
                            "request_id": {"type": "string"},
                            "timestamp": {"type": "string", "format": "date-time"}
                        }
                    }
                }
            },
            "PredictionRequest": {
                "type": "object",
                "required": ["rainfall_mm", "humidity_percent", "temperature_c"],
                "properties": {
                    "rainfall_mm": {"type": "number", "format": "float", "example": 25.5},
                    "humidity_percent": {"type": "number", "format": "float", "example": 85.0},
                    "temperature_c": {"type": "number", "format": "float", "example": 28.5},
                    "wind_speed_kph": {"type": "number", "format": "float", "example": 15.0},
                    "pressure_hpa": {"type": "number", "format": "float", "example": 1010.0}
                }
            },
            "PredictionResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "data": {
                        "type": "object",
                        "properties": {
                            "risk_level": {"type": "integer", "enum": [0, 1, 2]},
                            "risk_label": {"type": "string", "enum": ["Safe", "Alert", "Critical"]},
                            "confidence": {"type": "number", "format": "float"},
                            "probabilities": {
                                "type": "object",
                                "properties": {
                                    "safe": {"type": "number"},
                                    "alert": {"type": "number"},
                                    "critical": {"type": "number"}
                                }
                            }
                        }
                    },
                    "request_id": {"type": "string"}
                }
            },
            "Alert": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "risk_level": {"type": "integer", "enum": [0, 1, 2]},
                    "risk_label": {"type": "string"},
                    "location": {"type": "string"},
                    "message": {"type": "string"},
                    "delivery_status": {"type": "string", "enum": ["delivered", "pending", "failed"]},
                    "created_at": {"type": "string", "format": "date-time"}
                }
            },
            "SSEEvent": {
                "type": "object",
                "properties": {
                    "event": {"type": "string", "enum": ["alert", "heartbeat", "connected"]},
                    "data": {"type": "object"}
                }
            }
        },
        "responses": {
            "BadRequest": {
                "description": "Bad request - invalid input",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "Unauthorized": {
                "description": "Authentication required",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "NotFound": {
                "description": "Resource not found",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "RateLimited": {
                "description": "Rate limit exceeded",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            },
            "InternalError": {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                }
            }
        }
    },
    "security": [
        {"BearerAuth": []},
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
            "description": "Historical weather data management"
        },
        {
            "name": "Predictions",
            "description": "Prediction history and analytics"
        },
        {
            "name": "Alerts",
            "description": "Alert management and history"
        },
        {
            "name": "Real-time",
            "description": "Server-Sent Events for live updates"
        },
        {
            "name": "Dashboard",
            "description": "Dashboard summary and statistics"
        },
        {
            "name": "Authentication",
            "description": "User authentication and session management"
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
        },
        {
            "name": "Models",
            "description": "ML model information and management"
        },
        {
            "name": "Tasks",
            "description": "Background task management"
        },
        {
            "name": "Tides",
            "description": "Tide data and predictions"
        },
        {
            "name": "Performance",
            "description": "API performance monitoring"
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
    "specs_route": "/apidocs",
    "openapi": "3.0.3"
}


def init_swagger(app):
    """Initialize Swagger documentation and OpenAPI export."""
    swagger = Swagger(app, template=SWAGGER_TEMPLATE, config=SWAGGER_CONFIG)
    
    # Add route to export OpenAPI schema as JSON file
    @app.route('/openapi.json', methods=['GET'])
    def export_openapi_schema():
        """Export OpenAPI 3.0 schema as JSON for frontend codegen.
        
        Returns:
            JSON: Complete OpenAPI 3.0 specification
        """
        # Get the generated spec from Swagger
        spec = swagger.get_apispecs()
        
        return Response(
            json.dumps(spec, indent=2, default=str),
            mimetype='application/json',
            headers={
                'Content-Disposition': 'attachment; filename=openapi.json',
                'Access-Control-Allow-Origin': '*'
            }
        )
    
    return swagger
