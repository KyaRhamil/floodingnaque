"""
Pytest Configuration and Shared Fixtures.

Provides reusable fixtures and configuration for all tests.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


# ============================================================================
# Flask Application Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def app():
    """Create a Flask application for testing."""
    from app.api.app import create_app

    # Create app with testing configuration
    # Note: FLASK_ENV is deprecated in Flask 2.3+ - use FLASK_DEBUG instead
    os.environ["FLASK_DEBUG"] = "true"
    os.environ["TESTING"] = "true"
    os.environ["AUTH_BYPASS_ENABLED"] = "true"

    application = create_app()
    application.config["TESTING"] = True

    yield application


@pytest.fixture(scope="function")
def client(app):
    """Create a test client for making HTTP requests."""
    with app.test_client() as client:
        yield client


@pytest.fixture(scope="function")
def app_context(app):
    """Provide an application context."""
    with app.app_context():
        yield


# ============================================================================
# Mock Model Fixtures
# ============================================================================


@pytest.fixture
def mock_model():
    """Create a mock ML model for testing."""
    model = MagicMock()
    model.predict.return_value = [0]  # No flood by default
    model.predict_proba.return_value = [[0.8, 0.2]]  # 80% no flood, 20% flood
    model.feature_names_in_ = ["temperature", "humidity", "precipitation"]
    return model


@pytest.fixture
def mock_model_flood():
    """Create a mock ML model that predicts flood."""
    model = MagicMock()
    model.predict.return_value = [1]  # Flood predicted
    model.predict_proba.return_value = [[0.15, 0.85]]  # 85% flood probability
    model.feature_names_in_ = ["temperature", "humidity", "precipitation"]
    return model


@pytest.fixture
def mock_model_loader(mock_model):
    """Patch the ModelLoader to use mock model."""
    with patch("app.services.predict._get_model_loader") as mock_loader:
        loader_instance = MagicMock()
        loader_instance.model = mock_model
        loader_instance.model_path = "models/test_model.joblib"
        loader_instance.metadata = {"version": 1, "checksum": "abc123"}
        loader_instance.checksum = "abc123456789"
        mock_loader.return_value = loader_instance
        yield loader_instance


# ============================================================================
# Sample Data Fixtures
# ============================================================================


@pytest.fixture
def valid_weather_data() -> Dict[str, Any]:
    """Valid weather input data for predictions."""
    return {"temperature": 298.15, "humidity": 75.0, "precipitation": 5.0}  # 25°C in Kelvin


@pytest.fixture
def extreme_weather_data() -> Dict[str, Any]:
    """Extreme weather conditions for edge case testing."""
    return {
        "temperature": 315.0,  # 42°C - very hot
        "humidity": 95.0,  # Very humid
        "precipitation": 100.0,  # Heavy rain (100mm)
    }


@pytest.fixture
def boundary_weather_data():
    """Boundary value test data."""
    return [
        # Minimum valid values
        {"temperature": 200.0, "humidity": 0.0, "precipitation": 0.0},
        # Maximum valid values
        {"temperature": 330.0, "humidity": 100.0, "precipitation": 500.0},
        # Edge cases
        {"temperature": 273.15, "humidity": 50.0, "precipitation": 0.0},  # 0°C
        {"temperature": 298.15, "humidity": 85.1, "precipitation": 10.0},  # High humidity
        {"temperature": 298.15, "humidity": 50.0, "precipitation": 30.1},  # Heavy rain
    ]


@pytest.fixture
def invalid_weather_data():
    """Invalid weather data for error testing."""
    return [
        # Invalid humidity (out of range)
        {"temperature": 298.15, "humidity": 150.0, "precipitation": 5.0},
        {"temperature": 298.15, "humidity": -10.0, "precipitation": 5.0},
        # Invalid precipitation
        {"temperature": 298.15, "humidity": 50.0, "precipitation": -5.0},
        # Missing required fields
        {"temperature": 298.15},
        {"humidity": 50.0},
        # Invalid types
        {"temperature": "hot", "humidity": 50.0, "precipitation": 5.0},
        {"temperature": 298.15, "humidity": "wet", "precipitation": 5.0},
    ]


@pytest.fixture
def sample_coordinates():
    """Sample geographic coordinates for testing."""
    return {
        "paranaque": {"lat": 14.4793, "lon": 121.0198},
        "manila": {"lat": 14.5995, "lon": 120.9842},
        "invalid_lat": {"lat": 91.0, "lon": 121.0198},
        "invalid_lon": {"lat": 14.4793, "lon": 181.0},
        "boundary_lat": {"lat": 90.0, "lon": 0.0},
        "boundary_lon": {"lat": 0.0, "lon": 180.0},
    }


# ============================================================================
# API Key Fixtures
# ============================================================================


@pytest.fixture
def valid_api_key():
    """Generate a valid API key for testing."""
    return "test-api-key-12345-valid"


@pytest.fixture
def invalid_api_key():
    """An invalid API key for testing."""
    return "invalid-api-key-xyz"


@pytest.fixture
def api_headers(valid_api_key):
    """Headers with valid API key."""
    return {"X-API-Key": valid_api_key, "Content-Type": "application/json"}


@pytest.fixture
def api_headers_invalid(invalid_api_key):
    """Headers with invalid API key."""
    return {"X-API-Key": invalid_api_key, "Content-Type": "application/json"}


# ============================================================================
# Environment Fixtures
# ============================================================================


@pytest.fixture
def mock_env_production():
    """Mock production environment variables."""
    env_vars = {
        "FLASK_DEBUG": "false",  # FLASK_ENV deprecated in Flask 2.3+
        "DEBUG": "false",
        "AUTH_BYPASS_ENABLED": "false",
        "VALID_API_KEYS": "prod-key-1,prod-key-2",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield


@pytest.fixture
def mock_env_development():
    """Mock development environment variables."""
    env_vars = {
        "FLASK_DEBUG": "true",  # FLASK_ENV deprecated in Flask 2.3+
        "DEBUG": "true",
        "AUTH_BYPASS_ENABLED": "true",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        yield


# ============================================================================
# Risk Classification Fixtures
# ============================================================================


@pytest.fixture
def risk_test_cases():
    """Test cases for risk classification."""
    return [
        # (prediction, probability, precipitation, humidity, expected_risk_level)
        # Safe cases
        (0, {"no_flood": 0.95, "flood": 0.05}, 0.0, 50.0, 0),
        (0, {"no_flood": 0.80, "flood": 0.20}, 5.0, 60.0, 0),
        # Alert cases
        (0, {"no_flood": 0.65, "flood": 0.35}, 15.0, 80.0, 1),
        (1, {"no_flood": 0.45, "flood": 0.55}, 20.0, 85.0, 1),
        # Critical cases
        (1, {"no_flood": 0.20, "flood": 0.80}, 50.0, 95.0, 2),
        (1, {"no_flood": 0.10, "flood": 0.90}, 100.0, 98.0, 2),
    ]


# ============================================================================
# Response Schema Validators
# ============================================================================


def validate_health_response(response_data: Dict) -> bool:
    """Validate health endpoint response schema."""
    required_fields = ["status"]
    return all(field in response_data for field in required_fields)


def validate_prediction_response(response_data: Dict) -> bool:
    """Validate prediction endpoint response schema."""
    required_fields = ["prediction", "flood_risk", "request_id"]
    return all(field in response_data for field in required_fields)


def validate_error_response(response_data: Dict) -> bool:
    """Validate error response schema."""
    required_fields = ["error"]
    return all(field in response_data for field in required_fields)


# Register validators as fixtures
@pytest.fixture
def response_validators():
    """Return response validation functions."""
    return {
        "health": validate_health_response,
        "prediction": validate_prediction_response,
        "error": validate_error_response,
    }


# ============================================================================
# Utility Functions
# ============================================================================


@pytest.fixture
def assert_json_response():
    """Helper to assert JSON response structure."""

    def _assert(response, expected_status, required_fields=None):
        assert response.status_code == expected_status
        data = response.get_json()
        assert data is not None
        if required_fields:
            for field in required_fields:
                assert field in data, f"Missing field: {field}"
        return data

    return _assert


# ============================================================================
# Database Mocking Fixtures
# ============================================================================


@pytest.fixture
def mock_db():
    """Mock database session for testing."""
    mock = MagicMock()
    mock.session = MagicMock()
    mock.session.add = MagicMock()
    mock.session.commit = MagicMock()
    mock.session.rollback = MagicMock()
    mock.session.query = MagicMock()
    mock.session.execute = MagicMock()
    mock.session.close = MagicMock()
    return mock


@pytest.fixture
def mock_db_session(mock_db):
    """Patch database session for testing."""
    with patch("app.models.db", mock_db):
        with patch("app.services.db", mock_db):
            yield mock_db


@pytest.fixture
def mock_sqlalchemy_engine():
    """Mock SQLAlchemy engine for connection testing."""
    engine = MagicMock()
    engine.connect.return_value.__enter__ = MagicMock(return_value=MagicMock())
    engine.connect.return_value.__exit__ = MagicMock(return_value=None)
    engine.execute = MagicMock()
    engine.dispose = MagicMock()
    return engine


@pytest.fixture
def sample_db_records():
    """Sample database records for testing."""
    return [
        {
            "id": 1,
            "timestamp": "2025-01-15T10:00:00Z",
            "temperature": 298.15,
            "humidity": 75.0,
            "precipitation": 5.0,
            "prediction": 0,
            "flood_risk": "low",
        },
        {
            "id": 2,
            "timestamp": "2025-01-15T11:00:00Z",
            "temperature": 300.0,
            "humidity": 85.0,
            "precipitation": 25.0,
            "prediction": 1,
            "flood_risk": "high",
        },
    ]


# ============================================================================
# Redis/Cache Mocking Fixtures
# ============================================================================


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = MagicMock()
    redis_mock.get = MagicMock(return_value=None)
    redis_mock.set = MagicMock(return_value=True)
    redis_mock.setex = MagicMock(return_value=True)
    redis_mock.delete = MagicMock(return_value=1)
    redis_mock.exists = MagicMock(return_value=0)
    redis_mock.incr = MagicMock(return_value=1)
    redis_mock.expire = MagicMock(return_value=True)
    redis_mock.ttl = MagicMock(return_value=3600)
    redis_mock.keys = MagicMock(return_value=[])
    redis_mock.flushdb = MagicMock(return_value=True)
    redis_mock.pipeline = MagicMock(return_value=MagicMock())
    redis_mock.ping = MagicMock(return_value=True)
    # Sorted set operations for sliding window rate limiting
    redis_mock.zcard = MagicMock(return_value=0)
    redis_mock.zadd = MagicMock(return_value=1)
    redis_mock.zremrangebyscore = MagicMock(return_value=0)
    return redis_mock


@pytest.fixture
def mock_redis_client(mock_redis):
    """Patch Redis client globally for testing."""
    with patch("app.utils.cache.redis_client", mock_redis):
        with patch("app.utils.rate_limit.redis_client", mock_redis):
            with patch("app.api.middleware.rate_limit.redis_client", mock_redis):
                yield mock_redis


@pytest.fixture
def mock_cache():
    """Mock cache decorator/manager for testing."""
    cache_mock = MagicMock()
    cache_mock.get = MagicMock(return_value=None)
    cache_mock.set = MagicMock(return_value=True)
    cache_mock.delete = MagicMock(return_value=True)
    cache_mock.clear = MagicMock(return_value=True)
    cache_mock.mget = MagicMock(return_value=[])
    cache_mock.mset = MagicMock(return_value=True)
    return cache_mock


# ============================================================================
# Celery/Task Queue Mocking Fixtures
# ============================================================================


@pytest.fixture
def mock_celery():
    """Mock Celery app for testing."""
    celery_mock = MagicMock()
    celery_mock.send_task = MagicMock()
    celery_mock.AsyncResult = MagicMock()
    return celery_mock


@pytest.fixture
def mock_celery_task():
    """Mock Celery task for testing."""
    task_mock = MagicMock()
    task_mock.delay = MagicMock()
    task_mock.apply_async = MagicMock()
    task_mock.s = MagicMock()  # Signature shortcut

    # Task result
    result_mock = MagicMock()
    result_mock.id = "task-id-12345"
    result_mock.state = "PENDING"
    result_mock.result = None
    result_mock.ready = MagicMock(return_value=False)
    result_mock.successful = MagicMock(return_value=False)
    result_mock.failed = MagicMock(return_value=False)
    result_mock.get = MagicMock(return_value=None)

    task_mock.delay.return_value = result_mock
    task_mock.apply_async.return_value = result_mock

    return task_mock


@pytest.fixture
def mock_task_queue(mock_celery_task):
    """Patch task queue for testing."""
    with patch("app.tasks.prediction_tasks.predict_flood", mock_celery_task):
        with patch("app.tasks.data_tasks.ingest_weather_data", mock_celery_task):
            yield mock_celery_task


# ============================================================================
# External API Mocking Fixtures
# ============================================================================


@pytest.fixture
def mock_weather_api():
    """Mock weather API responses."""
    return {
        "current": {
            "temperature": 298.15,
            "humidity": 75.0,
            "precipitation": 5.0,
            "wind_speed": 10.0,
            "pressure": 1013.25,
        },
        "forecast": [
            {"time": "2025-01-15T12:00:00Z", "temperature": 299.0, "precipitation": 2.0},
            {"time": "2025-01-15T15:00:00Z", "temperature": 301.0, "precipitation": 0.0},
            {"time": "2025-01-15T18:00:00Z", "temperature": 298.0, "precipitation": 10.0},
        ],
    }


@pytest.fixture
def mock_requests(mock_weather_api):
    """Mock requests library for external API calls."""
    with patch("requests.get") as mock_get:
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = mock_weather_api
        response_mock.text = '{"status": "ok"}'
        response_mock.headers = {"Content-Type": "application/json"}
        response_mock.raise_for_status = MagicMock()
        mock_get.return_value = response_mock
        yield mock_get


@pytest.fixture
def mock_httpx():
    """Mock httpx library for async HTTP calls."""
    with patch("httpx.AsyncClient") as mock_client:
        client_instance = MagicMock()
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = {"status": "ok"}
        client_instance.get = MagicMock(return_value=response_mock)
        client_instance.post = MagicMock(return_value=response_mock)
        mock_client.return_value.__aenter__ = MagicMock(return_value=client_instance)
        mock_client.return_value.__aexit__ = MagicMock(return_value=None)
        yield mock_client


# ============================================================================
# Metrics/Monitoring Fixtures
# ============================================================================


@pytest.fixture
def mock_prometheus():
    """Mock Prometheus metrics for testing."""
    prometheus_mock = MagicMock()
    prometheus_mock.Counter = MagicMock(return_value=MagicMock())
    prometheus_mock.Histogram = MagicMock(return_value=MagicMock())
    prometheus_mock.Gauge = MagicMock(return_value=MagicMock())
    prometheus_mock.Summary = MagicMock(return_value=MagicMock())
    return prometheus_mock


@pytest.fixture
def mock_metrics(mock_prometheus):
    """Patch Prometheus metrics for testing."""
    with patch("app.utils.metrics.prometheus_client", mock_prometheus):
        yield mock_prometheus


# ============================================================================
# Logging Fixtures
# ============================================================================


@pytest.fixture
def mock_logger():
    """Mock logger for testing log output."""
    logger = MagicMock()
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.critical = MagicMock()
    logger.exception = MagicMock()
    return logger


@pytest.fixture
def capture_logs(caplog):
    """Capture log output for assertions."""
    import logging

    caplog.set_level(logging.DEBUG)
    return caplog


# ============================================================================
# Time/Date Fixtures
# ============================================================================


@pytest.fixture
def freeze_time():
    """Fixture to freeze time for testing."""
    from datetime import datetime
    from unittest.mock import patch

    frozen_time = datetime(2025, 1, 15, 12, 0, 0)

    with patch("datetime.datetime") as mock_datetime:
        mock_datetime.now.return_value = frozen_time
        mock_datetime.utcnow.return_value = frozen_time
        mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
        yield frozen_time


@pytest.fixture
def mock_time():
    """Mock time module for testing."""
    import time

    with patch("time.time") as mock_time_func:
        mock_time_func.return_value = 1736942400.0  # 2025-01-15 12:00:00 UTC
        yield mock_time_func


# ============================================================================
# File/IO Fixtures
# ============================================================================


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file for testing."""
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("test content")
    return file_path


@pytest.fixture
def temp_json_file(tmp_path):
    """Create a temporary JSON file for testing."""
    import json

    file_path = tmp_path / "test_data.json"
    data = {"key": "value", "nested": {"a": 1, "b": 2}}
    file_path.write_text(json.dumps(data))
    return file_path


@pytest.fixture
def temp_csv_file(tmp_path):
    """Create a temporary CSV file for testing."""
    file_path = tmp_path / "test_data.csv"
    content = "timestamp,temperature,humidity,precipitation\n"
    content += "2025-01-15T10:00:00Z,298.15,75.0,5.0\n"
    content += "2025-01-15T11:00:00Z,300.0,85.0,25.0\n"
    file_path.write_text(content)
    return file_path


# ============================================================================
# Security Testing Fixtures
# ============================================================================


@pytest.fixture
def malicious_inputs():
    """Common malicious inputs for security testing."""
    return {
        "sql_injection": [
            "'; DROP TABLE users; --",
            "1 OR 1=1",
            "UNION SELECT * FROM passwords",
            "1; DELETE FROM predictions WHERE 1=1",
        ],
        "xss": [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert(1)>",
            "javascript:alert('xss')",
            "<svg onload=alert(1)>",
        ],
        "path_traversal": ["../../../etc/passwd", "..\\..\\..\\windows\\system32", "....//....//etc/passwd"],
        "command_injection": ["; ls -la", "| cat /etc/shadow", "`id`", "$(uname -a)"],
    }


@pytest.fixture
def security_headers():
    """Expected security headers for responses."""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'",
    }


# ============================================================================
# Performance Testing Fixtures
# ============================================================================


@pytest.fixture
def performance_timer():
    """Timer for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self):
            self.end_time = time.perf_counter()

        @property
        def elapsed_ms(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time) * 1000
            return None

        def assert_under(self, max_ms):
            assert self.elapsed_ms is not None, "Timer not stopped"
            assert self.elapsed_ms < max_ms, f"Elapsed {self.elapsed_ms}ms exceeds {max_ms}ms"

    return Timer()


@pytest.fixture
def benchmark_requests(client, api_headers):
    """Benchmark helper for request performance."""
    import time

    def _benchmark(endpoint, method="GET", data=None, iterations=10):
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            if method == "GET":
                client.get(endpoint, headers=api_headers)
            elif method == "POST":
                client.post(endpoint, json=data, headers=api_headers)
            times.append((time.perf_counter() - start) * 1000)

        return {"min_ms": min(times), "max_ms": max(times), "avg_ms": sum(times) / len(times), "iterations": iterations}

    return _benchmark


# ============================================================================
# Database Integration Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def test_database_url():
    """Get test database URL (uses SQLite for testing)."""
    import tempfile

    temp_dir = tempfile.gettempdir()
    return f"sqlite:///{temp_dir}/test_floodingnaque.db"


@pytest.fixture(scope="function")
def db_session(app, test_database_url):
    """Create a database session for testing with automatic cleanup."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import scoped_session, sessionmaker

    # Use test database URL
    with patch.dict(os.environ, {"DATABASE_URL": test_database_url}):
        engine = create_engine(test_database_url, echo=False)

        # Import models to create tables
        try:
            from app.models.db import Base

            Base.metadata.create_all(engine)
        except ImportError:
            pass

        Session = scoped_session(sessionmaker(bind=engine))
        session = Session()

        yield session

        # Cleanup
        session.rollback()
        session.close()
        Session.remove()


@pytest.fixture
def mock_db_context():
    """Context manager for mocking database operations."""

    class DBContextMock:
        def __init__(self):
            self.session = MagicMock()
            self.queries = []

        def __enter__(self):
            return self.session

        def __exit__(self, *args):
            pass

        def record_query(self, query):
            self.queries.append(query)

    return DBContextMock()


# ============================================================================
# External API Mock Fixtures (Contract Testing)
# ============================================================================


@pytest.fixture
def mock_google_weather_response():
    """Mock Google Weather API response."""
    return {
        "currentConditions": {
            "temperature": {"value": 28.5},
            "humidity": {"value": 75},
            "precipitation": {"value": 5.0},
            "windSpeed": {"value": 15.0},
            "pressure": {"value": 1013.25},
            "uvIndex": {"value": 7},
            "cloudCover": {"value": 40},
        },
        "forecast": {
            "days": [
                {
                    "date": "2025-01-21",
                    "maxTemperature": {"value": 32.0},
                    "minTemperature": {"value": 24.0},
                    "precipitation": {"value": 10.0},
                },
                {
                    "date": "2025-01-22",
                    "maxTemperature": {"value": 30.0},
                    "minTemperature": {"value": 25.0},
                    "precipitation": {"value": 25.0},
                },
            ]
        },
    }


@pytest.fixture
def mock_meteostat_response():
    """Mock Meteostat API response."""
    return {
        "data": [
            {
                "date": "2025-01-20",
                "tavg": 27.5,
                "tmin": 24.0,
                "tmax": 31.0,
                "prcp": 5.2,
                "rhum": 78,
                "wspd": 12.5,
                "pres": 1012.0,
            },
            {
                "date": "2025-01-21",
                "tavg": 28.0,
                "tmin": 25.0,
                "tmax": 32.0,
                "prcp": 15.0,
                "rhum": 82,
                "wspd": 10.0,
                "pres": 1010.5,
            },
        ],
        "meta": {"generated": "2025-01-21T12:00:00Z", "stations": ["RPLL0"]},
    }


@pytest.fixture
def mock_worldtides_response():
    """Mock WorldTides API response."""
    return {
        "status": 200,
        "callCount": 1,
        "copyright": "WorldTides",
        "requestLat": 14.4793,
        "requestLon": 121.0198,
        "responseLat": 14.4793,
        "responseLon": 121.0198,
        "atlas": "TPXO",
        "station": "MANILA",
        "heights": [
            {"dt": 1737446400, "date": "2025-01-21T08:00+08:00", "height": 0.85},
            {"dt": 1737468000, "date": "2025-01-21T14:00+08:00", "height": 1.25},
            {"dt": 1737489600, "date": "2025-01-21T20:00+08:00", "height": 0.45},
        ],
        "extremes": [
            {"dt": 1737457200, "date": "2025-01-21T11:00+08:00", "height": 1.35, "type": "High"},
            {"dt": 1737500400, "date": "2025-01-21T23:00+08:00", "height": 0.25, "type": "Low"},
        ],
    }


@pytest.fixture
def mock_external_apis(mock_google_weather_response, mock_meteostat_response, mock_worldtides_response):
    """Mock all external API calls for integration testing."""
    with (
        patch("app.services.google_weather_service.requests.get") as mock_google,
        patch("app.services.meteostat_service.requests.get") as mock_meteostat,
        patch("app.services.worldtides_service.requests.get") as mock_tides,
    ):

        # Google Weather
        google_response = MagicMock()
        google_response.status_code = 200
        google_response.json.return_value = mock_google_weather_response
        google_response.raise_for_status = MagicMock()
        mock_google.return_value = google_response

        # Meteostat
        meteostat_response = MagicMock()
        meteostat_response.status_code = 200
        meteostat_response.json.return_value = mock_meteostat_response
        meteostat_response.raise_for_status = MagicMock()
        mock_meteostat.return_value = meteostat_response

        # WorldTides
        tides_response = MagicMock()
        tides_response.status_code = 200
        tides_response.json.return_value = mock_worldtides_response
        tides_response.raise_for_status = MagicMock()
        mock_tides.return_value = tides_response

        yield {"google": mock_google, "meteostat": mock_meteostat, "worldtides": mock_tides}


# ============================================================================
# Security Testing Fixtures (Extended)
# ============================================================================


@pytest.fixture
def sql_injection_payloads():
    """SQL injection test payloads."""
    return [
        # Basic injection
        "'; DROP TABLE users; --",
        "1 OR 1=1",
        "1' OR '1'='1",
        "1; DELETE FROM predictions WHERE 1=1",
        "UNION SELECT * FROM passwords",
        # Blind SQL injection
        "1 AND 1=1",
        "1 AND 1=2",
        "1' AND '1'='1",
        "1' AND SLEEP(5)--",
        # Second-order injection
        "admin'--",
        "' OR ''='",
        # Database-specific
        "'; EXEC xp_cmdshell('dir'); --",  # SQL Server
        "1; SELECT pg_sleep(5)--",  # PostgreSQL
    ]


@pytest.fixture
def xss_payloads():
    """XSS test payloads."""
    return [
        # Basic XSS
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert(1)>",
        "<svg onload=alert(1)>",
        # Event handlers
        "<body onload=alert(1)>",
        "<div onmouseover=alert(1)>",
        "<input onfocus=alert(1) autofocus>",
        # Protocol handlers
        "javascript:alert('xss')",
        "data:text/html,<script>alert('xss')</script>",
        # Encoded payloads
        "%3Cscript%3Ealert('xss')%3C/script%3E",
        "&#60;script&#62;alert('xss')&#60;/script&#62;",
        # DOM-based XSS
        "<img src='x' onerror='this.onerror=null;alert(1)'>",
        "<iframe src='javascript:alert(1)'></iframe>",
    ]


@pytest.fixture
def csrf_test_data():
    """CSRF test configuration."""
    return {
        "valid_token": "valid-csrf-token-12345",
        "invalid_token": "invalid-csrf-token",
        "expired_token": "expired-csrf-token",
        "missing_token": None,
        "protected_endpoints": ["/api/v1/predict", "/ingest", "/api/v1/upload", "/api/v1/export"],
    }


@pytest.fixture
def path_traversal_payloads():
    """Path traversal attack payloads."""
    return [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "....//....//....//etc/passwd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc/passwd",
        "..%252f..%252f..%252fetc/passwd",
        "/etc/passwd%00.txt",
        "....\\....\\....\\windows\\system32",
    ]


@pytest.fixture
def command_injection_payloads():
    """Command injection test payloads."""
    return [
        "; ls -la",
        "| cat /etc/passwd",
        "& whoami",
        "`id`",
        "$(whoami)",
        "; ping -c 5 127.0.0.1",
        "|| cat /etc/passwd",
        "&& dir",
    ]


@pytest.fixture
def header_injection_payloads():
    """HTTP header injection payloads."""
    return [
        "test\r\nX-Injected: header",
        "test\r\nSet-Cookie: malicious=value",
        "test%0d%0aX-Injected:%20header",
        "test\nX-Forwarded-For: 127.0.0.1",
    ]


# ============================================================================
# Network Failure Simulation Fixtures
# ============================================================================


@pytest.fixture
def mock_network_failure():
    """Simulate network failures for negative path testing."""
    import requests.exceptions

    class NetworkFailureMock:
        def __init__(self):
            self.failure_type = None

        def timeout(self):
            """Simulate connection timeout."""
            self.failure_type = "timeout"
            raise requests.exceptions.Timeout("Connection timed out")

        def connection_error(self):
            """Simulate connection error."""
            self.failure_type = "connection"
            raise requests.exceptions.ConnectionError("Failed to connect")

        def dns_failure(self):
            """Simulate DNS failure."""
            self.failure_type = "dns"
            raise requests.exceptions.ConnectionError("DNS lookup failed")

        def ssl_error(self):
            """Simulate SSL error."""
            self.failure_type = "ssl"
            raise requests.exceptions.SSLError("SSL certificate verification failed")

        def http_error(self, status_code=500):
            """Simulate HTTP error response."""
            self.failure_type = f"http_{status_code}"
            response = MagicMock()
            response.status_code = status_code
            response.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status_code} Error")
            return response

    return NetworkFailureMock()


@pytest.fixture
def mock_slow_response():
    """Simulate slow API responses."""
    import time

    def _slow_response(delay_seconds=5):
        time.sleep(delay_seconds)
        return {"status": "ok", "delayed": True}

    return _slow_response


# ============================================================================
# Rate Limiting Test Fixtures
# ============================================================================


@pytest.fixture
def rate_limit_test_config():
    """Configuration for rate limiting tests."""
    return {
        "default_limit": "100 per minute",
        "burst_limit": "10 per second",
        "prediction_limit": "30 per minute",
        "ingest_limit": "60 per minute",
        "test_iterations": 150,  # Exceed default limit
    }


@pytest.fixture
def rapid_requests(client, api_headers):
    """Helper to make rapid requests for rate limit testing."""

    def _rapid_requests(endpoint, count=100, method="GET", data=None):
        responses = []
        for i in range(count):
            if method == "GET":
                resp = client.get(endpoint, headers=api_headers)
            else:
                resp = client.post(endpoint, json=data, headers=api_headers)
            responses.append(
                {
                    "iteration": i + 1,
                    "status_code": resp.status_code,
                    "rate_limited": resp.status_code == 429,
                    "headers": dict(resp.headers),
                }
            )
            if resp.status_code == 429:
                break
        return responses

    return _rapid_requests


# ============================================================================
# API Versioning Test Fixtures
# ============================================================================


@pytest.fixture
def api_v1_endpoints():
    """V1 API endpoints for backward compatibility testing."""
    return [
        {"path": "/api/v1/predict", "method": "POST", "required_fields": ["prediction", "flood_risk"]},
        {"path": "/api/v1/health", "method": "GET", "required_fields": ["status"]},
        {"path": "/api/v1/data", "method": "GET", "required_fields": ["data", "total"]},
        {"path": "/api/v1/models", "method": "GET", "required_fields": ["models"]},
    ]


@pytest.fixture
def deprecated_endpoints():
    """Endpoints that are deprecated but should still work."""
    return [
        {"path": "/predict", "new_path": "/api/v1/predict"},
        {"path": "/health", "new_path": "/api/v1/health"},
    ]
