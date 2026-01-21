"""
Unit tests for core constants module.

Tests for app/core/constants.py
"""

import pytest


class TestAPIConstants:
    """Tests for API-related constants."""

    def test_api_version(self):
        """Test API version constant."""
        from app.core.constants import API_VERSION

        assert isinstance(API_VERSION, str)
        assert len(API_VERSION) > 0

    def test_api_name(self):
        """Test API name constant."""
        from app.core.constants import API_NAME

        assert isinstance(API_NAME, str)
        assert "Floodingnaque" in API_NAME


class TestLocationConstants:
    """Tests for location-related constants."""

    def test_default_latitude(self):
        """Test default latitude constant."""
        from app.core.constants import DEFAULT_LATITUDE

        assert isinstance(DEFAULT_LATITUDE, float)
        assert -90 <= DEFAULT_LATITUDE <= 90

    def test_default_longitude(self):
        """Test default longitude constant."""
        from app.core.constants import DEFAULT_LONGITUDE

        assert isinstance(DEFAULT_LONGITUDE, float)
        assert -180 <= DEFAULT_LONGITUDE <= 180

    def test_default_location_name(self):
        """Test default location name constant."""
        from app.core.constants import DEFAULT_LOCATION_NAME

        assert isinstance(DEFAULT_LOCATION_NAME, str)
        assert "Parañaque" in DEFAULT_LOCATION_NAME


class TestWeatherDataLimits:
    """Tests for weather data limit constants."""

    def test_temperature_kelvin_limits(self):
        """Test temperature limits in Kelvin."""
        from app.core.constants import MAX_TEMPERATURE_KELVIN, MIN_TEMPERATURE_KELVIN

        assert MIN_TEMPERATURE_KELVIN < MAX_TEMPERATURE_KELVIN
        assert MIN_TEMPERATURE_KELVIN >= 0  # Absolute zero is ~0K

    def test_temperature_celsius_limits(self):
        """Test temperature limits in Celsius."""
        from app.core.constants import MAX_TEMPERATURE_CELSIUS, MIN_TEMPERATURE_CELSIUS

        assert MIN_TEMPERATURE_CELSIUS < MAX_TEMPERATURE_CELSIUS

    def test_humidity_limits(self):
        """Test humidity limits."""
        from app.core.constants import MAX_HUMIDITY, MIN_HUMIDITY

        assert MIN_HUMIDITY == 0.0
        assert MAX_HUMIDITY == 100.0

    def test_precipitation_limits(self):
        """Test precipitation limits."""
        from app.core.constants import MAX_PRECIPITATION, MIN_PRECIPITATION

        assert MIN_PRECIPITATION == 0.0
        assert MAX_PRECIPITATION > 0


class TestCoordinateLimits:
    """Tests for coordinate limit constants."""

    def test_latitude_limits(self):
        """Test latitude limits."""
        from app.core.constants import MAX_LATITUDE, MIN_LATITUDE

        assert MIN_LATITUDE == -90.0
        assert MAX_LATITUDE == 90.0

    def test_longitude_limits(self):
        """Test longitude limits."""
        from app.core.constants import MAX_LONGITUDE, MIN_LONGITUDE

        assert MIN_LONGITUDE == -180.0
        assert MAX_LONGITUDE == 180.0


class TestPaginationDefaults:
    """Tests for pagination default constants."""

    def test_default_page_size(self):
        """Test default page size constant."""
        from app.core.constants import DEFAULT_PAGE_SIZE

        assert DEFAULT_PAGE_SIZE > 0

    def test_min_page_size(self):
        """Test minimum page size constant."""
        from app.core.constants import MIN_PAGE_SIZE

        assert MIN_PAGE_SIZE >= 1

    def test_max_page_size(self):
        """Test maximum page size constant."""
        from app.core.constants import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

        assert MAX_PAGE_SIZE > 0
        assert MAX_PAGE_SIZE >= DEFAULT_PAGE_SIZE

    def test_default_offset(self):
        """Test default offset constant."""
        from app.core.constants import DEFAULT_OFFSET

        assert DEFAULT_OFFSET == 0


class TestRateLimitConstants:
    """Tests for rate limiting constants."""

    def test_default_rate_limit(self):
        """Test default rate limit constant."""
        from app.core.constants import DEFAULT_RATE_LIMIT

        assert DEFAULT_RATE_LIMIT > 0

    def test_default_rate_window(self):
        """Test default rate limit window constant."""
        from app.core.constants import DEFAULT_RATE_WINDOW_SECONDS

        assert DEFAULT_RATE_WINDOW_SECONDS > 0

    def test_rate_limit_predict(self):
        """Test predict endpoint rate limit."""
        from app.core.constants import RATE_LIMIT_PREDICT

        assert isinstance(RATE_LIMIT_PREDICT, str)

    def test_rate_limit_ingest(self):
        """Test ingest endpoint rate limit."""
        from app.core.constants import RATE_LIMIT_INGEST

        assert isinstance(RATE_LIMIT_INGEST, str)


class TestRiskLevels:
    """Tests for risk level constants."""

    def test_risk_levels_exist(self):
        """Test RISK_LEVELS constant exists."""
        from app.core.constants import RISK_LEVELS

        assert isinstance(RISK_LEVELS, dict)
        assert len(RISK_LEVELS) > 0

    def test_risk_levels_have_required_keys(self):
        """Test each risk level has required keys."""
        from app.core.constants import RISK_LEVELS

        required_keys = ["label", "color", "description"]

        for level, data in RISK_LEVELS.items():
            for key in required_keys:
                assert key in data, f"Risk level {level} missing {key}"

    def test_risk_levels_have_thresholds(self):
        """Test risk levels have threshold values."""
        from app.core.constants import RISK_LEVELS

        for level, data in RISK_LEVELS.items():
            assert "threshold_min" in data
            assert "threshold_max" in data
            assert data["threshold_min"] < data["threshold_max"]


class TestModelConfiguration:
    """Tests for model configuration constants."""

    def test_default_model_dir(self):
        """Test default model directory constant."""
        from app.core.constants import DEFAULT_MODEL_DIR

        assert isinstance(DEFAULT_MODEL_DIR, str)

    def test_default_model_name(self):
        """Test default model name constant."""
        from app.core.constants import DEFAULT_MODEL_NAME

        assert isinstance(DEFAULT_MODEL_NAME, str)

    def test_model_file_extension(self):
        """Test model file extension constant."""
        from app.core.constants import MODEL_FILE_EXTENSION

        assert MODEL_FILE_EXTENSION.startswith(".")


class TestDatabaseConfiguration:
    """Tests for database configuration constants."""

    def test_default_db_pool_size(self):
        """Test default database pool size."""
        from app.core.constants import DEFAULT_DB_POOL_SIZE

        assert DEFAULT_DB_POOL_SIZE > 0

    def test_default_db_max_overflow(self):
        """Test default database max overflow."""
        from app.core.constants import DEFAULT_DB_MAX_OVERFLOW

        assert DEFAULT_DB_MAX_OVERFLOW >= 0

    def test_default_db_pool_recycle(self):
        """Test default database pool recycle time."""
        from app.core.constants import DEFAULT_DB_POOL_RECYCLE

        assert DEFAULT_DB_POOL_RECYCLE > 0


class TestHTTPStatusCodes:
    """Tests for HTTP status code constants."""

    def test_http_ok(self):
        """Test HTTP 200 OK constant."""
        from app.core.constants import HTTP_OK

        assert HTTP_OK == 200

    def test_http_created(self):
        """Test HTTP 201 Created constant."""
        from app.core.constants import HTTP_CREATED

        assert HTTP_CREATED == 201

    def test_http_bad_request(self):
        """Test HTTP 400 Bad Request constant."""
        from app.core.constants import HTTP_BAD_REQUEST

        assert HTTP_BAD_REQUEST == 400

    def test_http_unauthorized(self):
        """Test HTTP 401 Unauthorized constant."""
        from app.core.constants import HTTP_UNAUTHORIZED

        assert HTTP_UNAUTHORIZED == 401

    def test_http_forbidden(self):
        """Test HTTP 403 Forbidden constant."""
        from app.core.constants import HTTP_FORBIDDEN

        assert HTTP_FORBIDDEN == 403

    def test_http_not_found(self):
        """Test HTTP 404 Not Found constant."""
        from app.core.constants import HTTP_NOT_FOUND

        assert HTTP_NOT_FOUND == 404

    def test_http_too_many_requests(self):
        """Test HTTP 429 Too Many Requests constant."""
        from app.core.constants import HTTP_TOO_MANY_REQUESTS

        assert HTTP_TOO_MANY_REQUESTS == 429

    def test_http_internal_error(self):
        """Test HTTP 500 Internal Server Error constant."""
        from app.core.constants import HTTP_INTERNAL_ERROR

        assert HTTP_INTERNAL_ERROR == 500

    def test_http_service_unavailable(self):
        """Test HTTP 503 Service Unavailable constant."""
        from app.core.constants import HTTP_SERVICE_UNAVAILABLE

        assert HTTP_SERVICE_UNAVAILABLE == 503


class TestDateTimeFormats:
    """Tests for date/time format constants."""

    def test_iso_date_format(self):
        """Test ISO date format constant."""
        from app.core.constants import ISO_DATE_FORMAT

        assert "%Y" in ISO_DATE_FORMAT
        assert "%m" in ISO_DATE_FORMAT
        assert "%d" in ISO_DATE_FORMAT

    def test_iso_datetime_format(self):
        """Test ISO datetime format constant."""
        from app.core.constants import ISO_DATETIME_FORMAT

        assert "%Y" in ISO_DATETIME_FORMAT
        assert "%H" in ISO_DATETIME_FORMAT


class TestAPITimeouts:
    """Tests for API timeout constants."""

    def test_api_timeout_default(self):
        """Test default API timeout constant."""
        from app.core.constants import API_TIMEOUT_DEFAULT

        assert API_TIMEOUT_DEFAULT > 0

    def test_api_timeout_weather(self):
        """Test weather API timeout constant."""
        from app.core.constants import API_TIMEOUT_WEATHER

        assert API_TIMEOUT_WEATHER > 0

    def test_api_retry_count(self):
        """Test API retry count constant."""
        from app.core.constants import API_RETRY_COUNT

        assert API_RETRY_COUNT >= 0


class TestSecurityDefaults:
    """Tests for security default constants."""

    def test_min_api_key_length(self):
        """Test minimum API key length constant."""
        from app.core.constants import MIN_API_KEY_LENGTH

        assert MIN_API_KEY_LENGTH >= 16

    def test_max_input_length(self):
        """Test maximum input length constant."""
        from app.core.constants import MAX_INPUT_LENGTH

        assert MAX_INPUT_LENGTH > 0

    def test_session_timeout_hours(self):
        """Test session timeout hours constant."""
        from app.core.constants import SESSION_TIMEOUT_HOURS

        assert SESSION_TIMEOUT_HOURS > 0

    def test_jwt_expiry_hours(self):
        """Test JWT expiry hours constant."""
        from app.core.constants import JWT_EXPIRY_HOURS

        assert JWT_EXPIRY_HOURS > 0


class TestLogLevels:
    """Tests for log level constants."""

    def test_log_levels_exist(self):
        """Test LOG_LEVELS constant exists."""
        from app.core.constants import LOG_LEVELS

        assert isinstance(LOG_LEVELS, dict)

    def test_log_levels_have_standard_levels(self):
        """Test LOG_LEVELS has standard logging levels."""
        from app.core.constants import LOG_LEVELS

        assert "DEBUG" in LOG_LEVELS
        assert "INFO" in LOG_LEVELS
        assert "WARNING" in LOG_LEVELS
        assert "ERROR" in LOG_LEVELS
        assert "CRITICAL" in LOG_LEVELS

    def test_log_levels_are_numeric(self):
        """Test log level values are numeric."""
        from app.core.constants import LOG_LEVELS

        for level, value in LOG_LEVELS.items():
            assert isinstance(value, int)
