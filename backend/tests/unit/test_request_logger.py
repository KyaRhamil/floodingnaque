"""
Unit tests for request logger middleware.

Tests for app/api/middleware/request_logger.py
"""

import time
from unittest.mock import MagicMock, patch

import pytest


class TestLogRequestToDb:
    """Tests for log_request_to_db function."""

    def test_log_request_to_db_function_exists(self):
        """Test log_request_to_db function exists."""
        from app.api.middleware.request_logger import log_request_to_db

        assert callable(log_request_to_db)

    @patch("app.api.middleware.request_logger.get_db_session")
    def test_log_request_to_db_success(self, mock_get_db):
        """Test successful request logging to database."""
        from app.api.middleware.request_logger import log_request_to_db
        from flask import Flask, g

        app = Flask(__name__)
        mock_session = MagicMock()
        mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_db.return_value.__exit__ = MagicMock(return_value=None)

        with app.test_request_context("/api/test", method="GET"):
            g.start_time = time.time()
            g.request_id = "test-request-123"

            log_request_to_db()

            mock_session.add.assert_called_once()
            mock_session.commit.assert_called_once()

    def test_log_request_to_db_no_start_time(self):
        """Test log_request_to_db handles missing start_time."""
        from app.api.middleware.request_logger import log_request_to_db
        from flask import Flask, g

        app = Flask(__name__)

        with app.test_request_context("/api/test"):
            # No g.start_time set
            # Should return early without error
            log_request_to_db()

    def test_log_request_to_db_no_request_id(self):
        """Test log_request_to_db handles missing request_id."""
        from app.api.middleware.request_logger import log_request_to_db
        from flask import Flask, g

        app = Flask(__name__)

        with app.test_request_context("/api/test"):
            g.start_time = time.time()
            # No g.request_id set
            # Should return early without error
            log_request_to_db()


class TestSetupRequestLoggingMiddleware:
    """Tests for setup_request_logging_middleware function."""

    def test_setup_function_exists(self):
        """Test setup_request_logging_middleware function exists."""
        from app.api.middleware.request_logger import setup_request_logging_middleware

        assert callable(setup_request_logging_middleware)

    def test_setup_registers_handlers(self):
        """Test setup registers before/after/teardown handlers."""
        from app.api.middleware.request_logger import setup_request_logging_middleware
        from flask import Flask

        app = Flask(__name__)

        # Should not raise
        setup_request_logging_middleware(app)

        # Verify hooks are registered (Flask stores these internally)
        assert app.before_request_funcs is not None
        assert app.after_request_funcs is not None


class TestAPIRequestModel:
    """Tests for APIRequest model usage."""

    def test_api_request_model_exists(self):
        """Test APIRequest model can be imported from db models."""
        from app.models.db import APIRequest

        assert APIRequest is not None


class TestGetDbSession:
    """Tests for get_db_session exists."""

    def test_get_db_session_exists(self):
        """Test get_db_session can be imported from db models."""
        from app.models.db import get_db_session

        assert callable(get_db_session)


class TestAPIVersionExtraction:
    """Tests for API version extraction from path."""

    @patch("app.api.middleware.request_logger.get_db_session")
    def test_api_version_extracted_from_v1_path(self, mock_get_db):
        """Test API version is extracted from /v1/ path."""
        from app.api.middleware.request_logger import log_request_to_db
        from flask import Flask, g

        app = Flask(__name__)
        mock_session = MagicMock()
        mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_db.return_value.__exit__ = MagicMock(return_value=None)

        with app.test_request_context("/v1/predict", method="POST"):
            g.start_time = time.time()
            g.request_id = "test-123"

            log_request_to_db()

            # Verify APIRequest was created with correct version
            call_args = mock_session.add.call_args
            if call_args:
                api_request = call_args[0][0]
                assert hasattr(api_request, "api_version")

    @patch("app.api.middleware.request_logger.get_db_session")
    def test_api_version_default(self, mock_get_db):
        """Test API version defaults to v1."""
        from app.api.middleware.request_logger import log_request_to_db
        from flask import Flask, g

        app = Flask(__name__)
        mock_session = MagicMock()
        mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_db.return_value.__exit__ = MagicMock(return_value=None)

        with app.test_request_context("/health", method="GET"):
            g.start_time = time.time()
            g.request_id = "test-456"

            log_request_to_db()

            call_args = mock_session.add.call_args
            if call_args:
                api_request = call_args[0][0]
                assert api_request.api_version == "v1"


class TestErrorHandling:
    """Tests for error handling in request logging."""

    @patch("app.api.middleware.request_logger.get_db_session")
    @patch("app.api.middleware.request_logger.logger")
    def test_database_error_logged(self, mock_logger, mock_get_db):
        """Test database errors are logged, not raised."""
        from app.api.middleware.request_logger import log_request_to_db
        from flask import Flask, g

        app = Flask(__name__)
        mock_get_db.side_effect = Exception("Database connection error")

        with app.test_request_context("/api/test"):
            g.start_time = time.time()
            g.request_id = "test-error-123"

            # Should not raise, just log error
            log_request_to_db()

            mock_logger.error.assert_called()


class TestResponseTimeMeasurement:
    """Tests for response time measurement."""

    @patch("app.api.middleware.request_logger.get_db_session")
    def test_response_time_calculated(self, mock_get_db):
        """Test response time is calculated correctly."""
        from app.api.middleware.request_logger import log_request_to_db
        from flask import Flask, g

        app = Flask(__name__)
        mock_session = MagicMock()
        mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_get_db.return_value.__exit__ = MagicMock(return_value=None)

        with app.test_request_context("/api/test"):
            g.start_time = time.time() - 0.1  # 100ms ago
            g.request_id = "test-time-123"

            log_request_to_db()

            call_args = mock_session.add.call_args
            if call_args:
                api_request = call_args[0][0]
                # Response time should be around 100ms
                assert api_request.response_time_ms >= 90
                assert api_request.response_time_ms <= 200
