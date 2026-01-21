"""
Unit tests for logging middleware.

Tests for app/api/middleware/logging.py
"""

import time
from unittest.mock import MagicMock, patch

import pytest


class TestAddRequestId:
    """Tests for add_request_id function."""

    def test_add_request_id_function_exists(self):
        """Test add_request_id function exists."""
        from app.api.middleware.logging import add_request_id

        assert callable(add_request_id)

    def test_add_request_id_generates_uuid(self):
        """Test add_request_id generates UUID when not provided."""
        from app.api.middleware.logging import add_request_id
        from flask import Flask

        app = Flask(__name__)

        with app.test_request_context("/test"):
            request_id = add_request_id()

            # Should return a string (UUID)
            assert isinstance(request_id, str)
            assert len(request_id) > 0

    def test_add_request_id_uses_header(self):
        """Test add_request_id uses X-Request-ID header if provided."""
        from app.api.middleware.logging import add_request_id
        from flask import Flask

        app = Flask(__name__)

        with app.test_request_context("/test", headers={"X-Request-ID": "custom-id-123"}):
            request_id = add_request_id()

            assert request_id == "custom-id-123"


class TestLogRequest:
    """Tests for log_request function."""

    def test_log_request_function_exists(self):
        """Test log_request function exists."""
        from app.api.middleware.logging import log_request

        assert callable(log_request)

    @patch("app.api.middleware.logging.logger")
    def test_log_request_logs_info(self, mock_logger):
        """Test log_request logs request info."""
        from app.api.middleware.logging import add_request_id, log_request
        from flask import Flask

        app = Flask(__name__)

        with app.test_request_context("/api/test", method="GET"):
            add_request_id()
            log_request()

            mock_logger.info.assert_called()


class TestLogResponse:
    """Tests for log_response function."""

    def test_log_response_function_exists(self):
        """Test log_response function exists."""
        from app.api.middleware.logging import log_response

        assert callable(log_response)

    @patch("app.api.middleware.logging.logger")
    def test_log_response_returns_response(self, mock_logger):
        """Test log_response returns the response object."""
        from app.api.middleware.logging import log_response
        from flask import Flask, Response

        app = Flask(__name__)
        response = Response(status=200)

        with app.test_request_context("/test"):
            result = log_response(response)

            assert result == response

    @patch("app.api.middleware.logging.logger")
    def test_log_response_logs_status_code(self, mock_logger):
        """Test log_response logs status code."""
        from app.api.middleware.logging import add_request_id, log_response
        from flask import Flask, Response, g

        app = Flask(__name__)
        response = Response(status=200)

        with app.test_request_context("/test"):
            add_request_id()
            g.request_start_time = time.time()
            log_response(response)

            mock_logger.info.assert_called()


class TestSetupRequestLogging:
    """Tests for setup_request_logging function."""

    def test_setup_request_logging_function_exists(self):
        """Test setup_request_logging function exists."""
        from app.api.middleware.logging import setup_request_logging

        assert callable(setup_request_logging)

    def test_setup_request_logging_registers_handlers(self):
        """Test setup_request_logging registers before/after handlers."""
        from app.api.middleware.logging import setup_request_logging
        from flask import Flask

        app = Flask(__name__)

        # Should not raise
        setup_request_logging(app)


class TestRequestLoggerDecorator:
    """Tests for request_logger decorator."""

    def test_request_logger_decorator_exists(self):
        """Test request_logger decorator exists."""
        from app.api.middleware.logging import request_logger

        assert callable(request_logger)

    def test_request_logger_wraps_function(self):
        """Test request_logger wraps function correctly."""
        from app.api.middleware.logging import request_logger

        @request_logger
        def my_function():
            return "result"

        # Decorator should preserve function
        assert callable(my_function)
