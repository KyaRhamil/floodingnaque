"""
Unit tests for core exceptions module.

Tests for app/core/exceptions.py which re-exports from app/utils/api_errors.py
"""

from unittest.mock import MagicMock, patch

import pytest


class TestAppException:
    """Tests for base AppException class."""

    def test_app_exception_exists(self):
        """Test AppException class exists."""
        from app.core.exceptions import AppException

        assert AppException is not None

    def test_app_exception_inherits_from_exception(self):
        """Test AppException inherits from Exception."""
        from app.core.exceptions import AppException

        assert issubclass(AppException, Exception)

    def test_app_exception_with_message(self):
        """Test AppException with message."""
        from app.core.exceptions import AppException

        exc = AppException("Test error message")

        assert exc.message == "Test error message"
        assert str(exc) == "Test error message"

    def test_app_exception_with_status_code(self):
        """Test AppException with status code."""
        from app.core.exceptions import AppException

        exc = AppException("Error", status_code=500)

        assert exc.status_code == 500

    def test_app_exception_default_status_code(self):
        """Test AppException default status code is 400."""
        from app.core.exceptions import AppException

        exc = AppException("Error")

        assert exc.status_code == 400

    def test_app_exception_to_dict(self):
        """Test AppException to_dict method."""
        from app.core.exceptions import AppException

        exc = AppException("Test error", status_code=400)

        result = exc.to_dict()

        assert isinstance(result, dict)
        assert "error" in result
        assert result["error"]["status"] == 400


class TestValidationError:
    """Tests for ValidationError class."""

    def test_validation_error_exists(self):
        """Test ValidationError class exists."""
        from app.core.exceptions import ValidationError

        assert ValidationError is not None

    def test_validation_error_default_message(self):
        """Test ValidationError with default message."""
        from app.core.exceptions import ValidationError

        exc = ValidationError()

        assert exc.status_code == 400
        assert "Validation" in exc.message or "validation" in exc.message.lower()

    def test_validation_error_with_field_errors(self):
        """Test ValidationError with field errors."""
        from app.core.exceptions import ValidationError

        field_errors = [{"field": "temperature", "message": "Must be a number"}]

        exc = ValidationError("Validation failed", field_errors=field_errors)

        assert exc.errors == field_errors

    def test_validation_error_from_fields(self):
        """Test ValidationError.from_fields class method."""
        from app.core.exceptions import ValidationError

        field_errors = {"temperature": "Required field", "humidity": "Must be positive"}

        exc = ValidationError.from_fields(field_errors)

        assert len(exc.errors) == 2


class TestNotFoundError:
    """Tests for NotFoundError class."""

    def test_not_found_error_exists(self):
        """Test NotFoundError class exists."""
        from app.core.exceptions import NotFoundError

        assert NotFoundError is not None

    def test_not_found_error_status_code(self):
        """Test NotFoundError has 404 status code."""
        from app.core.exceptions import NotFoundError

        exc = NotFoundError("Resource not found")

        assert exc.status_code == 404


class TestUnauthorizedError:
    """Tests for UnauthorizedError class."""

    def test_unauthorized_error_exists(self):
        """Test UnauthorizedError class exists."""
        from app.core.exceptions import UnauthorizedError

        assert UnauthorizedError is not None

    def test_unauthorized_error_status_code(self):
        """Test UnauthorizedError has 401 status code."""
        from app.core.exceptions import UnauthorizedError

        exc = UnauthorizedError("Not authenticated")

        assert exc.status_code == 401


class TestAuthenticationError:
    """Tests for AuthenticationError class."""

    def test_authentication_error_exists(self):
        """Test AuthenticationError class exists."""
        from app.core.exceptions import AuthenticationError

        assert AuthenticationError is not None

    def test_authentication_error_status_code(self):
        """Test AuthenticationError has 401 status code."""
        from app.core.exceptions import AuthenticationError

        exc = AuthenticationError("Authentication failed")

        assert exc.status_code == 401


class TestForbiddenError:
    """Tests for ForbiddenError class."""

    def test_forbidden_error_exists(self):
        """Test ForbiddenError class exists."""
        from app.core.exceptions import ForbiddenError

        assert ForbiddenError is not None

    def test_forbidden_error_status_code(self):
        """Test ForbiddenError has 403 status code."""
        from app.core.exceptions import ForbiddenError

        exc = ForbiddenError("Access denied")

        assert exc.status_code == 403


class TestConflictError:
    """Tests for ConflictError class."""

    def test_conflict_error_exists(self):
        """Test ConflictError class exists."""
        from app.core.exceptions import ConflictError

        assert ConflictError is not None

    def test_conflict_error_status_code(self):
        """Test ConflictError has 409 status code."""
        from app.core.exceptions import ConflictError

        exc = ConflictError("Resource already exists")

        assert exc.status_code == 409


class TestRateLimitExceededError:
    """Tests for RateLimitExceededError class."""

    def test_rate_limit_error_exists(self):
        """Test RateLimitExceededError class exists."""
        from app.core.exceptions import RateLimitExceededError

        assert RateLimitExceededError is not None

    def test_rate_limit_error_status_code(self):
        """Test RateLimitExceededError has 429 status code."""
        from app.core.exceptions import RateLimitExceededError

        exc = RateLimitExceededError("Too many requests")

        assert exc.status_code == 429


class TestInternalServerError:
    """Tests for InternalServerError class."""

    def test_internal_server_error_exists(self):
        """Test InternalServerError class exists."""
        from app.core.exceptions import InternalServerError

        assert InternalServerError is not None

    def test_internal_server_error_status_code(self):
        """Test InternalServerError has 500 status code."""
        from app.core.exceptions import InternalServerError

        exc = InternalServerError("Server error")

        assert exc.status_code == 500


class TestServiceUnavailableError:
    """Tests for ServiceUnavailableError class."""

    def test_service_unavailable_error_exists(self):
        """Test ServiceUnavailableError class exists."""
        from app.core.exceptions import ServiceUnavailableError

        assert ServiceUnavailableError is not None

    def test_service_unavailable_error_status_code(self):
        """Test ServiceUnavailableError has 503 status code."""
        from app.core.exceptions import ServiceUnavailableError

        exc = ServiceUnavailableError("Service is down")

        assert exc.status_code == 503


class TestBadRequestError:
    """Tests for BadRequestError class."""

    def test_bad_request_error_exists(self):
        """Test BadRequestError class exists."""
        from app.core.exceptions import BadRequestError

        assert BadRequestError is not None

    def test_bad_request_error_status_code(self):
        """Test BadRequestError has 400 status code."""
        from app.core.exceptions import BadRequestError

        exc = BadRequestError("Bad request")

        assert exc.status_code == 400


class TestModelError:
    """Tests for ModelError class."""

    def test_model_error_exists(self):
        """Test ModelError class exists."""
        from app.core.exceptions import ModelError

        assert ModelError is not None


class TestExternalServiceError:
    """Tests for ExternalServiceError class."""

    def test_external_service_error_exists(self):
        """Test ExternalServiceError class exists."""
        from app.core.exceptions import ExternalServiceError

        assert ExternalServiceError is not None


class TestExternalAPIError:
    """Tests for ExternalAPIError class."""

    def test_external_api_error_exists(self):
        """Test ExternalAPIError class exists."""
        from app.core.exceptions import ExternalAPIError

        assert ExternalAPIError is not None


class TestDatabaseError:
    """Tests for DatabaseError class."""

    def test_database_error_exists(self):
        """Test DatabaseError class exists."""
        from app.core.exceptions import DatabaseError

        assert DatabaseError is not None


class TestConfigurationError:
    """Tests for ConfigurationError class."""

    def test_configuration_error_exists(self):
        """Test ConfigurationError class exists."""
        from app.core.exceptions import ConfigurationError

        assert ConfigurationError is not None


class TestResponseHelpers:
    """Tests for response helper functions."""

    def test_api_error_exists(self):
        """Test api_error function exists."""
        from app.core.exceptions import api_error

        assert callable(api_error)

    def test_api_error_from_exception_exists(self):
        """Test api_error_from_exception function exists."""
        from app.core.exceptions import api_error_from_exception

        assert callable(api_error_from_exception)

    def test_api_success_exists(self):
        """Test api_success function exists."""
        from app.core.exceptions import api_success

        assert callable(api_success)

    def test_api_created_exists(self):
        """Test api_created function exists."""
        from app.core.exceptions import api_created

        assert callable(api_created)

    def test_api_accepted_exists(self):
        """Test api_accepted function exists."""
        from app.core.exceptions import api_accepted

        assert callable(api_accepted)


class TestAllExports:
    """Tests for module exports."""

    def test_all_exports_defined(self):
        """Test __all__ exports are defined."""
        from app.core import exceptions

        assert hasattr(exceptions, "__all__")
        assert len(exceptions.__all__) > 0

    def test_all_exports_importable(self):
        """Test all exported names are importable."""
        import app.core.exceptions as exc_module
        from app.core.exceptions import __all__

        for name in __all__:
            assert hasattr(exc_module, name), f"Missing export: {name}"
