"""
Unit tests for core config module.

Tests for app/core/config.py
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestIsDebugMode:
    """Tests for is_debug_mode function."""

    def test_is_debug_mode_function_exists(self):
        """Test is_debug_mode function exists."""
        from app.core.config import is_debug_mode

        assert callable(is_debug_mode)

    @patch.dict(os.environ, {"FLASK_DEBUG": "true"})
    def test_debug_mode_enabled(self):
        """Test debug mode when FLASK_DEBUG is true."""
        from app.core.config import is_debug_mode

        assert is_debug_mode() is True

    @patch.dict(os.environ, {"FLASK_DEBUG": "false"})
    def test_debug_mode_disabled(self):
        """Test debug mode when FLASK_DEBUG is false."""
        from app.core.config import is_debug_mode

        assert is_debug_mode() is False


class TestSecretKeyGeneration:
    """Tests for _get_secret_key function."""

    @patch.dict(
        os.environ, {"SECRET_KEY": "test-secret-key-12345678901234567890", "FLASK_DEBUG": "true"}
    )  # pragma: allowlist secret
    def test_secret_key_from_env(self):
        """Test secret key from environment variable."""
        from app.core.config import _get_secret_key

        key = _get_secret_key()
        assert key == "test-secret-key-12345678901234567890"

    @patch.dict(os.environ, {"SECRET_KEY": "", "FLASK_DEBUG": "true"}, clear=False)
    def test_secret_key_generated_in_debug(self):
        """Test secret key is generated in debug mode."""
        from app.core.config import _get_secret_key

        key = _get_secret_key()
        assert len(key) >= 32  # Generated key should be long enough

    @patch.dict(os.environ, {"SECRET_KEY": "", "FLASK_DEBUG": "false"}, clear=False)
    def test_secret_key_required_in_production(self):
        """Test secret key is required in production."""
        from app.core.config import _get_secret_key

        with pytest.raises(ValueError, match="SECRET_KEY must be set"):
            _get_secret_key()


class TestJWTSecretKey:
    """Tests for _get_jwt_secret_key function."""

    @patch.dict(
        os.environ, {"JWT_SECRET_KEY": "jwt-test-key-1234567890123456", "FLASK_DEBUG": "true"}
    )  # pragma: allowlist secret
    def test_jwt_secret_key_from_env(self):
        """Test JWT secret key from environment variable."""
        from app.core.config import _get_jwt_secret_key

        key = _get_jwt_secret_key()
        assert key == "jwt-test-key-1234567890123456"


class TestDatabaseURL:
    """Tests for _get_database_url function."""

    @patch.dict(
        os.environ, {"DATABASE_URL": "postgresql://user:pass@localhost/db", "FLASK_DEBUG": "true"}
    )  # pragma: allowlist secret
    def test_database_url_from_env(self):
        """Test database URL from environment variable."""
        from app.core.config import _get_database_url

        url = _get_database_url()
        assert "postgresql" in url

    @patch.dict(os.environ, {"DATABASE_URL": "", "FLASK_DEBUG": "true", "APP_ENV": "development"})
    def test_database_url_fallback_development(self):
        """Test database URL fallback in development."""
        from app.core.config import _get_database_url

        url = _get_database_url()
        assert "sqlite" in url

    @patch.dict(os.environ, {"DATABASE_URL": "", "FLASK_DEBUG": "false", "APP_ENV": "production"})
    def test_database_url_required_production(self):
        """Test database URL required in production."""
        from app.core.config import _get_database_url

        with pytest.raises(ValueError, match="DATABASE_URL must be set"):
            _get_database_url()


class TestDBSSLMode:
    """Tests for _get_db_ssl_mode function."""

    @patch.dict(os.environ, {"DB_SSL_MODE": "verify-full"})
    def test_ssl_mode_from_env(self):
        """Test SSL mode from environment variable."""
        from app.core.config import _get_db_ssl_mode

        mode = _get_db_ssl_mode()
        assert mode == "verify-full"

    @patch.dict(os.environ, {"DB_SSL_MODE": "", "APP_ENV": "production"})
    def test_ssl_mode_default_production(self):
        """Test SSL mode default in production."""
        from app.core.config import _get_db_ssl_mode

        mode = _get_db_ssl_mode()
        assert mode == "verify-full"

    @patch.dict(os.environ, {"DB_SSL_MODE": "", "APP_ENV": "development"})
    def test_ssl_mode_default_development(self):
        """Test SSL mode default in development."""
        from app.core.config import _get_db_ssl_mode

        mode = _get_db_ssl_mode()
        assert mode == "require"


class TestConfigDataclass:
    """Tests for Config dataclass."""

    def test_config_dataclass_exists(self):
        """Test Config dataclass exists."""
        from app.core.config import Config

        assert Config is not None

    @patch.dict(os.environ, {"FLASK_DEBUG": "true", "SECRET_KEY": "test-secret-key-12345"})  # pragma: allowlist secret
    def test_config_has_required_attributes(self):
        """Test Config has required attributes."""
        from app.core.config import Config

        # Check that the class has expected attributes
        config = Config()
        assert hasattr(config, "SECRET_KEY") or hasattr(config, "secret_key")


class TestBaseDir:
    """Tests for BASE_DIR constant."""

    def test_base_dir_exists(self):
        """Test BASE_DIR constant exists."""
        from app.core.config import BASE_DIR

        assert BASE_DIR is not None

    def test_base_dir_is_path(self):
        """Test BASE_DIR is a Path object."""
        from pathlib import Path

        from app.core.config import BASE_DIR

        assert isinstance(BASE_DIR, Path)

    def test_base_dir_exists_on_filesystem(self):
        """Test BASE_DIR points to existing directory."""
        from app.core.config import BASE_DIR

        assert BASE_DIR.exists()
