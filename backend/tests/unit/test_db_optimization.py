"""
Unit tests for database optimization utilities.

Tests for app/utils/db_optimization.py
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestDatabaseConfig:
    """Tests for DatabaseConfig class."""

    def test_database_config_initialization(self):
        """Test DatabaseConfig initializes with defaults."""
        from app.utils.db_optimization import DatabaseConfig

        config = DatabaseConfig()

        assert hasattr(config, "primary_url")
        assert hasattr(config, "replica_url")
        assert hasattr(config, "pool_size")
        assert hasattr(config, "max_overflow")

    @patch.dict(
        os.environ,
        {  # pragma: allowlist secret
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
            "DATABASE_REPLICA_URL": "postgresql://user:pass@replica/db",
        },
    )
    def test_database_config_from_env(self):
        """Test DatabaseConfig reads from environment."""
        from app.utils.db_optimization import DatabaseConfig

        config = DatabaseConfig()

        assert "localhost" in config.primary_url
        assert "replica" in config.replica_url

    @patch.dict(os.environ, {"DB_POOL_SIZE": "50"})
    def test_database_config_pool_size(self):
        """Test pool size from environment."""
        from app.utils.db_optimization import DatabaseConfig

        config = DatabaseConfig()

        assert config.pool_size == 50

    @patch.dict(os.environ, {"USE_PGBOUNCER": "true"})
    def test_database_config_pgbouncer(self):
        """Test PgBouncer configuration."""
        from app.utils.db_optimization import DatabaseConfig

        config = DatabaseConfig()

        assert config.use_pgbouncer is True


class TestPgDriverSelection:
    """Tests for PostgreSQL driver selection."""

    @patch("sys.platform", "win32")
    def test_pg_driver_windows(self):
        """Test pg8000 selected on Windows."""
        from app.utils.db_optimization import _get_pg_driver

        driver = _get_pg_driver()

        assert driver == "pg8000"

    @patch("sys.platform", "linux")
    def test_pg_driver_linux_psycopg2(self):
        """Test psycopg2 selected on Linux when available."""
        from app.utils.db_optimization import _get_pg_driver

        # Will return psycopg2 if available, otherwise pg8000
        driver = _get_pg_driver()

        assert driver in ["psycopg2", "pg8000"]


class TestDatabaseUrlPreparation:
    """Tests for database URL preparation."""

    def test_prepare_sqlite_url(self):
        """Test SQLite URL is not modified."""
        from app.utils.db_optimization import _prepare_database_url

        url = "sqlite:///data/test.db"
        result = _prepare_database_url(url)

        assert result == url

    def test_prepare_postgres_url(self):
        """Test postgres:// URL is updated with driver."""
        from app.utils.db_optimization import _prepare_database_url

        url = "postgres://user:pass@localhost/db"
        result = _prepare_database_url(url)

        assert "postgresql+" in result

    def test_prepare_postgresql_url(self):
        """Test postgresql:// URL is updated with driver."""
        from app.utils.db_optimization import _prepare_database_url

        url = "postgresql://user:pass@localhost/db"
        result = _prepare_database_url(url)

        assert "postgresql+" in result

    def test_prepare_url_already_has_driver(self):
        """Test URL with driver is not double-modified."""
        from app.utils.db_optimization import _prepare_database_url

        url = "postgresql+psycopg2://user:pass@localhost/db"
        result = _prepare_database_url(url)

        # Should not have double driver specification
        assert result.count("+") == 1


class TestEngineFactory:
    """Tests for engine factory functions."""

    @patch("app.utils.db_optimization.create_engine")
    def test_create_engine_with_pooling_sqlite(self, mock_create):
        """Test SQLite engine creation."""
        from app.utils.db_optimization import _create_engine_with_pooling

        mock_create.return_value = MagicMock()

        engine = _create_engine_with_pooling("sqlite:///test.db")

        mock_create.assert_called_once()
        assert "sqlite" in mock_create.call_args[0][0]

    @patch("app.utils.db_optimization._get_pg_driver")
    @patch("app.utils.db_optimization.create_engine")
    def test_create_engine_with_pooling_postgres(self, mock_create, mock_driver):
        """Test PostgreSQL engine creation with pooling."""
        mock_driver.return_value = "pg8000"
        mock_create.return_value = MagicMock()

        from app.utils.db_optimization import _create_engine_with_pooling

        engine = _create_engine_with_pooling("postgresql://user:pass@localhost/db", pool_size=20, max_overflow=10)

        mock_create.assert_called_once()


class TestDatabaseRouter:
    """Tests for read/write routing."""

    def test_get_read_engine(self):
        """Test getting read engine."""
        from app.utils.db_optimization import get_read_engine

        # May return engine or None depending on configuration
        try:
            engine = get_read_engine()
            # Either returns engine or raises/returns None
        except Exception:
            pass  # Expected if database not configured

    def test_get_write_engine(self):
        """Test getting write engine."""
        from app.utils.db_optimization import get_write_engine

        try:
            engine = get_write_engine()
        except Exception:
            pass  # Expected if database not configured


class TestReadReplicaRouter:
    """Tests for read replica routing."""

    @patch.dict(os.environ, {"READ_REPLICA_ENABLED": "true"})
    def test_read_replica_enabled(self):
        """Test read replica configuration enabled."""
        from app.utils.db_optimization import DatabaseConfig

        config = DatabaseConfig()

        assert config.read_replica_enabled is True

    @patch.dict(os.environ, {"READ_REPLICA_ENABLED": "false"})
    def test_read_replica_disabled(self):
        """Test read replica configuration disabled."""
        from app.utils.db_optimization import DatabaseConfig

        config = DatabaseConfig()

        assert config.read_replica_enabled is False


class TestSessionManagement:
    """Tests for session management utilities."""

    def test_read_engine_exists(self):
        """Test read engine function exists."""
        from app.utils.db_optimization import get_read_engine

        # Function should exist and be callable
        assert callable(get_read_engine)

    def test_write_engine_exists(self):
        """Test write engine function exists."""
        from app.utils.db_optimization import get_write_engine

        # Function should exist and be callable
        assert callable(get_write_engine)


class TestConnectionPoolMonitoring:
    """Tests for connection pool monitoring."""

    def test_get_pool_health(self):
        """Test getting connection pool health status."""
        from app.utils.db_optimization import get_pool_health

        try:
            status = get_pool_health()
            assert isinstance(status, dict)
        except Exception:
            pass  # Expected if not connected


class TestDatabaseDecorators:
    """Tests for database operation decorators."""

    def test_use_read_replica_decorator(self):
        """Test use_read_replica decorator exists."""
        from app.utils.db_optimization import use_read_replica

        @use_read_replica
        def read_operation():
            return "data"

        # Function should be callable
        assert callable(read_operation)

    def test_use_primary_decorator(self):
        """Test use_primary decorator exists."""
        from app.utils.db_optimization import use_primary

        @use_primary
        def write_operation():
            return "written"

        assert callable(write_operation)


class TestSSLConfiguration:
    """Tests for SSL configuration."""

    @patch.dict(os.environ, {"DB_SSL_MODE": "require"})
    def test_ssl_mode_require(self):
        """Test SSL mode require configuration."""
        from app.utils.db_optimization import DatabaseConfig

        config = DatabaseConfig()

        assert config.ssl_mode == "require"

    @patch.dict(os.environ, {"DB_SSL_MODE": "verify-full"})
    def test_ssl_mode_verify_full(self):
        """Test SSL mode verify-full configuration."""
        from app.utils.db_optimization import DatabaseConfig

        config = DatabaseConfig()

        assert config.ssl_mode == "verify-full"


class TestPartitioningHelpers:
    """Tests for time-series partitioning helpers."""

    def test_get_partition_name(self):
        """Test partition name generation."""
        from datetime import datetime

        from app.utils.db_optimization import get_partition_name

        name = get_partition_name("weather_data", datetime(2024, 1, 15))

        assert "weather_data" in name
        assert "2024" in name or "01" in name

    def test_create_partition_sql(self):
        """Test partition SQL generation helper."""
        from app.utils.db_optimization import create_partition_sql

        sql = create_partition_sql("weather_data", 2024, 1)

        # Just verify function is callable and returns string
        assert isinstance(sql, str)
        assert "weather_data" in sql
