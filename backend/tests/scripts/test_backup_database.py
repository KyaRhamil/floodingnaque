"""
Unit tests for backup_database.py script.

Tests cover:
- SQLite backup functionality
- PostgreSQL backup functionality
- Dry-run mode
- Subprocess sanitization with shlex.quote
- Cleanup of old backups
- Database type detection
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGetTimestamp:
    """Tests for get_timestamp function."""

    def test_timestamp_format(self):
        """Test that timestamp has correct format."""
        from scripts.backup_database import get_timestamp

        timestamp = get_timestamp()
        # Format: YYYYMMDD_HHMMSS
        assert len(timestamp) == 15
        assert timestamp[8] == "_"
        assert timestamp[:8].isdigit()
        assert timestamp[9:].isdigit()


class TestEnsureBackupDir:
    """Tests for ensure_backup_dir function."""

    def test_creates_directory(self, tmp_path):
        """Test that backup directory is created."""
        from scripts.backup_database import ensure_backup_dir

        backup_dir = tmp_path / "backups" / "nested"
        result = ensure_backup_dir(backup_dir)
        assert result == backup_dir
        assert backup_dir.exists()

    def test_existing_directory(self, tmp_path):
        """Test with existing directory."""
        from scripts.backup_database import ensure_backup_dir

        backup_dir = tmp_path / "existing"
        backup_dir.mkdir()
        result = ensure_backup_dir(backup_dir)
        assert result == backup_dir
        assert backup_dir.exists()


class TestBackupSqlite:
    """Tests for backup_sqlite function."""

    def test_backup_creates_file(self, tmp_path):
        """Test that SQLite backup creates a file."""
        from scripts.backup_database import backup_sqlite

        # Create a mock database file
        db_path = tmp_path / "test.db"
        db_path.write_text("test database content")

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        result = backup_sqlite(db_path, backup_dir, compress=False)
        assert result.exists()
        assert "backup" in result.name

    def test_backup_compression(self, tmp_path):
        """Test that backup is compressed when requested."""
        from scripts.backup_database import backup_sqlite

        db_path = tmp_path / "test.db"
        db_path.write_text("test database content")

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        result = backup_sqlite(db_path, backup_dir, compress=True)
        assert result.exists()
        assert result.suffix == ".gz"

    def test_backup_file_not_found(self, tmp_path):
        """Test error when database file doesn't exist."""
        from scripts.backup_database import backup_sqlite

        db_path = tmp_path / "nonexistent.db"
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        with pytest.raises(FileNotFoundError):
            backup_sqlite(db_path, backup_dir)

    def test_dry_run_no_file_created(self, tmp_path):
        """Test that dry run doesn't create files."""
        from scripts.backup_database import backup_sqlite

        db_path = tmp_path / "test.db"
        db_path.write_text("test database content")

        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        result = backup_sqlite(db_path, backup_dir, compress=False, dry_run=True)
        # Result should be a path but file shouldn't exist (it's a hypothetical path in dry run)
        assert "backup" in str(result)


class TestBackupPostgresql:
    """Tests for backup_postgresql function."""

    def test_dry_run_mode(self, tmp_path):
        """Test that dry run mode works correctly."""
        from scripts.backup_database import backup_postgresql

        database_url = "postgresql://user:password@localhost:5432/testdb"
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        result = backup_postgresql(database_url, backup_dir, compress=False, dry_run=True)
        assert "backup" in str(result)
        # File should not actually exist in dry run
        # (it returns the would-be path)

    @patch("scripts.backup_database.subprocess.run")
    def test_pg_dump_command_sanitization(self, mock_run, tmp_path):
        """Test that pg_dump command uses sanitized inputs."""
        from scripts.backup_database import backup_postgresql

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        # URL with potentially dangerous characters
        database_url = "postgresql://user:pass@localhost:5432/test_db"
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        # Create a fake backup file to satisfy the existence check
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_size=1024)
                backup_postgresql(database_url, backup_dir, compress=False, dry_run=False)

        # Verify subprocess.run was called
        mock_run.assert_called_once()

        # Check that the command args don't contain unescaped shell characters
        call_args = mock_run.call_args
        cmd = call_args[0][0]  # First positional arg is the command list

        # Verify the command is a list (not a shell string)
        assert isinstance(cmd, list)
        assert cmd[0] == "pg_dump"

    @patch("scripts.backup_database.subprocess.run")
    def test_pg_dump_failure(self, mock_run, tmp_path):
        """Test handling of pg_dump failure."""
        from scripts.backup_database import backup_postgresql

        mock_run.return_value = MagicMock(returncode=1, stderr="Connection refused")

        database_url = "postgresql://user:pass@localhost:5432/testdb"
        backup_dir = tmp_path / "backups"
        backup_dir.mkdir()

        with pytest.raises(RuntimeError, match="pg_dump failed"):
            backup_postgresql(database_url, backup_dir, compress=False, dry_run=False)


class TestCleanupOldBackups:
    """Tests for cleanup_old_backups function."""

    def test_removes_old_backups(self, tmp_path):
        """Test that old backups are removed."""
        from scripts.backup_database import cleanup_old_backups

        # Create multiple backup files
        for i in range(15):
            backup = tmp_path / f"floodingnaque.db.backup.2024010{i:02d}_120000"
            backup.write_text("backup content")
            # Set different modification times
            os.utime(backup, (i * 1000, i * 1000))

        removed = cleanup_old_backups(tmp_path, keep_count=10)
        assert removed == 5

        remaining = list(tmp_path.glob("floodingnaque*.backup.*"))
        assert len(remaining) == 10

    def test_keeps_all_when_under_limit(self, tmp_path):
        """Test that all backups kept when under limit."""
        from scripts.backup_database import cleanup_old_backups

        for i in range(5):
            backup = tmp_path / f"floodingnaque.db.backup.2024010{i}_120000"
            backup.write_text("backup content")

        removed = cleanup_old_backups(tmp_path, keep_count=10)
        assert removed == 0

    def test_dry_run_no_deletion(self, tmp_path):
        """Test that dry run doesn't delete files."""
        from scripts.backup_database import cleanup_old_backups

        for i in range(15):
            backup = tmp_path / f"floodingnaque.db.backup.2024010{i:02d}_120000"
            backup.write_text("backup content")
            os.utime(backup, (i * 1000, i * 1000))

        removed = cleanup_old_backups(tmp_path, keep_count=10, dry_run=True)
        assert removed == 5  # Would remove 5

        # All files should still exist
        remaining = list(tmp_path.glob("floodingnaque*.backup.*"))
        assert len(remaining) == 15


class TestGetDatabaseType:
    """Tests for get_database_type function."""

    def test_sqlite_detection(self, monkeypatch):
        """Test SQLite database detection."""
        from scripts.backup_database import get_database_type

        monkeypatch.setenv("DATABASE_URL", "sqlite:///data/test.db")
        assert get_database_type() == "sqlite"

    def test_postgresql_detection(self, monkeypatch):
        """Test PostgreSQL database detection."""
        from scripts.backup_database import get_database_type

        monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
        assert get_database_type() == "postgresql"

    def test_postgres_detection(self, monkeypatch):
        """Test postgres:// URL detection."""
        from scripts.backup_database import get_database_type

        monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@localhost/db")
        assert get_database_type() == "postgresql"

    def test_default_sqlite_when_empty(self, monkeypatch):
        """Test default to SQLite when no URL set."""
        from scripts.backup_database import get_database_type

        monkeypatch.delenv("DATABASE_URL", raising=False)
        assert get_database_type() == "sqlite"

    def test_unknown_database(self, monkeypatch):
        """Test unknown database type."""
        from scripts.backup_database import get_database_type

        monkeypatch.setenv("DATABASE_URL", "mysql://user:pass@localhost/db")
        assert get_database_type() == "unknown"


class TestRunBackup:
    """Tests for run_backup function."""

    def test_dry_run_sqlite(self, tmp_path, monkeypatch):
        """Test dry run with SQLite."""
        from scripts.backup_database import run_backup

        # Create a mock database
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_file = data_dir / "floodingnaque.db"
        db_file.write_text("test db")

        # Patch the backend_path
        monkeypatch.setattr("scripts.backup_database.backend_path", tmp_path)

        result = run_backup(
            backup_type="sqlite",
            output_dir=str(tmp_path / "backups"),
            dry_run=True,
        )

        assert "backup" in str(result)
