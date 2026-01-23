"""
Database Backup Utility for Floodingnaque.

Provides automated and manual database backup functionality for both
SQLite (development) and PostgreSQL (production) databases.

Usage:
    python scripts/backup_database.py                    # Auto-detect and backup
    python scripts/backup_database.py --sqlite           # SQLite backup only
    python scripts/backup_database.py --postgres         # PostgreSQL backup only
    python scripts/backup_database.py --output ./backups # Custom output directory
    python scripts/backup_database.py --max-backups 5    # Keep only 5 most recent backups
    python scripts/backup_database.py --verbose          # Enable verbose logging
    python scripts/backup_database.py --force            # Overwrite without confirmation

Environment Variables:
    FLOODINGNAQUE_BACKUP_DIR    Default backup directory
    FLOODINGNAQUE_MAX_BACKUPS   Default max backups to keep
    FLOODINGNAQUE_LOG_LEVEL     Logging level (DEBUG, INFO, etc.)
"""

import argparse
import logging
import os
import shlex
import shutil
import subprocess  # nosec B404
import sys
from datetime import datetime
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from scripts.cli_utils import (
    add_common_arguments,
    get_backup_dir,
    get_env_int,
    setup_logging,
)

logger = logging.getLogger(__name__)


def get_timestamp() -> str:
    """Generate timestamp string for backup files."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_backup_dir(backup_dir: Path) -> Path:
    """Ensure backup directory exists."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def backup_sqlite(db_path: Path, backup_dir: Path, compress: bool = True, dry_run: bool = False) -> Path:
    """
    Create a backup of SQLite database.

    Args:
        db_path: Path to SQLite database file
        backup_dir: Directory to store backups
        compress: Whether to compress the backup
        dry_run: If True, only show what would be done without executing

    Returns:
        Path to the backup file
    """
    if not db_path.exists():
        logger.error(f"SQLite database not found: {db_path}")
        raise FileNotFoundError(f"Database file not found: {db_path}")

    timestamp = get_timestamp()
    backup_name = f"floodingnaque.db.backup.{timestamp}"
    backup_path = backup_dir / backup_name

    logger.info(f"Creating SQLite backup: {backup_path}")

    if dry_run:
        logger.info(f"[DRY RUN] Would copy {db_path} to {backup_path}")
        if compress:
            logger.info(f"[DRY RUN] Would compress to {backup_path}.gz")
        return backup_path

    # Copy the database file
    shutil.copy2(db_path, backup_path)

    # Optionally compress
    if compress:
        import gzip

        compressed_path = backup_path.with_suffix(backup_path.suffix + ".gz")

        with open(backup_path, "rb") as f_in:
            with gzip.open(compressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove uncompressed backup
        backup_path.unlink()
        backup_path = compressed_path
        logger.info(f"Compressed backup created: {backup_path}")

    # Verify backup
    if backup_path.exists():
        size_mb = backup_path.stat().st_size / (1024 * 1024)
        logger.info(f"Backup successful: {backup_path} ({size_mb:.2f} MB)")
        return backup_path
    else:
        raise RuntimeError("Backup file was not created")


def backup_postgresql(database_url: str, backup_dir: Path, compress: bool = True, dry_run: bool = False) -> Path:
    """
    Create a backup of PostgreSQL database using pg_dump.

    Args:
        database_url: PostgreSQL connection URL
        backup_dir: Directory to store backups
        compress: Whether to compress the backup
        dry_run: If True, only show what would be done without executing

    Returns:
        Path to the backup file
    """
    from urllib.parse import urlparse

    # Parse database URL
    parsed = urlparse(database_url)

    timestamp = get_timestamp()
    backup_name = f"floodingnaque_postgres.backup.{timestamp}.sql"
    backup_path = backup_dir / backup_name

    # Sanitize all inputs using shlex.quote for shell safety
    hostname = shlex.quote(parsed.hostname or "localhost")
    port = shlex.quote(str(parsed.port or 5432))
    username = shlex.quote(parsed.username or "postgres")
    database = shlex.quote(parsed.path.lstrip("/"))
    output_path = shlex.quote(str(backup_path))

    # Build pg_dump command with sanitized inputs
    env = os.environ.copy()
    env["PGPASSWORD"] = parsed.password or ""

    pg_dump_cmd = [
        "pg_dump",
        "-h",
        hostname.strip("'"),  # Remove quotes for list-based subprocess
        "-p",
        port.strip("'"),
        "-U",
        username.strip("'"),
        "-d",
        database.strip("'"),
        "-f",
        output_path.strip("'"),
        "--format=plain",
        "--no-owner",
        "--no-privileges",
    ]

    logger.info(f"Creating PostgreSQL backup: {backup_path}")
    logger.info(f"Host: {parsed.hostname}, Database: {parsed.path.lstrip('/')}")

    if dry_run:
        logger.info("[DRY RUN] Would execute pg_dump command")
        logger.info(f"[DRY RUN] Output would be: {backup_path}")
        return backup_path

    try:
        result = subprocess.run(pg_dump_cmd, env=env, capture_output=True, text=True, timeout=600)  # nosec B603

        if result.returncode != 0:
            logger.error(f"pg_dump failed: {result.stderr}")
            raise RuntimeError(f"pg_dump failed: {result.stderr}")

    except FileNotFoundError:
        logger.error("pg_dump not found. Install PostgreSQL client tools.")
        raise RuntimeError("pg_dump command not found")
    except subprocess.TimeoutExpired:
        logger.error("pg_dump timed out")
        raise RuntimeError("Backup timed out")

    # Optionally compress
    if compress and backup_path.exists():
        import gzip

        compressed_path = backup_path.with_suffix(backup_path.suffix + ".gz")

        with open(backup_path, "rb") as f_in:
            with gzip.open(compressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        backup_path.unlink()
        backup_path = compressed_path
        logger.info(f"Compressed backup created: {backup_path}")

    if backup_path.exists():
        size_mb = backup_path.stat().st_size / (1024 * 1024)
        logger.info(f"Backup successful: {backup_path} ({size_mb:.2f} MB)")
        return backup_path
    else:
        raise RuntimeError("Backup file was not created")


def cleanup_old_backups(backup_dir: Path, keep_count: int = 10, dry_run: bool = False) -> int:
    """
    Remove old backups, keeping only the most recent ones.

    Args:
        backup_dir: Directory containing backups
        keep_count: Number of recent backups to keep
        dry_run: If True, only show what would be done without executing

    Returns:
        Number of backups removed
    """
    backups = sorted(backup_dir.glob("floodingnaque*.backup.*"), key=lambda p: p.stat().st_mtime, reverse=True)

    removed_count = 0
    for old_backup in backups[keep_count:]:
        if dry_run:
            logger.info(f"[DRY RUN] Would remove old backup: {old_backup}")
        else:
            logger.info(f"Removing old backup: {old_backup}")
            old_backup.unlink()
        removed_count += 1

    if removed_count > 0:
        if dry_run:
            logger.info(f"[DRY RUN] Would clean up {removed_count} old backups")
        else:
            logger.info(f"Cleaned up {removed_count} old backups")

    return removed_count


def get_database_type() -> str:
    """Detect database type from environment."""
    database_url = os.getenv("DATABASE_URL", "")

    if database_url.startswith("sqlite"):
        return "sqlite"
    elif "postgresql" in database_url or "postgres" in database_url:
        return "postgresql"
    elif not database_url:
        # Default to SQLite for development
        return "sqlite"
    else:
        return "unknown"


def run_backup(
    backup_type: str = "auto",
    output_dir: str | None = None,
    compress: bool = True,
    cleanup: bool = True,
    keep_count: int = 10,
    dry_run: bool = False,
) -> Path:
    """
    Run database backup.

    Args:
        backup_type: 'auto', 'sqlite', or 'postgresql'
        output_dir: Custom output directory
        compress: Whether to compress backups
        cleanup: Whether to clean up old backups
        keep_count: Number of backups to keep (--max-backups)
        dry_run: If True, only show what would be done without executing

    Returns:
        Path to the created backup
    """
    # Determine backup directory (supports FLOODINGNAQUE_BACKUP_DIR env var)
    if output_dir:
        backup_dir = Path(output_dir)
    else:
        backup_dir = get_backup_dir()

    if not dry_run:
        ensure_backup_dir(backup_dir)
    else:
        logger.info(f"[DRY RUN] Would ensure backup directory exists: {backup_dir}")

    # Determine backup type
    if backup_type == "auto":
        backup_type = get_database_type()
        logger.info(f"Auto-detected database type: {backup_type}")

    # Perform backup
    if backup_type == "sqlite":
        db_path = backend_path / "data" / "floodingnaque.db"
        backup_path = backup_sqlite(db_path, backup_dir, compress, dry_run)
    elif backup_type == "postgresql":
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        backup_path = backup_postgresql(database_url, backup_dir, compress, dry_run)
    else:
        raise ValueError(f"Unknown database type: {backup_type}")

    # Cleanup old backups
    if cleanup:
        cleanup_old_backups(backup_dir, keep_count, dry_run)

    return backup_path


def main():
    """Main entry point for backup script."""
    # Get default max backups from env or use 10
    default_max_backups = get_env_int("FLOODINGNAQUE_MAX_BACKUPS", 10)

    parser = argparse.ArgumentParser(
        description="Backup Floodingnaque database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Database type selection
    parser.add_argument("--sqlite", action="store_true", help="Backup SQLite database")
    parser.add_argument("--postgres", action="store_true", help="Backup PostgreSQL database")

    # Standardized common arguments
    add_common_arguments(parser, include=["verbose", "output", "force", "dry-run"])

    # Backup-specific options
    parser.add_argument("--no-compress", action="store_true", help="Skip compression")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup of old backups")

    # Backup rotation with --max-backups (preferred) or --keep (legacy)
    parser.add_argument(
        "--max-backups",
        type=int,
        default=default_max_backups,
        help=f"Maximum number of backups to keep (default: {default_max_backups})",
    )
    parser.add_argument(
        "--keep",
        type=int,
        default=None,
        help="[Deprecated] Use --max-backups instead. Number of backups to keep.",
    )

    args = parser.parse_args()

    # Setup logging based on --verbose flag
    setup_logging(verbose=args.verbose)

    # Handle deprecated --keep flag
    keep_count = args.max_backups
    if args.keep is not None:
        logger.warning("--keep is deprecated. Use --max-backups instead.")
        keep_count = args.keep

    # Determine backup type
    if args.sqlite:
        backup_type = "sqlite"
    elif args.postgres:
        backup_type = "postgresql"
    else:
        backup_type = "auto"

    if args.dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN MODE - No changes will be made")
        print("=" * 60 + "\n")

    try:
        backup_path = run_backup(
            backup_type=backup_type,
            output_dir=args.output,
            compress=not args.no_compress,
            cleanup=not args.no_cleanup,
            keep_count=keep_count,
            dry_run=args.dry_run,
        )

        print(f"\n{'='*60}")
        if args.dry_run:
            print("Dry run completed successfully!")
            print(f"Would create backup file: {backup_path}")
        else:
            print("Backup completed successfully!")
            print(f"Backup file: {backup_path}")
        print(f"{'='*60}\n")

    except Exception as e:
        logger.error(f"Backup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
