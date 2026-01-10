"""
Database Backup Utility for Floodingnaque.

Provides automated and manual database backup functionality for both
SQLite (development) and PostgreSQL (production) databases.

Usage:
    python scripts/backup_database.py                    # Auto-detect and backup
    python scripts/backup_database.py --sqlite           # SQLite backup only
    python scripts/backup_database.py --postgres         # PostgreSQL backup only
    python scripts/backup_database.py --output ./backups # Custom output directory
"""

import os
import sys
import shutil
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import logging

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_timestamp() -> str:
    """Generate timestamp string for backup files."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_backup_dir(backup_dir: Path) -> Path:
    """Ensure backup directory exists."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def backup_sqlite(
    db_path: Path,
    backup_dir: Path,
    compress: bool = True
) -> Path:
    """
    Create a backup of SQLite database.
    
    Args:
        db_path: Path to SQLite database file
        backup_dir: Directory to store backups
        compress: Whether to compress the backup
    
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
    
    # Copy the database file
    shutil.copy2(db_path, backup_path)
    
    # Optionally compress
    if compress:
        import gzip
        compressed_path = backup_path.with_suffix(backup_path.suffix + '.gz')
        
        with open(backup_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
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


def backup_postgresql(
    database_url: str,
    backup_dir: Path,
    compress: bool = True
) -> Path:
    """
    Create a backup of PostgreSQL database using pg_dump.
    
    Args:
        database_url: PostgreSQL connection URL
        backup_dir: Directory to store backups
        compress: Whether to compress the backup
    
    Returns:
        Path to the backup file
    """
    from urllib.parse import urlparse
    
    # Parse database URL
    parsed = urlparse(database_url)
    
    timestamp = get_timestamp()
    backup_name = f"floodingnaque_postgres.backup.{timestamp}.sql"
    backup_path = backup_dir / backup_name
    
    # Build pg_dump command
    env = os.environ.copy()
    env['PGPASSWORD'] = parsed.password or ''
    
    pg_dump_cmd = [
        'pg_dump',
        '-h', parsed.hostname or 'localhost',
        '-p', str(parsed.port or 5432),
        '-U', parsed.username or 'postgres',
        '-d', parsed.path.lstrip('/'),
        '-f', str(backup_path),
        '--format=plain',
        '--no-owner',
        '--no-privileges'
    ]
    
    logger.info(f"Creating PostgreSQL backup: {backup_path}")
    logger.info(f"Host: {parsed.hostname}, Database: {parsed.path.lstrip('/')}")
    
    try:
        result = subprocess.run(
            pg_dump_cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
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
        compressed_path = backup_path.with_suffix(backup_path.suffix + '.gz')
        
        with open(backup_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
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


def cleanup_old_backups(backup_dir: Path, keep_count: int = 10) -> int:
    """
    Remove old backups, keeping only the most recent ones.
    
    Args:
        backup_dir: Directory containing backups
        keep_count: Number of recent backups to keep
    
    Returns:
        Number of backups removed
    """
    backups = sorted(
        backup_dir.glob('floodingnaque*.backup.*'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    removed_count = 0
    for old_backup in backups[keep_count:]:
        logger.info(f"Removing old backup: {old_backup}")
        old_backup.unlink()
        removed_count += 1
    
    if removed_count > 0:
        logger.info(f"Cleaned up {removed_count} old backups")
    
    return removed_count


def get_database_type() -> str:
    """Detect database type from environment."""
    database_url = os.getenv('DATABASE_URL', '')
    
    if database_url.startswith('sqlite'):
        return 'sqlite'
    elif 'postgresql' in database_url or 'postgres' in database_url:
        return 'postgresql'
    elif not database_url:
        # Default to SQLite for development
        return 'sqlite'
    else:
        return 'unknown'


def run_backup(
    backup_type: str = 'auto',
    output_dir: str = None,
    compress: bool = True,
    cleanup: bool = True,
    keep_count: int = 10
) -> Path:
    """
    Run database backup.
    
    Args:
        backup_type: 'auto', 'sqlite', or 'postgresql'
        output_dir: Custom output directory
        compress: Whether to compress backups
        cleanup: Whether to clean up old backups
        keep_count: Number of backups to keep
    
    Returns:
        Path to the created backup
    """
    # Determine backup directory
    if output_dir:
        backup_dir = Path(output_dir)
    else:
        backup_dir = backend_path / 'backups'
    
    ensure_backup_dir(backup_dir)
    
    # Determine backup type
    if backup_type == 'auto':
        backup_type = get_database_type()
        logger.info(f"Auto-detected database type: {backup_type}")
    
    # Perform backup
    if backup_type == 'sqlite':
        db_path = backend_path / 'data' / 'floodingnaque.db'
        backup_path = backup_sqlite(db_path, backup_dir, compress)
    elif backup_type == 'postgresql':
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        backup_path = backup_postgresql(database_url, backup_dir, compress)
    else:
        raise ValueError(f"Unknown database type: {backup_type}")
    
    # Cleanup old backups
    if cleanup:
        cleanup_old_backups(backup_dir, keep_count)
    
    return backup_path


def main():
    """Main entry point for backup script."""
    parser = argparse.ArgumentParser(
        description='Backup Floodingnaque database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--sqlite',
        action='store_true',
        help='Backup SQLite database'
    )
    parser.add_argument(
        '--postgres',
        action='store_true',
        help='Backup PostgreSQL database'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for backups'
    )
    parser.add_argument(
        '--no-compress',
        action='store_true',
        help='Skip compression'
    )
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Skip cleanup of old backups'
    )
    parser.add_argument(
        '--keep',
        type=int,
        default=10,
        help='Number of backups to keep (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Determine backup type
    if args.sqlite:
        backup_type = 'sqlite'
    elif args.postgres:
        backup_type = 'postgresql'
    else:
        backup_type = 'auto'
    
    try:
        backup_path = run_backup(
            backup_type=backup_type,
            output_dir=args.output,
            compress=not args.no_compress,
            cleanup=not args.no_cleanup,
            keep_count=args.keep
        )
        
        print(f"\n{'='*60}")
        print(f"Backup completed successfully!")
        print(f"Backup file: {backup_path}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
