"""
Verify Database Schema Configuration
====================================

Verifies that the database schema matches expected configuration.
Supports both PostgreSQL (Supabase) and SQLite (development) databases.

Usage:
    python scripts/verify_supabase_schema.py           # Auto-detect and verify
    python scripts/verify_supabase_schema.py --verbose # With debug output
    python scripts/verify_supabase_schema.py --force   # Force check even on SQLite

Environment Variables:
    DATABASE_URL                Database connection URL
    FLOODINGNAQUE_LOG_LEVEL     Logging level (DEBUG, INFO, etc.)
"""

import argparse
import logging
import os
import sys

# Add backend to path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

from scripts.cli_utils import (
    add_common_arguments,
    get_database_type,
    is_postgresql,
    is_sqlite,
    setup_logging,
)

logger = logging.getLogger(__name__)

# Expected tables and their key columns
EXPECTED_SCHEMA = {
    "weather_data": ["id", "temperature", "humidity", "precipitation", "timestamp"],
    "predictions": ["id", "temperature", "humidity", "precipitation", "prediction", "confidence"],
    "api_requests": ["id", "endpoint", "method", "timestamp", "response_time_ms"],
    "users": ["id", "email", "created_at"],
}


def verify_schema():
    """Verify database schema matches expectations."""
    try:
        from app.models.db import engine
        from sqlalchemy import inspect, text

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        logger.info("Checking database schema...")
        all_valid = True

        for expected_table, expected_columns in EXPECTED_SCHEMA.items():
            if expected_table in tables:
                actual_columns = [col["name"] for col in inspector.get_columns(expected_table)]
                missing = [c for c in expected_columns if c not in actual_columns]

                if missing:
                    logger.warning(f"  ⚠ {expected_table}: Missing columns: {missing}")
                    all_valid = False
                else:
                    logger.info(f"  ✓ {expected_table}: Schema valid")
            else:
                logger.warning(f"  ✗ {expected_table}: Table not found")
                all_valid = False

        # Check for additional tables
        extra_tables = [t for t in tables if t not in EXPECTED_SCHEMA and not t.startswith("_")]
        if extra_tables:
            logger.info(f"\nAdditional tables: {extra_tables}")

        return all_valid

    except ImportError:
        logger.error("Database module not available")
        return False
    except Exception as e:
        logger.error(f"Error verifying schema: {e}")
        return False


def verify_indexes():
    """
    Verify performance indexes exist.

    Note: Index listing is PostgreSQL-specific. For SQLite, uses a different query.
    """
    try:
        from app.models.db import engine
        from sqlalchemy import text

        db_type = get_database_type()

        with engine.connect() as conn:
            if db_type == "postgresql":
                # PostgreSQL-specific query
                result = conn.execute(
                    text(
                        """
                    SELECT tablename, indexname
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                    ORDER BY tablename
                """
                    )
                )
                indexes = result.fetchall()
                logger.info(f"\nFound {len(indexes)} indexes:")
                for idx in indexes[:20]:  # Limit output
                    logger.info(f"  - {idx[0]}.{idx[1]}")

                if len(indexes) > 20:
                    logger.info(f"  ... and {len(indexes) - 20} more")

            elif db_type == "sqlite":
                # SQLite-specific query
                result = conn.execute(
                    text("SELECT name, tbl_name FROM sqlite_master WHERE type='index' ORDER BY tbl_name")
                )
                indexes = result.fetchall()
                logger.info(f"\nFound {len(indexes)} indexes (SQLite):")
                for idx in indexes[:20]:
                    logger.info(f"  - {idx[1]}.{idx[0]}")

                if len(indexes) > 20:
                    logger.info(f"  ... and {len(indexes) - 20} more")

            else:
                logger.warning(f"Unknown database type: {db_type}. Skipping index verification.")
                return False

            return True

    except Exception as e:
        logger.error(f"Error checking indexes: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify database schema configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Standardized common arguments
    add_common_arguments(parser, include=["verbose", "force"])

    args = parser.parse_args()

    # Setup logging based on --verbose flag
    setup_logging(verbose=args.verbose)

    # Detect database type
    db_type = get_database_type()

    print("\n" + "=" * 50)
    print("DATABASE SCHEMA VERIFICATION")
    print("=" * 50)
    print(f"Database type: {db_type}")

    if db_type == "sqlite":
        print("Note: Running on SQLite (development database)")
        print("      For production, use PostgreSQL (Supabase)")

    schema_valid = verify_schema()
    verify_indexes()

    print("\n" + "=" * 50)
    if schema_valid:
        print("Schema verification: PASSED")
    else:
        print("Schema verification: WARNINGS")
    print("=" * 50)


if __name__ == "__main__":
    main()
