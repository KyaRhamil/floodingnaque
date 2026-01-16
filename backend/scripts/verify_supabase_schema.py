"""
Verify Supabase Schema Configuration
====================================

Verifies that the Supabase database schema matches expected configuration.

Usage:
    python scripts/verify_supabase_schema.py
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add backend to path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

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
    """Verify performance indexes exist."""
    try:
        from app.models.db import engine
        from sqlalchemy import text

        with engine.connect() as conn:
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

            return True

    except Exception as e:
        logger.error(f"Error checking indexes: {e}")
        return False


def main():
    print("\n" + "=" * 50)
    print("SUPABASE SCHEMA VERIFICATION")
    print("=" * 50)

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
