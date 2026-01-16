"""
Verify Supabase Row Level Security (RLS) Configuration
======================================================

Verifies that RLS policies are correctly configured for the Floodingnaque database.

Usage:
    python scripts/verify_rls.py
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add backend to path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)


def verify_rls_policies():
    """Verify RLS policies are enabled and configured correctly."""
    try:
        from app.models.db import engine
        from sqlalchemy import text

        with engine.connect() as conn:
            # Check if RLS is enabled on key tables
            tables = ["weather_data", "predictions", "api_requests"]

            logger.info("Checking RLS configuration...")

            for table in tables:
                result = conn.execute(
                    text(
                        """
                    SELECT relrowsecurity, relforcerowsecurity
                    FROM pg_class
                    WHERE relname = :table
                """
                    ),
                    {"table": table},
                )

                row = result.fetchone()
                if row:
                    rls_enabled = row[0]
                    force_rls = row[1]
                    status = "✓" if rls_enabled else "✗"
                    logger.info(f"  {status} {table}: RLS={rls_enabled}, Force={force_rls}")
                else:
                    logger.warning(f"  ? {table}: Table not found")

            # Check existing policies
            result = conn.execute(
                text(
                    """
                SELECT schemaname, tablename, policyname, permissive, cmd
                FROM pg_policies
                WHERE schemaname = 'public'
                ORDER BY tablename, policyname
            """
                )
            )

            policies = result.fetchall()
            if policies:
                logger.info(f"\nFound {len(policies)} RLS policies:")
                for policy in policies:
                    logger.info(f"  - {policy[1]}.{policy[2]} ({policy[4]})")
            else:
                logger.warning("No RLS policies found")

            return True

    except ImportError:
        logger.error("Database module not available")
        return False
    except Exception as e:
        logger.error(f"Error verifying RLS: {e}")
        return False


def main():
    print("\n" + "=" * 50)
    print("RLS VERIFICATION")
    print("=" * 50)

    success = verify_rls_policies()

    print("\n" + "=" * 50)
    if success:
        print("RLS verification complete")
    else:
        print("RLS verification failed")
    print("=" * 50)


if __name__ == "__main__":
    main()
