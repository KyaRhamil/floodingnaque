"""
Partition Management Script for Floodingnaque

Manages time-series table partitions:
- Creates future partitions
- Drops old partitions based on retention policy
- Reports partition statistics

Usage:
    # Create partitions for next 3 months
    python manage_partitions.py --create --months 3

    # Drop partitions older than 24 months
    python manage_partitions.py --cleanup --retention 24

    # Show partition statistics
    python manage_partitions.py --stats

    # Full maintenance (create + cleanup + stats)
    python manage_partitions.py --all
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from app.models.db import engine
from sqlalchemy import text

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Tables that use time-series partitioning
PARTITIONED_TABLES = ["weather_data", "predictions"]


def create_partitions(months_ahead: int = 3) -> dict:
    """
    Create future partitions for all partitioned tables.

    Args:
        months_ahead: Number of months to create partitions for

    Returns:
        Dictionary with creation results
    """
    results = {"created": [], "errors": []}

    with engine.connect() as conn:
        for table_name in PARTITIONED_TABLES:
            try:
                logger.info(f"Creating partitions for {table_name} ({months_ahead} months ahead)...")

                conn.execute(
                    text(
                        """
                    SELECT create_monthly_partitions(:table_name, :months_ahead)
                """
                    ),
                    {"table_name": table_name, "months_ahead": months_ahead},
                )

                conn.commit()
                results["created"].append({"table": table_name, "months": months_ahead, "status": "success"})
                logger.info(f"  Created partitions for {table_name}")

            except Exception as e:
                error_msg = str(e)
                if "function create_monthly_partitions" in error_msg.lower():
                    logger.warning(f"  Partition function not found for {table_name} - table may not be partitioned")
                else:
                    logger.error(f"  Error creating partitions for {table_name}: {e}")
                results["errors"].append({"table": table_name, "error": error_msg})

    return results


def cleanup_old_partitions(retention_months: int = 24) -> dict:
    """
    Drop partitions older than retention period.

    Args:
        retention_months: Number of months to keep

    Returns:
        Dictionary with cleanup results
    """
    results = {"dropped": [], "errors": []}

    with engine.connect() as conn:
        for table_name in PARTITIONED_TABLES:
            try:
                logger.info(f"Cleaning up old partitions for {table_name} (keeping {retention_months} months)...")

                result = conn.execute(
                    text(
                        """
                    SELECT drop_old_partitions(:table_name, :retention_months)
                """
                    ),
                    {"table_name": table_name, "retention_months": retention_months},
                )

                dropped_count = result.scalar() or 0
                conn.commit()

                results["dropped"].append(
                    {"table": table_name, "partitions_dropped": dropped_count, "status": "success"}
                )
                logger.info(f"  Dropped {dropped_count} partitions from {table_name}")

            except Exception as e:
                error_msg = str(e)
                if "function drop_old_partitions" in error_msg.lower():
                    logger.warning(f"  Partition function not found for {table_name} - table may not be partitioned")
                else:
                    logger.error(f"  Error cleaning up partitions for {table_name}: {e}")
                results["errors"].append({"table": table_name, "error": error_msg})

    return results


def get_partition_statistics() -> dict:
    """
    Get statistics for all partitioned tables.

    Returns:
        Dictionary with partition statistics
    """
    results = {"tables": {}, "errors": []}

    with engine.connect() as conn:
        for table_name in PARTITIONED_TABLES:
            try:
                logger.info(f"Getting partition statistics for {table_name}...")

                result = conn.execute(
                    text(
                        """
                    SELECT * FROM get_partition_stats(:table_name)
                """
                    ),
                    {"table_name": table_name},
                )

                partitions = []
                total_rows = 0
                # Note: total_size is tracked per partition in partition_info

                for row in result:
                    partition_info = {
                        "name": row[0],
                        "row_count": row[1],
                        "table_size": row[2],
                        "index_size": row[3],
                        "total_size": row[4],
                    }
                    partitions.append(partition_info)
                    total_rows += row[1] or 0

                results["tables"][table_name] = {
                    "partition_count": len(partitions),
                    "total_rows": total_rows,
                    "partitions": partitions,
                }
                logger.info(f"  Found {len(partitions)} partitions with {total_rows:,} total rows")

            except Exception as e:
                error_msg = str(e)
                if "function get_partition_stats" in error_msg.lower():
                    # Table might not be partitioned yet
                    logger.info(f"  {table_name} is not partitioned (or function not available)")

                    # Get basic table stats instead
                    try:
                        row_count = conn.execute(
                            text(
                                f"""
                            SELECT COUNT(*) FROM {table_name} WHERE is_deleted = false
                        """  # nosec B608
                            )
                        ).scalar()

                        results["tables"][table_name] = {
                            "partition_count": 0,
                            "total_rows": row_count or 0,
                            "partitions": [],
                            "note": "Table not partitioned",
                        }
                    except Exception:  # nosec B110
                        pass
                else:
                    logger.error(f"  Error getting stats for {table_name}: {e}")
                results["errors"].append({"table": table_name, "error": error_msg})

    return results


def print_statistics(stats: dict):
    """Print partition statistics in a formatted way."""
    print("\n" + "=" * 60)
    print("PARTITION STATISTICS REPORT")
    print("=" * 60)
    print(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    print()

    for table_name, table_stats in stats.get("tables", {}).items():
        print(f"\n📊 {table_name.upper()}")
        print("-" * 40)
        print(f"  Partition Count: {table_stats['partition_count']}")
        print(f"  Total Rows: {table_stats['total_rows']:,}")

        if table_stats.get("note"):
            print(f"  Note: {table_stats['note']}")

        if table_stats.get("partitions"):
            print("\n  Partitions:")
            for partition in table_stats["partitions"][-6:]:  # Show last 6
                print(f"    - {partition['name']}: {partition['row_count']:,} rows, {partition['total_size']}")

            if len(table_stats["partitions"]) > 6:
                print(f"    ... and {len(table_stats['partitions']) - 6} more partitions")

    if stats.get("errors"):
        print("\n⚠️  ERRORS:")
        for error in stats["errors"]:
            print(f"  - {error['table']}: {error['error'][:100]}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Manage time-series table partitions for Floodingnaque")

    parser.add_argument("--create", "-c", action="store_true", help="Create future partitions")

    parser.add_argument("--months", "-m", type=int, default=3, help="Months ahead to create partitions (default: 3)")

    parser.add_argument("--cleanup", "-d", action="store_true", help="Drop old partitions based on retention policy")

    parser.add_argument("--retention", "-r", type=int, default=24, help="Months to retain partitions (default: 24)")

    parser.add_argument("--stats", "-s", action="store_true", help="Show partition statistics")

    parser.add_argument("--all", "-a", action="store_true", help="Run all operations: create, cleanup, and stats")

    args = parser.parse_args()

    # Default to stats if no action specified
    if not any([args.create, args.cleanup, args.stats, args.all]):
        args.stats = True

    print("\n🗄️  Floodingnaque Partition Manager")
    print("=" * 40)

    # Run requested operations
    if args.all or args.create:
        print(f"\n📅 Creating partitions ({args.months} months ahead)...")
        results = create_partitions(args.months)
        print(f"   Created: {len(results['created'])} tables")
        if results["errors"]:
            print(f"   Errors: {len(results['errors'])}")

    if args.all or args.cleanup:
        print(f"\n🗑️  Cleaning up old partitions (retention: {args.retention} months)...")
        results = cleanup_old_partitions(args.retention)
        total_dropped = sum(r["partitions_dropped"] for r in results["dropped"])
        print(f"   Dropped: {total_dropped} partitions total")
        if results["errors"]:
            print(f"   Errors: {len(results['errors'])}")

    if args.all or args.stats:
        print("\n📊 Gathering partition statistics...")
        stats = get_partition_statistics()
        print_statistics(stats)

    print("\n✅ Partition management complete!")


if __name__ == "__main__":
    main()
