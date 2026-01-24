#!/usr/bin/env python
"""
Data Freshness Checker
======================

Checks the freshness of cached data files (earthengine_cache, meteostat_cache)
and alerts when data is stale beyond configurable thresholds.

Usage:
    # Check all cached data
    python scripts/check_data_freshness.py

    # Check with custom thresholds (in days)
    python scripts/check_data_freshness.py --threshold 14

    # JSON output for CI/CD integration
    python scripts/check_data_freshness.py --json

    # Check specific cache directory
    python scripts/check_data_freshness.py --cache earthengine_cache

    # Alert mode (exit code 1 if stale data found)
    python scripts/check_data_freshness.py --alert

Exit Codes:
    0: All data is fresh
    1: Stale data found (when --alert is used)
    2: Error occurred

Author: Floodingnaque Team
Date: 2026-01-24
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
DATA_DIR = BACKEND_DIR / "data"

# Default freshness thresholds (in days)
DEFAULT_THRESHOLDS = {
    "earthengine_cache": 30,  # Earth Engine data - monthly refresh
    "meteostat_cache": 7,  # Meteostat weather data - weekly refresh
    "processed": 14,  # Processed datasets - bi-weekly
    "cleaned": 30,  # Cleaned records - monthly
    "raw": 90,  # Raw PAGASA/flood files - quarterly
}

# File patterns to check
CACHE_PATTERNS = {
    "earthengine_cache": ["*.csv", "*.json", "*.tif", "*.geojson"],
    "meteostat_cache": ["*.csv", "*.json", "*.parquet"],
    "processed": ["*.csv"],
    "cleaned": ["*.csv"],
}


class DataFreshnessChecker:
    """Check and report on data freshness across cache directories."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        thresholds: Optional[Dict[str, int]] = None,
    ):
        """
        Initialize the freshness checker.

        Args:
            data_dir: Base data directory
            thresholds: Custom freshness thresholds (days) by cache name
        """
        self.data_dir = data_dir or DATA_DIR
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.results: Dict[str, List[Dict]] = {}

    def get_file_age_days(self, filepath: Path) -> float:
        """Get the age of a file in days."""
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
        age = datetime.now() - mtime
        return age.total_seconds() / 86400  # Convert to days

    def check_directory(
        self,
        dir_name: str,
        patterns: Optional[List[str]] = None,
        threshold_days: Optional[int] = None,
    ) -> List[Dict]:
        """
        Check all files in a directory for freshness.

        Args:
            dir_name: Name of the directory under data/
            patterns: File patterns to check
            threshold_days: Days after which data is considered stale

        Returns:
            List of file status dictionaries
        """
        dir_path = self.data_dir / dir_name

        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return [
                {
                    "directory": dir_name,
                    "status": "missing",
                    "message": f"Cache directory does not exist: {dir_path}",
                }
            ]

        patterns = patterns or CACHE_PATTERNS.get(dir_name, ["*.*"])
        threshold = threshold_days or self.thresholds.get(dir_name, 30)

        results = []

        for pattern in patterns:
            for filepath in dir_path.glob(pattern):
                if filepath.is_file():
                    age_days = self.get_file_age_days(filepath)
                    is_stale = age_days > threshold

                    result = {
                        "file": str(filepath.relative_to(self.data_dir)),
                        "age_days": round(age_days, 1),
                        "threshold_days": threshold,
                        "is_stale": is_stale,
                        "status": "stale" if is_stale else "fresh",
                        "last_modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
                        "size_bytes": filepath.stat().st_size,
                    }

                    results.append(result)

        if not results:
            results.append(
                {
                    "directory": dir_name,
                    "status": "empty",
                    "message": f"No cached data files found in {dir_path}",
                }
            )

        return results

    def check_all_caches(self) -> Dict[str, List[Dict]]:
        """
        Check all cache directories for data freshness.

        Returns:
            Dictionary mapping cache names to their file status lists
        """
        self.results = {}

        for cache_name in CACHE_PATTERNS.keys():
            self.results[cache_name] = self.check_directory(cache_name)

        return self.results

    def get_stale_files(self) -> List[Dict]:
        """Get all stale files across all caches."""
        stale = []
        for cache_name, files in self.results.items():
            for f in files:
                if f.get("is_stale"):
                    f["cache"] = cache_name
                    stale.append(f)
        return stale

    def get_summary(self) -> Dict:
        """
        Get a summary of the freshness check.

        Returns:
            Summary dictionary with counts and statistics
        """
        total_files = 0
        stale_files = 0
        fresh_files = 0
        missing_dirs = 0
        empty_dirs = 0

        by_cache = {}

        for cache_name, files in self.results.items():
            cache_stats = {
                "total": 0,
                "fresh": 0,
                "stale": 0,
                "status": "unknown",
            }

            for f in files:
                if f.get("status") == "missing":
                    missing_dirs += 1
                    cache_stats["status"] = "missing"
                elif f.get("status") == "empty":
                    empty_dirs += 1
                    cache_stats["status"] = "empty"
                elif f.get("is_stale"):
                    stale_files += 1
                    cache_stats["stale"] += 1
                    cache_stats["total"] += 1
                else:
                    fresh_files += 1
                    cache_stats["fresh"] += 1
                    cache_stats["total"] += 1

            if cache_stats["total"] > 0:
                if cache_stats["stale"] > 0:
                    cache_stats["status"] = "stale"
                else:
                    cache_stats["status"] = "fresh"

            by_cache[cache_name] = cache_stats

        total_files = stale_files + fresh_files

        return {
            "checked_at": datetime.now().isoformat(),
            "data_directory": str(self.data_dir),
            "total_files": total_files,
            "fresh_files": fresh_files,
            "stale_files": stale_files,
            "missing_directories": missing_dirs,
            "empty_directories": empty_dirs,
            "overall_status": "stale" if stale_files > 0 else "fresh",
            "by_cache": by_cache,
        }

    def print_report(self, verbose: bool = False) -> None:
        """Print a human-readable freshness report."""
        summary = self.get_summary()

        print("\n" + "=" * 70)
        print("DATA FRESHNESS REPORT")
        print("=" * 70)
        print(f"Checked at: {summary['checked_at']}")
        print(f"Data directory: {summary['data_directory']}")
        print("-" * 70)

        # Cache summary
        print("\nCache Status:")
        for cache_name, stats in summary["by_cache"].items():
            status_icon = {
                "fresh": "✓",
                "stale": "⚠",
                "missing": "✗",
                "empty": "○",
            }.get(stats["status"], "?")

            threshold = self.thresholds.get(cache_name, 30)

            if stats["status"] in ("missing", "empty"):
                print(f"  [{status_icon}] {cache_name:20} - {stats['status'].upper()}")
            else:
                print(
                    f"  [{status_icon}] {cache_name:20} - "
                    f"{stats['fresh']}/{stats['total']} fresh "
                    f"(threshold: {threshold} days)"
                )

        print("-" * 70)

        # Stale files detail
        stale = self.get_stale_files()
        if stale:
            print(f"\n⚠ STALE FILES ({len(stale)}):")
            for f in sorted(stale, key=lambda x: -x.get("age_days", 0)):
                age = f.get("age_days", 0)
                threshold = f.get("threshold_days", 0)
                print(f"  • {f['file']}")
                print(f"    Age: {age:.1f} days (threshold: {threshold} days)")
                print(f"    Last modified: {f['last_modified']}")
        else:
            print("\n✓ All cached data is fresh!")

        # Overall status
        print("\n" + "=" * 70)
        if summary["overall_status"] == "stale":
            print("⚠ OVERALL STATUS: STALE DATA DETECTED")
            print("  Run data ingestion to refresh stale caches:")
            print("  ./models/run_training_pipeline.ps1 -Ingest -IngestMeteostat")
        else:
            print("✓ OVERALL STATUS: ALL DATA FRESH")
        print("=" * 70 + "\n")

        if verbose:
            print("\nDetailed Results:")
            print(json.dumps(self.results, indent=2))

    def save_report(self, output_path: Optional[Path] = None) -> Path:
        """
        Save the freshness report to a JSON file.

        Args:
            output_path: Output file path

        Returns:
            Path to saved report
        """
        output_path = output_path or (BACKEND_DIR / "reports" / "data_freshness_report.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "summary": self.get_summary(),
            "details": self.results,
            "stale_files": self.get_stale_files(),
            "thresholds": self.thresholds,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to: {output_path}")
        return output_path


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Check data freshness and alert on stale cached data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--threshold",
        type=int,
        help="Override freshness threshold for all caches (in days)",
    )
    parser.add_argument(
        "--cache",
        choices=["earthengine_cache", "meteostat_cache", "processed", "cleaned"],
        help="Check only a specific cache directory",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--alert",
        action="store_true",
        help="Exit with code 1 if stale data is found",
    )
    parser.add_argument(
        "--save",
        type=Path,
        nargs="?",
        const=BACKEND_DIR / "reports" / "data_freshness_report.json",
        help="Save report to file (default: reports/data_freshness_report.json)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed results",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override data directory path",
    )

    args = parser.parse_args()

    # Build custom thresholds if override provided
    thresholds = None
    if args.threshold:
        thresholds = {k: args.threshold for k in DEFAULT_THRESHOLDS}

    checker = DataFreshnessChecker(
        data_dir=args.data_dir,
        thresholds=thresholds,
    )

    # Check specific cache or all
    if args.cache:
        checker.results = {args.cache: checker.check_directory(args.cache)}
    else:
        checker.check_all_caches()

    # Output format
    if args.json:
        output = {
            "summary": checker.get_summary(),
            "stale_files": checker.get_stale_files(),
        }
        print(json.dumps(output, indent=2))
    else:
        checker.print_report(verbose=args.verbose)

    # Save report if requested
    if args.save:
        checker.save_report(args.save)

    # Alert mode
    if args.alert:
        stale = checker.get_stale_files()
        if stale:
            logger.warning(f"Found {len(stale)} stale files")
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
