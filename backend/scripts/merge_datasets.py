"""
Dataset Merging Utility
=======================

Merges multiple datasets into a single training dataset.

Usage:
    python scripts/merge_datasets.py
    python scripts/merge_datasets.py --output merged_dataset.csv
"""

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def find_datasets() -> List[Path]:
    """Find all available processed datasets."""
    datasets = []
    patterns = ["cumulative_*.csv", "pagasa_*.csv", "processed_flood_*.csv"]

    for pattern in patterns:
        datasets.extend(PROCESSED_DIR.glob(pattern))

    return sorted(set(datasets))


def merge_datasets(datasets: List[Path], output_name: str = "merged_dataset.csv") -> pd.DataFrame:
    """Merge multiple datasets into one."""
    if not datasets:
        raise ValueError("No datasets to merge")

    dfs = []
    for path in datasets:
        try:
            df = pd.read_csv(path)
            df["source_file"] = path.name
            dfs.append(df)
            logger.info(f"Loaded: {path.name} ({len(df)} records)")
        except Exception as e:
            logger.warning(f"Error loading {path}: {e}")

    if not dfs:
        raise ValueError("Could not load any datasets")

    merged = pd.concat(dfs, ignore_index=True)

    # Remove duplicates based on key columns
    key_cols = ["temperature", "humidity", "precipitation", "flood"]
    available_keys = [c for c in key_cols if c in merged.columns]
    if available_keys:
        original_len = len(merged)
        merged = merged.drop_duplicates(subset=available_keys)
        logger.info(f"Removed {original_len - len(merged)} duplicates")

    # Save merged dataset
    output_path = PROCESSED_DIR / output_name
    merged.to_csv(output_path, index=False)
    logger.info(f"Saved: {output_path} ({len(merged)} records)")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge multiple datasets")
    parser.add_argument("--output", type=str, default="merged_dataset.csv", help="Output filename")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to merge")
    args = parser.parse_args()

    if args.datasets:
        datasets = [PROCESSED_DIR / d for d in args.datasets if (PROCESSED_DIR / d).exists()]
    else:
        datasets = find_datasets()

    if not datasets:
        logger.error("No datasets found to merge")
        return

    logger.info(f"Found {len(datasets)} datasets to merge")
    merged = merge_datasets(datasets, args.output)

    print(f"\nMerged {len(datasets)} datasets into {len(merged)} records")


if __name__ == "__main__":
    main()
