#!/usr/bin/env python
"""
Data Directory Reorganization Script
=====================================

Moves raw PAGASA weather station CSVs and flood record CSVs to data/raw/
subdirectory while preserving the existing processed/ and cleaned/ structure.

This provides a cleaner separation between:
- data/raw/         - Original source files (PAGASA, flood records)
- data/cleaned/     - Cleaned versions of raw files
- data/processed/   - Training-ready datasets

Usage:
    # Preview what would be moved (dry run)
    python scripts/reorganize_data_directory.py --dry-run

    # Execute the reorganization
    python scripts/reorganize_data_directory.py

    # Revert if needed (move files back)
    python scripts/reorganize_data_directory.py --revert

Author: Floodingnaque Team
Date: 2026-01-24
"""

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

# Files to move to raw/ directory
RAW_FILE_PATTERNS = [
    # PAGASA weather station data
    "Floodingnaque_CADS-S0126006_*.csv",
    # Official flood records
    "Floodingnaque_Paranaque_Official_Flood_Records_*.csv",
    # PAGASA readme
    "Floodingnaque_CADS-S0126006_A.ReadMe.txt",
]

# Files to leave in place (not move)
KEEP_IN_PLACE = [
    "checksums.json",
    "data_lineage.json",
    "data_folder_contents.txt",
    "SCHEMA.md",
    "DATA_DICTIONARY.md",
    "synthetic_dataset.csv",  # This is a processed/generated file
]

# Subdirectories to leave in place
KEEP_DIRS = [
    "cleaned",
    "processed",
    "earthengine_cache",
    "meteostat_cache",
    "raw",  # Target directory
]


class DataReorganizer:
    """Reorganize data directory structure."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.manifest: Dict[str, str] = {}
        self.moved_files: List[Tuple[Path, Path]] = []

    def find_raw_files(self) -> List[Path]:
        """Find all raw files that should be moved."""
        raw_files = []

        for pattern in RAW_FILE_PATTERNS:
            for filepath in self.data_dir.glob(pattern):
                if filepath.is_file() and filepath.name not in KEEP_IN_PLACE:
                    raw_files.append(filepath)

        return sorted(raw_files)

    def create_raw_subdirs(self) -> None:
        """Create raw/ subdirectory structure."""
        subdirs = [
            self.raw_dir / "pagasa",
            self.raw_dir / "flood_records",
        ]

        for subdir in subdirs:
            subdir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {subdir}")

    def get_target_path(self, source: Path) -> Path:
        """Determine the target path for a raw file."""
        filename = source.name

        # PAGASA data goes to raw/pagasa/
        if "CADS-S0126006" in filename:
            return self.raw_dir / "pagasa" / filename

        # Flood records go to raw/flood_records/
        if "Official_Flood_Records" in filename:
            return self.raw_dir / "flood_records" / filename

        # Default: just put in raw/
        return self.raw_dir / filename

    def move_files(self, dry_run: bool = False) -> List[Tuple[Path, Path]]:
        """
        Move raw files to the raw/ directory.

        Args:
            dry_run: If True, only preview what would be moved

        Returns:
            List of (source, target) tuples for moved files
        """
        raw_files = self.find_raw_files()

        if not raw_files:
            logger.info("No raw files found to move")
            return []

        logger.info(f"Found {len(raw_files)} files to move")

        moves = []

        for source in raw_files:
            target = self.get_target_path(source)
            moves.append((source, target))

            if dry_run:
                logger.info(f"  [DRY-RUN] Would move: {source.name}")
                logger.info(f"            To: {target.relative_to(self.data_dir)}")
            else:
                # Create parent directory if needed
                target.parent.mkdir(parents=True, exist_ok=True)

                # Move the file
                shutil.move(str(source), str(target))
                logger.info(f"  Moved: {source.name} -> {target.relative_to(self.data_dir)}")

        self.moved_files = moves
        return moves

    def create_readme(self, dry_run: bool = False) -> None:
        """Create a README for the raw/ directory."""
        readme_content = """# Raw Data Directory

This directory contains the original, unprocessed source data files for the
Floodingnaque flood prediction system.

## Directory Structure

```
raw/
├── pagasa/               # PAGASA weather station data
│   ├── *_NAIA Daily Data.csv
│   ├── *_Port Area Daily Data.csv
│   └── *_Science Garden Daily Data.csv
└── flood_records/        # Official flood records from Parañaque CDRRMO
    ├── *_2022.csv
    ├── *_2023.csv
    ├── *_2024.csv
    └── *_2025.csv
```

## Data Sources

### PAGASA Weather Data
- **Provider:** DOST-PAGASA (Philippine Atmospheric, Geophysical and Astronomical Services Administration)
- **Portal:** http://bagong.pagasa.dost.gov.ph/climate/climate-data
- **Stations:** NAIA, Port Area, Science Garden

### Flood Records
- **Provider:** City of Parañaque CDRRMO
- **Format:** CSV with irregular headers (requires preprocessing)

## Processing Pipeline

1. **Raw data** (this directory) → Original source files
2. **Cleaned data** (`../cleaned/`) → Standardized format, fixed headers
3. **Processed data** (`../processed/`) → Training-ready datasets

## Important Notes

- Do NOT modify files in this directory directly
- Use preprocessing scripts to generate cleaned versions
- See `../DATA_DICTIONARY.md` for column definitions
- See `../SCHEMA.md` for expected data schemas

## Regenerating Cleaned Data

```bash
# Clean PAGASA data
python scripts/preprocess_pagasa_data.py

# Clean flood records
python scripts/clean_raw_flood_records.py

# Merge all datasets
python scripts/merge_datasets.py
```

---
*Last updated: {date}*
"""

        readme_path = self.raw_dir / "README.md"

        if dry_run:
            logger.info("[DRY-RUN] Would create: raw/README.md")
        else:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            with open(readme_path, "w") as f:
                f.write(readme_content.format(date=datetime.now().strftime("%Y-%m-%d")))
            logger.info(f"Created: {readme_path.relative_to(self.data_dir)}")

    def save_manifest(self, dry_run: bool = False) -> Path:
        """Save a manifest of file moves for potential reversion."""
        manifest = {
            "reorganized_at": datetime.now().isoformat(),
            "data_directory": str(self.data_dir),
            "moves": [
                {"source": str(s.relative_to(self.data_dir)), "target": str(t.relative_to(self.data_dir))}
                for s, t in self.moved_files
            ],
        }

        manifest_path = self.data_dir / "reorganization_manifest.json"

        if dry_run:
            logger.info("[DRY-RUN] Would save manifest to: reorganization_manifest.json")
        else:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Saved manifest: {manifest_path.name}")

        return manifest_path

    def revert(self) -> bool:
        """Revert file moves using the saved manifest."""
        manifest_path = self.data_dir / "reorganization_manifest.json"

        if not manifest_path.exists():
            logger.error("No manifest found. Cannot revert.")
            return False

        with open(manifest_path) as f:
            manifest = json.load(f)

        logger.info(f"Reverting {len(manifest['moves'])} file moves...")

        for move in manifest["moves"]:
            source = self.data_dir / move["target"]  # Current location
            target = self.data_dir / move["source"]  # Original location

            if source.exists():
                shutil.move(str(source), str(target))
                logger.info(f"  Reverted: {move['target']} -> {move['source']}")
            else:
                logger.warning(f"  File not found: {move['target']}")

        # Remove empty raw directories
        for subdir in ["pagasa", "flood_records"]:
            subdir_path = self.raw_dir / subdir
            if subdir_path.exists() and not any(subdir_path.iterdir()):
                subdir_path.rmdir()
                logger.info(f"  Removed empty directory: raw/{subdir}")

        # Remove manifest
        manifest_path.unlink()
        logger.info("Removed manifest file")

        return True

    def update_scripts(self, dry_run: bool = False) -> None:
        """
        Print instructions for updating scripts to use new paths.

        In a real implementation, this could automatically update script imports.
        """
        print("\n" + "=" * 70)
        print("SCRIPT UPDATE INSTRUCTIONS")
        print("=" * 70)
        print(
            """
After reorganization, update your scripts to use the new paths:

OLD PATH:
    data/Floodingnaque_CADS-S0126006_NAIA Daily Data.csv

NEW PATH:
    data/raw/pagasa/Floodingnaque_CADS-S0126006_NAIA Daily Data.csv

Scripts that may need updates:
    - scripts/preprocess_pagasa_data.py
    - scripts/clean_raw_flood_records.py
    - scripts/merge_datasets.py

You can use these constants:
    RAW_PAGASA_DIR = DATA_DIR / "raw" / "pagasa"
    RAW_FLOOD_DIR = DATA_DIR / "raw" / "flood_records"
"""
        )
        print("=" * 70)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Reorganize data directory to separate raw from processed files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without moving files",
    )
    parser.add_argument(
        "--revert",
        action="store_true",
        help="Revert previous reorganization using manifest",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Override data directory path",
    )

    args = parser.parse_args()

    reorganizer = DataReorganizer(data_dir=args.data_dir)

    if args.revert:
        success = reorganizer.revert()
        if success:
            print("\n✓ Reorganization reverted successfully")
        else:
            print("\n✗ Revert failed")
        return

    print("\n" + "=" * 70)
    print("DATA DIRECTORY REORGANIZATION")
    print("=" * 70)

    if args.dry_run:
        print("\n*** DRY RUN - No files will be moved ***\n")

    # Create directory structure
    if not args.dry_run:
        reorganizer.create_raw_subdirs()

    # Move files
    moves = reorganizer.move_files(dry_run=args.dry_run)

    if moves:
        # Create README
        reorganizer.create_readme(dry_run=args.dry_run)

        # Save manifest
        reorganizer.save_manifest(dry_run=args.dry_run)

        # Show update instructions
        reorganizer.update_scripts(dry_run=args.dry_run)

    print("\n" + "=" * 70)
    if args.dry_run:
        print(f"DRY RUN COMPLETE - {len(moves)} files would be moved")
        print("Run without --dry-run to execute")
    else:
        print(f"REORGANIZATION COMPLETE - {len(moves)} files moved")
        print("Use --revert to undo if needed")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
