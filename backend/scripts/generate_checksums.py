#!/usr/bin/env python3
"""
Generate checksums.json for data integrity verification.

This script computes SHA256 and MD5 checksums for all CSV files in the
data directory to enable integrity verification.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def compute_file_checksum(filepath: Path) -> dict:
    """Compute SHA256 and MD5 checksums for a file."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)  # MD5 used for integrity, not security

    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
            md5.update(chunk)

    return {
        "sha256": sha256.hexdigest(),
        "md5": md5.hexdigest(),
        "size_bytes": os.path.getsize(filepath),
        "last_modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat(),
    }


def generate_checksums(data_dir: Path) -> dict:
    """Generate checksums for all data files."""
    checksums = {
        "_metadata": {
            "generated_at": datetime.now().isoformat(),
            "generator": "scripts/generate_checksums.py",
            "description": "Data integrity checksums for Floodingnaque datasets",
            "usage": "Run 'python scripts/validate_checksums.py' to verify data integrity",
        },
        "files": {},
    }

    # Patterns to include
    patterns = ["*.csv", "processed/*.csv", "cleaned/*.csv"]

    for pattern in patterns:
        for filepath in sorted(data_dir.glob(pattern)):
            if filepath.is_file():
                rel_path = str(filepath.relative_to(data_dir)).replace("\\", "/")
                checksums["files"][rel_path] = compute_file_checksum(filepath)
                print(f"  ✓ {rel_path}")

    return checksums


def main():
    """Main entry point."""
    # Find data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    print("Generating data checksums...")
    print("-" * 50)

    checksums = generate_checksums(data_dir)

    output_path = data_dir / "checksums.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(checksums, f, indent=2)

    print("-" * 50)
    print(f"Total files: {len(checksums['files'])}")
    print(f"Saved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
