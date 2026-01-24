#!/usr/bin/env python3
"""
Validate data integrity using checksums.json.

This script verifies that all data files match their expected checksums
to detect corruption or unauthorized modifications.
"""

import hashlib
import json
import sys
from pathlib import Path


def compute_sha256(filepath: Path) -> str:
    """Compute SHA256 checksum for a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_checksums(data_dir: Path) -> tuple[int, int, int]:
    """
    Validate all files against checksums.json.

    Returns:
        Tuple of (passed, failed, missing) counts
    """
    checksums_path = data_dir / "checksums.json"

    if not checksums_path.exists():
        print("Error: checksums.json not found!")
        print("Run 'python scripts/generate_checksums.py' to create it.")
        return 0, 0, 0

    with open(checksums_path, "r", encoding="utf-8") as f:
        checksums = json.load(f)

    passed = 0
    failed = 0
    missing = 0

    print("Validating data integrity...")
    print("-" * 60)

    for rel_path, expected in checksums.get("files", {}).items():
        filepath = data_dir / rel_path

        if not filepath.exists():
            print(f"  ✗ MISSING: {rel_path}")
            missing += 1
            continue

        actual_hash = compute_sha256(filepath)

        if actual_hash == expected["sha256"]:
            print(f"  ✓ OK: {rel_path}")
            passed += 1
        else:
            print(f"  ✗ MISMATCH: {rel_path}")
            print(f"      Expected: {expected['sha256'][:16]}...")
            print(f"      Actual:   {actual_hash[:16]}...")
            failed += 1

    return passed, failed, missing


def main():
    """Main entry point."""
    # Find data directory
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "data"

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    passed, failed, missing = validate_checksums(data_dir)

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed, {missing} missing")

    if failed > 0 or missing > 0:
        print("\n⚠️  Data integrity check FAILED!")
        print("Some files may be corrupted or have been modified.")
        return 1

    print("\n✓ All files passed integrity check!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
