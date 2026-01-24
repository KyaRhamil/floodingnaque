#!/usr/bin/env python
"""
CSV Compression Utility
=======================

Compresses large CSV files using gzip to reduce storage space while maintaining
data integrity. Provides both compression and decompression functionality.

Usage:
    # Compress a single file
    python scripts/compress_csv.py data/processed/cumulative_v2_up_to_2025.csv

    # Compress all large CSVs in a directory
    python scripts/compress_csv.py --dir data/processed --threshold 5

    # Decompress a file
    python scripts/compress_csv.py --decompress data/processed/cumulative_v2_up_to_2025.csv.gz

    # List compression candidates
    python scripts/compress_csv.py --scan data/processed

Options:
    --threshold     Minimum file size in MB to compress (default: 1)
    --keep-original Keep original file after compression
    --level         Compression level 1-9 (default: 9, best compression)
    --decompress    Decompress a .gz file

Author: Floodingnaque Team
Date: 2026-01-24
"""

import argparse
import gzip
import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def get_file_hash(filepath: Path, chunk_size: int = 8192) -> str:
    """Calculate MD5 hash of a file."""
    hasher = hashlib.md5(usedforsecurity=False)  # Used for integrity, not security
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in megabytes."""
    return filepath.stat().st_size / (1024 * 1024)


def compress_csv(
    filepath: Path,
    output_path: Optional[Path] = None,
    compression_level: int = 9,
    keep_original: bool = False,
    verify: bool = True,
) -> Tuple[Path, Dict]:
    """
    Compress a CSV file using gzip.

    Args:
        filepath: Path to the CSV file
        output_path: Output path for compressed file (default: filepath.gz)
        compression_level: Gzip compression level (1-9)
        keep_original: Keep the original file after compression
        verify: Verify compressed file integrity

    Returns:
        Tuple of (output_path, stats_dict)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.suffix == ".gz":
        logger.warning(f"File already compressed: {filepath}")
        return filepath, {}

    output_path = output_path or filepath.with_suffix(filepath.suffix + ".gz")

    original_size = filepath.stat().st_size
    original_hash = get_file_hash(filepath) if verify else None

    logger.info(f"Compressing: {filepath}")
    logger.info(f"  Original size: {original_size / (1024*1024):.2f} MB")

    start_time = datetime.now()

    with open(filepath, "rb") as f_in:
        with gzip.open(output_path, "wb", compresslevel=compression_level) as f_out:
            shutil.copyfileobj(f_in, f_out)

    elapsed = (datetime.now() - start_time).total_seconds()
    compressed_size = output_path.stat().st_size
    ratio = (1 - compressed_size / original_size) * 100

    stats = {
        "original_file": str(filepath),
        "compressed_file": str(output_path),
        "original_size_bytes": original_size,
        "compressed_size_bytes": compressed_size,
        "compression_ratio": f"{ratio:.1f}%",
        "compression_level": compression_level,
        "original_hash": original_hash,
        "compressed_at": datetime.now().isoformat(),
        "duration_seconds": elapsed,
    }

    logger.info(f"  Compressed size: {compressed_size / (1024*1024):.2f} MB")
    logger.info(f"  Compression ratio: {ratio:.1f}%")
    logger.info(f"  Duration: {elapsed:.2f}s")

    # Verify integrity
    if verify:
        logger.info("  Verifying integrity...")
        # Decompress to memory and check hash
        with gzip.open(output_path, "rb") as f:
            hasher = hashlib.md5(usedforsecurity=False)  # Used for integrity, not security
            while chunk := f.read(8192):
                hasher.update(chunk)
            decompressed_hash = hasher.hexdigest()

        if decompressed_hash != original_hash:
            output_path.unlink()
            raise ValueError("Compression verification failed - hashes don't match!")

        logger.info("  Integrity verified ✓")
        stats["verified"] = True

    # Remove original if requested
    if not keep_original:
        filepath.unlink()
        logger.info(f"  Removed original file")
        stats["original_removed"] = True

    return output_path, stats


def decompress_csv(
    filepath: Path,
    output_path: Optional[Path] = None,
    keep_compressed: bool = False,
) -> Tuple[Path, Dict]:
    """
    Decompress a gzip-compressed CSV file.

    Args:
        filepath: Path to the .gz file
        output_path: Output path for decompressed file
        keep_compressed: Keep the compressed file after decompression

    Returns:
        Tuple of (output_path, stats_dict)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if not filepath.suffix == ".gz":
        raise ValueError(f"File is not gzip compressed: {filepath}")

    # Remove .gz suffix
    if output_path is None:
        output_path = filepath.with_suffix("")

    compressed_size = filepath.stat().st_size

    logger.info(f"Decompressing: {filepath}")

    start_time = datetime.now()

    with gzip.open(filepath, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    elapsed = (datetime.now() - start_time).total_seconds()
    decompressed_size = output_path.stat().st_size

    stats = {
        "compressed_file": str(filepath),
        "decompressed_file": str(output_path),
        "compressed_size_bytes": compressed_size,
        "decompressed_size_bytes": decompressed_size,
        "decompressed_at": datetime.now().isoformat(),
        "duration_seconds": elapsed,
    }

    logger.info(f"  Decompressed size: {decompressed_size / (1024*1024):.2f} MB")
    logger.info(f"  Duration: {elapsed:.2f}s")

    if not keep_compressed:
        filepath.unlink()
        logger.info(f"  Removed compressed file")
        stats["compressed_removed"] = True

    return output_path, stats


def scan_for_large_csvs(
    directory: Path,
    threshold_mb: float = 1.0,
) -> List[Dict]:
    """
    Scan a directory for large CSV files that could benefit from compression.

    Args:
        directory: Directory to scan
        threshold_mb: Minimum file size in MB to consider

    Returns:
        List of file info dictionaries
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    candidates = []

    for csv_file in directory.rglob("*.csv"):
        size_mb = get_file_size_mb(csv_file)

        if size_mb >= threshold_mb:
            # Check if already has compressed version
            gz_file = csv_file.with_suffix(csv_file.suffix + ".gz")
            has_compressed = gz_file.exists()

            candidates.append(
                {
                    "file": str(csv_file),
                    "size_mb": round(size_mb, 2),
                    "has_compressed": has_compressed,
                    "estimated_compressed_mb": round(size_mb * 0.15, 2),  # ~85% compression typical
                }
            )

    # Sort by size descending
    candidates.sort(key=lambda x: x["size_mb"], reverse=True)

    return candidates


def compress_directory(
    directory: Path,
    threshold_mb: float = 1.0,
    compression_level: int = 9,
    keep_original: bool = False,
) -> List[Dict]:
    """
    Compress all large CSV files in a directory.

    Args:
        directory: Directory to process
        threshold_mb: Minimum file size to compress
        compression_level: Gzip compression level
        keep_original: Keep original files

    Returns:
        List of compression stats for each file
    """
    candidates = scan_for_large_csvs(directory, threshold_mb)

    if not candidates:
        logger.info(f"No CSV files >= {threshold_mb}MB found in {directory}")
        return []

    results = []

    for candidate in candidates:
        if candidate["has_compressed"]:
            logger.info(f"Skipping (already compressed): {candidate['file']}")
            continue

        try:
            _, stats = compress_csv(
                Path(candidate["file"]),
                compression_level=compression_level,
                keep_original=keep_original,
            )
            results.append(stats)
        except Exception as e:
            logger.error(f"Failed to compress {candidate['file']}: {e}")
            results.append({"file": candidate["file"], "error": str(e)})

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compress or decompress CSV files using gzip",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="Path to CSV file or directory",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        help="Directory to process (compress all large CSVs)",
    )
    parser.add_argument(
        "--decompress",
        action="store_true",
        help="Decompress a .gz file",
    )
    parser.add_argument(
        "--scan",
        type=Path,
        help="Scan directory for compression candidates",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Minimum file size in MB to compress (default: 1)",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Keep original file after compression",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=9,
        choices=range(1, 10),
        help="Compression level 1-9 (default: 9)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path",
    )

    args = parser.parse_args()

    # Scan mode
    if args.scan:
        candidates = scan_for_large_csvs(args.scan, args.threshold)

        if not candidates:
            logger.info(f"No CSV files >= {args.threshold}MB found")
            return

        print(f"\nCompression Candidates (>= {args.threshold}MB):")
        print("-" * 70)

        total_size = 0
        total_estimated = 0

        for c in candidates:
            status = "[HAS .gz]" if c["has_compressed"] else ""
            print(f"  {c['size_mb']:8.2f} MB -> ~{c['estimated_compressed_mb']:6.2f} MB  {c['file']} {status}")
            total_size += c["size_mb"]
            total_estimated += c["estimated_compressed_mb"]

        print("-" * 70)
        print(
            f"  Total: {total_size:.2f} MB -> ~{total_estimated:.2f} MB (est. {(1-total_estimated/total_size)*100:.0f}% savings)"
        )
        return

    # Directory mode
    if args.dir:
        results = compress_directory(
            args.dir,
            threshold_mb=args.threshold,
            compression_level=args.level,
            keep_original=args.keep_original,
        )

        print(f"\nCompressed {len(results)} files")
        return

    # Single file mode
    if args.path:
        if args.decompress:
            output_path, stats = decompress_csv(
                args.path,
                output_path=args.output,
                keep_compressed=args.keep_original,
            )
            print(f"Decompressed: {output_path}")
        else:
            output_path, stats = compress_csv(
                args.path,
                output_path=args.output,
                compression_level=args.level,
                keep_original=args.keep_original,
            )
            print(f"Compressed: {output_path}")
            print(f"Compression ratio: {stats.get('compression_ratio', 'N/A')}")
        return

    # No arguments - compress default large files
    logger.info("No path specified. Scanning data/processed for large CSVs...")
    candidates = scan_for_large_csvs(PROCESSED_DIR, args.threshold)

    if candidates:
        print(f"\nFound {len(candidates)} large CSV files:")
        for c in candidates:
            print(f"  {c['size_mb']:8.2f} MB  {c['file']}")
        print(f"\nRun with --dir {PROCESSED_DIR} to compress all")
    else:
        print(f"No CSV files >= {args.threshold}MB found in {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
