"""
Unit tests for merge_datasets.py script.

Tests cover:
- Dataset discovery and loading
- Merge operations
- Error handling for file operations
- Duplicate removal
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


class TestFindDatasets:
    """Tests for find_datasets function."""

    def test_finds_matching_files(self, tmp_path):
        """Test that matching datasets are found."""
        from scripts.merge_datasets import find_datasets

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        # Create test files
        (processed_dir / "cumulative_2024.csv").write_text("data")
        (processed_dir / "pagasa_daily.csv").write_text("data")
        (processed_dir / "processed_flood_records.csv").write_text("data")
        (processed_dir / "other_file.csv").write_text("data")  # Should not match

        with patch("scripts.merge_datasets.PROCESSED_DIR", processed_dir):
            datasets = find_datasets()

        assert len(datasets) == 3

    def test_returns_empty_when_no_matches(self, tmp_path):
        """Test empty list when no matching files."""
        from scripts.merge_datasets import find_datasets

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        with patch("scripts.merge_datasets.PROCESSED_DIR", processed_dir):
            datasets = find_datasets()

        assert datasets == []


class TestMergeDatasets:
    """Tests for merge_datasets function."""

    def test_merges_multiple_datasets(self, tmp_path):
        """Test successful merge of multiple datasets."""
        from scripts.merge_datasets import merge_datasets

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        # Create test datasets
        df1 = pd.DataFrame({"temperature": [25, 26], "humidity": [80, 85], "precipitation": [10, 20], "flood": [0, 1]})
        df2 = pd.DataFrame({"temperature": [27, 28], "humidity": [75, 90], "precipitation": [5, 15], "flood": [0, 0]})

        file1 = processed_dir / "dataset1.csv"
        file2 = processed_dir / "dataset2.csv"
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)

        with patch("scripts.merge_datasets.PROCESSED_DIR", processed_dir):
            result = merge_datasets([file1, file2])

        assert len(result) == 4
        assert "source_file" in result.columns

    def test_empty_datasets_list(self):
        """Test error when no datasets provided."""
        from scripts.merge_datasets import merge_datasets

        with pytest.raises(ValueError, match="No datasets to merge"):
            merge_datasets([])

    def test_removes_duplicates(self, tmp_path):
        """Test that duplicates are removed."""
        from scripts.merge_datasets import merge_datasets

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        # Create datasets with duplicates
        df1 = pd.DataFrame({"temperature": [25, 26], "humidity": [80, 85], "precipitation": [10, 20], "flood": [0, 1]})
        df2 = pd.DataFrame(
            {"temperature": [25, 27], "humidity": [80, 90], "precipitation": [10, 15], "flood": [0, 0]}  # Duplicate
        )

        file1 = processed_dir / "dataset1.csv"
        file2 = processed_dir / "dataset2.csv"
        df1.to_csv(file1, index=False)
        df2.to_csv(file2, index=False)

        with patch("scripts.merge_datasets.PROCESSED_DIR", processed_dir):
            result = merge_datasets([file1, file2])

        # Should have 3 rows (one duplicate removed)
        assert len(result) == 3

    def test_handles_missing_file(self, tmp_path, caplog):
        """Test handling of missing files."""
        from scripts.merge_datasets import merge_datasets

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        # Create one valid file
        df1 = pd.DataFrame({"temperature": [25], "humidity": [80], "precipitation": [10], "flood": [0]})
        file1 = processed_dir / "dataset1.csv"
        df1.to_csv(file1, index=False)

        # Reference a non-existent file
        file2 = processed_dir / "nonexistent.csv"

        with patch("scripts.merge_datasets.PROCESSED_DIR", processed_dir):
            result = merge_datasets([file1, file2])

        # Should still work with just the valid file
        assert len(result) == 1

    def test_handles_empty_csv(self, tmp_path, caplog):
        """Test handling of empty CSV files."""
        from scripts.merge_datasets import merge_datasets

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        # Create valid file
        df1 = pd.DataFrame({"temperature": [25], "humidity": [80], "precipitation": [10], "flood": [0]})
        file1 = processed_dir / "dataset1.csv"
        df1.to_csv(file1, index=False)

        # Create empty file
        file2 = processed_dir / "empty.csv"
        file2.write_text("")

        with patch("scripts.merge_datasets.PROCESSED_DIR", processed_dir):
            result = merge_datasets([file1, file2])

        assert len(result) == 1

    def test_handles_parse_error(self, tmp_path, caplog):
        """Test handling of CSV parse errors."""
        from scripts.merge_datasets import merge_datasets

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        # Create valid file
        df1 = pd.DataFrame({"temperature": [25], "humidity": [80], "precipitation": [10], "flood": [0]})
        file1 = processed_dir / "dataset1.csv"
        df1.to_csv(file1, index=False)

        # Create file with invalid content (binary)
        file2 = processed_dir / "malformed.csv"
        file2.write_bytes(b"\x00\x01\x02\x03")

        with patch("scripts.merge_datasets.PROCESSED_DIR", processed_dir):
            result = merge_datasets([file1, file2])

        assert len(result) == 1

    def test_all_files_fail(self, tmp_path):
        """Test error when all files fail to load."""
        from scripts.merge_datasets import merge_datasets

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        # Create only malformed files
        file1 = processed_dir / "malformed1.csv"
        file1.write_text("")

        file2 = processed_dir / "malformed2.csv"
        file2.write_text("")

        with patch("scripts.merge_datasets.PROCESSED_DIR", processed_dir):
            with pytest.raises(ValueError, match="Could not load any datasets"):
                merge_datasets([file1, file2])

    def test_output_permission_error(self, tmp_path):
        """Test handling of write permission error."""
        from scripts.merge_datasets import merge_datasets

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        df1 = pd.DataFrame({"temperature": [25], "humidity": [80], "precipitation": [10], "flood": [0]})
        file1 = processed_dir / "dataset1.csv"
        df1.to_csv(file1, index=False)

        # Mock to_csv to raise permission error
        with patch("scripts.merge_datasets.PROCESSED_DIR", processed_dir):
            with patch.object(pd.DataFrame, "to_csv", side_effect=PermissionError("Permission denied")):
                with pytest.raises(IOError, match="permission denied"):
                    merge_datasets([file1])
