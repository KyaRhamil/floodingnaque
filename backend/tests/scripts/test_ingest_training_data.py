"""
Unit tests for ingest_training_data.py script.

Tests cover:
- File ingestion
- Directory ingestion
- Data validation
- Dry-run mode
- Weather and tide data merging
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestValidateData:
    """Tests for validate_data function."""

    def test_valid_data(self):
        """Test validation passes for valid data."""
        from scripts.ingest_training_data import validate_data

        df = pd.DataFrame({"temperature": [25], "humidity": [80], "precipitation": [10], "flood": [0]})

        assert validate_data(df) is True

    def test_missing_required_columns(self):
        """Test validation fails for missing columns."""
        from scripts.ingest_training_data import validate_data

        df = pd.DataFrame({"temperature": [25], "humidity": [80]})  # Missing precipitation and flood

        assert validate_data(df) is False


class TestIngestFile:
    """Tests for ingest_file function."""

    def test_successful_ingestion(self, tmp_path):
        """Test successful file ingestion."""
        from scripts.ingest_training_data import ingest_file

        # Create test file
        data_file = tmp_path / "test_data.csv"
        df = pd.DataFrame({"temperature": [25], "humidity": [80], "precipitation": [10], "flood": [0]})
        df.to_csv(data_file, index=False)

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        with patch("scripts.ingest_training_data.PROCESSED_DIR", processed_dir):
            result = ingest_file(data_file)

        assert len(result) == 1
        assert (processed_dir / "ingested_test_data.csv").exists()

    def test_file_not_found(self, tmp_path):
        """Test error when file doesn't exist."""
        from scripts.ingest_training_data import ingest_file

        with pytest.raises(FileNotFoundError):
            ingest_file(tmp_path / "nonexistent.csv")

    def test_invalid_data_format(self, tmp_path):
        """Test error for invalid data format."""
        from scripts.ingest_training_data import ingest_file

        data_file = tmp_path / "invalid.csv"
        df = pd.DataFrame({"col1": [1], "col2": [2]})  # Missing required columns
        df.to_csv(data_file, index=False)

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        with patch("scripts.ingest_training_data.PROCESSED_DIR", processed_dir):
            with pytest.raises(ValueError, match="Invalid data format"):
                ingest_file(data_file)

    def test_dry_run_no_file_created(self, tmp_path):
        """Test dry run doesn't create output file."""
        from scripts.ingest_training_data import ingest_file

        data_file = tmp_path / "test_data.csv"
        df = pd.DataFrame({"temperature": [25], "humidity": [80], "precipitation": [10], "flood": [0]})
        df.to_csv(data_file, index=False)

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        with patch("scripts.ingest_training_data.PROCESSED_DIR", processed_dir):
            result = ingest_file(data_file, dry_run=True)

        assert len(result) == 1
        assert not (processed_dir / "ingested_test_data.csv").exists()


class TestIngestDirectory:
    """Tests for ingest_directory function."""

    def test_ingests_all_csv_files(self, tmp_path):
        """Test all CSV files in directory are ingested."""
        from scripts.ingest_training_data import ingest_directory

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create multiple test files
        for i in range(3):
            df = pd.DataFrame({"temperature": [25 + i], "humidity": [80], "precipitation": [10], "flood": [0]})
            df.to_csv(data_dir / f"test_{i}.csv", index=False)

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        with patch("scripts.ingest_training_data.PROCESSED_DIR", processed_dir):
            count = ingest_directory(data_dir)

        assert count == 3

    def test_directory_not_found(self, tmp_path):
        """Test error when directory doesn't exist."""
        from scripts.ingest_training_data import ingest_directory

        with pytest.raises(FileNotFoundError):
            ingest_directory(tmp_path / "nonexistent")

    def test_dry_run_no_files_created(self, tmp_path):
        """Test dry run doesn't create files."""
        from scripts.ingest_training_data import ingest_directory

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        df = pd.DataFrame({"temperature": [25], "humidity": [80], "precipitation": [10], "flood": [0]})
        df.to_csv(data_dir / "test.csv", index=False)

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        with patch("scripts.ingest_training_data.PROCESSED_DIR", processed_dir):
            count = ingest_directory(data_dir, dry_run=True)

        assert count == 1
        assert not list(processed_dir.glob("ingested_*.csv"))


class TestMergeWeatherAndTideData:
    """Tests for merge_weather_and_tide_data function."""

    def test_merges_weather_and_tide(self):
        """Test successful merge of weather and tide data."""
        from scripts.ingest_training_data import merge_weather_and_tide_data

        weather_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
                "temperature": [25, 26, 27],
                "humidity": [80, 81, 82],
            }
        )

        tide_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
                "tide_height": [1.5, 1.8, 2.0],
                "datum": ["MSL", "MSL", "MSL"],
            }
        )

        result = merge_weather_and_tide_data(weather_df, tide_df)

        assert "tide_height" in result.columns
        assert "has_tide_data" in result.columns
        assert len(result) == 3

    def test_handles_no_tide_data(self):
        """Test handling when no tide data available."""
        from scripts.ingest_training_data import merge_weather_and_tide_data

        weather_df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-01", periods=3, freq="h"), "temperature": [25, 26, 27]}
        )

        result = merge_weather_and_tide_data(weather_df, None)

        assert "tide_height" in result.columns
        assert all(result["tide_height"] == 0.0)
        assert all(result["has_tide_data"] == False)  # noqa: E712

    def test_handles_empty_weather_data(self):
        """Test handling of empty weather data."""
        from scripts.ingest_training_data import merge_weather_and_tide_data

        weather_df = pd.DataFrame()
        tide_df = pd.DataFrame({"timestamp": ["2024-01-01"], "tide_height": [1.5]})

        result = merge_weather_and_tide_data(weather_df, tide_df)

        assert result.empty


class TestProcessAndMergeFetchedData:
    """Tests for process_and_merge_fetched_data function."""

    def test_dry_run_no_files_saved(self, tmp_path):
        """Test dry run doesn't save files."""
        from scripts.ingest_training_data import process_and_merge_fetched_data

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        weather_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
                "temperature": [25, 26, 27],
                "humidity": [80, 81, 82],
                "precipitation": [0, 5, 10],
                "flood": [0, 0, 1],
            }
        )

        fetched_data = {"Meteostat": weather_df}

        with patch("scripts.ingest_training_data.PROCESSED_DIR", processed_dir):
            args = MagicMock()
            count = process_and_merge_fetched_data(fetched_data, args, dry_run=True)

        assert count > 0
        # Files should not be created in dry run
        assert not list(processed_dir.glob("fetched_*.csv"))
        assert not (processed_dir / "training_data_merged.csv").exists()

    def test_saves_files_when_not_dry_run(self, tmp_path):
        """Test files are saved when not in dry run."""
        from scripts.ingest_training_data import process_and_merge_fetched_data

        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        weather_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="h"),
                "temperature": [25, 26, 27],
                "humidity": [80, 81, 82],
                "precipitation": [0, 5, 10],
                "flood": [0, 0, 1],
            }
        )

        fetched_data = {"Meteostat": weather_df}

        with patch("scripts.ingest_training_data.PROCESSED_DIR", processed_dir):
            args = MagicMock()
            count = process_and_merge_fetched_data(fetched_data, args, dry_run=False)

        assert count > 0
        assert (processed_dir / "fetched_meteostat.csv").exists()
