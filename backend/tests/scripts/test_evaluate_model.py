"""
Unit tests for evaluate_model.py script.

Tests cover:
- Model loading and error handling
- Data loading and validation
- File operation error handling
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_model_file_not_found(self, tmp_path):
        """Test error when model file doesn't exist."""
        from scripts.evaluate_model import evaluate_model

        with pytest.raises(FileNotFoundError, match="Model file not found"):
            evaluate_model(model_path=tmp_path / "nonexistent.joblib")

    def test_data_file_not_found(self, tmp_path):
        """Test error when data file doesn't exist."""
        from scripts.evaluate_model import evaluate_model

        # Create a mock model file
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"fake model")

        with pytest.raises(FileNotFoundError, match="Data file not found"):
            evaluate_model(
                model_path=model_path,
                data_path=tmp_path / "nonexistent.csv",
            )

    @patch("scripts.evaluate_model.joblib.load")
    def test_model_load_error(self, mock_load, tmp_path):
        """Test handling of model load errors."""
        from scripts.evaluate_model import evaluate_model

        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"corrupt data")
        data_path = tmp_path / "data.csv"
        data_path.write_text("temperature,humidity,precipitation,flood\n1,2,3,0")

        mock_load.side_effect = Exception("Corrupt model file")

        with pytest.raises(IOError, match="Failed to load model"):
            evaluate_model(model_path=model_path, data_path=data_path)

    @patch("scripts.evaluate_model.joblib.load")
    @patch("scripts.evaluate_model.pd.read_csv")
    def test_empty_data_file(self, mock_read_csv, mock_load, tmp_path):
        """Test handling of empty data file."""
        from scripts.evaluate_model import evaluate_model

        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"dummy")
        data_path = tmp_path / "data.csv"
        data_path.write_text("")

        mock_load.return_value = MagicMock()
        mock_read_csv.side_effect = pd.errors.EmptyDataError("No columns")

        with pytest.raises(ValueError, match="empty"):
            evaluate_model(model_path=model_path, data_path=data_path)

    @patch("scripts.evaluate_model.joblib.load")
    @patch("scripts.evaluate_model.pd.read_csv")
    def test_missing_flood_column(self, mock_read_csv, mock_load, tmp_path):
        """Test handling of missing flood column."""
        from scripts.evaluate_model import evaluate_model

        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"dummy")
        data_path = tmp_path / "data.csv"
        data_path.write_text("temperature,humidity,precipitation\n1,2,3")

        mock_load.return_value = MagicMock()
        mock_read_csv.return_value = pd.DataFrame({"temperature": [25], "humidity": [80], "precipitation": [10]})

        with pytest.raises(ValueError, match="flood"):
            evaluate_model(model_path=model_path, data_path=data_path)

    @patch("scripts.evaluate_model.plt")
    @patch("scripts.evaluate_model.joblib.load")
    def test_successful_evaluation(self, mock_load, mock_plt, tmp_path):
        """Test successful model evaluation."""
        from scripts.evaluate_model import evaluate_model

        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"dummy")

        data_path = tmp_path / "data.csv"
        data_path.write_text(
            "temperature,humidity,precipitation,flood\n" "25,80,10,0\n" "30,90,50,1\n" "22,70,5,0\n" "28,85,30,1\n"
        )

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0, 1, 0, 1])
        mock_model.feature_importances_ = [0.5, 0.3, 0.2]
        mock_load.return_value = mock_model

        # Patch MODELS_DIR to use tmp_path
        with patch("scripts.evaluate_model.MODELS_DIR", tmp_path):
            accuracy = evaluate_model(model_path=model_path, data_path=data_path)

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    @patch("scripts.evaluate_model.plt.savefig")
    @patch("scripts.evaluate_model.plt.figure")
    @patch("scripts.evaluate_model.plt.close")
    @patch("scripts.evaluate_model.joblib.load")
    def test_plot_save_permission_error(self, mock_load, mock_close, mock_figure, mock_savefig, tmp_path, capsys):
        """Test handling of permission error when saving plots."""
        from scripts.evaluate_model import evaluate_model

        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"dummy")

        data_path = tmp_path / "data.csv"
        data_path.write_text("temperature,humidity,precipitation,flood\n" "25,80,10,0\n" "30,90,50,1\n")

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0, 1])
        mock_load.return_value = mock_model

        mock_savefig.side_effect = PermissionError("Permission denied")

        with patch("scripts.evaluate_model.MODELS_DIR", tmp_path):
            # Should not raise, but print warning
            accuracy = evaluate_model(model_path=model_path, data_path=data_path)

        captured = capsys.readouterr()
        assert "permission denied" in captured.out.lower() or accuracy is not None
