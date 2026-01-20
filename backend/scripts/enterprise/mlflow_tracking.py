"""
MLflow Experiment Tracking Module
=================================

This module provides MLflow integration for experiment tracking, model logging,
and artifact management in the Floodingnaque training pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# Check if mlflow is available
try:
    import mlflow as _mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    _mlflow = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import mlflow
    from mlflow.entities import Run
    from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracker for Floodingnaque ML pipeline.

    Provides:
    - Automatic experiment creation
    - Run management with context managers
    - Metric and parameter logging
    - Model artifact storage
    - Comparison utilities
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "floodingnaque_flood_prediction",
        tags: Optional[Dict[str, str]] = None,
        enabled: bool = True,
    ):
        """
        Initialize MLflow tracker.

        Args:
            tracking_uri: MLflow tracking server URI (local or remote)
            experiment_name: Name of the experiment
            tags: Default tags to apply to all runs
            enabled: Whether tracking is enabled
        """
        self.enabled = enabled and MLFLOW_AVAILABLE
        self.experiment_name = experiment_name
        self.default_tags = tags or {}

        if not self.enabled:
            if not MLFLOW_AVAILABLE:
                logger.warning("MLflow not installed. Tracking disabled.")
            return

        # Set tracking URI
        if tracking_uri:
            _mlflow.set_tracking_uri(tracking_uri)  # type: ignore[union-attr]

        # Set or create experiment
        self._setup_experiment()

        # Initialize client
        from mlflow.tracking import MlflowClient as _MlflowClient

        self.client: Optional[MlflowClient] = _MlflowClient()

        logger.info(f"MLflow tracking initialized: {experiment_name}")

    def _setup_experiment(self) -> None:
        """Set up or create MLflow experiment."""
        experiment = _mlflow.get_experiment_by_name(self.experiment_name)  # type: ignore[union-attr]

        if experiment is None:
            experiment_id = _mlflow.create_experiment(  # type: ignore[union-attr]
                self.experiment_name, tags=self.default_tags
            )
            logger.info(f"Created new experiment: {self.experiment_name} (ID: {experiment_id})")
        else:
            _mlflow.set_experiment(self.experiment_name)  # type: ignore[union-attr]
            logger.info(f"Using existing experiment: {self.experiment_name}")

    @contextmanager
    def start_run(
        self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None, description: Optional[str] = None
    ):
        """
        Context manager for MLflow run.

        Args:
            run_name: Name for the run
            tags: Additional tags for this run
            description: Run description

        Yields:
            MLflow run object or None if disabled
        """
        if not self.enabled:
            yield None
            return

        # Combine default and run-specific tags
        all_tags = {**self.default_tags, **(tags or {})}

        if description:
            all_tags["mlflow.note.content"] = description

        with _mlflow.start_run(run_name=run_name, tags=all_tags) as run:  # type: ignore[union-attr]
            logger.info(f"Started MLflow run: {run.info.run_id}")
            yield run
            logger.info(f"Completed MLflow run: {run.info.run_id}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log parameters to the active run."""
        if not self.enabled:
            return

        # Flatten nested params
        flat_params = self._flatten_dict(params)

        # MLflow has param value length limit, truncate if needed
        for key, value in flat_params.items():
            str_value = str(value)
            if len(str_value) > 500:
                str_value = str_value[:497] + "..."
            _mlflow.log_param(key, str_value)  # type: ignore[union-attr]

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None) -> None:
        """Log metrics to the active run."""
        if not self.enabled:
            return

        # Flatten nested metrics
        flat_metrics = self._flatten_dict(metrics)

        for key, value in flat_metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                _mlflow.log_metric(key, float(value), step=step)  # type: ignore[union-attr]

    def log_model(
        self,
        model: Any,  # BaseEstimator but may have predict method
        artifact_path: str = "model",
        input_example: Optional[pd.DataFrame] = None,
        registered_model_name: Optional[str] = None,
    ) -> None:
        """
        Log sklearn model to MLflow.

        Args:
            model: Trained sklearn model
            artifact_path: Path in artifact store
            input_example: Example input for signature inference
            registered_model_name: If provided, register model
        """
        if not self.enabled:
            return

        # Infer signature if example provided
        signature = None
        if input_example is not None:
            try:
                from mlflow.models import infer_signature

                predictions = model.predict(input_example)
                signature = infer_signature(input_example, predictions)
            except Exception as e:
                logger.warning(f"Could not infer signature: {e}")

        _mlflow.sklearn.log_model(  # type: ignore[union-attr]
            model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )

        logger.info(f"Logged model to {artifact_path}")

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file as artifact."""
        if not self.enabled:
            return

        _mlflow.log_artifact(local_path, artifact_path)  # type: ignore[union-attr]

    def log_figure(self, figure: Any, artifact_file: str) -> None:
        """Log a matplotlib figure as artifact."""
        if not self.enabled:
            return

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            figure.savefig(tmp.name, dpi=150, bbox_inches="tight")
            _mlflow.log_artifact(tmp.name, artifact_path="figures")  # type: ignore[union-attr]
            os.unlink(tmp.name)

    def log_dict(self, data: Dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary as JSON artifact."""
        if not self.enabled:
            return

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(data, tmp, indent=2, default=str)
            tmp.flush()
            _mlflow.log_artifact(tmp.name, artifact_path="data")  # type: ignore[union-attr]
            os.unlink(tmp.name)

    def log_dataframe(self, df: pd.DataFrame, artifact_file: str) -> None:
        """Log a DataFrame as CSV artifact."""
        if not self.enabled:
            return

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
            df.to_csv(tmp.name, index=False)
            _mlflow.log_artifact(tmp.name, artifact_path="data")  # type: ignore[union-attr]
            os.unlink(tmp.name)

    def log_training_info(
        self, version: int, features: List[str], dataset_size: int, class_distribution: Dict[int, int]
    ) -> None:
        """Log training metadata."""
        if not self.enabled:
            return

        _mlflow.log_params(
            {  # type: ignore[union-attr]
                "version": version,
                "n_features": len(features),
                "dataset_size": dataset_size,
                "n_flood_samples": class_distribution.get(1, 0),
                "n_non_flood_samples": class_distribution.get(0, 0),
            }
        )

        # Log features as artifact
        self.log_dict({"features": features}, "features.json")

    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the active run."""
        if not self.enabled:
            return
        _mlflow.set_tag(key, value)  # type: ignore[union-attr]

    def get_best_run(self, metric: str = "f1_score", ascending: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.

        Args:
            metric: Metric to optimize
            ascending: If True, lower is better

        Returns:
            Best run info or None
        """
        if not self.enabled or self.client is None:
            return None

        experiment = _mlflow.get_experiment_by_name(self.experiment_name)  # type: ignore[union-attr]
        if experiment is None:
            return None

        order = "ASC" if ascending else "DESC"
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id], order_by=[f"metrics.{metric} {order}"], max_results=1
        )

        if not runs:
            return None

        best_run = runs[0]
        return {
            "run_id": best_run.info.run_id,
            "metrics": best_run.data.metrics,
            "params": best_run.data.params,
            "tags": best_run.data.tags,
        }

    def compare_runs(self, metrics: Optional[List[str]] = None, max_runs: int = 10) -> pd.DataFrame:
        """
        Get comparison DataFrame of recent runs.

        Args:
            metrics: Metrics to include
            max_runs: Maximum runs to compare

        Returns:
            DataFrame with run comparison
        """
        if not self.enabled or self.client is None:
            return pd.DataFrame()

        if metrics is None:
            metrics = ["accuracy", "f1_score", "roc_auc", "cv_mean"]

        experiment = _mlflow.get_experiment_by_name(self.experiment_name)  # type: ignore[union-attr]
        if experiment is None:
            return pd.DataFrame()

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id], max_results=max_runs, order_by=["start_time DESC"]
        )

        data = []
        for run in runs:
            row: Dict[str, Any] = {
                "run_id": run.info.run_id[:8],
                "run_name": run.data.tags.get("mlflow.runName", ""),
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M"),
            }

            # Add metrics
            for metric in metrics:
                row[metric] = run.data.metrics.get(metric)

            # Add key params
            row["version"] = run.data.params.get("version")
            row["n_estimators"] = run.data.params.get("n_estimators")

            data.append(row)

        return pd.DataFrame(data)

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, (list, tuple)):
                # Convert lists to string representation
                items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)


def create_tracker_from_config(config: Dict[str, Any]) -> MLflowTracker:
    """
    Create MLflow tracker from configuration dictionary.

    Args:
        config: MLflow configuration from YAML

    Returns:
        Configured MLflowTracker instance
    """
    return MLflowTracker(
        tracking_uri=config.get("tracking_uri"),
        experiment_name=config.get("experiment_name", "floodingnaque_flood_prediction"),
        tags=config.get("tags"),
        enabled=True,
    )


# Convenience function for simple use cases
def log_training_run(
    model: Any,
    metrics: Dict[str, float],
    params: Dict[str, Any],
    features: List[str],
    version: int,
    experiment_name: str = "floodingnaque_flood_prediction",
) -> Optional[str]:
    """
    Simple function to log a complete training run.

    Args:
        model: Trained model
        metrics: Model metrics
        params: Model parameters
        features: Feature list
        version: Model version
        experiment_name: MLflow experiment name

    Returns:
        Run ID if successful, None otherwise
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available. Skipping tracking.")
        return None

    tracker = MLflowTracker(experiment_name=experiment_name)

    with tracker.start_run(run_name=f"v{version}_{datetime.now():%Y%m%d_%H%M%S}") as run:
        tracker.log_params(params)
        tracker.log_metrics(metrics)
        tracker.log_training_info(
            version=version,
            features=features,
            dataset_size=params.get("dataset_size", 0),
            class_distribution=params.get("class_distribution", {}),
        )
        tracker.log_model(model, registered_model_name=f"flood_model_v{version}")

        return run.info.run_id if run else None
