#!/usr/bin/env python
"""
Enterprise Training Script
==========================

This script integrates all enterprise features:
- Centralized YAML configuration
- MLflow experiment tracking
- Pandera data validation
- Structured JSON logging
- Model registry with staging

Usage:
    python train_enterprise.py                     # Train default version (v6)
    python train_enterprise.py --version 4         # Train specific version
    python train_enterprise.py --grid-search       # Enable grid search
    python train_enterprise.py --promote staging   # Auto-promote if criteria met

Author: Floodingnaque Team
Date: 2026-01-19
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

# Import enterprise modules
from scripts.enterprise import (
    FloodDataValidator,
    MLflowTracker,
    ModelRegistry,
    ModelStage,
    TrainingLogger,
    create_registry,
    new_correlation_id,
    validate_training_data,
)

# Import configuration
try:
    from config import get_config

    config = get_config()
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    config = None

# Directories
MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = BACKEND_DIR / "reports"
LOGS_DIR = BACKEND_DIR / "logs"

# Ensure directories exist
for dir_path in [MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class EnterpriseTrainer:
    """
    Enterprise-grade model trainer with full MLOps integration.

    Features:
    - Configuration-driven training
    - Experiment tracking
    - Data validation
    - Structured logging
    - Model registry integration
    """

    # Default parameters (used if config not available)
    DEFAULT_PARAMS = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    GRID_PARAMS = {
        "n_estimators": [100, 150, 200, 250, 300],
        "max_depth": [10, 15, 20, 25, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.5],
    }

    VERSION_REGISTRY = {
        1: {"name": "v1_2022", "file": "cumulative_up_to_2022.csv", "desc": "2022 data only"},
        2: {"name": "v2_2023", "file": "cumulative_up_to_2023.csv", "desc": "2022-2023 data"},
        3: {"name": "v3_2024", "file": "cumulative_up_to_2024.csv", "desc": "2022-2024 data"},
        4: {"name": "v4_2025", "file": "cumulative_up_to_2025.csv", "desc": "2022-2025 data"},
        5: {"name": "v5_synthetic", "file": "synthetic_dataset.csv", "desc": "Synthetic data"},
        6: {"name": "v6_ultimate", "file": "cumulative_up_to_2025.csv", "desc": "Ultimate combined"},
    }

    CORE_FEATURES = [
        "temperature",
        "humidity",
        "precipitation",
        "is_monsoon_season",
        "month",
    ]

    INTERACTION_FEATURES = [
        "temp_humidity_interaction",
        "humidity_precip_interaction",
        "temp_precip_interaction",
        "monsoon_precip_interaction",
    ]

    def __init__(self, enable_mlflow: bool = True, enable_validation: bool = True, log_level: str = "INFO"):
        """
        Initialize enterprise trainer.

        Args:
            enable_mlflow: Enable MLflow experiment tracking
            enable_validation: Enable data validation
            log_level: Logging level
        """
        # Generate correlation ID for this training session
        self.correlation_id = new_correlation_id()

        # Setup logging
        self.logger = TrainingLogger(
            name="floodingnaque.enterprise",
            log_level=log_level,
            log_file=LOGS_DIR / "enterprise_training.log",
            json_format=True,
        )

        # Setup MLflow tracker
        self.mlflow_enabled = enable_mlflow
        if enable_mlflow:
            mlflow_config: Dict[str, Any] = {}
            if CONFIG_AVAILABLE and config is not None:
                mlflow_config = config.get_mlflow_config()
            self.tracker: Optional[MLflowTracker] = MLflowTracker(
                tracking_uri=mlflow_config.get("tracking_uri", str(BACKEND_DIR / "mlruns")),
                experiment_name=mlflow_config.get("experiment_name", "floodingnaque_enterprise"),
                tags={"correlation_id": self.correlation_id},
            )
        else:
            self.tracker = None

        # Setup data validator
        self.validation_enabled = enable_validation
        validation_config: Dict[str, Any] = {}
        if CONFIG_AVAILABLE and config is not None:
            validation_config = config.get("validation")  # type: ignore[assignment]
        self.validator = FloodDataValidator(validation_config)

        # Setup model registry
        self.registry = create_registry(MODELS_DIR)

        # Training state
        self.model: Optional[RandomForestClassifier] = None
        self.metrics: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.training_info: Dict[str, Any] = {}

        self.logger.info(
            "Enterprise trainer initialized",
            correlation_id=self.correlation_id,
            mlflow_enabled=enable_mlflow,
            validation_enabled=enable_validation,
        )

    def load_data(self, version: int) -> Optional[pd.DataFrame]:
        """
        Load data for a specific version.

        Args:
            version: Version number

        Returns:
            DataFrame or None if not found
        """
        if version not in self.VERSION_REGISTRY:
            self.logger.error(f"Unknown version: {version}")
            return None

        version_info = self.VERSION_REGISTRY[version]
        data_path = PROCESSED_DIR / version_info["file"]

        if not data_path.exists():
            self.logger.error(f"Data file not found: {data_path}")
            return None

        self.logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        self.logger.info("Data loaded", rows=len(df), columns=len(df.columns), version=version)

        return df

    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and optionally fix training data.

        Args:
            df: Input DataFrame

        Returns:
            Validated DataFrame
        """
        if not self.validation_enabled:
            return df

        self.logger.info("Validating training data")

        validated_df, errors = self.validator.validate(df, raise_on_error=False, fix_errors=True)

        if errors:
            self.logger.warning(
                f"Data validation found {len(errors)} issues", error_count=len(errors), errors=errors[:5]
            )
        else:
            self.logger.info("Data validation passed")

        return validated_df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (features, target)
        """
        df = df.copy()

        # Create interaction features
        if "temperature" in df.columns and "humidity" in df.columns:
            df["temp_humidity_interaction"] = df["temperature"] * df["humidity"] / 100

        if "humidity" in df.columns and "precipitation" in df.columns:
            df["humidity_precip_interaction"] = df["humidity"] * np.log1p(df["precipitation"])

        if "temperature" in df.columns and "precipitation" in df.columns:
            df["temp_precip_interaction"] = df["temperature"] * np.log1p(df["precipitation"])

        if "is_monsoon_season" in df.columns and "precipitation" in df.columns:
            df["monsoon_precip_interaction"] = df["is_monsoon_season"] * df["precipitation"]

        # Select available features
        all_features = self.CORE_FEATURES + self.INTERACTION_FEATURES
        available_features = [f for f in all_features if f in df.columns]

        X = df[available_features].copy()
        y = df["flood"].copy()

        # Handle missing values
        X = X.fillna(X.median())

        self.feature_names = list(X.columns)

        self.logger.info("Features prepared", n_features=len(available_features), features=available_features)

        return X, y

    def train(
        self, X: pd.DataFrame, y: pd.Series, version: int, use_grid_search: bool = False, cv_folds: int = 10
    ) -> Optional[RandomForestClassifier]:
        """
        Train model with optional hyperparameter optimization.

        Args:
            X: Features
            y: Target
            version: Model version
            use_grid_search: Enable grid search
            cv_folds: Cross-validation folds

        Returns:
            Trained model
        """
        start_time = time.time()

        # Log training start
        self.logger.training_started(version=version, features=len(X.columns), samples=len(X))

        # Split data
        random_state = 42
        test_size = 0.2
        if CONFIG_AVAILABLE and config is not None:
            random_state = config.random_state
            test_size = config.test_size

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.logger.info("Data split", train_size=len(X_train), test_size=len(X_test))

        # Get model parameters
        params: Dict[str, Any]
        if CONFIG_AVAILABLE and config is not None:
            params = config.get_model_params()
        else:
            params = self.DEFAULT_PARAMS.copy()

        # Train with or without grid search
        if use_grid_search:
            self.logger.info("Starting hyperparameter optimization (this may take a while)")

            grid_params = self.GRID_PARAMS
            if CONFIG_AVAILABLE and config is not None:
                grid_params = config.get_grid_search_params()

            base_model = RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1)

            search = RandomizedSearchCV(
                base_model,
                grid_params,
                n_iter=50,
                cv=StratifiedKFold(cv_folds, shuffle=True, random_state=random_state),
                scoring="f1_weighted",
                n_jobs=-1,
                random_state=random_state,
                verbose=1,
            )

            search.fit(X_train, y_train)
            # Cast to RandomForestClassifier since we know the base model type
            best_model = search.best_estimator_
            if isinstance(best_model, RandomForestClassifier):
                self.model = best_model
            else:
                self.model = RandomForestClassifier(**search.best_params_)
                self.model.fit(X_train, y_train)
            params = search.best_params_

            self.logger.info("Grid search complete", best_params=params, best_cv_score=float(search.best_score_))
        else:
            self.model = RandomForestClassifier(**params)
            self.model.fit(X_train, y_train)

        # Evaluate model
        self._evaluate(X_test, y_test, X, y, cv_folds)

        # Log training completion
        duration = time.time() - start_time
        self.logger.training_completed(version=version, metrics=self.metrics, duration_seconds=duration)

        # Store training info
        self.training_info = {
            "version": version,
            "parameters": params,
            "features": self.feature_names,
            "dataset_size": len(X),
            "class_distribution": y.value_counts().to_dict(),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "cv_folds": cv_folds,
            "training_duration_seconds": duration,
            "correlation_id": self.correlation_id,
        }

        return self.model

    def _evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series, X_full: pd.DataFrame, y_full: pd.Series, cv_folds: int
    ) -> None:
        """Evaluate model and store metrics."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Basic metrics
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        }

        # Cross-validation
        random_state = 42
        if CONFIG_AVAILABLE and config is not None:
            random_state = config.random_state
        cv = StratifiedKFold(cv_folds, shuffle=True, random_state=random_state)

        cv_scores = cross_val_score(self.model, X_full, y_full, cv=cv, scoring="f1_weighted", n_jobs=-1)

        self.metrics["cv_mean"] = float(cv_scores.mean())
        self.metrics["cv_std"] = float(cv_scores.std())
        self.metrics["cv_scores"] = [float(s) for s in cv_scores]

        # Feature importance
        feature_importance = dict(zip(self.feature_names, [float(imp) for imp in self.model.feature_importances_]))
        self.metrics["feature_importance"] = feature_importance

        self.logger.log_metrics(self.metrics)

    def save_model(self, version: int) -> Path:
        """
        Save model and metadata.

        Args:
            version: Model version

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Register in model registry
        _ = self.registry.register(
            model=self.model,
            version=version,
            name=self.VERSION_REGISTRY[version]["name"],
            metrics=self.metrics,
            parameters=self.training_info.get("parameters", {}),
            features=self.feature_names,
            description=self.VERSION_REGISTRY[version]["desc"],
            tags={"correlation_id": self.correlation_id},
        )

        # Also save to standard location for compatibility
        model_path = MODELS_DIR / f"flood_rf_model_v{version}.joblib"
        joblib.dump(self.model, model_path)

        # Save as latest
        latest_path = MODELS_DIR / "flood_rf_model_latest.joblib"
        joblib.dump(self.model, latest_path)

        # Save metadata
        metadata = {
            "version": version,
            "model_type": "RandomForestClassifier",
            "created_at": datetime.now().isoformat(),
            "correlation_id": self.correlation_id,
            **self.training_info,
            "metrics": self.metrics,
        }

        metadata_path = model_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.model_saved(path=str(model_path), version=version, metrics=self.metrics)

        return model_path

    def log_to_mlflow(self, version: int, X_sample: Optional[pd.DataFrame] = None) -> None:
        """Log run to MLflow."""
        if not self.mlflow_enabled or self.tracker is None:
            return

        version_info = self.VERSION_REGISTRY[version]

        with self.tracker.start_run(
            run_name=f"v{version}_{datetime.now():%Y%m%d_%H%M%S}", description=version_info["desc"]
        ):
            # Log parameters
            self.tracker.log_params(self.training_info.get("parameters", {}))
            self.tracker.log_params(
                {
                    "version": version,
                    "n_features": len(self.feature_names),
                    "dataset_size": self.training_info.get("dataset_size", 0),
                }
            )

            # Log metrics
            self.tracker.log_metrics(self.metrics)

            # Log model
            self.tracker.log_model(
                self.model,
                input_example=X_sample.head(5) if X_sample is not None else None,
                registered_model_name=f"flood_model_v{version}",
            )

            # Log training info
            self.tracker.log_dict(self.training_info, "training_info.json")

    def generate_report(self, version: int) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        report = {
            "report_type": "enterprise_training",
            "generated_at": datetime.now().isoformat(),
            "correlation_id": self.correlation_id,
            "version": version,
            "version_info": self.VERSION_REGISTRY[version],
            "training_info": self.training_info,
            "metrics": self.metrics,
            "registry_summary": self.registry.get_summary(),
        }

        # Save report
        report_path = REPORTS_DIR / "enterprise_training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Report saved to {report_path}")

        return report

    def promote(self, version: int, stage: str) -> Tuple[bool, str]:
        """
        Promote model to specified stage.

        Args:
            version: Model version
            stage: Target stage (staging/production)

        Returns:
            Tuple of (success, message)
        """
        target_stage = ModelStage.STAGING if stage == "staging" else ModelStage.PRODUCTION

        success, message = self.registry.promote(version, target_stage)

        if success:
            self.logger.info(f"Model v{version} promoted to {stage}")
        else:
            self.logger.warning(f"Promotion failed: {message}")

        return success, message

    def run(
        self, version: int = 6, use_grid_search: bool = False, cv_folds: int = 10, promote_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline.

        Args:
            version: Model version to train
            use_grid_search: Enable hyperparameter optimization
            cv_folds: Cross-validation folds
            promote_to: Auto-promote to stage (staging/production)

        Returns:
            Training report
        """
        with self.logger.operation("enterprise_training", version=version):
            # Load data
            df = self.load_data(version)
            if df is None:
                return {"error": f"Failed to load data for version {version}"}

            # Validate data
            df = self.validate_data(df)

            # Prepare features
            X, y = self.prepare_features(df)

            # Train model
            self.train(X, y, version, use_grid_search, cv_folds)

            # Save model
            self.save_model(version)

            # Log to MLflow
            self.log_to_mlflow(version, X)

            # Generate report
            report = self.generate_report(version)

            # Auto-promote if requested
            if promote_to:
                success, message = self.promote(version, promote_to)
                report["promotion"] = {"success": success, "message": message}

            return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enterprise Model Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_enterprise.py                      # Train v6 with defaults
    python train_enterprise.py --version 4          # Train v4
    python train_enterprise.py --grid-search        # Enable hyperparameter search
    python train_enterprise.py --promote staging    # Auto-promote to staging
        """,
    )

    parser.add_argument(
        "--version", "-v", type=int, default=6, choices=[1, 2, 3, 4, 5, 6], help="Model version to train (default: 6)"
    )

    parser.add_argument("--grid-search", "-g", action="store_true", help="Enable hyperparameter grid search")

    parser.add_argument("--cv-folds", type=int, default=10, help="Number of cross-validation folds (default: 10)")

    parser.add_argument(
        "--promote", choices=["staging", "production"], help="Auto-promote model to stage if criteria met"
    )

    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")

    parser.add_argument("--no-validation", action="store_true", help="Disable data validation")

    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = EnterpriseTrainer(
        enable_mlflow=not args.no_mlflow, enable_validation=not args.no_validation, log_level=args.log_level
    )

    # Run training pipeline
    report = trainer.run(
        version=args.version, use_grid_search=args.grid_search, cv_folds=args.cv_folds, promote_to=args.promote
    )

    # Print summary
    print("\n" + "=" * 60)
    print("ENTERPRISE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Version: {args.version}")
    print(f"Accuracy: {report.get('metrics', {}).get('accuracy', 0):.4f}")
    print(f"F1 Score: {report.get('metrics', {}).get('f1_score', 0):.4f}")
    print(f"ROC-AUC:  {report.get('metrics', {}).get('roc_auc', 0):.4f}")
    print(f"CV Mean:  {report.get('metrics', {}).get('cv_mean', 0):.4f}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
