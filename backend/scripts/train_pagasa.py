"""
PAGASA-Enhanced Flood Prediction Model Training
================================================

Upgraded training pipeline specifically designed for PAGASA weather station data.
Leverages the comprehensive features from DOST-PAGASA climate data (2020-2025).

Features:
- Multi-station support (NAIA, Port Area, Science Garden)
- Rolling precipitation windows (3-day, 7-day, 14-day)
- Rain streak detection
- Monsoon season modeling
- Heat index calculations
- Interaction features
- Integration with official Parañaque flood records

Usage:
    # Basic training with NAIA station (closest to Parañaque)
    python scripts/train_pagasa.py

    # Full pipeline with all stations
    python scripts/train_pagasa.py --all-stations

    # With hyperparameter tuning
    python scripts/train_pagasa.py --grid-search

    # Production-ready model
    python scripts/train_pagasa.py --production

Author: Floodingnaque Team
Last Updated: January 2026
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Optional imports
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
MODELS_DIR = BACKEND_DIR / "models"
REPORTS_DIR = BACKEND_DIR / "reports"
DATA_DIR = BACKEND_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# PAGASA-specific feature groups
PAGASA_FEATURE_GROUPS = {
    "core": ["temperature", "humidity", "precipitation"],
    "temporal": ["month", "year", "is_monsoon_season"],
    "rolling_precip": [
        "precip_3day_sum",
        "precip_7day_sum",
        "precip_14day_sum",
        "precip_3day_avg",
        "precip_7day_avg",
        "precip_max_3day",
        "precip_max_7day",
        "precip_lag1",
        "precip_lag2",
        "precip_lag3",
        "rain_streak",
    ],
    "rolling_other": ["humidity_3day_avg", "humidity_7day_avg", "humidity_lag1", "temp_3day_avg", "temp_7day_avg"],
    "derived": [
        "temp_range",
        "heat_index",
        "temp_humidity_interaction",
        "humidity_precip_interaction",
        "temp_precip_interaction",
        "monsoon_precip_interaction",
        "wind_rain_interaction",
        "saturation_risk",
    ],
    "spatial": ["latitude", "longitude", "elevation"],
}

# Optimized hyperparameters for PAGASA data (from previous experiments)
PAGASA_OPTIMIZED_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


class PAGASAModelTrainer:
    """
    Production-grade trainer specifically designed for PAGASA weather data.
    """

    def __init__(self, models_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR, random_state: int = 42):
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.random_state = random_state
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.feature_names: List[str] = []
        self.training_metrics: Dict = {}
        self.data_info: Dict = {}

    def preprocess_pagasa_data(self, force_reprocess: bool = False) -> Path:
        """
        Run PAGASA preprocessing to create training dataset.

        Returns:
            Path to the processed training dataset
        """
        training_file = PROCESSED_DIR / "pagasa_training_dataset.csv"

        if training_file.exists() and not force_reprocess:
            logger.info(f"Using existing processed data: {training_file}")
            return training_file

        logger.info("Running PAGASA data preprocessing...")

        # Import and run the preprocessing script
        sys.path.insert(0, str(SCRIPT_DIR))
        from preprocess_pagasa_data import create_training_dataset

        create_training_dataset(use_naia_only=True, include_flood_records=True)

        if not training_file.exists():
            raise FileNotFoundError(f"Preprocessing failed to create: {training_file}")

        return training_file

    def load_data(self, data_path: Optional[str] = None, use_all_stations: bool = False) -> pd.DataFrame:
        """
        Load PAGASA training data.

        Args:
            data_path: Path to CSV file. If None, uses default processed file.
            use_all_stations: If True, use merged data from all stations.

        Returns:
            DataFrame with training data
        """
        if data_path:
            path = Path(data_path)
        elif use_all_stations:
            path = PROCESSED_DIR / "pagasa_all_stations_merged.csv"
        else:
            path = PROCESSED_DIR / "pagasa_training_dataset.csv"

        if not path.exists():
            logger.warning(f"Data file not found: {path}")
            logger.info("Running preprocessing...")
            path = self.preprocess_pagasa_data(force_reprocess=True)

        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} records from {path.name}")

        # Store data info
        self.data_info = {
            "source_file": str(path),
            "total_records": len(df),
            "date_range": f"{df['year'].min()}-{df['year'].max()}" if "year" in df.columns else "unknown",
            "columns": list(df.columns),
        }

        return df

    def prepare_features(
        self, df: pd.DataFrame, feature_groups: Optional[List[str]] = None, target_column: str = "flood"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for training using PAGASA-specific feature groups.

        Args:
            df: Input DataFrame
            feature_groups: List of feature group names to include
            target_column: Name of target column

        Returns:
            Tuple of (X, y)
        """
        if feature_groups is None:
            # Use all available feature groups
            feature_groups = list(PAGASA_FEATURE_GROUPS.keys())

        # Collect features from specified groups
        selected_features = []
        for group in feature_groups:
            if group in PAGASA_FEATURE_GROUPS:
                selected_features.extend(PAGASA_FEATURE_GROUPS[group])

        # Filter to features that exist in the data
        available_features = [f for f in selected_features if f in df.columns]
        missing_features = [f for f in selected_features if f not in df.columns]

        if missing_features:
            logger.warning(f"Missing features: {missing_features[:10]}...")

        logger.info(f"Using {len(available_features)} features from groups: {feature_groups}")

        # Prepare X and y
        X = df[available_features].copy()
        y = df[target_column].copy()

        # Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        # Store feature names
        self.feature_names = list(X.columns)

        # Log feature summary
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        use_grid_search: bool = False,
        use_time_split: bool = True,
        n_folds: int = 5,
        custom_params: Optional[Dict] = None,
    ) -> RandomForestClassifier:
        """
        Train Random Forest model on PAGASA data.

        Args:
            X: Feature matrix
            y: Target vector
            use_grid_search: Perform hyperparameter tuning
            use_time_split: Use time-series split (recommended for temporal data)
            n_folds: Number of CV folds
            custom_params: Custom model parameters

        Returns:
            Trained model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        logger.info(f"Train set: {len(X_train)} | Test set: {len(X_test)}")

        if use_grid_search:
            logger.info("Performing hyperparameter optimization...")
            model = self._grid_search_train(X_train, y_train, n_folds)
        else:
            # Use optimized parameters
            params = custom_params or PAGASA_OPTIMIZED_PARAMS
            logger.info(f"Training with parameters: {params}")
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        self.training_metrics = metrics

        # Log results
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"{'='*60}\n")

        # Feature importance
        self._log_feature_importance(model)

        # Cross-validation score
        cv_scores = cross_val_score(model, X, y, cv=n_folds, scoring="f1_weighted", n_jobs=-1)
        logger.info(f"Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

        self.model = model
        return model

    def _grid_search_train(self, X_train: pd.DataFrame, y_train: pd.Series, n_folds: int) -> RandomForestClassifier:
        """Perform grid search for hyperparameter optimization."""
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
        }

        base_model = RandomForestClassifier(class_weight="balanced", random_state=self.random_state, n_jobs=-1)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=StratifiedKFold(n_folds, shuffle=True, random_state=self.random_state),
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics["classification_report"] = report

        return metrics

    def _log_feature_importance(self, model: RandomForestClassifier, top_n: int = 15):
        """Log top feature importances."""
        importances = pd.DataFrame(
            {"feature": self.feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        logger.info(f"\nTop {top_n} Feature Importances:")
        for _, row in importances.head(top_n).iterrows():
            logger.info(f"  {row['feature']:30s} {row['importance']:.4f}")

    def generate_shap_analysis(self, X: pd.DataFrame, max_samples: int = 500):
        """Generate SHAP explainability analysis."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not installed. Run: pip install shap")
            return None

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        logger.info("Generating SHAP analysis...")

        # Sample data
        if len(X) > max_samples:
            X_sample = X.sample(max_samples, random_state=self.random_state)
        else:
            X_sample = X

        # Create explainer and calculate SHAP values
        explainer = shap.TreeExplainer(self.model)  # type: ignore[name-defined]
        shap_values = explainer.shap_values(X_sample)

        # For binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        if PLOT_AVAILABLE:
            # Summary plot
            plt.figure(figsize=(12, 8))  # type: ignore[name-defined]
            shap.summary_plot(shap_values, X_sample, show=False)  # type: ignore[name-defined]
            plt.tight_layout()  # type: ignore[name-defined]
            plt.savefig(self.reports_dir / "pagasa_shap_summary.png", dpi=300)  # type: ignore[name-defined]
            plt.close()  # type: ignore[name-defined]

            # Bar plot
            plt.figure(figsize=(10, 8))  # type: ignore[name-defined]
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)  # type: ignore[name-defined]
            plt.tight_layout()  # type: ignore[name-defined]
            plt.savefig(self.reports_dir / "pagasa_shap_importance.png", dpi=300)  # type: ignore[name-defined]
            plt.close()  # type: ignore[name-defined]

            logger.info(f"SHAP plots saved to {self.reports_dir}")

        # Return mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.feature_names, mean_shap))

    def save_model(self, version: Optional[int] = None) -> Path:
        """
        Save trained model and metadata.

        Args:
            version: Model version number (auto-incremented if None)

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save. Call train() first.")

        # Auto-increment version
        if version is None:
            version = self._get_next_version()

        # Save model
        model_filename = f"flood_rf_model_v{version}_pagasa.joblib"
        model_path = self.models_dir / model_filename
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved: {model_path}")

        # Also save as latest
        latest_path = self.models_dir / "flood_rf_model_latest.joblib"
        joblib.dump(self.model, latest_path)
        logger.info(f"Model saved as latest: {latest_path}")

        # Save metadata
        metadata = {
            "version": version,
            "model_type": "RandomForestClassifier",
            "model_path": str(model_path),
            "created_at": datetime.now().isoformat(),
            "data_source": "PAGASA Climate Data (NAIA, Port Area, Science Garden)",
            "data_info": self.data_info,
            "features": self.feature_names,
            "feature_count": len(self.feature_names),
            "model_parameters": {
                "n_estimators": getattr(self.model, "n_estimators", None),
                "max_depth": getattr(self.model, "max_depth", None),
                "min_samples_split": getattr(self.model, "min_samples_split", None),
                "min_samples_leaf": getattr(self.model, "min_samples_leaf", None),
                "class_weight": str(getattr(self.model, "class_weight", None)),
            },
            "metrics": self.training_metrics,
            "feature_importance": dict(zip(self.feature_names, [float(x) for x in self.model.feature_importances_])),
        }

        metadata_path = model_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved: {metadata_path}")

        return model_path

    def _get_next_version(self) -> int:
        """Get next available version number."""
        existing = list(self.models_dir.glob("flood_rf_model_v*_pagasa.joblib"))
        if not existing:
            return 1

        versions = []
        for f in existing:
            try:
                v = int(f.stem.split("_v")[1].split("_")[0])
                versions.append(v)
            except (ValueError, IndexError):
                continue

        return max(versions) + 1 if versions else 1

    def generate_training_report(self) -> str:
        """Generate a comprehensive training report."""
        report_lines = [
            "=" * 70,
            "PAGASA FLOOD PREDICTION MODEL - TRAINING REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            "DATA SOURCE",
            "-" * 40,
            f"Source: {self.data_info.get('source_file', 'N/A')}",
            f"Records: {self.data_info.get('total_records', 'N/A')}",
            f"Date Range: {self.data_info.get('date_range', 'N/A')}",
            f"Features: {len(self.feature_names)}",
            "",
            "MODEL PERFORMANCE",
            "-" * 40,
            f"Accuracy:  {self.training_metrics.get('accuracy', 0):.4f}",
            f"Precision: {self.training_metrics.get('precision', 0):.4f}",
            f"Recall:    {self.training_metrics.get('recall', 0):.4f}",
            f"F1 Score:  {self.training_metrics.get('f1_score', 0):.4f}",
            f"ROC-AUC:   {self.training_metrics.get('roc_auc', 0):.4f}",
            "",
            "TOP 10 FEATURES",
            "-" * 40,
        ]

        if self.model is not None:
            importances = sorted(
                zip(self.feature_names, self.model.feature_importances_), key=lambda x: x[1], reverse=True
            )
            for feat, imp in importances[:10]:
                report_lines.append(f"  {feat:30s} {imp:.4f}")

        report_lines.extend(
            [
                "",
                "=" * 70,
            ]
        )

        report = "\n".join(report_lines)

        # Save report
        report_path = self.reports_dir / f'pagasa_training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved: {report_path}")

        return report


def main():
    """Main entry point for PAGASA model training."""
    parser = argparse.ArgumentParser(description="Train flood prediction model using PAGASA weather data")
    parser.add_argument("--data-path", type=str, help="Path to training CSV (default: auto-generated PAGASA dataset)")
    parser.add_argument(
        "--all-stations", action="store_true", help="Use data from all 3 PAGASA stations (default: NAIA only)"
    )
    parser.add_argument(
        "--grid-search", action="store_true", help="Perform hyperparameter optimization with GridSearchCV"
    )
    parser.add_argument(
        "--production", action="store_true", help="Full production pipeline (grid search + SHAP + report)"
    )
    parser.add_argument("--reprocess", action="store_true", help="Force reprocessing of PAGASA data")
    parser.add_argument("--version", type=int, help="Model version number (default: auto-increment)")
    parser.add_argument("--no-save", action="store_true", help="Do not save the trained model")

    args = parser.parse_args()

    # Initialize trainer
    trainer = PAGASAModelTrainer()

    # Preprocess data if needed
    if args.reprocess or not (PROCESSED_DIR / "pagasa_training_dataset.csv").exists():
        trainer.preprocess_pagasa_data(force_reprocess=args.reprocess)

    # Load data
    df = trainer.load_data(data_path=args.data_path, use_all_stations=args.all_stations)

    # Prepare features
    X, y = trainer.prepare_features(df)

    # Train model
    use_grid_search = args.grid_search or args.production
    trainer.train(X, y, use_grid_search=use_grid_search)

    # Generate SHAP analysis for production
    if args.production and SHAP_AVAILABLE:
        trainer.generate_shap_analysis(X)

    # Generate report
    if args.production:
        report = trainer.generate_training_report()
        print(report)

    # Save model
    if not args.no_save:
        model_path = trainer.save_model(version=args.version)
        logger.info(f"\n✅ Training complete! Model saved to: {model_path}")

    return trainer


if __name__ == "__main__":
    main()
