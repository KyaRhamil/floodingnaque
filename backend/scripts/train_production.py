"""
Production-Ready Model Training Script
======================================

.. deprecated:: 1.0.0
    This script is deprecated. Use the unified CLI instead:

    python -m scripts train --mode production        # Production training
    python -m scripts train --mode production --shap # With SHAP

    Or use the UnifiedTrainer class:

    from scripts.train_unified import UnifiedTrainer, TrainingMode
    trainer = UnifiedTrainer(mode=TrainingMode.PRODUCTION)
    trainer.train()

Comprehensive training pipeline for production deployment with:
- Full hyperparameter optimization
- Extensive cross-validation
- Model calibration
- Feature importance analysis
- SHAP explainability (optional)
- Comprehensive validation
- Deployment-ready artifacts

Usage:
    python scripts/train_production.py
    python scripts/train_production.py --production
    python scripts/train_production.py --production --shap
"""

import warnings

warnings.warn(
    "train_production.py is deprecated. Use 'python -m scripts train --mode production' instead.",
    DeprecationWarning,
    stacklevel=2,
)

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
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
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
MODELS_DIR = BACKEND_DIR / "models"
REPORTS_DIR = BACKEND_DIR / "reports"
DATA_DIR = BACKEND_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Production features (comprehensive set)
PRODUCTION_FEATURES = [
    # Core weather features
    "temperature",
    "humidity",
    "precipitation",
    # Temporal features
    "month",
    "is_monsoon_season",
    # Interaction features
    "temp_humidity_interaction",
    "humidity_precip_interaction",
    "temp_precip_interaction",
    "monsoon_precip_interaction",
    "saturation_risk",
]

# Extensive hyperparameter grid for production
PRODUCTION_PARAM_GRID = {
    "n_estimators": [200, 300, 500],
    "max_depth": [10, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
    "criterion": ["gini", "entropy"],
}

# Quick training parameters
QUICK_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


class ProductionModelTrainer:
    """Production-grade model trainer with comprehensive validation."""

    def __init__(self, models_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR):
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.calibrated_model = None
        self.feature_names: List[str] = []
        self.metrics: Dict = {}
        self.training_info: Dict = {}

    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load training data."""
        if data_path:
            path = Path(data_path)
        else:
            # Default to most comprehensive dataset
            path = PROCESSED_DIR / "cumulative_up_to_2025.csv"
            if not path.exists():
                path = PROCESSED_DIR / "pagasa_training_dataset.csv"

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")

        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} records from {path.name}")

        self.training_info["data_source"] = str(path)
        self.training_info["total_records"] = len(df)

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix with comprehensive feature set."""
        available = [f for f in PRODUCTION_FEATURES if f in df.columns]
        missing = [f for f in PRODUCTION_FEATURES if f not in df.columns]

        if missing:
            logger.warning(f"Missing features: {missing}")

        X = df[available].copy()
        y = df["flood"].copy()

        # Handle missing values
        X = X.fillna(X.median())

        self.feature_names = list(X.columns)
        self.training_info["features"] = self.feature_names
        self.training_info["feature_count"] = len(self.feature_names)

        logger.info(f"Features: {len(available)}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        return X, y

    def train(
        self, X: pd.DataFrame, y: pd.Series, production_mode: bool = False, cv_folds: int = 10
    ) -> Optional[RandomForestClassifier]:
        """Train model with optional production-grade optimization."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

        if production_mode:
            logger.info("Running production-grade hyperparameter optimization...")
            logger.info("This may take 30-60 minutes...")

            base_model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)

            # Use RandomizedSearchCV for faster search with production grid
            grid_search = RandomizedSearchCV(
                base_model,
                PRODUCTION_PARAM_GRID,
                n_iter=50,
                cv=StratifiedKFold(cv_folds, shuffle=True, random_state=42),
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=2,
                random_state=42,
            )
            grid_search.fit(X_train, y_train)

            # best_estimator_ is guaranteed to be RandomForestClassifier here
            self.model = cast(RandomForestClassifier, grid_search.best_estimator_)
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

            self.training_info["grid_search"] = {
                "best_params": grid_search.best_params_,
                "best_cv_score": float(grid_search.best_score_),
            }
        else:
            self.model = RandomForestClassifier(**QUICK_PARAMS)
            self.model.fit(X_train, y_train)

        # Evaluate
        self._evaluate(X_test, y_test, X, y, cv_folds)

        # Calibrate model for better probability estimates
        self._calibrate_model(X_train, y_train)

        return self.model

    def _evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series, X_full: pd.DataFrame, y_full: pd.Series, cv_folds: int
    ):
        """Comprehensive model evaluation."""
        assert self.model is not None, "Model not trained. Call train() first."  # nosec B101
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Basic metrics
        self.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
            "brier_score": float(brier_score_loss(y_test, y_pred_proba)),
            "avg_precision": float(average_precision_score(y_test, y_pred_proba)),
        }

        # Cross-validation with multiple metrics
        cv = StratifiedKFold(cv_folds, shuffle=True, random_state=42)
        cv_acc = cross_val_score(self.model, X_full, y_full, cv=cv, scoring="accuracy")
        cv_f1 = cross_val_score(self.model, X_full, y_full, cv=cv, scoring="f1_weighted")
        cv_roc = cross_val_score(self.model, X_full, y_full, cv=cv, scoring="roc_auc")

        self.metrics["cross_validation"] = {
            "cv_folds": cv_folds,
            "accuracy_mean": float(cv_acc.mean()),
            "accuracy_std": float(cv_acc.std()),
            "f1_mean": float(cv_f1.mean()),
            "f1_std": float(cv_f1.std()),
            "roc_auc_mean": float(cv_roc.mean()),
            "roc_auc_std": float(cv_roc.std()),
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.metrics["confusion_matrix"] = cm.tolist()

        # Classification report
        self.metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True)

        # Log results
        logger.info(f"\n{'='*60}")
        logger.info("PRODUCTION MODEL RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy:      {self.metrics['accuracy']:.4f}")
        logger.info(f"Precision:     {self.metrics['precision']:.4f}")
        logger.info(f"Recall:        {self.metrics['recall']:.4f}")
        logger.info(f"F1 Score:      {self.metrics['f1_score']:.4f}")
        logger.info(f"ROC-AUC:       {self.metrics['roc_auc']:.4f}")
        logger.info(f"Brier Score:   {self.metrics['brier_score']:.4f}")
        logger.info(f"Avg Precision: {self.metrics['avg_precision']:.4f}")
        logger.info(f"\nCross-Validation ({cv_folds}-fold):")
        cv = self.metrics["cross_validation"]
        logger.info(f"  Accuracy: {cv['accuracy_mean']:.4f} (+/- {cv['accuracy_std']*2:.4f})")
        logger.info(f"  F1 Score: {cv['f1_mean']:.4f} (+/- {cv['f1_std']*2:.4f})")
        logger.info(f"  ROC-AUC:  {cv['roc_auc_mean']:.4f} (+/- {cv['roc_auc_std']*2:.4f})")
        logger.info(f"{'='*60}\n")

    def _calibrate_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Calibrate model for better probability estimates."""
        logger.info("Calibrating model for better probability estimates...")

        self.calibrated_model = CalibratedClassifierCV(self.model, method="isotonic", cv=5)
        self.calibrated_model.fit(X_train, y_train)

        logger.info("Model calibration complete")

    def generate_shap_analysis(self, X: pd.DataFrame, max_samples: int = 500):
        """Generate SHAP explainability analysis."""
        try:
            import shap

            logger.info("Generating SHAP analysis...")

            # Sample for faster computation
            if len(X) > max_samples:
                X_sample = X.sample(max_samples, random_state=42)
            else:
                X_sample = X

            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)

            # For binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            shap_importance = dict(zip(self.feature_names, mean_shap))
            self.metrics["shap_importance"] = shap_importance

            # Generate plots
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, show=False)
                plt.tight_layout()
                plt.savefig(self.reports_dir / "production_shap_summary.png", dpi=300)
                plt.close()
                logger.info(f"SHAP plot saved to {self.reports_dir}")
            except ImportError:
                pass

        except ImportError:
            logger.warning("SHAP not installed. Run: pip install shap")

    def save_model(self, version: str = "production") -> Path:
        """Save production model and comprehensive metadata."""
        # Save main model
        model_path = self.models_dir / f"flood_rf_model_{version}.joblib"
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved: {model_path}")

        # Save calibrated model
        if self.calibrated_model:
            calibrated_path = self.models_dir / f"flood_rf_model_{version}_calibrated.joblib"
            joblib.dump(self.calibrated_model, calibrated_path)
            logger.info(f"Calibrated model saved: {calibrated_path}")

        # Save as latest
        latest_path = self.models_dir / "flood_rf_model.joblib"
        joblib.dump(self.model, latest_path)
        logger.info(f"Saved as latest: {latest_path}")

        # Comprehensive metadata
        assert self.model is not None, "Model not trained"  # nosec B101
        metadata = {
            "version": version,
            "model_type": "RandomForestClassifier",
            "model_path": str(model_path),
            "created_at": datetime.now().isoformat(),
            "training_info": self.training_info,
            "training_data": {
                "file": self.training_info.get("data_source"),
                "records": self.training_info.get("total_records"),
                "features": self.feature_names,
            },
            "model_parameters": {
                "n_estimators": getattr(self.model, "n_estimators", None),
                "max_depth": getattr(self.model, "max_depth", None),
                "min_samples_split": getattr(self.model, "min_samples_split", None),
                "min_samples_leaf": getattr(self.model, "min_samples_leaf", None),
                "max_features": getattr(self.model, "max_features", None),
                "class_weight": str(getattr(self.model, "class_weight", None)),
            },
            "metrics": self.metrics,
            "feature_importance": dict(zip(self.feature_names, [float(x) for x in self.model.feature_importances_])),
        }

        metadata_path = model_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved: {metadata_path}")

        return model_path

    def generate_report(self):
        """Generate comprehensive production report."""
        report_path = self.reports_dir / f"production_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("PRODUCTION MODEL TRAINING REPORT\n")
            f.write("Flood Prediction System - Parañaque City\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("DATA SOURCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"File: {self.training_info.get('data_source', 'N/A')}\n")
            f.write(f"Records: {self.training_info.get('total_records', 'N/A')}\n")
            f.write(f"Features: {len(self.feature_names)}\n\n")

            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy:      {self.metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Precision:     {self.metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall:        {self.metrics.get('recall', 0):.4f}\n")
            f.write(f"F1 Score:      {self.metrics.get('f1_score', 0):.4f}\n")
            f.write(f"ROC-AUC:       {self.metrics.get('roc_auc', 0):.4f}\n")
            f.write(f"Brier Score:   {self.metrics.get('brier_score', 0):.4f}\n\n")

            cv = self.metrics.get("cross_validation", {})
            f.write("CROSS-VALIDATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Folds: {cv.get('cv_folds', 'N/A')}\n")
            f.write(f"Accuracy: {cv.get('accuracy_mean', 0):.4f} (+/- {cv.get('accuracy_std', 0)*2:.4f})\n")
            f.write(f"F1 Score: {cv.get('f1_mean', 0):.4f} (+/- {cv.get('f1_std', 0)*2:.4f})\n\n")

            f.write("FEATURE IMPORTANCE (Top 10)\n")
            f.write("-" * 40 + "\n")
            assert self.model is not None, "Model not trained"  # nosec B101
            importance = sorted(
                zip(self.feature_names, self.model.feature_importances_), key=lambda x: x[1], reverse=True
            )
            for feat, imp in importance[:10]:
                f.write(f"  {feat:35s} {imp:.4f}\n")

            f.write("\n" + "=" * 70 + "\n")

        logger.info(f"Report saved: {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Production-ready model training")
    parser.add_argument("--data", type=str, help="Path to training data")
    parser.add_argument("--production", action="store_true", help="Full production pipeline with optimization")
    parser.add_argument("--shap", action="store_true", help="Generate SHAP explainability analysis")
    parser.add_argument("--cv-folds", type=int, default=10, help="Number of cross-validation folds")

    args = parser.parse_args()

    trainer = ProductionModelTrainer()

    # Load data
    df = trainer.load_data(args.data)

    # Prepare features
    X, y = trainer.prepare_features(df)

    # Train model
    trainer.train(X, y, production_mode=args.production, cv_folds=args.cv_folds)

    # SHAP analysis
    if args.shap:
        trainer.generate_shap_analysis(X)

    # Save model
    trainer.save_model()

    # Generate report
    trainer.generate_report()

    logger.info("\n✅ Production training complete!")


if __name__ == "__main__":
    main()
