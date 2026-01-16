"""
Enhanced Model Training with Multi-Level Classification
========================================================

Trains both binary and multi-level flood risk classifiers:
- Binary: Flood (1) vs No Flood (0)
- Multi-Level: LOW (0), MODERATE (1), HIGH (2) flood risk

Features:
- Randomized hyperparameter search
- Ensemble methods comparison
- Class imbalance handling
- Feature engineering for multi-class

Usage:
    python scripts/train_enhanced.py
    python scripts/train_enhanced.py --multi-level
    python scripts/train_enhanced.py --multi-level --randomized-search
    python scripts/train_enhanced.py --compare-models
"""

import argparse
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

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

# Features for enhanced training
ENHANCED_FEATURES = [
    "temperature",
    "humidity",
    "precipitation",
    "is_monsoon_season",
    "month",
    "temp_humidity_interaction",
    "humidity_precip_interaction",
    "temp_precip_interaction",
    "monsoon_precip_interaction",
    "saturation_risk",
]

# Multi-level classification labels
RISK_LABELS = {0: "LOW", 1: "MODERATE", 2: "HIGH"}

# Randomized search parameter distributions
RF_PARAM_DIST = {
    "n_estimators": [100, 150, 200, 250, 300],
    "max_depth": [10, 15, 20, 25, None],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
}

GB_PARAM_DIST = {
    "n_estimators": [100, 150, 200],
    "max_depth": [3, 5, 7, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "min_samples_split": [2, 5, 10],
    "subsample": [0.8, 0.9, 1.0],
}


class EnhancedModelTrainer:
    """Enhanced trainer with multi-level classification support."""

    def __init__(self, models_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR):
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.binary_model = None
        self.multi_level_model = None
        self.feature_names: List[str] = []
        self.label_encoder = LabelEncoder()

    def load_data(self, data_path: Optional[str] = None) -> pd.DataFrame:
        """Load training data."""
        if data_path:
            path = Path(data_path)
        else:
            path = PROCESSED_DIR / "cumulative_up_to_2025.csv"
            if not path.exists():
                path = PROCESSED_DIR / "pagasa_training_dataset.csv"

        if not path.exists():
            raise FileNotFoundError(f"Data not found: {path}")

        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} records from {path.name}")
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Prepare features with both binary and multi-level targets."""
        available = [f for f in ENHANCED_FEATURES if f in df.columns]
        X = df[available].copy().fillna(df[available].median())

        # Binary target
        y_binary = df["flood"].copy()

        # Multi-level target
        if "risk_level" in df.columns:
            y_multi = df["risk_level"].copy()
        else:
            # Create risk levels from precipitation if not available
            conditions = [
                df["precipitation"] < 20,
                (df["precipitation"] >= 20) & (df["precipitation"] < 50),
                df["precipitation"] >= 50,
            ]
            y_multi = pd.Series(np.select(conditions, [0, 1, 2], default=0), index=df.index)

        self.feature_names = list(X.columns)

        logger.info(f"Features: {len(self.feature_names)}")
        logger.info(f"Binary distribution: {y_binary.value_counts().to_dict()}")
        logger.info(f"Multi-level distribution: {y_multi.value_counts().to_dict()}")

        return X, y_binary, y_multi

    def train_binary_model(
        self, X: pd.DataFrame, y: pd.Series, use_randomized_search: bool = False, cv_folds: int = 5
    ) -> Tuple[RandomForestClassifier, Dict]:
        """Train binary flood classifier."""
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING BINARY CLASSIFIER (Flood vs No Flood)")
        logger.info("=" * 60)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        if use_randomized_search:
            logger.info("Running randomized search...")
            base_model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
            search = RandomizedSearchCV(
                base_model,
                RF_PARAM_DIST,
                n_iter=30,
                cv=StratifiedKFold(cv_folds, shuffle=True, random_state=42),
                scoring="f1_weighted",
                n_jobs=-1,
                random_state=42,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            logger.info(f"Best params: {search.best_params_}")
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        }

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="f1_weighted", n_jobs=-1)
        metrics["cv_mean"] = float(cv_scores.mean())
        metrics["cv_std"] = float(cv_scores.std())

        logger.info(f"\nBinary Classifier Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"  CV Score:  {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")

        self.binary_model = model
        return model, metrics

    def train_multi_level_model(
        self, X: pd.DataFrame, y: pd.Series, use_randomized_search: bool = False, cv_folds: int = 5
    ) -> Tuple[RandomForestClassifier, Dict]:
        """Train multi-level risk classifier (LOW/MODERATE/HIGH)."""
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING MULTI-LEVEL CLASSIFIER (LOW/MODERATE/HIGH)")
        logger.info("=" * 60)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        if use_randomized_search:
            logger.info("Running randomized search...")
            base_model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
            search = RandomizedSearchCV(
                base_model,
                RF_PARAM_DIST,
                n_iter=30,
                cv=StratifiedKFold(cv_folds, shuffle=True, random_state=42),
                scoring="f1_weighted",
                n_jobs=-1,
                random_state=42,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            logger.info(f"Best params: {search.best_params_}")
        else:
            model = RandomForestClassifier(
                n_estimators=250,
                max_depth=20,
                min_samples_split=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }

        # Per-class metrics
        class_report = classification_report(
            y_test, y_pred, target_names=[RISK_LABELS[i] for i in sorted(y.unique())], output_dict=True
        )
        metrics["classification_report"] = class_report

        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="f1_weighted", n_jobs=-1)
        metrics["cv_mean"] = float(cv_scores.mean())
        metrics["cv_std"] = float(cv_scores.std())

        logger.info(f"\nMulti-Level Classifier Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  CV Score:  {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")

        # Per-class performance
        logger.info(f"\n  Per-Class F1:")
        for label_id in sorted(y.unique()):
            label = RISK_LABELS.get(label_id, str(label_id))
            if label in class_report:
                logger.info(f"    {label}: {class_report[label]['f1-score']:.4f}")

        self.multi_level_model = model
        return model, metrics

    def compare_models(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict:
        """Compare different model architectures."""
        logger.info("\n" + "=" * 60)
        logger.info("MODEL ARCHITECTURE COMPARISON")
        logger.info("=" * 60)

        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=200, max_depth=15, class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=150, max_depth=7, learning_rate=0.1, random_state=42
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=200, max_depth=15, random_state=42),
        }

        results = {}
        cv = StratifiedKFold(cv_folds, shuffle=True, random_state=42)

        for name, model in models.items():
            logger.info(f"\nEvaluating {name}...")

            scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
            results[name] = {
                "cv_mean": float(scores.mean()),
                "cv_std": float(scores.std()),
                "cv_scores": [float(s) for s in scores],
            }
            logger.info(f"  CV F1: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

        # Determine best model
        best_model = max(results.keys(), key=lambda k: results[k]["cv_mean"])
        logger.info(f"\nBest Model: {best_model}")

        return results

    def save_models(self, binary_metrics: Dict, multi_level_metrics: Optional[Dict] = None):
        """Save trained models with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save binary model
        if self.binary_model:
            binary_path = self.models_dir / "flood_enhanced_binary.joblib"
            joblib.dump(self.binary_model, binary_path)

            metadata = {
                "model_type": "RandomForestClassifier",
                "classification_type": "binary",
                "created_at": datetime.now().isoformat(),
                "features": self.feature_names,
                "metrics": binary_metrics,
            }

            with open(binary_path.with_suffix(".json"), "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Binary model saved: {binary_path}")

        # Save multi-level model
        if self.multi_level_model and multi_level_metrics:
            multi_path = self.models_dir / "flood_enhanced_multilevel.joblib"
            joblib.dump(self.multi_level_model, multi_path)

            metadata = {
                "model_type": "RandomForestClassifier",
                "classification_type": "multi-level",
                "risk_labels": RISK_LABELS,
                "created_at": datetime.now().isoformat(),
                "features": self.feature_names,
                "metrics": multi_level_metrics,
            }

            with open(multi_path.with_suffix(".json"), "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Multi-level model saved: {multi_path}")

        # Save as latest
        if self.binary_model:
            latest_path = self.models_dir / "flood_rf_model.joblib"
            joblib.dump(self.binary_model, latest_path)
            logger.info(f"Saved as latest: {latest_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Enhanced model training with multi-level classification")
    parser.add_argument("--data", type=str, help="Path to training data")
    parser.add_argument("--multi-level", action="store_true", help="Train multi-level classifier")
    parser.add_argument("--randomized-search", action="store_true", help="Use randomized hyperparameter search")
    parser.add_argument("--compare-models", action="store_true", help="Compare different model architectures")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")

    args = parser.parse_args()

    trainer = EnhancedModelTrainer()

    # Load data
    df = trainer.load_data(args.data)

    # Prepare features
    X, y_binary, y_multi = trainer.prepare_features(df)

    # Compare models if requested
    if args.compare_models:
        comparison = trainer.compare_models(X, y_binary, args.cv_folds)
        comparison_path = trainer.reports_dir / "model_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Comparison saved: {comparison_path}")

    # Train binary model
    binary_model, binary_metrics = trainer.train_binary_model(
        X, y_binary, use_randomized_search=args.randomized_search, cv_folds=args.cv_folds
    )

    # Train multi-level model if requested
    multi_metrics = None
    if args.multi_level:
        multi_model, multi_metrics = trainer.train_multi_level_model(
            X, y_multi, use_randomized_search=args.randomized_search, cv_folds=args.cv_folds
        )

    # Save models
    trainer.save_models(binary_metrics, multi_metrics)

    print("\n" + "=" * 60)
    print("ENHANCED TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBinary Classifier F1: {binary_metrics['f1_score']:.4f}")
    if multi_metrics:
        print(f"Multi-Level Classifier F1: {multi_metrics['f1_score']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
