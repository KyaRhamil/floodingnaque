"""
Basic Flood Prediction Model Training Script
============================================

Simple training script for the Random Forest flood prediction model.
For advanced features, use train_pagasa.py or train_production.py.

Usage:
    python scripts/train.py
    python scripts/train.py --data data/processed/cumulative_up_to_2025.csv
    python scripts/train.py --grid-search
    python scripts/train.py --cv-folds 10
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Default features
DEFAULT_FEATURES = [
    "temperature",
    "humidity",
    "precipitation",
    "is_monsoon_season",
    "month",
]

# Default model parameters
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def load_data(data_path: str) -> pd.DataFrame:
    """Load training data from CSV."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} records from {path.name}")
    return df


def prepare_features(df: pd.DataFrame, features: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target vector."""
    if features is None:
        features = DEFAULT_FEATURES

    # Filter to available features
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    if missing:
        logger.warning(f"Missing features: {missing}")

    if not available:
        raise ValueError("No features available for training")

    X = df[available].copy()
    y = df["flood"].copy()

    # Fill missing values
    X = X.fillna(X.median())

    logger.info(f"Features: {available}")
    logger.info(f"X shape: {X.shape}, y distribution: {y.value_counts().to_dict()}")

    return X, y


def train_model(
    X: pd.DataFrame, y: pd.Series, use_grid_search: bool = False, cv_folds: int = 5
) -> Tuple[RandomForestClassifier, Dict]:
    """Train Random Forest model."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    if use_grid_search:
        logger.info("Running grid search for hyperparameter optimization...")
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        base_model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=StratifiedKFold(cv_folds, shuffle=True, random_state=42),
            scoring="f1_weighted",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_
        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    else:
        model = RandomForestClassifier(**DEFAULT_PARAMS)
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

    # Log results
    logger.info(f"\n{'='*50}")
    logger.info("TRAINING RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
    logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info(f"CV Score:  {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")
    logger.info(f"{'='*50}\n")

    # Log feature importance
    logger.info("Feature Importance:")
    importance = sorted(zip(X.columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
    for feat, imp in importance[:10]:
        logger.info(f"  {feat:30s} {imp:.4f}")

    return model, metrics


def save_model(model: RandomForestClassifier, metrics: Dict, feature_names: List[str], version: Optional[int] = None):
    """Save model and metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Auto-version
    if version is None:
        existing = list(MODELS_DIR.glob("flood_rf_model_v*.joblib"))
        versions = []
        for f in existing:
            try:
                v = int(f.stem.split("_v")[1].split("_")[0])
                versions.append(v)
            except (ValueError, IndexError):
                pass
        version = max(versions) + 1 if versions else 1

    # Save model
    model_path = MODELS_DIR / f"flood_rf_model_v{version}.joblib"
    joblib.dump(model, model_path)
    logger.info(f"Model saved: {model_path}")

    # Save as latest
    latest_path = MODELS_DIR / "flood_rf_model.joblib"
    joblib.dump(model, latest_path)
    logger.info(f"Saved as latest: {latest_path}")

    # Save metadata
    metadata = {
        "version": version,
        "model_type": "RandomForestClassifier",
        "model_path": str(model_path),
        "created_at": datetime.now().isoformat(),
        "metrics": metrics,
        "training_data": {"features": feature_names, "feature_count": len(feature_names)},
        "model_parameters": {
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
        },
        "cross_validation": {"cv_folds": 5, "cv_mean": metrics.get("cv_mean"), "cv_std": metrics.get("cv_std")},
    }

    metadata_path = model_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved: {metadata_path}")

    return model_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train flood prediction model")
    parser.add_argument(
        "--data", type=str, default="data/processed/cumulative_up_to_2025.csv", help="Path to training data"
    )
    parser.add_argument("--grid-search", action="store_true", help="Use grid search for hyperparameters")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--version", type=int, help="Model version number")
    parser.add_argument("--no-save", action="store_true", help="Don't save the model")

    args = parser.parse_args()

    # Resolve data path
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = BACKEND_DIR / data_path

    # Load data
    df = load_data(str(data_path))

    # Prepare features
    X, y = prepare_features(df)

    # Train model
    model, metrics = train_model(X, y, use_grid_search=args.grid_search, cv_folds=args.cv_folds)

    # Save model
    if not args.no_save:
        save_model(model, metrics, list(X.columns), version=args.version)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
