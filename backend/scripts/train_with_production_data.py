"""
Training with Production Data
=============================

Training script optimized for production deployment using real-world data.
Focuses on robustness, reproducibility, and deployment readiness.

Usage:
    python scripts/train_with_production_data.py
    python scripts/train_with_production_data.py --optimize
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
MODELS_DIR = BACKEND_DIR / "models"
DATA_DIR = BACKEND_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Production data sources (in priority order)
PRODUCTION_DATA_SOURCES = [
    "pagasa_training_dataset.csv",
    "cumulative_up_to_2025.csv",
    "pagasa_naia_processed.csv",
]

# Production features
PRODUCTION_FEATURES = [
    "temperature",
    "humidity",
    "precipitation",
    "is_monsoon_season",
    "month",
    "temp_humidity_interaction",
    "humidity_precip_interaction",
    "monsoon_precip_interaction",
]

# Production-optimized parameters
PRODUCTION_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def find_production_data() -> Path:
    """Find the best available production data source."""
    for data_file in PRODUCTION_DATA_SOURCES:
        path = PROCESSED_DIR / data_file
        if path.exists():
            return path
    raise FileNotFoundError("No production data source found")


def load_and_prepare(data_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare production data."""
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records from {data_path.name}")

    # Select available features
    available = [f for f in PRODUCTION_FEATURES if f in df.columns]
    X = df[available].copy().fillna(df[available].median())
    y = df["flood"].copy()

    logger.info(f"Using {len(available)} features")
    return X, y


def train_production_model(
    X: pd.DataFrame, y: pd.Series, optimize: bool = False
) -> Tuple[RandomForestClassifier, Dict]:
    """Train production-ready model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(**PRODUCTION_PARAMS)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1_weighted", n_jobs=-1)
    metrics["cv_mean"] = float(cv_scores.mean())
    metrics["cv_std"] = float(cv_scores.std())

    logger.info(f"F1: {metrics['f1_score']:.4f}, CV: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")

    return model, metrics


def save_production_model(model: RandomForestClassifier, metrics: Dict, features: List[str]):
    """Save production model with metadata."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = MODELS_DIR / "flood_rf_model_production.joblib"
    joblib.dump(model, model_path)

    # Save as default
    default_path = MODELS_DIR / "flood_rf_model.joblib"
    joblib.dump(model, default_path)

    # Metadata
    metadata = {
        "version": "production",
        "model_type": "RandomForestClassifier",
        "created_at": datetime.now().isoformat(),
        "features": features,
        "metrics": metrics,
        "parameters": PRODUCTION_PARAMS,
    }

    with open(model_path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model saved: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train with production data")
    parser.add_argument("--data", type=str, help="Custom data path")
    parser.add_argument("--optimize", action="store_true", help="Run optimization")
    args = parser.parse_args()

    # Find and load data
    data_path = Path(args.data) if args.data else find_production_data()
    X, y = load_and_prepare(data_path)

    # Train
    model, metrics = train_production_model(X, y, optimize=args.optimize)

    # Save
    save_production_model(model, metrics, list(X.columns))

    print(f"\n{'='*50}")
    print("Production Training Complete")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
