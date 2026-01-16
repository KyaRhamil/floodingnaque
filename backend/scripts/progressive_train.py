"""
Progressive Model Training for Thesis Demonstration
====================================================

Trains multiple model versions progressively using cumulative datasets
(2022 → 2023 → 2024 → 2025) to demonstrate model improvement over time.

This is ideal for thesis presentations showing how the model evolves
as more data becomes available.

Output:
    - flood_rf_model_v1.joblib (2022 only)
    - flood_rf_model_v2.joblib (2022-2023)
    - flood_rf_model_v3.joblib (2022-2024)
    - flood_rf_model_v4.joblib (2022-2025)
    - progression_report.json (comparison metrics)
    - metrics_evolution.png (visualization)

Usage:
    python scripts/progressive_train.py
    python scripts/progressive_train.py --grid-search
    python scripts/progressive_train.py --cv-folds 10
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
REPORTS_DIR = BACKEND_DIR / "reports"
DATA_DIR = BACKEND_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Progressive versions configuration
PROGRESSIVE_VERSIONS = [
    {"version": 1, "data_file": "cumulative_up_to_2022.csv", "description": "Baseline: 2022 Only"},
    {"version": 2, "data_file": "cumulative_up_to_2023.csv", "description": "2022-2023"},
    {"version": 3, "data_file": "cumulative_up_to_2024.csv", "description": "2022-2024"},
    {"version": 4, "data_file": "cumulative_up_to_2025.csv", "description": "Full Dataset: 2022-2025"},
]

# Model features
TRAINING_FEATURES = [
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

# Default parameters
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


class ProgressiveTrainer:
    """Trains models progressively across cumulative datasets."""

    def __init__(self, models_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR):
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict] = []

    def load_data(self, data_file: str) -> pd.DataFrame:
        """Load cumulative dataset."""
        path = PROCESSED_DIR / data_file
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return pd.read_csv(path)

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for training."""
        available = [f for f in TRAINING_FEATURES if f in df.columns]
        X = df[available].copy().fillna(df[available].median())
        y = df["flood"].copy()
        return X, y

    def train_version(
        self, version_config: Dict, use_grid_search: bool = False, cv_folds: int = 5
    ) -> Tuple[RandomForestClassifier, Dict]:
        """Train a single model version."""
        version = version_config["version"]
        data_file = version_config["data_file"]
        description = version_config["description"]

        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING VERSION {version}: {description}")
        logger.info(f"{'='*60}")

        # Load data
        df = self.load_data(data_file)
        X, y = self.prepare_features(df)

        logger.info(f"Data: {len(df)} records, {len(X.columns)} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train model
        if use_grid_search:
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 15, 20],
                "min_samples_split": [2, 5],
            }
            base_model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=StratifiedKFold(cv_folds, shuffle=True, random_state=42),
                scoring="f1_weighted",
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            model = RandomForestClassifier(**DEFAULT_PARAMS)
            model.fit(X_train, y_train)
            best_params = DEFAULT_PARAMS

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

        logger.info(f"\nVersion {version} Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"  CV Score:  {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")

        # Store result
        result = {
            "version": version,
            "description": description,
            "data_file": data_file,
            "dataset_size": len(df),
            "metrics": metrics,
            "best_params": best_params,
            "features": list(X.columns),
        }
        self.results.append(result)

        return model, result

    def save_model(self, model: RandomForestClassifier, result: Dict):
        """Save model and metadata."""
        version = result["version"]
        model_path = self.models_dir / f"flood_rf_model_v{version}.joblib"
        joblib.dump(model, model_path)

        metadata = {
            "version": version,
            "model_type": "RandomForestClassifier",
            "created_at": datetime.now().isoformat(),
            "description": result["description"],
            "training_data": {
                "file": result["data_file"],
                "shape": [result["dataset_size"], len(result["features"])],
                "features": result["features"],
            },
            "metrics": result["metrics"],
            "model_parameters": result["best_params"],
            "cross_validation": {
                "cv_folds": 5,
                "cv_mean": result["metrics"].get("cv_mean"),
                "cv_std": result["metrics"].get("cv_std"),
            },
        }

        metadata_path = model_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved: {model_path}")

        # Save latest version as default
        if version == max(v["version"] for v in PROGRESSIVE_VERSIONS):
            latest_path = self.models_dir / "flood_rf_model.joblib"
            joblib.dump(model, latest_path)
            logger.info(f"Saved as latest: {latest_path}")

    def train_all(self, use_grid_search: bool = False, cv_folds: int = 5):
        """Train all progressive versions."""
        logger.info("\n" + "=" * 70)
        logger.info("PROGRESSIVE MODEL TRAINING")
        logger.info("Training models across cumulative datasets")
        logger.info("=" * 70)

        for version_config in PROGRESSIVE_VERSIONS:
            try:
                model, result = self.train_version(version_config, use_grid_search, cv_folds)
                self.save_model(model, result)
            except FileNotFoundError as e:
                logger.warning(f"Skipping version {version_config['version']}: {e}")
                continue

        # Generate reports
        self.generate_progression_report()
        self.generate_progression_chart()

    def generate_progression_report(self):
        """Generate JSON report of progression results."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "versions": self.results,
            "improvement_summary": self._calculate_improvements(),
        }

        report_path = self.reports_dir / "progression_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"\nProgression report saved: {report_path}")

    def _calculate_improvements(self) -> Dict:
        """Calculate improvement metrics between versions."""
        if len(self.results) < 2:
            return {}

        first = self.results[0]["metrics"]
        last = self.results[-1]["metrics"]

        return {
            "accuracy_improvement": last["accuracy"] - first["accuracy"],
            "f1_improvement": last["f1_score"] - first["f1_score"],
            "roc_auc_improvement": last["roc_auc"] - first["roc_auc"],
            "first_version": self.results[0]["version"],
            "last_version": self.results[-1]["version"],
        }

    def generate_progression_chart(self):
        """Generate visualization of model progression."""
        try:
            import matplotlib.pyplot as plt

            if not self.results:
                return

            versions = [r["version"] for r in self.results]
            accuracy = [r["metrics"]["accuracy"] for r in self.results]
            f1_scores = [r["metrics"]["f1_score"] for r in self.results]
            roc_auc = [r["metrics"]["roc_auc"] for r in self.results]

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(versions, accuracy, "o-", label="Accuracy", linewidth=2, markersize=8)
            ax.plot(versions, f1_scores, "s-", label="F1 Score", linewidth=2, markersize=8)
            ax.plot(versions, roc_auc, "^-", label="ROC-AUC", linewidth=2, markersize=8)

            ax.set_xlabel("Model Version", fontsize=12, fontweight="bold")
            ax.set_ylabel("Score", fontsize=12, fontweight="bold")
            ax.set_title(
                "Model Performance Evolution\nProgressive Training (2022 → 2025)", fontsize=14, fontweight="bold"
            )
            ax.legend(loc="lower right", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.1])
            ax.set_xticks(versions)

            plt.tight_layout()
            chart_path = self.reports_dir / "metrics_evolution.png"
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Progression chart saved: {chart_path}")

        except ImportError:
            logger.warning("matplotlib not available, skipping chart generation")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Progressive model training for thesis demonstration")
    parser.add_argument("--grid-search", action="store_true", help="Use grid search for each version")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of cross-validation folds")

    args = parser.parse_args()

    trainer = ProgressiveTrainer()
    trainer.train_all(use_grid_search=args.grid_search, cv_folds=args.cv_folds)

    # Print summary
    print("\n" + "=" * 60)
    print("PROGRESSIVE TRAINING COMPLETE")
    print("=" * 60)

    if trainer.results:
        print("\nVersion Summary:")
        for r in trainer.results:
            print(f"  v{r['version']}: {r['description']}")
            print(f"    F1: {r['metrics']['f1_score']:.4f}, Accuracy: {r['metrics']['accuracy']:.4f}")

        improvements = trainer._calculate_improvements()
        if improvements:
            print(f"\nOverall Improvement (v{improvements['first_version']} → v{improvements['last_version']}):")
            print(f"  Accuracy: {improvements['accuracy_improvement']:+.4f}")
            print(f"  F1 Score: {improvements['f1_improvement']:+.4f}")
            print(f"  ROC-AUC:  {improvements['roc_auc_improvement']:+.4f}")

    print("=" * 60)


if __name__ == "__main__":
    main()
