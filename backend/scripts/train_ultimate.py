"""
Ultimate Progressive Model Training
====================================

Comprehensive training pipeline that creates all model versions progressively:
    v1: Official Records 2022 (Baseline)
    v2: Official Records 2022-2023
    v3: Official Records 2022-2024
    v4: Official Records 2022-2025
    v5: PAGASA Weather Data (2020-2025)
    v6: ULTIMATE - All datasets combined

Perfect for thesis demonstrations showing model evolution from baseline to best.

Usage:
    python scripts/train_ultimate.py
    python scripts/train_ultimate.py --progressive
    python scripts/train_ultimate.py --production
    python scripts/train_ultimate.py --latest-only
"""

import argparse
import json
import logging
import warnings
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split

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

# Version Registry - defines all model versions and their data sources
VERSION_REGISTRY = {
    1: {
        "name": "Baseline_2022",
        "description": "Official Records 2022 Only",
        "data_file": "cumulative_up_to_2022.csv",
        "data_type": "official_records",
    },
    2: {
        "name": "Extended_2023",
        "description": "Official Records 2022-2023",
        "data_file": "cumulative_up_to_2023.csv",
        "data_type": "official_records",
    },
    3: {
        "name": "Extended_2024",
        "description": "Official Records 2022-2024",
        "data_file": "cumulative_up_to_2024.csv",
        "data_type": "official_records",
    },
    4: {
        "name": "Full_Official",
        "description": "Official Records 2022-2025",
        "data_file": "cumulative_up_to_2025.csv",
        "data_type": "official_records",
    },
    5: {
        "name": "PAGASA",
        "description": "PAGASA Weather Data (2020-2025)",
        "data_file": "pagasa_training_dataset.csv",
        "data_type": "pagasa",
    },
    6: {
        "name": "ULTIMATE",
        "description": "Combined: Official + PAGASA",
        "data_files": ["cumulative_up_to_2025.csv", "pagasa_training_dataset.csv"],
        "data_type": "combined",
    },
}

# Feature sets for different data types
FEATURE_SETS = {
    "official_records": ["temperature", "humidity", "precipitation", "is_monsoon_season", "month"],
    "pagasa": [
        "temperature",
        "humidity",
        "precipitation",
        "is_monsoon_season",
        "month",
        "precip_3day_sum",
        "precip_7day_sum",
        "temp_humidity_interaction",
        "humidity_precip_interaction",
        "monsoon_precip_interaction",
        "rain_streak",
    ],
    "combined": [
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
    ],
}

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

# Grid search parameters
GRID_PARAMS = {
    "n_estimators": [150, 200, 300],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}


class UltimateTrainer:
    """Ultimate progressive training across all data sources."""

    def __init__(self, models_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR):
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[Dict] = []
        self.best_version: Optional[int] = None

    def load_version_data(self, version: int) -> Optional[pd.DataFrame]:
        """Load data for a specific version."""
        if version not in VERSION_REGISTRY:
            logger.error(f"Unknown version: {version}")
            return None

        config = VERSION_REGISTRY[version]

        if config["data_type"] == "combined":
            # Combine multiple data sources
            dfs = []
            for data_file in config["data_files"]:
                path = PROCESSED_DIR / data_file
                if path.exists():
                    df = pd.read_csv(path)
                    dfs.append(df)
                else:
                    logger.warning(f"Data file not found: {path}")

            if not dfs:
                return None

            # Combine and deduplicate
            combined = pd.concat(dfs, ignore_index=True)
            combined = combined.drop_duplicates(subset=["temperature", "humidity", "precipitation", "flood"])
            return combined
        else:
            path = PROCESSED_DIR / config["data_file"]
            if not path.exists():
                logger.warning(f"Data file not found: {path}")
                return None
            return pd.read_csv(path)

    def prepare_features(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features based on data type."""
        features = FEATURE_SETS.get(data_type, FEATURE_SETS["official_records"])
        available = [f for f in features if f in df.columns]

        X = df[available].copy()
        y = df["flood"].copy()

        # Handle missing values
        X = X.fillna(X.median())

        return X, y

    def train_version(
        self, version: int, use_grid_search: bool = False, cv_folds: int = 5
    ) -> Optional[Tuple[RandomForestClassifier, Dict]]:
        """Train a single version."""
        config = VERSION_REGISTRY[version]

        logger.info(f"\n{'='*60}")
        logger.info(f"VERSION {version}: {config['description']}")
        logger.info(f"{'='*60}")

        # Load data
        df = self.load_version_data(version)
        if df is None:
            logger.warning(f"Skipping version {version}: No data available")
            return None

        # Prepare features
        X, y = self.prepare_features(df, config["data_type"])

        logger.info(f"Data: {len(df)} records, {len(X.columns)} features")
        logger.info(f"Target: {y.value_counts().to_dict()}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train model
        if use_grid_search:
            logger.info("Running grid search...")
            base_model = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                base_model,
                GRID_PARAMS,
                cv=StratifiedKFold(cv_folds, shuffle=True, random_state=42),
                scoring="f1_weighted",
                n_jobs=-1,
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            logger.info(f"Best params: {best_params}")
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
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"  CV Score:  {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")

        # Store result
        result = {
            "version": version,
            "name": config["name"],
            "description": config["description"],
            "data_type": config["data_type"],
            "dataset_size": len(df),
            "feature_count": len(X.columns),
            "features": list(X.columns),
            "metrics": metrics,
            "best_params": best_params,
        }
        self.results.append(result)

        return model, result

    def save_version(self, model: RandomForestClassifier, result: Dict):
        """Save a trained version."""
        version = result["version"]

        # Save model
        model_path = self.models_dir / f"flood_rf_model_v{version}.joblib"
        joblib.dump(model, model_path)

        # Save metadata
        metadata = {
            "version": version,
            "model_type": "RandomForestClassifier",
            "created_at": datetime.now().isoformat(),
            **result,
        }

        metadata_path = model_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved: {model_path}")

        # Track best version
        if self.best_version is None or result["metrics"]["f1_score"] > self._get_best_f1():
            self.best_version = version

            # Save as latest
            latest_path = self.models_dir / "flood_rf_model_latest.joblib"
            joblib.dump(model, latest_path)

            default_path = self.models_dir / "flood_rf_model.joblib"
            joblib.dump(model, default_path)

            logger.info(f"Saved as best/latest: {latest_path}")

    def _get_best_f1(self) -> float:
        """Get F1 score of current best version."""
        if not self.results:
            return 0.0
        for r in self.results:
            if r["version"] == self.best_version:
                return r["metrics"]["f1_score"]
        return 0.0

    def train_all(self, use_grid_search: bool = False, latest_only: bool = False, cv_folds: int = 5):
        """Train all versions progressively."""
        logger.info("\n" + "=" * 70)
        logger.info("ULTIMATE PROGRESSIVE TRAINING")
        logger.info("Training all model versions: v1 → v2 → ... → ULTIMATE")
        logger.info("=" * 70)

        versions_to_train = [6] if latest_only else list(VERSION_REGISTRY.keys())

        for version in versions_to_train:
            result = self.train_version(version, use_grid_search=use_grid_search, cv_folds=cv_folds)
            if result:
                model, result_data = result
                self.save_version(model, result_data)

        # Generate reports
        self.generate_progression_report()
        self.generate_progression_chart()

    def generate_progression_report(self):
        """Generate comprehensive progression report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_versions": len(self.results),
            "best_version": self.best_version,
            "versions": self.results,
        }

        # Calculate improvements
        if len(self.results) >= 2:
            first = self.results[0]["metrics"]
            last = self.results[-1]["metrics"]
            report["improvement"] = {
                "accuracy": last["accuracy"] - first["accuracy"],
                "f1_score": last["f1_score"] - first["f1_score"],
                "roc_auc": last["roc_auc"] - first["roc_auc"],
            }

        report_path = self.reports_dir / "progressive_training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"\nProgression report saved: {report_path}")

    def generate_progression_chart(self):
        """Generate model progression visualization."""
        try:
            import matplotlib.pyplot as plt

            if not self.results:
                return

            versions = [r["version"] for r in self.results]
            names = [r["name"] for r in self.results]
            f1_scores = [r["metrics"]["f1_score"] for r in self.results]
            accuracy = [r["metrics"]["accuracy"] for r in self.results]
            roc_auc = [r["metrics"]["roc_auc"] for r in self.results]

            fig, ax = plt.subplots(figsize=(12, 6))

            x = np.arange(len(versions))
            width = 0.25

            bars1 = ax.bar(x - width, accuracy, width, label="Accuracy", color="#2ecc71", alpha=0.8)
            bars2 = ax.bar(x, f1_scores, width, label="F1 Score", color="#3498db", alpha=0.8)
            bars3 = ax.bar(x + width, roc_auc, width, label="ROC-AUC", color="#e74c3c", alpha=0.8)

            ax.set_xlabel("Model Version", fontsize=12, fontweight="bold")
            ax.set_ylabel("Score", fontsize=12, fontweight="bold")
            ax.set_title(
                "Ultimate Progressive Training\nModel Performance Evolution", fontsize=14, fontweight="bold", pad=20
            )
            ax.set_xticks(x)
            ax.set_xticklabels([f"v{v}\n{n}" for v, n in zip(versions, names)], fontsize=9)
            ax.legend(loc="lower right", fontsize=10)
            ax.set_ylim([0, 1.1])
            ax.grid(True, axis="y", alpha=0.3)

            # Add value labels
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                    )

            plt.tight_layout()
            chart_path = self.reports_dir / "model_progression_chart.png"
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Progression chart saved: {chart_path}")

        except ImportError:
            logger.warning("matplotlib not available, skipping chart")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ultimate progressive model training")
    parser.add_argument("--progressive", action="store_true", help="Train all versions progressively")
    parser.add_argument("--production", action="store_true", help="Use grid search for optimization")
    parser.add_argument("--latest-only", action="store_true", help="Only train the ULTIMATE version")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds")

    args = parser.parse_args()

    trainer = UltimateTrainer()

    # Default to progressive if no specific mode
    if not args.progressive and not args.latest_only:
        args.progressive = True

    trainer.train_all(use_grid_search=args.production, latest_only=args.latest_only, cv_folds=args.cv_folds)

    # Print summary
    print("\n" + "=" * 70)
    print("ULTIMATE PROGRESSIVE TRAINING COMPLETE")
    print("=" * 70)

    if trainer.results:
        print("\nVersion Summary:")
        print("-" * 50)
        for r in trainer.results:
            print(f"  v{r['version']} ({r['name']})")
            print(f"    F1: {r['metrics']['f1_score']:.4f}, Accuracy: {r['metrics']['accuracy']:.4f}")
            print(f"    Data: {r['dataset_size']} records, {r['feature_count']} features")

        if trainer.best_version:
            best = next(r for r in trainer.results if r["version"] == trainer.best_version)
            print(f"\nBest Version: v{trainer.best_version} ({best['name']})")
            print(f"  F1 Score: {best['metrics']['f1_score']:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
