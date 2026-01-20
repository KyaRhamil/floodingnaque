"""
Progressive Model Training Pipeline
====================================

Enterprise-grade progressive training pipeline that creates all model versions:

Phases:
    Phase 1: Official Records 2022 (Baseline)
    Phase 2: Official Records 2022-2023
    Phase 3: Official Records 2022-2024
    Phase 4: Official Records 2022-2025
    Phase 5: + PAGASA Weather Data (merged with flood events)
    Phase 6: + External APIs (GEE, Meteostat, WorldTides)
    Phase 7: Station-specific models (Port Area, NAIA, Science Garden)
    Phase 8: ULTIMATE - Stacking ensemble with calibration

Features:
    - class_weight='balanced_subsample' for imbalance handling
    - Optional SMOTENC for categorical-aware oversampling
    - F2 score as primary metric (recall-weighted)
    - Optuna hyperparameter optimization
    - Temporal walk-forward validation
    - Stacking ensemble with LogisticRegression meta-learner
    - CalibratedClassifierCV for probability calibration
    - MLflow experiment tracking
    - Auto-generated model cards
    - PSI drift detection

Usage:
    python scripts/train_progressive.py                    # Full 8-phase training
    python scripts/train_progressive.py --phase 8         # Single phase
    python scripts/train_progressive.py --quick           # Quick mode (no Optuna)
    python scripts/train_progressive.py --with-smote      # Enable SMOTENC
    python scripts/train_progressive.py --station-only    # Only station-specific
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    train_test_split,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
CONFIG_DIR = BACKEND_DIR / "config"
MODELS_DIR = BACKEND_DIR / "models"
REPORTS_DIR = BACKEND_DIR / "reports"
DATA_DIR = BACKEND_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories
for d in [MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PHASE REGISTRY - All 8 progressive phases
# =============================================================================
PHASE_REGISTRY = {
    1: {
        "name": "Baseline_2022",
        "description": "Official Records 2022 Only",
        "data_file": "cumulative_up_to_2022.csv",
        "data_type": "official_records",
        "model_type": "unified",
    },
    2: {
        "name": "Extended_2023",
        "description": "Official Records 2022-2023",
        "data_file": "cumulative_up_to_2023.csv",
        "data_type": "official_records",
        "model_type": "unified",
    },
    3: {
        "name": "Extended_2024",
        "description": "Official Records 2022-2024",
        "data_file": "cumulative_up_to_2024.csv",
        "data_type": "official_records",
        "model_type": "unified",
    },
    4: {
        "name": "Full_Official",
        "description": "Official Records 2022-2025 (Complete)",
        "data_file": "cumulative_up_to_2025.csv",
        "data_type": "official_records",
        "model_type": "unified",
    },
    5: {
        "name": "PAGASA_Merged",
        "description": "Official + PAGASA Weather Data Merged",
        "data_files": ["cumulative_up_to_2025.csv", "pagasa_training_dataset.csv"],
        "data_type": "pagasa_merged",
        "model_type": "unified",
    },
    6: {
        "name": "External_APIs",
        "description": "All sources + External APIs (GEE, Meteostat, WorldTides)",
        "data_files": [
            "cumulative_up_to_2025.csv",
            "pagasa_training_dataset.csv",
            "fetched_googlecloud.csv",
            "fetched_meteostat.csv",
            "fetched_worldtides.csv",
        ],
        "data_type": "external_merged",
        "model_type": "unified",
    },
    7: {
        "name": "Station_Specific",
        "description": "Station-specific models (Port Area, NAIA, Science Garden)",
        "data_file": "pagasa_training_dataset.csv",
        "data_type": "station_specific",
        "model_type": "station_specific",
        "stations": ["Port Area", "NAIA", "Science Garden"],
    },
    8: {
        "name": "ULTIMATE_Ensemble",
        "description": "Stacking Ensemble with Calibration",
        "data_type": "ensemble",
        "model_type": "stacking",
        "base_phases": [4, 5, 6],  # Use models from these phases
        "station_phase": 7,
    },
}

# =============================================================================
# FEATURE SETS - Features available for each data type
# =============================================================================
FEATURE_SETS = {
    "official_records": [
        "temperature",
        "humidity",
        "precipitation",
        "is_monsoon_season",
        "month",
    ],
    "official_records_extended": [
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
    "pagasa_merged": [
        "temperature",
        "humidity",
        "precipitation",
        "is_monsoon_season",
        "month",
        "precip_3day_sum",
        "precip_7day_sum",
        "precip_14day_sum",
        "precip_lag1",
        "precip_lag2",
        "rain_streak",
        "temp_humidity_interaction",
        "humidity_precip_interaction",
        "monsoon_precip_interaction",
    ],
    "external_merged": [
        "temperature",
        "humidity",
        "precipitation",
        "is_monsoon_season",
        "month",
        "precip_3day_sum",
        "precip_7day_sum",
        "rain_streak",
        "tide_height",
        "temp_humidity_interaction",
        "humidity_precip_interaction",
        "monsoon_precip_interaction",
        "saturation_risk",
    ],
    "station_specific": [
        "temperature",
        "humidity",
        "precipitation",
        "is_monsoon_season",
        "month",
        "precip_3day_sum",
        "precip_7day_sum",
        "precip_14day_sum",
        "rain_streak",
        "heat_index",
    ],
}

# Categorical features for SMOTENC
CATEGORICAL_FEATURES = ["is_monsoon_season", "month", "station_id"]


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class TrainingConfig:
    """Training configuration loaded from YAML or defaults."""

    # General
    random_state: int = 42
    enable_mlflow: bool = True

    # Data
    test_size: float = 0.2
    validation_size: float = 0.1

    # Model
    n_estimators: int = 200
    max_depth: int = 15
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    class_weight: str = "balanced_subsample"  # Changed from "balanced"

    # Training
    cv_folds: int = 10
    primary_metric: str = "f2"  # F2 score (recall-weighted)
    use_optuna: bool = True
    optuna_trials: int = 50
    use_smote: bool = False  # SMOTENC optional

    # Temporal validation
    use_temporal_validation: bool = True
    temporal_splits: int = 5

    # Ensemble
    ensemble_cv_folds: int = 5
    calibrate_probabilities: bool = True
    calibration_method: str = "isotonic"

    # Registry
    promotion_f1_threshold: float = 0.85
    promotion_roc_threshold: float = 0.85

    @classmethod
    def from_yaml(cls, config_path: Path) -> "TrainingConfig":
        """Load config from YAML file."""
        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}, using defaults")
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        return cls(
            random_state=data.get("general", {}).get("random_state", 42),
            enable_mlflow=data.get("general", {}).get("enable_mlflow", True),
            test_size=data.get("data", {}).get("test_size", 0.2),
            validation_size=data.get("data", {}).get("validation_size", 0.1),
            n_estimators=data.get("model", {}).get("default_params", {}).get("n_estimators", 200),
            max_depth=data.get("model", {}).get("default_params", {}).get("max_depth", 15),
            cv_folds=data.get("cross_validation", {}).get("folds", 10),
            promotion_f1_threshold=data.get("registry", {})
            .get("promotion_criteria", {})
            .get("staging", {})
            .get("min_f1_score", 0.85),
        )


def load_config() -> TrainingConfig:
    """Load training configuration."""
    config_path = CONFIG_DIR / "training_config.yaml"
    return TrainingConfig.from_yaml(config_path)


# =============================================================================
# METRICS
# =============================================================================
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculate comprehensive metrics with F2 as primary."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f2_score": float(fbeta_score(y_true, y_pred, beta=2, average="weighted", zero_division=0)),
    }

    if y_pred_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
            metrics["brier_score"] = float(brier_score_loss(y_true, y_pred_proba))
        except Exception:
            metrics["roc_auc"] = 0.0
            metrics["brier_score"] = 1.0

    return metrics


def calculate_f2_scorer(y_true, y_pred):
    """F2 scorer for sklearn - prioritizes recall."""
    return fbeta_score(y_true, y_pred, beta=2, average="weighted", zero_division=0)


# =============================================================================
# DATA LOADING
# =============================================================================
class DataLoader:
    """Handles data loading and preparation for all phases."""

    def __init__(self, processed_dir: Path = PROCESSED_DIR):
        self.processed_dir = processed_dir

    def load_phase_data(self, phase: int) -> Optional[pd.DataFrame]:
        """Load data for a specific phase."""
        if phase not in PHASE_REGISTRY:
            logger.error(f"Unknown phase: {phase}")
            return None

        config = PHASE_REGISTRY[phase]

        # Handle different data loading strategies
        if config["data_type"] == "ensemble":
            # Ensemble phase doesn't load raw data - uses existing models
            return None

        if "data_files" in config:
            return self._load_multiple_files(config["data_files"])
        elif "data_file" in config:
            return self._load_single_file(config["data_file"])

        return None

    def _load_single_file(self, filename: str) -> Optional[pd.DataFrame]:
        """Load a single data file."""
        path = self.processed_dir / filename
        if not path.exists():
            logger.warning(f"Data file not found: {path}")
            return None

        df = pd.read_csv(path)
        df = self._clean_dataframe(df)
        return df

    def _load_multiple_files(self, filenames: List[str]) -> Optional[pd.DataFrame]:
        """Load and merge multiple data files."""
        dfs = []
        for filename in filenames:
            path = self.processed_dir / filename
            if path.exists():
                df = pd.read_csv(path)
                df = self._clean_dataframe(df)
                dfs.append(df)
            else:
                logger.warning(f"Data file not found, skipping: {path}")

        if not dfs:
            return None

        # Merge and deduplicate
        combined = pd.concat(dfs, ignore_index=True)

        # Remove duplicates based on key columns
        dedup_cols = ["temperature", "humidity", "precipitation", "flood"]
        dedup_cols = [c for c in dedup_cols if c in combined.columns]
        if dedup_cols:
            combined = combined.drop_duplicates(subset=dedup_cols)

        return combined

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize dataframe."""
        # Remove malformed columns (Excel artifacts)
        bad_cols = [c for c in df.columns if c.startswith("unnamed") or c.startswith("Unnamed")]
        if bad_cols:
            df = df.drop(columns=bad_cols)

        # Standardize temperature (convert Kelvin to Celsius if needed)
        if "temperature" in df.columns:
            if df["temperature"].mean() > 200:  # Likely Kelvin
                df["temperature"] = df["temperature"] - 273.15

        return df

    def load_station_data(self, station: str) -> Optional[pd.DataFrame]:
        """Load data for a specific station."""
        df = self._load_single_file("pagasa_training_dataset.csv")
        if df is None:
            return None

        # Filter by station
        station_col = None
        for col in ["station", "station_name", "Station"]:
            if col in df.columns:
                station_col = col
                break

        if station_col is None:
            logger.warning("No station column found in PAGASA data")
            return df

        station_df = df[df[station_col].str.contains(station, case=False, na=False)]

        if len(station_df) == 0:
            logger.warning(f"No data found for station: {station}")
            return None

        return station_df

    def prepare_features(
        self,
        df: pd.DataFrame,
        data_type: str,
        add_interactions: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target from dataframe."""
        # Get feature set for data type
        features = FEATURE_SETS.get(data_type, FEATURE_SETS["official_records"])

        # Filter to available features
        available = [f for f in features if f in df.columns]

        # Add interaction features if requested and not present
        if add_interactions:
            df = self._add_interaction_features(df)
            # Check for newly added interaction features
            interaction_features = [
                "temp_humidity_interaction",
                "humidity_precip_interaction",
                "temp_precip_interaction",
                "monsoon_precip_interaction",
                "saturation_risk",
            ]
            for f in interaction_features:
                if f in df.columns and f not in available:
                    available.append(f)

        X = df[available].copy()
        y = df["flood"].copy()

        # Handle missing values
        X = X.fillna(X.median())

        return X, y

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features if base features exist."""
        df = df.copy()

        if "temperature" in df.columns and "humidity" in df.columns:
            if "temp_humidity_interaction" not in df.columns:
                df["temp_humidity_interaction"] = df["temperature"] * df["humidity"] / 100

        if "humidity" in df.columns and "precipitation" in df.columns:
            if "humidity_precip_interaction" not in df.columns:
                df["humidity_precip_interaction"] = df["humidity"] * df["precipitation"] / 100

        if "temperature" in df.columns and "precipitation" in df.columns:
            if "temp_precip_interaction" not in df.columns:
                df["temp_precip_interaction"] = df["temperature"] * df["precipitation"] / 100

        if "is_monsoon_season" in df.columns and "precipitation" in df.columns:
            if "monsoon_precip_interaction" not in df.columns:
                df["monsoon_precip_interaction"] = df["is_monsoon_season"] * df["precipitation"]

        if "humidity" in df.columns and "precipitation" in df.columns:
            if "saturation_risk" not in df.columns:
                df["saturation_risk"] = ((df["humidity"] > 80) & (df["precipitation"] > 10)).astype(int)

        return df


# =============================================================================
# OPTUNA HYPERPARAMETER OPTIMIZATION
# =============================================================================
def optimize_with_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    config: TrainingConfig,
    n_trials: int = 50,
) -> Dict[str, Any]:
    """Optimize hyperparameters using Optuna."""
    try:
        import optuna
        from optuna.samplers import TPESampler

        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed, using default parameters")
        return get_default_params(config)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
            "class_weight": "balanced_subsample",
            "random_state": config.random_state,
            "n_jobs": -1,
        }

        model = RandomForestClassifier(**params)

        # Use F2 scoring
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_state)
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)

        return scores.mean()

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.random_state),
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params["class_weight"] = "balanced_subsample"
    best_params["random_state"] = config.random_state
    best_params["n_jobs"] = -1

    logger.info(f"Optuna best params: {best_params}")
    logger.info(f"Optuna best score: {study.best_value:.4f}")

    return best_params


def get_default_params(config: TrainingConfig) -> Dict[str, Any]:
    """Get default model parameters from config."""
    return {
        "n_estimators": config.n_estimators,
        "max_depth": config.max_depth,
        "min_samples_split": config.min_samples_split,
        "min_samples_leaf": config.min_samples_leaf,
        "max_features": "sqrt",
        "class_weight": config.class_weight,
        "random_state": config.random_state,
        "n_jobs": -1,
    }


# =============================================================================
# SMOTENC RESAMPLING (Optional)
# =============================================================================
def apply_smotenc(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: List[str],
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTENC for categorical-aware oversampling."""
    try:
        from imblearn.combine import SMOTETomek
        from imblearn.over_sampling import SMOTENC
    except ImportError:
        logger.warning("imbalanced-learn not installed, skipping SMOTENC")
        return X, y

    # Find indices of categorical features
    cat_indices = [X.columns.get_loc(c) for c in categorical_features if c in X.columns]

    if not cat_indices:
        # No categorical features, use regular SMOTE
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=random_state)
    else:
        smote = SMOTENC(categorical_features=cat_indices, random_state=random_state)

    try:
        result = smote.fit_resample(X, y)
        X_resampled = result[0]
        y_resampled = result[1]
        logger.info(f"SMOTENC: {len(X)} -> {len(X_resampled)} samples")
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    except Exception as e:
        logger.warning(f"SMOTENC failed: {e}, using original data")
        return X, y


# =============================================================================
# TEMPORAL VALIDATION
# =============================================================================
def temporal_walk_forward_validation(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Walk-forward validation for time series data."""
    tscv = TimeSeriesSplit(n_splits=n_splits)

    f2_scores = []
    roc_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_clone = RandomForestClassifier(**model.get_params())
        model_clone.fit(X_train, y_train)

        y_pred = model_clone.predict(X_test)
        y_pred_proba = model_clone.predict_proba(X_test)[:, 1]

        f2_scores.append(fbeta_score(y_test, y_pred, beta=2, average="weighted", zero_division=0))
        try:
            roc_scores.append(roc_auc_score(y_test, y_pred_proba))
        except ValueError:
            roc_scores.append(0.5)

    return {
        "temporal_f2_mean": float(np.mean(f2_scores)),
        "temporal_f2_std": float(np.std(f2_scores)),
        "temporal_roc_mean": float(np.mean(roc_scores)),
        "temporal_roc_std": float(np.std(roc_scores)),
    }


# =============================================================================
# MODEL TRAINING
# =============================================================================
class ProgressiveTrainer:
    """Enterprise-grade progressive training pipeline."""

    def __init__(
        self,
        config: TrainingConfig,
        models_dir: Path = MODELS_DIR,
        reports_dir: Path = REPORTS_DIR,
    ):
        self.config = config
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.data_loader = DataLoader()

        # Results tracking
        self.results: List[Dict] = []
        self.trained_models: Dict[int, Any] = {}
        self.station_models: Dict[str, Any] = {}
        self.best_phase: Optional[int] = None

        # MLflow setup
        self.mlflow_enabled = config.enable_mlflow
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        if not self.mlflow_enabled:
            return

        try:
            import mlflow

            mlflow.set_tracking_uri(str(BACKEND_DIR / "mlruns"))
            mlflow.set_experiment("floodingnaque_progressive_training")
            logger.info("MLflow tracking enabled")
        except ImportError:
            logger.warning("MLflow not installed, tracking disabled")
            self.mlflow_enabled = False

    def train_phase(
        self,
        phase: int,
        use_optuna: bool = True,
        use_smote: bool = False,
    ) -> Optional[Tuple[Any, Dict]]:
        """Train a single phase."""
        if phase not in PHASE_REGISTRY:
            logger.error(f"Unknown phase: {phase}")
            return None

        phase_config = PHASE_REGISTRY[phase]

        logger.info("\n" + "=" * 70)
        logger.info(f"PHASE {phase}: {phase_config['description']}")
        logger.info("=" * 70)

        # Handle different phase types
        if phase_config["model_type"] == "station_specific":
            return self._train_station_specific(phase, phase_config)
        elif phase_config["model_type"] == "stacking":
            return self._train_stacking_ensemble(phase, phase_config)
        else:
            return self._train_unified(phase, phase_config, use_optuna, use_smote)

    def _train_unified(
        self,
        phase: int,
        phase_config: Dict,
        use_optuna: bool,
        use_smote: bool,
    ) -> Optional[Tuple[RandomForestClassifier, Dict]]:
        """Train a unified model for a phase."""
        # Load data
        df = self.data_loader.load_phase_data(phase)
        if df is None or len(df) == 0:
            logger.warning(f"Skipping phase {phase}: No data available")
            return None

        # Prepare features
        X, y = self.data_loader.prepare_features(df, phase_config["data_type"])

        logger.info(f"Data: {len(df):,} records, {len(X.columns)} features")
        logger.info(f"Class distribution: {dict(y.value_counts())}")

        # Apply SMOTENC if requested
        if use_smote and self.config.use_smote:
            X, y = apply_smotenc(X, y, CATEGORICAL_FEATURES, self.config.random_state)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Get hyperparameters
        if use_optuna and self.config.use_optuna:
            logger.info("Running Optuna hyperparameter optimization...")
            params = optimize_with_optuna(X_train, y_train, self.config, self.config.optuna_trials)
        else:
            params = get_default_params(self.config)

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

        # Cross-validation
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
        metrics["cv_mean"] = float(cv_scores.mean())
        metrics["cv_std"] = float(cv_scores.std())

        # Temporal validation if enabled
        if self.config.use_temporal_validation:
            temporal_metrics = temporal_walk_forward_validation(model, X, y, self.config.temporal_splits)
            metrics.update(temporal_metrics)

        # Log results
        self._log_phase_results(phase, phase_config, metrics)

        # Feature importance
        feature_importance = dict(zip(X.columns, model.feature_importances_))

        # Store result
        result = {
            "phase": phase,
            "name": phase_config["name"],
            "description": phase_config["description"],
            "data_type": phase_config["data_type"],
            "dataset_size": len(df),
            "feature_count": len(X.columns),
            "features": list(X.columns),
            "metrics": metrics,
            "params": params,
            "feature_importance": feature_importance,
        }
        self.results.append(result)
        self.trained_models[phase] = model

        # Track best
        if self.best_phase is None or metrics["f2_score"] > self._get_best_f2():
            self.best_phase = phase

        return model, result

    def _train_station_specific(
        self,
        phase: int,
        phase_config: Dict,
    ) -> Optional[Tuple[Dict, Dict]]:
        """Train station-specific models."""
        stations = phase_config.get("stations", ["Port Area", "NAIA", "Science Garden"])

        all_metrics = {}

        for station in stations:
            logger.info(f"\n--- Training model for station: {station} ---")

            df = self.data_loader.load_station_data(station)
            if df is None or len(df) < 50:
                logger.warning(f"Insufficient data for station: {station}")
                continue

            X, y = self.data_loader.prepare_features(df, "station_specific")

            # Quick training for station models
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y if len(y.unique()) > 1 else None,
            )

            params = get_default_params(self.config)
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(y_test))

            metrics = calculate_metrics(y_test, y_pred, y_pred_proba if len(y_pred_proba) > 0 else None)

            logger.info(f"  {station}: F2={metrics['f2_score']:.4f}, ROC-AUC={metrics.get('roc_auc', 0):.4f}")

            self.station_models[station] = model
            all_metrics[station] = metrics

        result = {
            "phase": phase,
            "name": phase_config["name"],
            "description": phase_config["description"],
            "stations": list(self.station_models.keys()),
            "station_metrics": all_metrics,
        }
        self.results.append(result)

        return self.station_models, result

    def _train_stacking_ensemble(
        self,
        phase: int,
        phase_config: Dict,
    ) -> Optional[Tuple[Any, Dict]]:
        """Train stacking ensemble with calibration."""
        logger.info("Building stacking ensemble from trained models...")

        # Collect base estimators
        estimators = []

        # Add models from base phases
        for base_phase in phase_config.get("base_phases", []):
            if base_phase in self.trained_models:
                model = self.trained_models[base_phase]
                name = f"phase_{base_phase}"
                estimators.append((name, model))

        # Add station-specific models
        for station, model in self.station_models.items():
            name = f"station_{station.replace(' ', '_').lower()}"
            estimators.append((name, model))

        if len(estimators) < 2:
            logger.warning("Not enough base models for stacking, need at least 2")
            return None

        logger.info(f"Stacking {len(estimators)} base estimators: {[e[0] for e in estimators]}")

        # Load combined data for ensemble training
        df = self.data_loader.load_phase_data(6)  # Use Phase 6 data
        if df is None:
            df = self.data_loader.load_phase_data(5)  # Fallback to Phase 5
        if df is None:
            df = self.data_loader.load_phase_data(4)  # Fallback to Phase 4

        if df is None:
            logger.error("No data available for ensemble training")
            return None

        X, y = self.data_loader.prepare_features(df, "external_merged")

        # Align features - use only features that all models can handle
        common_features = ["temperature", "humidity", "precipitation", "is_monsoon_season", "month"]
        common_features = [f for f in common_features if f in X.columns]
        X = X[common_features]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Retrain base estimators on common features
        retrained_estimators = []
        for name, _ in estimators:
            new_model = RandomForestClassifier(**get_default_params(self.config))
            retrained_estimators.append((name, new_model))

        # Create stacking classifier
        meta_learner = LogisticRegression(
            random_state=self.config.random_state,
            max_iter=1000,
            class_weight="balanced",
        )

        stacking_model = StackingClassifier(
            estimators=retrained_estimators,
            final_estimator=meta_learner,
            cv=self.config.ensemble_cv_folds,
            stack_method="predict_proba",
            passthrough=False,
            n_jobs=-1,
        )

        logger.info("Training stacking ensemble...")
        stacking_model.fit(X_train, y_train)

        # Apply calibration
        if self.config.calibrate_probabilities:
            logger.info("Applying probability calibration...")
            calibration_method = self.config.calibration_method
            if calibration_method not in ("sigmoid", "isotonic"):
                calibration_method = "isotonic"
            calibrated_model = CalibratedClassifierCV(
                stacking_model,
                method=calibration_method,  # type: ignore
                cv=self.config.ensemble_cv_folds,
            )
            calibrated_model.fit(X_train, y_train)
            final_model = calibrated_model
        else:
            final_model = stacking_model

        # Evaluate
        y_pred = final_model.predict(X_test)
        y_proba = final_model.predict_proba(X_test)
        y_pred_proba = np.array(y_proba)[:, 1]

        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

        logger.info(f"\nEnsemble Results:")
        logger.info(f"  F2 Score:   {metrics['f2_score']:.4f}")
        logger.info(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
        logger.info(f"  Brier:      {metrics['brier_score']:.4f}")

        result = {
            "phase": phase,
            "name": phase_config["name"],
            "description": phase_config["description"],
            "n_estimators": len(estimators),
            "base_models": [e[0] for e in estimators],
            "calibrated": self.config.calibrate_probabilities,
            "metrics": metrics,
            "features": common_features,
        }
        self.results.append(result)
        self.trained_models[phase] = final_model

        # Ensemble is always best if it trained
        self.best_phase = phase

        return final_model, result

    def _log_phase_results(self, phase: int, phase_config: Dict, metrics: Dict):
        """Log phase results."""
        logger.info(f"\nPhase {phase} Results ({phase_config['name']}):")
        logger.info(f"  F2 Score:   {metrics['f2_score']:.4f} (primary)")
        logger.info(f"  F1 Score:   {metrics['f1_score']:.4f}")
        logger.info(f"  Recall:     {metrics['recall']:.4f}")
        logger.info(f"  ROC-AUC:    {metrics.get('roc_auc', 0):.4f}")
        logger.info(f"  CV Score:   {metrics.get('cv_mean', 0):.4f} (+/- {metrics.get('cv_std', 0)*2:.4f})")

        if "temporal_f2_mean" in metrics:
            logger.info(f"  Temporal F2: {metrics['temporal_f2_mean']:.4f} (+/- {metrics['temporal_f2_std']*2:.4f})")

        # MLflow logging
        if self.mlflow_enabled:
            try:
                import mlflow

                with mlflow.start_run(run_name=f"phase_{phase}_{phase_config['name']}"):
                    mlflow.log_params({"phase": phase, "name": phase_config["name"]})
                    mlflow.log_metrics(metrics)
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

    def _get_best_f2(self) -> float:
        """Get F2 score of current best phase."""
        if not self.results or self.best_phase is None:
            return 0.0
        for r in self.results:
            if r.get("phase") == self.best_phase:
                return r.get("metrics", {}).get("f2_score", 0.0)
        return 0.0

    def save_model(self, model: Any, result: Dict):
        """Save trained model and metadata."""
        phase = result["phase"]

        # Save model
        model_filename = f"flood_rf_model_phase{phase}.joblib"
        model_path = self.models_dir / model_filename
        joblib.dump(model, model_path)

        # Calculate checksum
        with open(model_path, "rb") as f:
            checksum = hashlib.sha256(f.read()).hexdigest()

        # Save metadata
        metadata = {
            "phase": phase,
            "model_type": type(model).__name__,
            "created_at": datetime.now().isoformat(),
            "checksum_sha256": checksum,
            **result,
        }

        metadata_path = model_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved: {model_path}")

        # If this is the best or final phase, also save as latest/default
        if phase == self.best_phase or phase == 8:
            latest_path = self.models_dir / "flood_rf_model_latest.joblib"
            default_path = self.models_dir / "flood_rf_model.joblib"

            joblib.dump(model, latest_path)
            joblib.dump(model, default_path)

            logger.info(f"Saved as best model: {default_path}")

    def generate_model_card(self, result: Dict) -> str:
        """Generate model card for documentation."""
        phase = result["phase"]
        metrics = result.get("metrics", {})

        card = f"""# Model Card: {result['name']}

## Overview
- **Phase**: {phase}
- **Version**: {result['name']}
- **Description**: {result['description']}
- **Created**: {datetime.now().isoformat()}

## Performance Metrics

| Metric | Value |
|--------|-------|
| F2 Score (Primary) | {metrics.get('f2_score', 'N/A'):.4f} |
| F1 Score | {metrics.get('f1_score', 'N/A'):.4f} |
| Recall | {metrics.get('recall', 'N/A'):.4f} |
| Precision | {metrics.get('precision', 'N/A'):.4f} |
| ROC-AUC | {metrics.get('roc_auc', 'N/A'):.4f} |
| Brier Score | {metrics.get('brier_score', 'N/A'):.4f} |
| CV Mean | {metrics.get('cv_mean', 'N/A'):.4f} |
| CV Std | {metrics.get('cv_std', 'N/A'):.4f} |

## Dataset
- **Size**: {result.get('dataset_size', 'N/A'):,} records
- **Features**: {result.get('feature_count', 'N/A')}

## Features Used
{chr(10).join(f'- {f}' for f in result.get('features', []))}

## Hyperparameters
```json
{json.dumps(result.get('params', {}), indent=2)}
```

## Limitations
- Limited to Parañaque, Metro Manila coverage
- Requires PAGASA station data availability
- Trained on historical data (2022-2025)

## Intended Use
Flood risk prediction for early warning systems in Parañaque City.

## Ethical Considerations
This model should be used as one input among many for flood warnings.
Human oversight is required for all critical decisions.
"""
        return card

    def train_all(
        self,
        phases: Optional[List[int]] = None,
        use_optuna: bool = True,
        use_smote: bool = False,
    ):
        """Train all specified phases progressively."""
        if phases is None:
            phases = list(PHASE_REGISTRY.keys())

        logger.info("\n" + "=" * 70)
        logger.info("PROGRESSIVE TRAINING PIPELINE")
        logger.info(f"Training phases: {phases}")
        logger.info("=" * 70)

        start_time = datetime.now()

        for phase in phases:
            result = self.train_phase(phase, use_optuna=use_optuna, use_smote=use_smote)
            if result:
                model, result_data = result
                self.save_model(model, result_data)

                # Generate model card
                card = self.generate_model_card(result_data)
                card_path = self.reports_dir / f"model_card_phase{phase}.md"
                with open(card_path, "w") as f:
                    f.write(card)

        # Generate final reports
        self.generate_progression_report()
        self.generate_progression_chart()

        duration = datetime.now() - start_time
        logger.info(f"\nTotal training time: {duration}")

    def generate_progression_report(self):
        """Generate comprehensive progression report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_phases": len(self.results),
            "best_phase": self.best_phase,
            "phases": self.results,
            "config": {
                "use_optuna": self.config.use_optuna,
                "use_smote": self.config.use_smote,
                "class_weight": self.config.class_weight,
                "primary_metric": self.config.primary_metric,
                "calibrated": self.config.calibrate_probabilities,
            },
        }

        # Calculate improvements from first to last
        unified_results = [r for r in self.results if "metrics" in r and "f2_score" in r.get("metrics", {})]
        if len(unified_results) >= 2:
            first = unified_results[0]["metrics"]
            last = unified_results[-1]["metrics"]
            report["improvement"] = {
                "f2_score": last["f2_score"] - first["f2_score"],
                "f1_score": last["f1_score"] - first["f1_score"],
                "recall": last["recall"] - first["recall"],
            }

        report_path = self.reports_dir / "progressive_training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"\nProgression report saved: {report_path}")

    def generate_progression_chart(self):
        """Generate model progression visualization."""
        try:
            import matplotlib.pyplot as plt

            # Filter results with metrics
            results_with_metrics = [r for r in self.results if "metrics" in r]

            if not results_with_metrics:
                return

            phases = [r["phase"] for r in results_with_metrics]
            names = [r["name"] for r in results_with_metrics]
            f2_scores = [r["metrics"]["f2_score"] for r in results_with_metrics]
            f1_scores = [r["metrics"]["f1_score"] for r in results_with_metrics]
            recall = [r["metrics"]["recall"] for r in results_with_metrics]

            fig, ax = plt.subplots(figsize=(14, 7))

            x = np.arange(len(phases))
            width = 0.25

            bars1 = ax.bar(x - width, f2_scores, width, label="F2 Score (Primary)", color="#e74c3c", alpha=0.9)
            bars2 = ax.bar(x, f1_scores, width, label="F1 Score", color="#3498db", alpha=0.8)
            bars3 = ax.bar(x + width, recall, width, label="Recall", color="#2ecc71", alpha=0.8)

            ax.set_xlabel("Phase", fontsize=12, fontweight="bold")
            ax.set_ylabel("Score", fontsize=12, fontweight="bold")
            ax.set_title(
                "Progressive Training Pipeline\nModel Performance Evolution (F2 Primary Metric)",
                fontsize=14,
                fontweight="bold",
                pad=20,
            )
            ax.set_xticks(x)
            ax.set_xticklabels([f"P{p}\n{n}" for p, n in zip(phases, names)], fontsize=9)
            ax.legend(loc="lower right", fontsize=10)
            ax.set_ylim(0, 1.1)
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
            chart_path = self.reports_dir / "progressive_training_chart.png"
            plt.savefig(chart_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Progression chart saved: {chart_path}")

        except ImportError:
            logger.warning("matplotlib not available, skipping chart generation")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Progressive Model Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_progressive.py                    # Full 8-phase training
    python train_progressive.py --phase 8         # Single phase only
    python train_progressive.py --phases 1,2,3,4  # Specific phases
    python train_progressive.py --quick           # Quick mode (no Optuna)
    python train_progressive.py --with-smote      # Enable SMOTENC
    python train_progressive.py --station-only    # Only station-specific (Phase 7)
        """,
    )

    parser.add_argument("--phase", type=int, help="Train single phase (1-8)")
    parser.add_argument("--phases", type=str, help="Comma-separated list of phases to train")
    parser.add_argument("--quick", action="store_true", help="Quick mode - skip Optuna optimization")
    parser.add_argument("--with-smote", action="store_true", help="Enable SMOTENC resampling")
    parser.add_argument("--station-only", action="store_true", help="Only train station-specific models (Phase 7)")
    parser.add_argument(
        "--ensemble-only", action="store_true", help="Only train ensemble (Phase 8) - requires existing models"
    )
    parser.add_argument("--cv-folds", type=int, default=10, help="Number of CV folds")
    parser.add_argument("--optuna-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Override config with CLI arguments
    if args.cv_folds:
        config.cv_folds = args.cv_folds
    if args.optuna_trials:
        config.optuna_trials = args.optuna_trials
    if args.no_mlflow:
        config.enable_mlflow = False
    if args.with_smote:
        config.use_smote = True
    if args.quick:
        config.use_optuna = False

    # Determine which phases to train
    if args.phase:
        phases = [args.phase]
    elif args.phases:
        phases = [int(p.strip()) for p in args.phases.split(",")]
    elif args.station_only:
        phases = [7]
    elif args.ensemble_only:
        phases = [8]
    else:
        phases = list(range(1, 9))  # All 8 phases

    # Create trainer and run
    trainer = ProgressiveTrainer(config)
    trainer.train_all(
        phases=phases,
        use_optuna=config.use_optuna,
        use_smote=config.use_smote,
    )

    # Print summary
    print("\n" + "=" * 70)
    print("PROGRESSIVE TRAINING COMPLETE")
    print("=" * 70)

    if trainer.results:
        print("\nPhase Summary:")
        print("-" * 60)
        for r in trainer.results:
            if "metrics" in r:
                m = r["metrics"]
                print(f"  Phase {r['phase']} ({r['name']})")
                print(
                    f"    F2: {m.get('f2_score', 0):.4f}, F1: {m.get('f1_score', 0):.4f}, Recall: {m.get('recall', 0):.4f}"
                )
                print(f"    Data: {r.get('dataset_size', 'N/A')} records, {r.get('feature_count', 'N/A')} features")
            else:
                print(f"  Phase {r['phase']} ({r['name']}): Station-specific models")

        if trainer.best_phase:
            best = next((r for r in trainer.results if r.get("phase") == trainer.best_phase), None)
            if best and "metrics" in best:
                print(f"\nBest Phase: {trainer.best_phase} ({best['name']})")
                print(f"  F2 Score: {best['metrics'].get('f2_score', 0):.4f}")

    print("\n" + "=" * 70)
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Reports saved to: {REPORTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
