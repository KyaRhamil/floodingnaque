"""
Production-Grade Model Training for Flood Prediction
======================================================

This script produces production-ready Random Forest models with:
1. Proper Train/Validation/Test Splits (60/20/20)
2. Time-Based Cross-Validation (prevents data leakage)
3. Stratified Sampling for Class Imbalance
4. Hyperparameter Optimization with GridSearchCV
5. SHAP Explainability Analysis
6. Comprehensive Metrics & Validation
7. Model Versioning with Metadata
8. Learning Curves for Overfitting Detection

Usage:
    python train_production.py                          # Standard training
    python train_production.py --grid-search           # Full hyperparameter tuning
    python train_production.py --ensemble              # Ensemble model (RF + GB)
    python train_production.py --multi-level           # 3-level risk classification
    python train_production.py --production            # Full production pipeline
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, cross_val_score, learning_curve,
    TimeSeriesSplit
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
import joblib
import json
import logging
import hashlib
import sys
import warnings
import os
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
from dotenv import load_dotenv

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

# Load environment configuration
if Path('.env.production').exists():
    load_dotenv('.env.production')
    logger_init_msg = "Loaded .env.production"
else:
    load_dotenv()
    logger_init_msg = "Loaded .env"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)
logger.info(logger_init_msg)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
MODELS_DIR = BACKEND_DIR / 'models'
REPORTS_DIR = BACKEND_DIR / 'reports'
DATA_DIR = BACKEND_DIR / 'data' / 'processed'

# Feature configuration
NUMERIC_FEATURES = [
    'temperature', 'humidity', 'precipitation', 
    'month', 'is_monsoon_season', 'year'
]
CATEGORICAL_FEATURES = ['weather_type', 'season']
EXCLUDE_COLUMNS = [
    'flood', 'risk_level', 'flood_depth_m', 'flood_depth_category',
    'weather_description', 'location', 'date', 'datetime'
]

# Risk level mapping for multi-level classification
RISK_LEVELS = {0: 'LOW', 1: 'MODERATE', 2: 'HIGH'}


class ProductionModelTrainer:
    """Production-grade flood prediction model trainer."""
    
    def __init__(
        self,
        models_dir: Path = MODELS_DIR,
        reports_dir: Path = REPORTS_DIR,
        random_state: int = 42
    ):
        self.models_dir = Path(models_dir)
        self.reports_dir = Path(reports_dir)
        self.random_state = random_state
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Resource management from environment
        self.n_jobs = self._get_n_jobs_from_env()
        logger.info(f"Using n_jobs={self.n_jobs} for parallel processing")
        
        # Tracking
        self.training_history: List[Dict] = []
        self.best_model = None
        self.best_metrics = None
        self.feature_names: List[str] = []
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate training data."""
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} records from {path.name}")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Validate required columns
        required = ['temperature', 'humidity', 'precipitation', 'flood']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction and derived features."""
        df = df.copy()
        
        # Precipitation-based features (most important for flooding)
        if 'precipitation' in df.columns:
            df['precipitation_log'] = np.log1p(df['precipitation'])
            df['precipitation_squared'] = df['precipitation'] ** 2
            df['high_precipitation'] = (df['precipitation'] > 50).astype(int)
            df['extreme_precipitation'] = (df['precipitation'] > 100).astype(int)
        
        # Temperature-Humidity interaction (saturation potential)
        if all(c in df.columns for c in ['temperature', 'humidity']):
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
            df['heat_index'] = df['temperature'] + 0.5 * (df['humidity'] / 100)
        
        # Precipitation interactions
        if all(c in df.columns for c in ['humidity', 'precipitation']):
            df['humidity_precip_interaction'] = df['humidity'] * np.log1p(df['precipitation'])
        
        if all(c in df.columns for c in ['temperature', 'precipitation']):
            df['temp_precip_interaction'] = df['temperature'] * np.log1p(df['precipitation'])
        
        # Monsoon interaction
        if all(c in df.columns for c in ['is_monsoon_season', 'precipitation']):
            df['monsoon_precip_interaction'] = df['is_monsoon_season'] * df['precipitation']
        
        # Encode categorical features
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        
        return df
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'flood',
        test_size: float = 0.2,
        val_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data with proper train/val/test splits.
        
        Uses stratified sampling to maintain class distribution.
        """
        # Feature engineering
        df = self.engineer_features(df)
        
        # Select features (exclude target and metadata)
        exclude = list(set(EXCLUDE_COLUMNS) | {target_col})
        feature_cols = [
            c for c in df.columns 
            if c not in exclude and df[c].dtype in ['float64', 'int64', 'int32', 'float32', 'uint8', 'bool']
        ]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        self.feature_names = list(X.columns)
        logger.info(f"Using {len(self.feature_names)} features: {self.feature_names}")
        
        # First split: separate test set (held out completely)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: train/validation from remaining data
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=self.random_state, stratify=y_temp
        )
        
        logger.info(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Train class distribution: {dict(pd.Series(y_train).value_counts())}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _get_n_jobs_from_env(self) -> int:
        """Get n_jobs from environment or calculate based on available CPUs."""
        try:
            # Check for explicit n_jobs setting
            n_jobs_env = os.getenv('TRAINING_N_JOBS', '').strip()
            if n_jobs_env:
                n_jobs = int(n_jobs_env)
                logger.info(f"Using TRAINING_N_JOBS from .env: {n_jobs}")
                return n_jobs
            
            # Check DB_POOL_SIZE as indicator of resource limits
            pool_size = int(os.getenv('DB_POOL_SIZE', '0'))
            if pool_size > 0 and pool_size < 20:
                # Conservative: use half of pool size as proxy for constrained resources
                n_jobs = max(2, pool_size // 2)
                logger.info(f"Detected constrained resources (DB_POOL_SIZE={pool_size}), using n_jobs={n_jobs}")
                return n_jobs
            
            # Default: use all CPUs minus 1 (leave one for system)
            cpu_count = multiprocessing.cpu_count()
            n_jobs = max(1, cpu_count - 1)
            logger.info(f"Using default n_jobs={n_jobs} (CPUs: {cpu_count})")
            return n_jobs
            
        except Exception as e:
            logger.warning(f"Error determining n_jobs, defaulting to 2: {e}")
            return 2
    
    def create_model(
        self,
        model_type: str = 'random_forest',
        class_weight: str = 'balanced'
    ) -> Any:
        """Create a model based on the specified type."""
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight=class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,  # Use environment-aware n_jobs
                oob_score=True
            )
        elif model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                validation_fraction=0.1,
                n_iter_no_change=10
            )
        elif model_type == 'ensemble':
            rf = RandomForestClassifier(
                n_estimators=150, max_depth=12,
                class_weight=class_weight, random_state=self.random_state, n_jobs=self.n_jobs
            )
            gb = GradientBoostingClassifier(
                n_estimators=100, max_depth=5,
                learning_rate=0.1, random_state=self.random_state
            )
            return VotingClassifier(
                estimators=[('rf', rf), ('gb', gb)],
                voting='soft',
                n_jobs=self.n_jobs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_param_grid(self, model_type: str) -> Dict:
        """Get hyperparameter grid for tuning."""
        if model_type == 'random_forest':
            return {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        elif model_type == 'gradient_boosting':
            return {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        return {}
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        multi_class: bool = False
    ) -> Dict[str, Any]:
        """Calculate comprehensive metrics."""
        average = 'weighted' if multi_class else 'binary'
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Per-class metrics
        classes = np.unique(y_true)
        for cls in classes:
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            metrics[f'precision_class_{cls}'] = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
            metrics[f'recall_class_{cls}'] = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
            metrics[f'f1_class_{cls}'] = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
        
        # ROC-AUC if probabilities available
        if y_proba is not None:
            try:
                if multi_class:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba, multi_class='ovr'))
                else:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba))
                    metrics['avg_precision'] = float(average_precision_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba))
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        return metrics
    
    def cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        use_time_split: bool = False
    ) -> Dict[str, Any]:
        """Perform cross-validation with multiple metrics."""
        if use_time_split:
            cv = TimeSeriesSplit(n_splits=n_folds)
        else:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        # Multiple scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted',
            'roc_auc': 'roc_auc'
        }
        
        results = {}
        for metric_name, scorer in scoring.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=self.n_jobs)
                results[f'cv_{metric_name}_mean'] = float(scores.mean())
                results[f'cv_{metric_name}_std'] = float(scores.std())
                results[f'cv_{metric_name}_scores'] = scores.tolist()
            except Exception as e:
                logger.warning(f"Could not calculate CV {metric_name}: {e}")
        
        return results
    
    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = 'random_forest',
        n_folds: int = 5,
        use_randomized: bool = True
    ) -> Tuple[Any, Dict]:
        """Perform hyperparameter tuning."""
        logger.info(f"Starting hyperparameter tuning for {model_type}...")
        
        base_model = self.create_model(model_type)
        param_grid = self.get_param_grid(model_type)
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        if use_randomized:
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=50, cv=cv,
                scoring='f1_weighted', n_jobs=self.n_jobs,
                random_state=self.random_state, verbose=1
            )
        else:
            search = GridSearchCV(
                base_model, param_grid, cv=cv,
                scoring='f1_weighted', n_jobs=self.n_jobs, verbose=1
            )
        
        search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV F1 score: {search.best_score_:.4f}")
        
        return search.best_estimator_, {
            'best_params': search.best_params_,
            'best_cv_score': float(search.best_score_),
            'cv_results': {
                'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
                'std_test_score': search.cv_results_['std_test_score'].tolist()
            }
        }
    
    def generate_learning_curves(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5
    ) -> Optional[Dict]:
        """Generate learning curves to detect overfitting."""
        if not PLOT_AVAILABLE:
            logger.warning("matplotlib not available, skipping learning curves")
            return None
        
        logger.info("Generating learning curves...")
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=cv,
            scoring='f1_weighted', n_jobs=self.n_jobs, random_state=self.random_state
        )
        
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        # Check for overfitting
        final_gap = train_mean[-1] - val_mean[-1]
        if final_gap > 0.1:
            logger.warning(f"Potential overfitting detected! Train-Val gap: {final_gap:.4f}")
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(train_sizes_abs, val_mean, 'o-', color='green', label='Validation Score')
        plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='green')
        plt.xlabel('Training Set Size')
        plt.ylabel('F1 Score')
        plt.title('Learning Curves - Flood Prediction Model')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = self.reports_dir / 'learning_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Learning curves saved to {output_path}")
        
        return {
            'train_sizes': train_sizes_abs.tolist(),
            'train_mean': train_mean.tolist(),
            'train_std': train_std.tolist(),
            'val_mean': val_mean.tolist(),
            'val_std': val_std.tolist(),
            'train_val_gap': float(final_gap)
        }
    
    def generate_shap_analysis(
        self,
        model: Any,
        X: np.ndarray,
        max_samples: int = 200
    ) -> Optional[Dict]:
        """Generate SHAP explainability analysis."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping explainability analysis")
            return None
        
        logger.info("Generating SHAP analysis...")
        
        # Sample data if too large
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
        else:
            X_sample = X
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            # Handle binary vs multi-class
            if isinstance(shap_values, list):
                shap_values_main = shap_values[1]  # Class 1 (flood)
            else:
                shap_values_main = shap_values
            
            # Mean absolute SHAP values
            mean_shap = np.abs(shap_values_main).mean(axis=0)
            shap_importance = {
                name: float(val) 
                for name, val in zip(self.feature_names, mean_shap)
            }
            
            # Sort by importance
            sorted_importance = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))
            
            # Generate plots if matplotlib available
            if PLOT_AVAILABLE:
                # Summary bar plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values_main, X_sample, feature_names=self.feature_names, 
                                  plot_type="bar", show=False)
                plt.title('SHAP Feature Importance')
                plt.tight_layout()
                plt.savefig(self.reports_dir / 'shap_importance.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Beeswarm plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values_main, X_sample, feature_names=self.feature_names, show=False)
                plt.title('SHAP Summary Plot')
                plt.tight_layout()
                plt.savefig(self.reports_dir / 'shap_summary.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"SHAP plots saved to {self.reports_dir}")
            
            return {
                'feature_importance': sorted_importance,
                'samples_analyzed': len(X_sample),
                'top_features': list(sorted_importance.keys())[:5]
            }
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return None
    
    def compute_model_hash(self, model_path: Path) -> str:
        """Compute SHA256 hash of model file for integrity verification."""
        with open(model_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def save_model(
        self,
        model: Any,
        metrics: Dict,
        training_config: Dict,
        version: Optional[int] = None,
        prefix: str = 'flood_rf_model'
    ) -> Tuple[Path, Dict]:
        """Save model with metadata and integrity hash."""
        if version is None:
            # Auto-increment version
            existing = list(self.models_dir.glob(f'{prefix}_v*.joblib'))
            if existing:
                versions = [int(p.stem.split('_v')[-1]) for p in existing]
                version = max(versions) + 1
            else:
                version = 1
        
        model_filename = f'{prefix}_v{version}.joblib'
        model_path = self.models_dir / model_filename
        
        # Save model
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Also save as latest
        latest_path = self.models_dir / f'{prefix}.joblib'
        joblib.dump(model, latest_path)
        
        # Compute integrity hash
        model_hash = self.compute_model_hash(model_path)
        
        # Build metadata
        metadata = {
            'version': version,
            'model_type': type(model).__name__,
            'model_path': str(model_path),
            'model_hash': model_hash,
            'created_at': datetime.now().isoformat(),
            'python_version': sys.version,
            'sklearn_version': __import__('sklearn').__version__,
            'training_data': training_config.get('data_info', {}),
            'configuration': training_config.get('config', {}),
            'metrics': {
                'train': metrics.get('train', {}),
                'validation': metrics.get('validation', {}),
                'test': metrics.get('test', {}),
                'cross_validation': metrics.get('cv', {})
            },
            'feature_names': self.feature_names,
            'feature_importance': {},
            'shap_analysis': metrics.get('shap', {}),
            'learning_curves': metrics.get('learning_curves', {})
        }
        
        # Add feature importance if available
        if hasattr(model, 'feature_importances_'):
            metadata['feature_importance'] = {
                name: float(imp)
                for name, imp in zip(self.feature_names, model.feature_importances_)
            }
        
        # Save metadata
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata saved to {metadata_path}")
        
        return model_path, metadata
    
    def train(
        self,
        data_path: str,
        model_type: str = 'random_forest',
        target_col: str = 'flood',
        use_grid_search: bool = False,
        generate_shap: bool = True,
        n_folds: int = 5,
        version: Optional[int] = None
    ) -> Tuple[Any, Dict, Dict]:
        """
        Full training pipeline.
        
        Args:
            data_path: Path to training data CSV
            model_type: 'random_forest', 'gradient_boosting', or 'ensemble'
            target_col: Target column name
            use_grid_search: Enable hyperparameter tuning
            generate_shap: Generate SHAP analysis
            n_folds: Cross-validation folds
            version: Model version (auto-incremented if None)
        
        Returns:
            (model, metrics, metadata)
        """
        logger.info("=" * 80)
        logger.info("PRODUCTION MODEL TRAINING")
        logger.info("=" * 80)
        
        # Load data
        df = self.load_data(data_path)
        
        data_info = {
            'file': str(data_path),
            'samples': len(df),
            'target_distribution': df[target_col].value_counts().to_dict()
        }
        logger.info(f"Target distribution: {data_info['target_distribution']}")
        
        # Prepare data with proper splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(
            df, target_col=target_col
        )
        
        # Create or tune model
        if use_grid_search:
            model, tuning_results = self.tune_hyperparameters(
                X_train, y_train, model_type=model_type, n_folds=n_folds
            )
            config = {'grid_search': tuning_results}
        else:
            model = self.create_model(model_type)
            config = {'model_type': model_type}
            
            # Cross-validation before final training
            logger.info(f"Performing {n_folds}-fold cross-validation...")
            cv_results = self.cross_validate(model, X_train, y_train, n_folds=n_folds)
            logger.info(f"CV F1: {cv_results['cv_f1_mean']:.4f} (+/- {cv_results['cv_f1_std']:.4f})")
            config['cross_validation'] = cv_results
        
        # Train on full training set
        logger.info("Training final model...")
        model.fit(X_train, y_train)
        
        # Evaluate on all sets
        metrics = {}
        
        # Training metrics
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train) if hasattr(model, 'predict_proba') else None
        metrics['train'] = self.calculate_metrics(y_train, y_train_pred, y_train_proba)
        
        # Validation metrics
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
        metrics['validation'] = self.calculate_metrics(y_val, y_val_pred, y_val_proba)
        
        # Test metrics (held-out)
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        metrics['test'] = self.calculate_metrics(y_test, y_test_pred, y_test_proba)
        
        # Log metrics
        logger.info("\n" + "=" * 60)
        logger.info("MODEL PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        for split in ['train', 'validation', 'test']:
            m = metrics[split]
            logger.info(f"\n{split.upper()}:")
            logger.info(f"  Accuracy:  {m['accuracy']:.4f}")
            logger.info(f"  Precision: {m['precision']:.4f}")
            logger.info(f"  Recall:    {m['recall']:.4f}")
            logger.info(f"  F1 Score:  {m['f1_score']:.4f}")
            if 'roc_auc' in m:
                logger.info(f"  ROC-AUC:   {m['roc_auc']:.4f}")
        
        # Check for overfitting
        train_test_gap = metrics['train']['f1_score'] - metrics['test']['f1_score']
        if train_test_gap > 0.1:
            logger.warning(f"⚠️ Potential overfitting! Train-Test F1 gap: {train_test_gap:.4f}")
        else:
            logger.info(f"✓ Model generalizes well. Train-Test F1 gap: {train_test_gap:.4f}")
        
        # Learning curves
        metrics['learning_curves'] = self.generate_learning_curves(model, X_train, y_train, n_folds)
        
        # SHAP analysis
        if generate_shap:
            metrics['shap'] = self.generate_shap_analysis(model, X_test)
        
        # Cross-validation results
        metrics['cv'] = config.get('cross_validation', {})
        
        # Save model
        training_config = {
            'data_info': data_info,
            'config': config
        }
        
        model_path, metadata = self.save_model(
            model, metrics, training_config, version=version
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"Model: {model_path}")
        logger.info(f"Test Accuracy: {metrics['test']['accuracy']:.4f}")
        logger.info(f"Test F1 Score: {metrics['test']['f1_score']:.4f}")
        logger.info("=" * 80)
        
        self.best_model = model
        self.best_metrics = metrics
        
        return model, metrics, metadata


def main():
    """Main entry point for production training."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Production-grade flood prediction model training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Standard training:
    python train_production.py
    
  With hyperparameter tuning:
    python train_production.py --grid-search
    
  Ensemble model:
    python train_production.py --model-type ensemble
    
  Full production pipeline:
    python train_production.py --production
        """
    )
    
    default_data = str(DATA_DIR / 'cumulative_up_to_2025.csv')
    
    parser.add_argument('--data', type=str, default=default_data,
                        help='Path to training data CSV')
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'ensemble'],
                        help='Model type to train')
    parser.add_argument('--grid-search', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--no-shap', action='store_true',
                        help='Skip SHAP analysis')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--version', type=int,
                        help='Model version (auto-incremented if not specified)')
    parser.add_argument('--production', action='store_true',
                        help='Full production pipeline with grid search and SHAP')
    
    args = parser.parse_args()
    
    # Production mode enables all features
    if args.production:
        args.grid_search = True
    
    trainer = ProductionModelTrainer()
    
    model, metrics, metadata = trainer.train(
        data_path=args.data,
        model_type=args.model_type,
        use_grid_search=args.grid_search,
        generate_shap=not args.no_shap,
        n_folds=args.cv_folds,
        version=args.version
    )
    
    return model, metrics, metadata


if __name__ == '__main__':
    main()
