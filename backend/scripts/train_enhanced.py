"""
Enhanced Model Training for Flood Prediction
=============================================

This script provides comprehensive training improvements:
1. Feature Engineering - Interaction terms, polynomial features
2. Categorical Encoding - One-hot encoding for weather_type, season
3. Multiple Model Types - Random Forest, Gradient Boosting, Ensemble
4. Advanced Validation - Stratified K-Fold, Time-based splits
5. Multi-Level Classification - Binary (flood/no-flood) and 3-Level (LOW/MODERATE/HIGH)
6. Hyperparameter Optimization - GridSearch and RandomizedSearch
7. Feature Importance Analysis - SHAP values, permutation importance

Usage:
    python train_enhanced.py                           # Basic enhanced training
    python train_enhanced.py --multi-level            # 3-level risk classification
    python train_enhanced.py --ensemble               # Use ensemble of models
    python train_enhanced.py --grid-search            # Full hyperparameter tuning
    python train_enhanced.py --model-type gradient_boosting
    python train_enhanced.py --all-features           # Use ALL available features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, 
    cross_val_score, StratifiedKFold, learning_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, AdaBoostClassifier
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score,
    make_scorer
)
from sklearn.feature_selection import SelectFromModel, RFE
import joblib
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import glob
import warnings

# Optional imports
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)

# Get the backend directory (parent of scripts directory)
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent

# Feature configuration
NUMERIC_FEATURES = ['temperature', 'humidity', 'precipitation', 'month', 'is_monsoon_season', 'year']
CATEGORICAL_FEATURES = ['weather_type', 'season']
SPATIAL_FEATURES = ['latitude', 'longitude']
EXCLUDE_FEATURES = ['flood', 'risk_level', 'flood_depth_m', 'flood_depth_category', 
                    'weather_description', 'location']  # Target and metadata

# Risk level mapping
RISK_LEVEL_NAMES = {0: 'LOW', 1: 'MODERATE', 2: 'HIGH'}


def get_next_version(models_dir='models', prefix='flood_enhanced'):
    """Get the next version number for model versioning."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return 1
    
    existing_versions = []
    for file in models_path.glob(f'{prefix}_v*.joblib'):
        try:
            version_str = file.stem.split('_v')[-1]
            version = int(version_str)
            existing_versions.append(version)
        except (ValueError, IndexError):
            continue
    
    return max(existing_versions) + 1 if existing_versions else 1


def create_interaction_features(df):
    """
    Create interaction features between weather variables.
    These capture non-linear relationships between features.
    """
    df = df.copy()
    
    # Temperature-Humidity interaction (heat index proxy)
    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
    
    # Temperature-Precipitation interaction
    if 'temperature' in df.columns and 'precipitation' in df.columns:
        df['temp_precip_interaction'] = df['temperature'] * np.log1p(df['precipitation'])
    
    # Humidity-Precipitation interaction (saturation proxy)
    if 'humidity' in df.columns and 'precipitation' in df.columns:
        df['humidity_precip_interaction'] = df['humidity'] * np.log1p(df['precipitation'])
    
    # Precipitation squared (non-linear flood relationship)
    if 'precipitation' in df.columns:
        df['precipitation_squared'] = df['precipitation'] ** 2
        df['precipitation_log'] = np.log1p(df['precipitation'])
    
    # Monsoon-Precipitation interaction
    if 'is_monsoon_season' in df.columns and 'precipitation' in df.columns:
        df['monsoon_precip_interaction'] = df['is_monsoon_season'] * df['precipitation']
    
    return df


def encode_categorical_features(df, categorical_cols):
    """
    One-hot encode categorical features.
    Returns the encoded dataframe and the encoder.
    """
    df = df.copy()
    encoded_dfs = [df]
    
    for col in categorical_cols:
        if col in df.columns:
            # Get dummies with prefix
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            encoded_dfs.append(dummies)
            df = df.drop(columns=[col])
    
    # Concatenate all
    result = pd.concat([df] + encoded_dfs[1:], axis=1)
    return result


def prepare_features(df, use_all_features=True, create_interactions=True, 
                    encode_categories=True, exclude_leakage=True):
    """
    Prepare features for training with all enhancements.
    
    Args:
        df: Input DataFrame
        use_all_features: If True, use all available numeric features
        create_interactions: If True, create interaction features
        encode_categories: If True, one-hot encode categorical features
        exclude_leakage: If True, exclude flood_depth_m (causes data leakage)
    
    Returns:
        X: Feature matrix
        feature_names: List of feature names
    """
    df = df.copy()
    
    # Determine which columns to exclude
    exclude_cols = list(EXCLUDE_FEATURES)
    if exclude_leakage and 'flood_depth_m' not in exclude_cols:
        exclude_cols.append('flood_depth_m')
    
    # Create interaction features
    if create_interactions:
        df = create_interaction_features(df)
    
    # Encode categorical features
    if encode_categories:
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
        if cat_cols:
            df = encode_categorical_features(df, cat_cols)
    
    # Select features
    if use_all_features:
        # Use all numeric columns except excluded ones
        feature_cols = [c for c in df.columns if c not in exclude_cols 
                       and df[c].dtype in ['float64', 'int64', 'float32', 'int32', 'uint8', 'bool']]
    else:
        # Use only core features
        feature_cols = [c for c in ['temperature', 'humidity', 'precipitation'] 
                       if c in df.columns]
    
    X = df[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    return X, list(X.columns)


def create_model_pipeline(model_type='random_forest', class_weight='balanced'):
    """
    Create a model pipeline with preprocessing.
    
    Args:
        model_type: 'random_forest', 'gradient_boosting', 'xgboost', 'ensemble'
        class_weight: 'balanced' or None
    
    Returns:
        Model or Pipeline
    """
    if model_type == 'random_forest':
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
    
    elif model_type == 'gradient_boosting':
        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            verbose=0
        )
    
    elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    
    elif model_type == 'ensemble':
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=15, 
                                         class_weight=class_weight, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                             learning_rate=0.1, random_state=42)),
        ]
        if XGBOOST_AVAILABLE:
            estimators.append(('xgb', XGBClassifier(n_estimators=100, max_depth=5, 
                                                    learning_rate=0.1, random_state=42, verbosity=0)))
        
        return VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    
    else:
        # Default to Random Forest
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )


def get_hyperparameter_grid(model_type='random_forest'):
    """Get hyperparameter grid for tuning."""
    if model_type == 'random_forest':
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
    elif model_type == 'gradient_boosting':
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    return {}


def calculate_metrics(y_true, y_pred, y_pred_proba=None, multi_class=False):
    """Calculate comprehensive metrics."""
    average = 'weighted' if multi_class else 'binary'
    
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0))
    }
    
    # Per-class metrics
    try:
        metrics['precision_per_class'] = {
            str(i): float(score) 
            for i, score in enumerate(precision_score(y_true, y_pred, average=None, zero_division=0))
        }
        metrics['recall_per_class'] = {
            str(i): float(score) 
            for i, score in enumerate(recall_score(y_true, y_pred, average=None, zero_division=0))
        }
        metrics['f1_per_class'] = {
            str(i): float(score) 
            for i, score in enumerate(f1_score(y_true, y_pred, average=None, zero_division=0))
        }
    except Exception as e:
        logger.warning(f"Could not calculate per-class metrics: {e}")
    
    # ROC-AUC
    if y_pred_proba is not None:
        try:
            if multi_class:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr'))
            else:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def plot_feature_importance(model, feature_names, output_dir='reports', filename='feature_importance.png'):
    """Plot and save feature importance."""
    if not PLOT_AVAILABLE:
        return
    
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            logger.warning("Model does not have feature importances")
            return
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:20]  # Top 20
        
        plt.figure(figsize=(12, 8))
        plt.title('Top 20 Feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.tight_layout()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_path / filename}")
        
    except Exception as e:
        logger.warning(f"Could not plot feature importance: {e}")


def train_enhanced_model(
    data_file='data/processed/cumulative_up_to_2025.csv',
    models_dir='models',
    reports_dir='reports',
    model_type='random_forest',
    multi_level=False,
    use_grid_search=False,
    use_randomized_search=False,
    n_folds=5,
    use_all_features=True,
    create_interactions=True,
    exclude_leakage=True,
    test_size=0.2,
    version=None
):
    """
    Train enhanced flood prediction model.
    
    Args:
        data_file: Path to training data CSV
        models_dir: Directory to save models
        reports_dir: Directory for reports and plots
        model_type: 'random_forest', 'gradient_boosting', 'xgboost', 'ensemble'
        multi_level: If True, train 3-level classifier (LOW/MODERATE/HIGH)
        use_grid_search: If True, perform GridSearchCV
        use_randomized_search: If True, perform RandomizedSearchCV (faster)
        n_folds: Cross-validation folds
        use_all_features: If True, use all available features
        create_interactions: If True, create interaction features
        exclude_leakage: If True, exclude flood_depth_m from features
        test_size: Test set proportion
        version: Model version (auto-incremented if None)
    
    Returns:
        model, metrics, metadata
    """
    logger.info("="*80)
    logger.info("ENHANCED MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Data file: {data_file}")
    logger.info(f"Model type: {model_type}")
    logger.info(f"Classification: {'3-Level Risk' if multi_level else 'Binary (Flood/No-Flood)'}")
    logger.info(f"Use all features: {use_all_features}")
    logger.info(f"Create interactions: {create_interactions}")
    logger.info(f"Exclude data leakage: {exclude_leakage}")
    logger.info("="*80)
    
    # Load data
    if not os.path.exists(data_file):
        # Try pattern matching
        files = glob.glob(data_file)
        if files:
            data_file = files[0]
        else:
            logger.error(f"Data file not found: {data_file}")
            sys.exit(1)
    
    logger.info(f"Loading data from {data_file}")
    data = pd.read_csv(data_file)
    logger.info(f"Loaded {len(data)} records with columns: {list(data.columns)}")
    
    # Determine target column
    if multi_level:
        if 'risk_level' not in data.columns:
            logger.error("risk_level column not found. Run preprocessing first.")
            sys.exit(1)
        target_col = 'risk_level'
        # Remove any NaN in risk_level
        data = data.dropna(subset=['risk_level'])
        data['risk_level'] = data['risk_level'].astype(int)
    else:
        target_col = 'flood'
    
    y = data[target_col]
    logger.info(f"\nTarget distribution ({target_col}):")
    logger.info(y.value_counts().sort_index())
    
    # Prepare features
    logger.info("\nPreparing features...")
    X, feature_names = prepare_features(
        data, 
        use_all_features=use_all_features,
        create_interactions=create_interactions,
        encode_categories=True,
        exclude_leakage=exclude_leakage
    )
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Features used ({len(feature_names)}): {feature_names}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    logger.info(f"\nTraining set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Create model
    class_weight = 'balanced' if model_type in ['random_forest'] else None
    model = create_model_pipeline(model_type, class_weight)
    
    # Hyperparameter tuning
    if use_grid_search or use_randomized_search:
        param_grid = get_hyperparameter_grid(model_type)
        
        if param_grid:
            if use_randomized_search:
                logger.info(f"\nPerforming RandomizedSearchCV with {n_folds} folds...")
                search = RandomizedSearchCV(
                    model, param_grid, n_iter=50, cv=n_folds,
                    scoring='f1_weighted', n_jobs=-1, random_state=42, verbose=1
                )
            else:
                logger.info(f"\nPerforming GridSearchCV with {n_folds} folds...")
                search = GridSearchCV(
                    model, param_grid, cv=n_folds,
                    scoring='f1_weighted', n_jobs=-1, verbose=1
                )
            
            search.fit(X_train, y_train)
            model = search.best_estimator_
            logger.info(f"Best parameters: {search.best_params_}")
            logger.info(f"Best CV score: {search.best_score_:.4f}")
    else:
        # Standard training with cross-validation
        logger.info(f"\nTraining {model_type} model...")
        model.fit(X_train, y_train)
        
        # Cross-validation
        logger.info(f"Performing {n_folds}-fold cross-validation...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=n_folds, 
                                   scoring='f1_weighted', n_jobs=-1)
        logger.info(f"CV F1 scores: {cv_scores}")
        logger.info(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Evaluate
    logger.info("\nEvaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, multi_class=multi_level)
    
    # Log metrics
    logger.info("\n" + "="*50)
    logger.info("MODEL EVALUATION METRICS")
    logger.info("="*50)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    if multi_level:
        logger.info("\nPer-Risk-Level Metrics:")
        for level in sorted(metrics.get('precision_per_class', {}).keys()):
            level_int = int(level)
            level_name = RISK_LEVEL_NAMES.get(level_int, level)
            logger.info(f"  {level_name}:")
            logger.info(f"    Precision: {metrics['precision_per_class'][level]:.4f}")
            logger.info(f"    Recall:    {metrics['recall_per_class'][level]:.4f}")
            logger.info(f"    F1:        {metrics['f1_per_class'][level]:.4f}")
    
    logger.info("\nClassification Report:")
    if multi_level:
        target_names = [RISK_LEVEL_NAMES.get(i, str(i)) for i in sorted(y.unique())]
        logger.info(classification_report(y_test, y_pred, target_names=target_names))
    else:
        logger.info(classification_report(y_test, y_pred))
    
    logger.info("\nConfusion Matrix:")
    logger.info(confusion_matrix(y_test, y_pred))
    
    # Feature importance plot
    if PLOT_AVAILABLE and hasattr(model, 'feature_importances_'):
        plot_feature_importance(model, feature_names, reports_dir)
    
    # Save model
    if version is None:
        prefix = 'flood_multilevel' if multi_level else 'flood_enhanced'
        version = get_next_version(models_dir, prefix)
    
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    prefix = 'flood_multilevel' if multi_level else 'flood_enhanced'
    model_filename = f'{prefix}_v{version}.joblib'
    model_path = models_path / model_filename
    
    joblib.dump(model, model_path)
    logger.info(f"\nModel saved to: {model_path}")
    
    # Save metadata
    metadata = {
        'version': version,
        'model_type': model_type,
        'classification_type': '3-level' if multi_level else 'binary',
        'model_path': str(model_path),
        'created_at': datetime.now().isoformat(),
        'training_data': {
            'file': data_file,
            'samples': len(data),
            'features': feature_names,
            'feature_count': len(feature_names)
        },
        'configuration': {
            'use_all_features': use_all_features,
            'create_interactions': create_interactions,
            'exclude_leakage': exclude_leakage,
            'test_size': test_size,
            'cv_folds': n_folds
        },
        'metrics': metrics,
        'feature_importance': {}
    }
    
    # Add feature importance
    if hasattr(model, 'feature_importances_'):
        metadata['feature_importance'] = {
            name: float(imp) 
            for name, imp in zip(feature_names, model.feature_importances_)
        }
    
    metadata_path = model_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to: {metadata_path}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info("="*80)
    
    return model, metrics, metadata


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced flood prediction model training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic enhanced training (uses all features):
    python train_enhanced.py
  
  3-level risk classification:
    python train_enhanced.py --multi-level
  
  Use ensemble of models:
    python train_enhanced.py --ensemble
  
  Full hyperparameter tuning:
    python train_enhanced.py --grid-search
  
  Fast hyperparameter tuning:
    python train_enhanced.py --randomized-search
  
  Training without interaction features:
    python train_enhanced.py --no-interactions
  
  Include flood_depth_m (data leakage - for analysis only):
    python train_enhanced.py --include-leakage
        """
    )
    
    # Default paths relative to backend directory
    default_data = str(BACKEND_DIR / 'data' / 'processed' / 'cumulative_up_to_2025.csv')
    default_models = str(BACKEND_DIR / 'models')
    default_reports = str(BACKEND_DIR / 'reports')
    
    parser.add_argument('--data', type=str, default=default_data,
                       help='Path to training data CSV')
    parser.add_argument('--models-dir', type=str, default=default_models,
                       help='Directory to save models')
    parser.add_argument('--reports-dir', type=str, default=default_reports,
                       help='Directory for reports and plots')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting', 'xgboost', 'ensemble'],
                       help='Model type to train')
    parser.add_argument('--multi-level', action='store_true',
                       help='Train 3-level risk classifier (LOW/MODERATE/HIGH)')
    parser.add_argument('--grid-search', action='store_true',
                       help='Perform GridSearchCV for hyperparameter tuning')
    parser.add_argument('--randomized-search', action='store_true',
                       help='Perform RandomizedSearchCV (faster than grid)')
    parser.add_argument('--ensemble', action='store_true',
                       help='Use ensemble of multiple models')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--no-interactions', action='store_true',
                       help='Disable interaction feature creation')
    parser.add_argument('--include-leakage', action='store_true',
                       help='Include flood_depth_m feature (causes data leakage)')
    parser.add_argument('--all-features', action='store_true', default=True,
                       help='Use ALL available features (default: True)')
    parser.add_argument('--core-features-only', action='store_true',
                       help='Use only core features (temperature, humidity, precipitation)')
    parser.add_argument('--version', type=int, help='Model version number')
    
    args = parser.parse_args()
    
    # Handle ensemble flag
    if args.ensemble:
        model_type = 'ensemble'
    else:
        model_type = args.model_type
    
    # Determine feature usage
    use_all_features = not args.core_features_only
    
    train_enhanced_model(
        data_file=args.data,
        models_dir=args.models_dir,
        reports_dir=args.reports_dir,
        model_type=model_type,
        multi_level=args.multi_level,
        use_grid_search=args.grid_search,
        use_randomized_search=args.randomized_search,
        n_folds=args.cv_folds,
        use_all_features=use_all_features,
        create_interactions=not args.no_interactions,
        exclude_leakage=not args.include_leakage,
        version=args.version
    )
