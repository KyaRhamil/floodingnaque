import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, make_scorer
)
import joblib
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import glob

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_next_version(models_dir='models'):
    """Get the next version number for model versioning."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return 1
    
    # Find existing model versions
    existing_versions = []
    for file in models_path.glob('flood_rf_model_v*.joblib'):
        try:
            # Extract version number from filename
            version_str = file.stem.split('_v')[-1]
            version = int(version_str)
            existing_versions.append(version)
        except (ValueError, IndexError):
            continue
    
    return max(existing_versions) + 1 if existing_versions else 1


def save_model_metadata(model, metrics, model_path, version, data_info, models_dir='models'):
    """Save model metadata to JSON file."""
    metadata = {
        'version': version,
        'model_type': 'RandomForestClassifier',
        'model_path': str(model_path),
        'created_at': datetime.now().isoformat(),
        'training_data': {
            'file': data_info.get('file', 'data/synthetic_dataset.csv'),
            'shape': data_info.get('shape'),
            'features': data_info.get('features', []),
            'target_distribution': data_info.get('target_distribution', {})
        },
        'model_parameters': {
            'n_estimators': model.n_estimators,
            'random_state': model.random_state,
            'max_depth': model.max_depth,
            'min_samples_split': model.min_samples_split,
            'min_samples_leaf': model.min_samples_leaf
        },
        'metrics': metrics,
        'feature_importance': {
            feature: float(importance) 
            for feature, importance in zip(
                model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [],
                model.feature_importances_
            )
        }
    }
    
    metadata_path = model_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Model metadata saved to {metadata_path}")
    return metadata


def calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba=None):
    """Calculate comprehensive evaluation metrics."""
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    }
    
    # Add per-class metrics
    try:
        metrics['precision_per_class'] = {
            str(i): float(score) 
            for i, score in enumerate(precision_score(y_test, y_pred, average=None, zero_division=0))
        }
        metrics['recall_per_class'] = {
            str(i): float(score) 
            for i, score in enumerate(recall_score(y_test, y_pred, average=None, zero_division=0))
        }
        metrics['f1_per_class'] = {
            str(i): float(score) 
            for i, score in enumerate(f1_score(y_test, y_pred, average=None, zero_division=0))
        }
    except Exception as e:
        logger.warning(f"Could not calculate per-class metrics: {str(e)}")
    
    # ROC-AUC if probabilities available
    if y_pred_proba is not None and len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba[:, 1]))
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {str(e)}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def train_model(version=None, models_dir='models', data_file='data/synthetic_dataset.csv', 
                use_grid_search=False, n_folds=5, merge_datasets=False):
    """
    Train the flood prediction model with versioning and comprehensive metrics.
    
    Args:
        version: Model version number (auto-incremented if None)
        models_dir: Directory to save models
        data_file: Path to training data CSV file (or pattern like 'data/*.csv' if merge_datasets=True)
        use_grid_search: If True, perform hyperparameter tuning with GridSearchCV
        n_folds: Number of cross-validation folds
        merge_datasets: If True, merge multiple CSV files matching the data_file pattern
    
    Returns:
        tuple: (model, metrics, metadata)
    """
    try:
        # Load data (single file or merge multiple files)
        if merge_datasets:
            logger.info(f"Merging datasets matching pattern: {data_file}")
            csv_files = glob.glob(data_file)
            if not csv_files:
                logger.error(f"No CSV files found matching pattern: {data_file}")
                sys.exit(1)
            logger.info(f"Found {len(csv_files)} CSV files to merge: {csv_files}")
            data_frames = []
            for file in csv_files:
                df = pd.read_csv(file)
                logger.info(f"  Loaded {file}: {df.shape[0]} rows")
                data_frames.append(df)
            data = pd.concat(data_frames, ignore_index=True)
            logger.info(f"Merged dataset shape: {data.shape}")
            # Update data_file reference for metadata
            data_file = f"merged_{len(csv_files)}_files"
        else:
            # Check if data file exists
            if not os.path.exists(data_file):
                logger.error(f"Data file not found: {data_file}")
                sys.exit(1)
            
            logger.info(f"Loading data from {data_file}")
            data = pd.read_csv(data_file)
        
        # Validate data
        if data.empty:
            logger.error("Dataset is empty")
            sys.exit(1)
        
        # Check for required columns
        required_cols = ['temperature', 'humidity', 'precipitation', 'flood']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {', '.join(missing_cols)}")
            sys.exit(1)
        
        # Prepare features and target
        X = data.drop('flood', axis=1)
        y = data['flood']
        
        # Store data info for metadata
        data_info = {
            'file': data_file,
            'shape': list(data.shape),
            'features': list(X.columns),
            'target_distribution': y.value_counts().to_dict()
        }
        
        logger.info(f"Dataset shape: {data.shape}")
        logger.info(f"Features: {list(X.columns)}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Train model with optional hyperparameter tuning
        if use_grid_search:
            logger.info("Performing hyperparameter tuning with GridSearchCV...")
            logger.info("This may take several minutes...")
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            # Create base model
            rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                estimator=rf_base,
                param_grid=param_grid,
                cv=n_folds,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=2,
                return_train_score=True
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            model = grid_search.best_estimator_
            
            logger.info(f"\nBest parameters found: {grid_search.best_params_}")
            logger.info(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
            
            # Store grid search results in data_info for metadata
            data_info['grid_search'] = {
                'best_params': grid_search.best_params_,
                'best_cv_score': float(grid_search.best_score_),
                'cv_folds': n_folds
            }
        else:
            logger.info("Training Random Forest model with default parameters...")
            model = RandomForestClassifier(
                n_estimators=200,  # Increased from 100
                max_depth=20,      # Added depth limit
                min_samples_split=5,  # Added to prevent overfitting
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
            model.fit(X_train, y_train)
            
            # Perform cross-validation for robustness
            logger.info(f"Performing {n_folds}-fold cross-validation...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=n_folds, 
                                       scoring='f1_weighted', n_jobs=-1)
            logger.info(f"Cross-validation F1 scores: {cv_scores}")
            logger.info(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            data_info['cross_validation'] = {
                'cv_folds': n_folds,
                'cv_scores': cv_scores.tolist(),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std())
            }
        
        # Evaluate with comprehensive metrics
        logger.info("Evaluating model...")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
        
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
        
        logger.info("\nPer-class Metrics:")
        if 'precision_per_class' in metrics:
            for class_id in metrics['precision_per_class']:
                logger.info(f"  Class {class_id}:")
                logger.info(f"    Precision: {metrics['precision_per_class'][class_id]:.4f}")
                logger.info(f"    Recall:    {metrics['recall_per_class'][class_id]:.4f}")
                logger.info(f"    F1:        {metrics['f1_per_class'][class_id]:.4f}")
        
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        logger.info("\nConfusion Matrix:")
        logger.info(confusion_matrix(y_test, y_pred))
        logger.info("="*50)
        
        # Determine version
        if version is None:
            version = get_next_version(models_dir)
        
        # Create models directory
        models_path = Path(models_dir)
        models_path.mkdir(exist_ok=True)
        
        # Save model with versioning
        model_filename = f'flood_rf_model_v{version}.joblib'
        model_path = models_path / model_filename
        
        # Also save as latest (for backward compatibility)
        latest_path = models_path / 'flood_rf_model.joblib'
        
        joblib.dump(model, model_path)
        joblib.dump(model, latest_path)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Latest model saved to {latest_path}")
        
        # Save metadata
        metadata = save_model_metadata(
            model, metrics, model_path, version, data_info, models_dir
        )
        
        # Update latest model link metadata
        latest_metadata_path = latest_path.with_suffix('.json')
        latest_metadata = metadata.copy()
        latest_metadata['is_latest'] = True
        latest_metadata['versioned_model'] = str(model_path)
        with open(latest_metadata_path, 'w') as f:
            json.dump(latest_metadata, f, indent=2)
        
        logger.info(f"\n✓ Model training completed successfully!")
        logger.info(f"  Version: {version}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Model file: {model_path}")
        logger.info(f"  Metadata: {model_path.with_suffix('.json')}")
        
        return model, metrics, metadata
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train flood prediction model with Random Forest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Basic training:
    python train.py
  
  Train with new dataset:
    python train.py --data data/flood_data_2025.csv
  
  Hyperparameter tuning (recommended for thesis):
    python train.py --grid-search --cv-folds 10
  
  Merge multiple datasets:
    python train.py --data "data/*.csv" --merge-datasets
  
  Full optimization with merged data:
    python train.py --data "data/*.csv" --merge-datasets --grid-search
        """
    )
    parser.add_argument('--version', type=int, help='Model version number (auto-incremented if not provided)')
    parser.add_argument('--data', type=str, default='data/synthetic_dataset.csv', 
                       help='Path to training data CSV (use quotes for patterns like "data/*.csv")')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--grid-search', action='store_true', 
                       help='Perform hyperparameter tuning with GridSearchCV (slow but optimal)')
    parser.add_argument('--cv-folds', type=int, default=5, 
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--merge-datasets', action='store_true',
                       help='Merge multiple CSV files matching the data pattern')
    
    args = parser.parse_args()
    
    train_model(
        version=args.version, 
        models_dir=args.models_dir, 
        data_file=args.data,
        use_grid_search=args.grid_search,
        n_folds=args.cv_folds,
        merge_datasets=args.merge_datasets
    )
