import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, make_scorer
)
from sklearn.feature_selection import SelectFromModel
import joblib
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import glob
import warnings

# Optional imports with graceful fallback
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


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


def detect_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Detect outliers using IQR method.
    
    Args:
        df: DataFrame with data
        columns: List of columns to check for outliers
        method: 'iqr' or 'zscore'
        threshold: IQR multiplier (1.5 for moderate, 3 for extreme) or z-score threshold
    
    Returns:
        Boolean mask where True indicates an outlier
    """
    outlier_mask = pd.Series([False] * len(df))
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = outlier_mask | ((df[col] < lower_bound) | (df[col] > upper_bound))
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_mask = outlier_mask | (z_scores > threshold)
    
    return outlier_mask


def generate_learning_curves(model, X, y, cv=5, output_dir='reports'):
    """
    Generate and save learning curves to visualize model performance.
    
    Args:
        model: Trained model or base estimator
        X: Features
        y: Target
        cv: Number of cross-validation folds
        output_dir: Directory to save the plot
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available. Skipping learning curves.")
        return None
    
    logger.info("Generating learning curves...")
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y, 
        train_sizes=train_sizes,
        cv=cv, 
        scoring='f1_weighted',
        n_jobs=-1,
        random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.plot(train_sizes_abs, val_mean, 'o-', color='green', label='Validation Score')
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='green')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('F1 Score (Weighted)')
    plt.title('Learning Curves - Random Forest Flood Prediction')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    plot_path = output_path / 'learning_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Learning curves saved to {plot_path}")
    
    return {
        'train_sizes': train_sizes_abs.tolist(),
        'train_mean': train_mean.tolist(),
        'train_std': train_std.tolist(),
        'val_mean': val_mean.tolist(),
        'val_std': val_std.tolist()
    }


def generate_shap_analysis(model, X, feature_names, output_dir='reports', max_samples=100):
    """
    Generate SHAP values for model explainability.
    
    Args:
        model: Trained Random Forest model
        X: Feature data (DataFrame or array)
        feature_names: List of feature names
        output_dir: Directory to save plots
        max_samples: Maximum samples to use for SHAP (for performance)
    
    Returns:
        Dictionary with SHAP summary
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not installed. Run: pip install shap")
        return None
    
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available. Skipping SHAP plots.")
        return None
    
    logger.info("Generating SHAP analysis for model explainability...")
    
    # Sample data if too large
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
    else:
        X_sample = X
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, shap_values is a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values_flood = shap_values[1]  # Class 1 = Flood
    else:
        shap_values_flood = shap_values
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Summary plot (bar)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_flood, X_sample, feature_names=feature_names, 
                      plot_type="bar", show=False)
    plt.title('SHAP Feature Importance (Flood Prediction)')
    plt.tight_layout()
    plt.savefig(output_path / 'shap_importance_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary plot (beeswarm)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_flood, X_sample, feature_names=feature_names, show=False)
    plt.title('SHAP Summary (Impact on Flood Prediction)')
    plt.tight_layout()
    plt.savefig(output_path / 'shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"SHAP analysis saved to {output_path}")
    
    # Calculate mean absolute SHAP values
    mean_shap = np.abs(shap_values_flood).mean(axis=0)
    shap_importance = {name: float(val) for name, val in zip(feature_names, mean_shap)}
    
    return {
        'mean_absolute_shap': shap_importance,
        'samples_analyzed': len(X_sample)
    }


def select_important_features(model, X, y, threshold='median'):
    """
    Select important features based on model feature importance.
    
    Args:
        model: Trained model with feature_importances_
        X: Features DataFrame
        y: Target
        threshold: 'median', 'mean', or float value
    
    Returns:
        Tuple of (selected_features, selector)
    """
    logger.info(f"Performing feature selection with threshold='{threshold}'...")
    
    selector = SelectFromModel(model, threshold=threshold, prefit=True)
    X_selected = selector.transform(X)
    
    # Get selected feature names
    if hasattr(X, 'columns'):
        feature_mask = selector.get_support()
        selected_features = X.columns[feature_mask].tolist()
    else:
        selected_features = [f"feature_{i}" for i in range(X_selected.shape[1])]
    
    logger.info(f"Selected {len(selected_features)}/{X.shape[1]} features: {selected_features}")
    
    return selected_features, selector


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
                use_grid_search=False, n_folds=5, merge_datasets=False,
                class_weight='balanced', remove_outliers=False, outlier_threshold=1.5,
                feature_selection=False, selection_threshold='median',
                generate_learning_curve=False, generate_shap=False,
                reports_dir='reports'):
    """
    Train the flood prediction model with versioning and comprehensive metrics.
    
    Args:
        version: Model version number (auto-incremented if None)
        models_dir: Directory to save models
        data_file: Path to training data CSV file (or pattern like 'data/*.csv' if merge_datasets=True)
        use_grid_search: If True, perform hyperparameter tuning with GridSearchCV
        n_folds: Number of cross-validation folds
        merge_datasets: If True, merge multiple CSV files matching the data_file pattern
        class_weight: 'balanced' to handle imbalanced data, None for equal weights
        remove_outliers: If True, remove outliers before training
        outlier_threshold: IQR threshold for outlier detection (default: 1.5)
        feature_selection: If True, perform feature selection after initial training
        selection_threshold: Threshold for feature selection ('median', 'mean', or float)
        generate_learning_curve: If True, generate learning curves plot
        generate_shap: If True, generate SHAP explainability analysis
        reports_dir: Directory to save reports and plots
    
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
        
        # Remove outliers if requested
        if remove_outliers:
            numeric_cols = [col for col in data.columns if col != 'flood' and data[col].dtype in ['float64', 'int64']]
            outlier_mask = detect_outliers(data, numeric_cols, threshold=outlier_threshold)
            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                logger.info(f"Removing {n_outliers} outliers ({n_outliers/len(data)*100:.1f}% of data)")
                data = data[~outlier_mask].reset_index(drop=True)
                logger.info(f"Data shape after outlier removal: {data.shape}")
        
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
        logger.info(f"Using class_weight='{class_weight}' for handling class imbalance")
        
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
            
            # Create base model with class_weight
            rf_base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight=class_weight)
            
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
            logger.info("Training Random Forest model with enhanced parameters...")
            model = RandomForestClassifier(
                n_estimators=200,  # Increased from 100
                max_depth=20,      # Added depth limit
                min_samples_split=5,  # Added to prevent overfitting
                random_state=42,
                n_jobs=-1,
                class_weight=class_weight,  # Handle class imbalance
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
        
        # Generate learning curves if requested
        learning_curve_data = None
        if generate_learning_curve:
            try:
                learning_curve_data = generate_learning_curves(
                    RandomForestClassifier(
                        n_estimators=100, max_depth=20, 
                        class_weight=class_weight, random_state=42, n_jobs=-1
                    ),
                    X, y, cv=min(n_folds, 5), output_dir=reports_dir
                )
                if learning_curve_data:
                    data_info['learning_curves'] = learning_curve_data
            except Exception as e:
                logger.warning(f"Failed to generate learning curves: {e}")
        
        # Generate SHAP analysis if requested
        shap_data = None
        if generate_shap:
            try:
                shap_data = generate_shap_analysis(
                    model, X_test, list(X.columns), 
                    output_dir=reports_dir, max_samples=min(100, len(X_test))
                )
                if shap_data:
                    data_info['shap_analysis'] = shap_data
            except Exception as e:
                logger.warning(f"Failed to generate SHAP analysis: {e}")
        
        # Feature selection (informational - doesn't retrain)
        if feature_selection:
            try:
                selected_features, _ = select_important_features(
                    model, X, y, threshold=selection_threshold
                )
                data_info['feature_selection'] = {
                    'threshold': str(selection_threshold),
                    'selected_features': selected_features,
                    'dropped_features': [f for f in X.columns if f not in selected_features]
                }
            except Exception as e:
                logger.warning(f"Failed to perform feature selection: {e}")
        
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
  
  Full optimization with all enhancements:
    python train.py --data "data/*.csv" --merge-datasets --grid-search --learning-curves --shap
  
  Remove outliers and handle class imbalance:
    python train.py --remove-outliers --class-weight balanced
        """
    )
    parser.add_argument('--version', type=int, help='Model version number (auto-incremented if not provided)')
    parser.add_argument('--data', type=str, default='data/synthetic_dataset.csv', 
                       help='Path to training data CSV (use quotes for patterns like "data/*.csv")')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--reports-dir', type=str, default='reports', help='Directory to save reports and plots')
    parser.add_argument('--grid-search', action='store_true', 
                       help='Perform hyperparameter tuning with GridSearchCV (slow but optimal)')
    parser.add_argument('--cv-folds', type=int, default=5, 
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--merge-datasets', action='store_true',
                       help='Merge multiple CSV files matching the data pattern')
    
    # New enhancement options
    parser.add_argument('--class-weight', type=str, default='balanced', choices=['balanced', 'none'],
                       help='Class weight strategy: balanced (recommended) or none')
    parser.add_argument('--remove-outliers', action='store_true',
                       help='Remove outliers from training data using IQR method')
    parser.add_argument('--outlier-threshold', type=float, default=1.5,
                       help='IQR threshold for outlier detection (default: 1.5)')
    parser.add_argument('--feature-selection', action='store_true',
                       help='Perform feature selection analysis')
    parser.add_argument('--selection-threshold', type=str, default='median',
                       help='Feature selection threshold: median, mean, or float value')
    parser.add_argument('--learning-curves', action='store_true',
                       help='Generate learning curves plot')
    parser.add_argument('--shap', action='store_true',
                       help='Generate SHAP explainability analysis (requires shap package)')
    
    args = parser.parse_args()
    
    # Convert class_weight argument
    cw = None if args.class_weight == 'none' else args.class_weight
    
    # Convert selection_threshold to float if numeric
    sel_threshold = args.selection_threshold
    try:
        sel_threshold = float(sel_threshold)
    except ValueError:
        pass  # Keep as string ('median' or 'mean')
    
    train_model(
        version=args.version, 
        models_dir=args.models_dir, 
        data_file=args.data,
        use_grid_search=args.grid_search,
        n_folds=args.cv_folds,
        merge_datasets=args.merge_datasets,
        class_weight=cw,
        remove_outliers=args.remove_outliers,
        outlier_threshold=args.outlier_threshold,
        feature_selection=args.feature_selection,
        selection_threshold=sel_threshold,
        generate_learning_curve=args.learning_curves,
        generate_shap=args.shap,
        reports_dir=args.reports_dir
    )
