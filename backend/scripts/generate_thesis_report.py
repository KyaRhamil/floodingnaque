"""
Comprehensive Model Analysis and Visualization for Thesis Defense
Generates publication-ready charts, metrics, and reports for Random Forest model.
"""

import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import learning_curve
from pathlib import Path
import logging
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_model_and_metadata(model_path):
    """Load model and its metadata."""
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    
    metadata_path = model_path.with_suffix('.json')
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return model, metadata


def plot_feature_importance(model, output_dir, top_n=20):
    """Plot feature importance with horizontal bar chart."""
    logger.info("Generating feature importance plot...")
    
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return
    
    feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else \
                    [f'Feature {i}' for i in range(len(model.feature_importances_))]
    
    # Create DataFrame for easier manipulation
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Take top N features
    importance_df = importance_df.head(top_n)
    
    # Create plot
    plt.figure(figsize=(10, max(6, len(importance_df) * 0.3)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    plt.barh(importance_df['feature'], importance_df['importance'], color=colors)
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title('Random Forest Feature Importance\nFlood Prediction Model', 
              fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        plt.text(row['importance'], i, f' {row["importance"]:.3f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'feature_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feature importance plot to {output_path}")


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot confusion matrix with annotations."""
    logger.info("Generating confusion matrix plot...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': 'Count'},
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix\nFlood Prediction Model', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add labels
    plt.gca().set_xticklabels(['No Flood (0)', 'Flood (1)'], fontsize=11)
    plt.gca().set_yticklabels(['No Flood (0)', 'Flood (1)'], fontsize=11)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def plot_roc_curve(y_true, y_pred_proba, output_dir):
    """Plot ROC curve with AUC score."""
    logger.info("Generating ROC curve plot...")
    
    if y_pred_proba.ndim > 1:
        y_pred_proba = y_pred_proba[:, 1]  # Use probability of positive class
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve\nFlood Prediction Model', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'roc_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC curve to {output_path}")


def plot_precision_recall_curve(y_true, y_pred_proba, output_dir):
    """Plot Precision-Recall curve."""
    logger.info("Generating Precision-Recall curve plot...")
    
    if y_pred_proba.ndim > 1:
        y_pred_proba = y_pred_proba[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curve\nFlood Prediction Model', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'precision_recall_curve.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Precision-Recall curve to {output_path}")


def plot_learning_curves(model, X, y, output_dir):
    """Plot learning curves to show model performance vs training size."""
    logger.info("Generating learning curves (this may take a while)...")
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1_weighted',
        random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-validation score')
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                     alpha=0.1, color='g')
    
    plt.xlabel('Training Set Size', fontsize=12, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=12, fontweight='bold')
    plt.title('Learning Curves\nRandom Forest Flood Prediction Model', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'learning_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved learning curves to {output_path}")


def plot_model_comparison(metadata, output_dir):
    """Plot metrics comparison if metadata contains multiple model info."""
    logger.info("Generating model metrics comparison plot...")
    
    metrics = metadata.get('metrics', {})
    
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_values = [metrics.get(m, 0) for m in metric_names]
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax.bar(metric_labels, metric_values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Metrics\nRandom Forest Flood Prediction', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved metrics comparison to {output_path}")


def generate_text_report(model, metadata, y_true, y_pred, y_pred_proba, output_dir):
    """Generate a comprehensive text report."""
    logger.info("Generating text report...")
    
    output_path = Path(output_dir) / 'model_report.txt'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" RANDOM FOREST FLOOD PREDICTION MODEL - COMPREHENSIVE REPORT\n")
        f.write(" Parañaque City Flood Detection System\n")
        f.write("="*80 + "\n\n")
        
        # Model Information
        f.write("MODEL INFORMATION\n")
        f.write("-"*80 + "\n")
        if metadata:
            f.write(f"Version: {metadata.get('version', 'N/A')}\n")
            f.write(f"Model Type: {metadata.get('model_type', 'N/A')}\n")
            f.write(f"Created: {metadata.get('created_at', 'N/A')}\n")
            f.write(f"Model Path: {metadata.get('model_path', 'N/A')}\n\n")
            
            # Training Data Info
            training_data = metadata.get('training_data', {})
            f.write("TRAINING DATA\n")
            f.write("-"*80 + "\n")
            f.write(f"Data Source: {training_data.get('file', 'N/A')}\n")
            f.write(f"Dataset Shape: {training_data.get('shape', 'N/A')}\n")
            f.write(f"Features: {', '.join(training_data.get('features', []))}\n")
            f.write(f"Target Distribution: {training_data.get('target_distribution', {})}\n\n")
            
            # Model Parameters
            params = metadata.get('model_parameters', {})
            f.write("MODEL PARAMETERS\n")
            f.write("-"*80 + "\n")
            for param, value in params.items():
                f.write(f"{param}: {value}\n")
            f.write("\n")
        
        # Performance Metrics
        f.write("PERFORMANCE METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}\n")
        f.write(f"Recall:    {recall_score(y_true, y_pred, average='weighted'):.4f}\n")
        f.write(f"F1 Score:  {f1_score(y_true, y_pred, average='weighted'):.4f}\n\n")
        
        # Classification Report
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(y_true, y_pred, 
                                     target_names=['No Flood', 'Flood']))
        f.write("\n")
        
        # Confusion Matrix
        f.write("CONFUSION MATRIX\n")
        f.write("-"*80 + "\n")
        cm = confusion_matrix(y_true, y_pred)
        f.write(f"                Predicted No Flood    Predicted Flood\n")
        f.write(f"Actual No Flood        {cm[0][0]:6d}              {cm[0][1]:6d}\n")
        f.write(f"Actual Flood           {cm[1][0]:6d}              {cm[1][1]:6d}\n\n")
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            f.write("FEATURE IMPORTANCE (Top 10)\n")
            f.write("-"*80 + "\n")
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else \
                           [f'Feature {i}' for i in range(len(model.feature_importances_))]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            for idx, row in importance_df.iterrows():
                f.write(f"{row['feature']:30s}: {row['importance']:.4f}\n")
            f.write("\n")
        
        # Cross-Validation Results
        if metadata and 'cross_validation' in metadata:
            cv = metadata['cross_validation']
            f.write("CROSS-VALIDATION RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(f"Folds: {cv.get('cv_folds', 'N/A')}\n")
            f.write(f"Mean F1 Score: {cv.get('cv_mean', 0):.4f} (+/- {cv.get('cv_std', 0):.4f})\n")
            f.write(f"CV Scores: {cv.get('cv_scores', [])}\n\n")
        
        # Grid Search Results
        if metadata and 'grid_search' in metadata:
            gs = metadata['grid_search']
            f.write("HYPERPARAMETER TUNING RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(f"Best CV Score: {gs.get('best_cv_score', 'N/A'):.4f}\n")
            f.write("Best Parameters:\n")
            for param, value in gs.get('best_params', {}).items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("Report generated on: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
        f.write("="*80 + "\n")
    
    logger.info(f"Saved text report to {output_path}")


def generate_thesis_report(model_path='models/flood_rf_model.joblib', 
                          data_file='data/synthetic_dataset.csv',
                          output_dir='reports'):
    """
    Generate comprehensive thesis report with all visualizations and metrics.
    
    Args:
        model_path: Path to the trained model
        data_file: Path to the test dataset
        output_dir: Directory to save reports and plots
    """
    logger.info("="*80)
    logger.info("GENERATING COMPREHENSIVE THESIS REPORT")
    logger.info("="*80)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_path.absolute()}")
    
    # Load model and metadata
    logger.info(f"Loading model from {model_path}...")
    model, metadata = load_model_and_metadata(model_path)
    
    # Load test data
    logger.info(f"Loading test data from {data_file}...")
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        sys.exit(1)
    
    data = pd.read_csv(data_file)
    X = data.drop('flood', axis=1)
    y = data['flood']
    
    # Make predictions
    logger.info("Generating predictions...")
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)
    
    # Generate all plots
    logger.info("\nGenerating visualizations...")
    plot_feature_importance(model, output_path)
    plot_confusion_matrix(y, y_pred, output_path)
    plot_roc_curve(y, y_pred_proba, output_path)
    plot_precision_recall_curve(y, y_pred_proba, output_path)
    plot_model_comparison(metadata if metadata else {}, output_path)
    plot_learning_curves(model, X, y, output_path)
    
    # Generate text report
    generate_text_report(model, metadata, y, y_pred, y_pred_proba, output_path)
    
    logger.info("\n" + "="*80)
    logger.info("✓ THESIS REPORT GENERATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nGenerated files in {output_path.absolute()}:")
    logger.info("  - feature_importance.png")
    logger.info("  - confusion_matrix.png")
    logger.info("  - roc_curve.png")
    logger.info("  - precision_recall_curve.png")
    logger.info("  - metrics_comparison.png")
    logger.info("  - learning_curves.png")
    logger.info("  - model_report.txt")
    logger.info("\nThese files are publication-ready for your thesis defense!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate comprehensive thesis report for Random Forest model'
    )
    parser.add_argument('--model', type=str, default='models/flood_rf_model.joblib',
                       help='Path to trained model file')
    parser.add_argument('--data', type=str, default='data/synthetic_dataset.csv',
                       help='Path to test data CSV file')
    parser.add_argument('--output', type=str, default='reports',
                       help='Output directory for reports and plots')
    
    args = parser.parse_args()
    
    generate_thesis_report(
        model_path=args.model,
        data_file=args.data,
        output_dir=args.output
    )
