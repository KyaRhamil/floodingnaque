"""
Model Comparison Tool - Compare performance across multiple model versions.
Useful for demonstrating model improvement over time in thesis presentations.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_metadata(model_path):
    """Load metadata for a model version."""
    metadata_path = Path(model_path).with_suffix('.json')
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None


def get_all_model_versions(models_dir='models'):
    """Get all available model versions with their metadata."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    
    versions = []
    for model_file in sorted(models_path.glob('flood_rf_model_v*.joblib')):
        try:
            version_str = model_file.stem.split('_v')[-1]
            version = int(version_str)
            metadata = load_model_metadata(str(model_file))
            
            versions.append({
                'version': version,
                'path': str(model_file),
                'metadata': metadata
            })
        except (ValueError, IndexError):
            continue
    
    return sorted(versions, key=lambda x: x['version'])


def compare_models(models_dir='models', output_dir='reports'):
    """
    Compare all model versions and generate comparison charts.
    
    Args:
        models_dir: Directory containing model files
        output_dir: Directory to save comparison charts
    """
    logger.info("="*80)
    logger.info("MODEL VERSION COMPARISON")
    logger.info("="*80)
    
    # Get all versions
    versions = get_all_model_versions(models_dir)
    
    if not versions:
        logger.error("No model versions found!")
        return
    
    if len(versions) < 2:
        logger.warning("Only one model version found. Need at least 2 for comparison.")
        return
    
    logger.info(f"\nFound {len(versions)} model versions to compare\n")
    
    # Extract data for comparison
    comparison_data = []
    for v in versions:
        metadata = v.get('metadata', {})
        metrics = metadata.get('metrics', {})
        
        data = {
            'version': v['version'],
            'created_at': metadata.get('created_at', 'N/A'),
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0)
        }
        
        # Extract model parameters
        params = metadata.get('model_parameters', {})
        data['n_estimators'] = params.get('n_estimators', 0)
        data['max_depth'] = params.get('max_depth', 'None')
        
        # Extract training data info
        training_data = metadata.get('training_data', {})
        data['dataset_size'] = training_data.get('shape', [0])[0] if training_data.get('shape') else 0
        
        comparison_data.append(data)
    
    df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    logger.info("COMPARISON TABLE")
    logger.info("-"*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    logger.info("\n" + df.to_string(index=False))
    logger.info("")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate comparison plots
    logger.info("\nGenerating comparison visualizations...")
    
    # 1. Metrics Evolution Plot
    plot_metrics_evolution(df, output_path)
    
    # 2. Metrics Comparison Bars
    plot_metrics_bars(df, output_path)
    
    # 3. Parameters Evolution
    plot_parameters_evolution(df, output_path)
    
    # Save comparison table to CSV
    csv_path = output_path / 'model_comparison.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Comparison table saved to: {csv_path}")
    
    # Generate text report
    generate_comparison_report(df, versions, output_path)
    
    logger.info("\n" + "="*80)
    logger.info("✓ MODEL COMPARISON COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nGenerated files in {output_path.absolute()}:")
    logger.info("  - metrics_evolution.png")
    logger.info("  - metrics_comparison.png")
    logger.info("  - parameters_evolution.png")
    logger.info("  - model_comparison.csv")
    logger.info("  - comparison_report.txt")


def plot_metrics_evolution(df, output_dir):
    """Plot how metrics evolved across versions."""
    logger.info("  Creating metrics evolution chart...")
    
    plt.figure(figsize=(12, 6))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    for metric, color in zip(metrics, colors):
        plt.plot(df['version'], df[metric], marker='o', linewidth=2, 
                label=metric.replace('_', ' ').title(), color=color)
    
    plt.xlabel('Model Version', fontsize=12, fontweight='bold')
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.title('Model Performance Evolution Across Versions\nFlood Prediction System', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.1])
    
    # Annotate latest version
    latest_idx = df.index[-1]
    for metric in metrics:
        plt.annotate(f'{df[metric].iloc[latest_idx]:.3f}',
                    xy=(df['version'].iloc[latest_idx], df[metric].iloc[latest_idx]),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    output_path = output_dir / 'metrics_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_bars(df, output_dir):
    """Plot metrics comparison as grouped bar chart."""
    logger.info("  Creating metrics comparison bar chart...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    x = range(len(df))
    width = 0.2
    
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = width * (i - 1.5)
        ax.bar([p + offset for p in x], df[metric], width, 
               label=metric.replace('_', ' ').title(), color=color, alpha=0.8)
    
    ax.set_xlabel('Model Version', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Metrics Comparison Across Versions\nFlood Prediction System', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'v{v}' for v in df['version']])
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    output_path = output_dir / 'metrics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_parameters_evolution(df, output_dir):
    """Plot how model parameters evolved."""
    logger.info("  Creating parameters evolution chart...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot n_estimators
    ax1.plot(df['version'], df['n_estimators'], marker='s', linewidth=2, 
            color='#9b59b6', markersize=8)
    ax1.set_xlabel('Model Version', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Trees (n_estimators)', fontsize=12, fontweight='bold')
    ax1.set_title('n_estimators Evolution', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Annotate values
    for idx, row in df.iterrows():
        ax1.annotate(f'{int(row["n_estimators"])}',
                    xy=(row['version'], row['n_estimators']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9)
    
    # Plot dataset size
    ax2.bar(df['version'], df['dataset_size'], color='#16a085', alpha=0.8)
    ax2.set_xlabel('Model Version', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Dataset Size', fontsize=12, fontweight='bold')
    ax2.set_title('Training Data Size Evolution', fontsize=12, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Annotate values
    for idx, row in df.iterrows():
        ax2.text(row['version'], row['dataset_size'], f'{int(row["dataset_size"])}',
                ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Model Configuration Evolution\nFlood Prediction System', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / 'parameters_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_comparison_report(df, versions, output_dir):
    """Generate detailed text comparison report."""
    logger.info("  Creating comparison text report...")
    
    output_path = output_dir / 'comparison_report.txt'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" MODEL VERSION COMPARISON REPORT\n")
        f.write(" Random Forest Flood Prediction System\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Versions Compared: {len(versions)}\n")
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary statistics
        f.write("PERFORMANCE IMPROVEMENT SUMMARY\n")
        f.write("-"*80 + "\n")
        
        first_version = df.iloc[0]
        latest_version = df.iloc[-1]
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            first_val = first_version[metric]
            latest_val = latest_version[metric]
            improvement = ((latest_val - first_val) / first_val * 100) if first_val > 0 else 0
            
            f.write(f"\n{metric.replace('_', ' ').title()}:\n")
            f.write(f"  v{int(first_version['version'])}: {first_val:.4f}\n")
            f.write(f"  v{int(latest_version['version'])}: {latest_val:.4f}\n")
            f.write(f"  Improvement: {improvement:+.2f}%\n")
        
        # Best performing version
        f.write("\nBEST PERFORMING VERSION\n")
        f.write("-"*80 + "\n")
        best_f1_idx = df['f1_score'].idxmax()
        best_version = df.iloc[best_f1_idx]
        f.write(f"Version: v{int(best_version['version'])}\n")
        f.write(f"F1 Score: {best_version['f1_score']:.4f}\n")
        f.write(f"Accuracy: {best_version['accuracy']:.4f}\n")
        f.write(f"Precision: {best_version['precision']:.4f}\n")
        f.write(f"Recall: {best_version['recall']:.4f}\n")
        
        # Detailed version breakdown
        f.write("\nDETAILED VERSION BREAKDOWN\n")
        f.write("-"*80 + "\n")
        
        for idx, row in df.iterrows():
            version_info = versions[idx]
            metadata = version_info.get('metadata', {})
            
            f.write(f"\n{'='*40}\n")
            f.write(f"VERSION {int(row['version'])}\n")
            f.write(f"{'='*40}\n")
            f.write(f"Created: {row['created_at']}\n")
            f.write(f"Dataset Size: {int(row['dataset_size'])} samples\n")
            f.write(f"n_estimators: {int(row['n_estimators'])}\n")
            f.write(f"max_depth: {row['max_depth']}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy:  {row['accuracy']:.4f}\n")
            f.write(f"  Precision: {row['precision']:.4f}\n")
            f.write(f"  Recall:    {row['recall']:.4f}\n")
            f.write(f"  F1 Score:  {row['f1_score']:.4f}\n")
            
            # Cross-validation info if available
            if 'cross_validation' in metadata:
                cv = metadata['cross_validation']
                f.write(f"\nCross-Validation:\n")
                f.write(f"  Folds: {cv.get('cv_folds')}\n")
                f.write(f"  Mean F1: {cv.get('cv_mean', 0):.4f} (+/- {cv.get('cv_std', 0):.4f})\n")
            
            # Grid search info if available
            if 'grid_search' in metadata:
                gs = metadata['grid_search']
                f.write(f"\nHyperparameter Tuning:\n")
                f.write(f"  Best CV Score: {gs.get('best_cv_score', 0):.4f}\n")
                f.write(f"  Optimized: Yes\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF COMPARISON REPORT\n")
        f.write("="*80 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare multiple model versions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool generates comparison charts and reports for all your model versions.
Perfect for showing model improvement over time in your thesis presentation!

Example:
  python compare_models.py
  python compare_models.py --models-dir models --output comparison_results
        """
    )
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory containing model files')
    parser.add_argument('--output', type=str, default='reports',
                       help='Output directory for comparison reports')
    
    args = parser.parse_args()
    
    compare_models(models_dir=args.models_dir, output_dir=args.output)
