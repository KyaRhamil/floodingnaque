"""
Progressive Model Training - Train models incrementally with cumulative data.

This script implements a progressive training strategy where each model
is trained on increasingly larger datasets:
  - Model v1: 2022 data only
  - Model v2: 2022 + 2023 data  
  - Model v3: 2022 + 2023 + 2024 data
  - Model v4: 2022 + 2023 + 2024 + 2025 data (complete)

Perfect for thesis defense - shows clear model evolution and improvement!
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
import sys

# Import from existing train.py
sys.path.insert(0, str(Path(__file__).parent))
from train import train_model, calculate_comprehensive_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_processed_data(year, data_dir='data/processed'):
    """Load preprocessed flood records for a specific year."""
    file_path = Path(data_dir) / f'processed_flood_records_{year}.csv'
    
    if not file_path.exists():
        logger.error(f"Processed data not found: {file_path}")
        logger.info("Please run preprocess_official_flood_records.py first!")
        return None
    
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {year} data: {len(df)} records")
    return df


def progressive_train(
    years=[2022, 2023, 2024, 2025],
    data_dir='data/processed',
    models_dir='models',
    use_grid_search=False,
    cv_folds=5
):
    """
    Train models progressively with cumulative data.
    
    Args:
        years: List of years to include (in order)
        data_dir: Directory containing processed CSV files
        models_dir: Directory to save models
        use_grid_search: If True, perform hyperparameter tuning
        cv_folds: Number of cross-validation folds
    
    Returns:
        dict: Training results for each model version
    """
    logger.info("="*80)
    logger.info("PROGRESSIVE MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Training strategy: Cumulative (each model learns from more data)")
    logger.info(f"Years: {years}")
    logger.info(f"Grid search: {use_grid_search}")
    logger.info(f"CV folds: {cv_folds}")
    logger.info("="*80)
    
    results = {}
    cumulative_data = None
    
    for i, year in enumerate(years, start=1):
        logger.info(f"\n{'#'*80}")
        logger.info(f"# TRAINING MODEL v{i} (Data up to {year})")
        logger.info(f"{'#'*80}\n")
        
        # Load this year's data
        year_data = load_processed_data(year, data_dir)
        
        if year_data is None:
            logger.error(f"Skipping {year} - data not available")
            continue
        
        # Cumulative approach: add this year's data to previous years
        if cumulative_data is None:
            cumulative_data = year_data
        else:
            cumulative_data = pd.concat([cumulative_data, year_data], ignore_index=True)
        
        logger.info(f"Cumulative dataset size: {len(cumulative_data)} records")
        logger.info(f"Years included: {years[:i]}")
        logger.info(f"Flood distribution: {cumulative_data['flood'].value_counts().to_dict()}")
        
        # Save cumulative dataset temporarily
        temp_csv = Path(data_dir) / f'cumulative_up_to_{year}.csv'
        cumulative_data.to_csv(temp_csv, index=False)
        logger.info(f"Saved cumulative dataset to: {temp_csv}")
        
        # Train model
        logger.info(f"\nTraining Model v{i}...")
        
        try:
            model, metrics, metadata = train_model(
                version=i,  # Explicit version number
                models_dir=models_dir,
                data_file=str(temp_csv),
                use_grid_search=use_grid_search,
                n_folds=cv_folds,
                merge_datasets=False  # Already merged
            )
            
            # Store results
            results[i] = {
                'version': i,
                'years_included': years[:i],
                'final_year': year,
                'dataset_size': len(cumulative_data),
                'metrics': metrics,
                'metadata_path': str(Path(models_dir) / f'flood_rf_model_v{i}.json')
            }
            
            logger.info(f"\n✓ Model v{i} training complete!")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training model v{i}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Generate comparison report
    logger.info(f"\n{'='*80}")
    logger.info("PROGRESSIVE TRAINING COMPLETE!")
    logger.info(f"{'='*80}\n")
    
    if results:
        generate_progression_report(results, models_dir)
    
    return results


def generate_progression_report(results, models_dir='models'):
    """Generate a report showing model progression and improvement."""
    logger.info("="*80)
    logger.info("MODEL PROGRESSION REPORT")
    logger.info("="*80)
    
    # Create comparison table
    logger.info(f"\n{'Version':<10} {'Years':<20} {'Records':<10} {'Accuracy':<12} {'F1 Score':<12} {'Improvement'}")
    logger.info("-"*80)
    
    previous_accuracy = None
    previous_f1 = None
    
    for version, data in sorted(results.items()):
        years_str = f"{data['years_included'][0]}-{data['final_year']}"
        accuracy = data['metrics']['accuracy']
        f1_score = data['metrics']['f1_score']
        
        # Calculate improvement
        if previous_accuracy is not None:
            acc_improvement = ((accuracy - previous_accuracy) / previous_accuracy) * 100
            f1_improvement = ((f1_score - previous_f1) / previous_f1) * 100
            improvement_str = f"+{acc_improvement:.2f}% / +{f1_improvement:.2f}%"
        else:
            improvement_str = "baseline"
        
        logger.info(f"v{version:<9} {years_str:<20} {data['dataset_size']:<10} "
                   f"{accuracy:<12.4f} {f1_score:<12.4f} {improvement_str}")
        
        previous_accuracy = accuracy
        previous_f1 = f1_score
    
    # Save detailed report
    report_path = Path(models_dir) / 'progressive_training_report.json'
    with open(report_path, 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'training_strategy': 'Progressive Cumulative',
            'results': results
        }, f, indent=2)
    
    logger.info(f"\n✓ Detailed report saved to: {report_path}")
    
    # Best model summary
    best_version = max(results.keys(), key=lambda k: results[k]['metrics']['f1_score'])
    best_data = results[best_version]
    
    logger.info(f"\n{'='*80}")
    logger.info("BEST MODEL")
    logger.info(f"{'='*80}")
    logger.info(f"Version: v{best_version}")
    logger.info(f"Years: {best_data['years_included'][0]}-{best_data['final_year']}")
    logger.info(f"Records: {best_data['dataset_size']}")
    logger.info(f"Accuracy: {best_data['metrics']['accuracy']:.4f}")
    logger.info(f"Precision: {best_data['metrics']['precision']:.4f}")
    logger.info(f"Recall: {best_data['metrics']['recall']:.4f}")
    logger.info(f"F1 Score: {best_data['metrics']['f1_score']:.4f}")
    logger.info(f"{'='*80}")


def year_specific_train(
    years=[2022, 2023, 2024, 2025],
    data_dir='data/processed',
    models_dir='models/year_specific',
    use_grid_search=False,
    cv_folds=5
):
    """
    Train separate models for each year (alternative strategy).
    
    This creates year-specific models instead of cumulative ones.
    Useful for analyzing year-specific patterns.
    """
    logger.info("="*80)
    logger.info("YEAR-SPECIFIC MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Training strategy: Year-specific (one model per year)")
    logger.info(f"Years: {years}")
    logger.info("="*80)
    
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for year in years:
        logger.info(f"\n{'#'*80}")
        logger.info(f"# TRAINING MODEL FOR {year}")
        logger.info(f"{'#'*80}\n")
        
        # Load year data
        year_data = load_processed_data(year, data_dir)
        
        if year_data is None:
            logger.error(f"Skipping {year} - data not available")
            continue
        
        logger.info(f"Dataset size: {len(year_data)} records")
        logger.info(f"Flood distribution: {year_data['flood'].value_counts().to_dict()}")
        
        # Save year dataset
        temp_csv = Path(data_dir) / f'year_{year}_only.csv'
        year_data.to_csv(temp_csv, index=False)
        
        # Train model
        try:
            model, metrics, metadata = train_model(
                version=year,  # Use year as version
                models_dir=models_dir,
                data_file=str(temp_csv),
                use_grid_search=use_grid_search,
                n_folds=cv_folds,
                merge_datasets=False
            )
            
            results[year] = {
                'year': year,
                'dataset_size': len(year_data),
                'metrics': metrics
            }
            
            logger.info(f"\n✓ Model for {year} complete!")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {year} model: {str(e)}")
            continue
    
    logger.info(f"\n{'='*80}")
    logger.info("YEAR-SPECIFIC TRAINING COMPLETE!")
    logger.info(f"{'='*80}\n")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Progressive model training with official flood records',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Training Strategies:

1. PROGRESSIVE (Recommended for Thesis):
   - Model v1: 2022 data
   - Model v2: 2022 + 2023 data
   - Model v3: 2022 + 2023 + 2024 data
   - Model v4: 2022 + 2023 + 2024 + 2025 data
   
   Shows clear evolution and improvement over time!

2. YEAR-SPECIFIC:
   - Model 2022: Only 2022 data
   - Model 2023: Only 2023 data
   - Model 2024: Only 2024 data
   - Model 2025: Only 2025 data
   
   Useful for year-to-year comparison.

Examples:
  Progressive training (recommended):
    python progressive_train.py
  
  With hyperparameter tuning:
    python progressive_train.py --grid-search --cv-folds 10
  
  Year-specific models:
    python progressive_train.py --year-specific
  
  Custom years:
    python progressive_train.py --years 2023 2024 2025
        """
    )
    parser.add_argument('--years', nargs='+', type=int,
                       default=[2022, 2023, 2024, 2025],
                       help='Years to include in training')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--grid-search', action='store_true',
                       help='Perform hyperparameter tuning (slow but optimal)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--year-specific', action='store_true',
                       help='Train separate models for each year instead of cumulative')
    
    args = parser.parse_args()
    
    if args.year_specific:
        # Year-specific training
        year_specific_train(
            years=args.years,
            data_dir=args.data_dir,
            models_dir=args.models_dir + '/year_specific',
            use_grid_search=args.grid_search,
            cv_folds=args.cv_folds
        )
    else:
        # Progressive cumulative training (recommended)
        progressive_train(
            years=args.years,
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            use_grid_search=args.grid_search,
            cv_folds=args.cv_folds
        )
