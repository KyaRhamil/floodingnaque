"""
Automated Training Pipeline with Production Data Sources
=========================================================

This script provides an end-to-end training workflow that:
1. Ingests fresh data from ALL production sources
2. Trains models with environment-aware resource allocation
3. Validates and evaluates the trained model
4. Generates comprehensive reports

This replaces the outdated manual workflow of:
- Using only static CSV files
- Training with hardcoded n_jobs=-1
- Missing real-time production data

Usage:
    # Full production pipeline (recommended)
    python scripts/train_with_production_data.py --production
    
    # Quick training with fresh data
    python scripts/train_with_production_data.py --days 180
    
    # Progressive training with production data
    python scripts/train_with_production_data.py --progressive --grid-search
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our enhanced scripts
from scripts.ingest_training_data import TrainingDataIngestion
from scripts.train_production import ProductionModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionTrainingPipeline:
    """
    Automated training pipeline using production data sources.
    
    Workflow:
    1. Data Ingestion: Fetch from Supabase, Earth Engine, Meteostat, WorldTides
    2. Data Processing: Clean, validate, engineer features
    3. Model Training: Train with proper resource allocation
    4. Validation: Test model performance
    5. Reporting: Generate comprehensive metrics
    """
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize the training pipeline.
        
        Args:
            env_file: Path to environment file (default: .env.production)
        """
        self.env_file = env_file
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Paths
        self.backend_dir = Path(__file__).parent.parent
        self.data_dir = self.backend_dir / 'data' / 'training'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.ingestion = TrainingDataIngestion(env_file=env_file)
        self.trainer = ProductionModelTrainer()
    
    def run_full_pipeline(
        self,
        days: int = 365,
        include_satellite: bool = True,
        include_tides: bool = True,
        include_meteostat: bool = True,
        use_grid_search: bool = False,
        model_type: str = 'random_forest',
        version: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            days: Days of data to fetch
            include_satellite: Include Google Earth Engine data
            include_tides: Include tide data
            include_meteostat: Include Meteostat data
            use_grid_search: Enable hyperparameter tuning
            model_type: Type of model to train
            version: Model version (auto-incremented if None)
            
        Returns:
            Dict with training results
        """
        logger.info("="*80)
        logger.info("PRODUCTION TRAINING PIPELINE")
        logger.info("="*80)
        logger.info(f"Timestamp: {self.timestamp}")
        logger.info(f"Environment: {self.env_file or '.env.production or .env'}")
        logger.info("="*80)
        
        results = {
            'started_at': datetime.now().isoformat(),
            'config': {
                'days': days,
                'include_satellite': include_satellite,
                'include_tides': include_tides,
                'include_meteostat': include_meteostat,
                'use_grid_search': use_grid_search,
                'model_type': model_type
            }
        }
        
        # Step 1: Data Ingestion
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA INGESTION")
        logger.info("="*80)
        
        dataset_path = self.data_dir / f'production_data_{self.timestamp}.csv'
        
        df = self.ingestion.ingest_all_sources(
            days=days,
            include_satellite=include_satellite,
            include_tides=include_tides,
            include_meteostat=include_meteostat,
            include_official=True  # Always include baseline official records
        )
        
        if df.empty:
            logger.error("❌ Data ingestion failed! No data available.")
            results['status'] = 'failed'
            results['error'] = 'No data ingested'
            return results
        
        # Save ingested data
        df.to_csv(dataset_path, index=False)
        logger.info(f"✓ Dataset saved to: {dataset_path}")
        
        results['ingestion'] = {
            'records': len(df),
            'dataset_path': str(dataset_path),
            'sources': df['source'].unique().tolist() if 'source' in df.columns else [],
            'date_range': {
                'start': str(df['timestamp'].min()) if 'timestamp' in df.columns else None,
                'end': str(df['timestamp'].max()) if 'timestamp' in df.columns else None
            }
        }
        
        # Step 2: Model Training
        logger.info("\n" + "="*80)
        logger.info("STEP 2: MODEL TRAINING")
        logger.info("="*80)
        
        try:
            model, metrics, metadata = self.trainer.train(
                data_path=str(dataset_path),
                model_type=model_type,
                use_grid_search=use_grid_search,
                generate_shap=True,
                version=version
            )
            
            results['training'] = {
                'status': 'success',
                'model_path': str(metadata.get('model_path')),
                'version': metadata.get('version'),
                'metrics': metrics
            }
            
            logger.info(f"\n✓ Model training completed successfully!")
            logger.info(f"  Model version: v{metadata.get('version')}")
            logger.info(f"  Test accuracy: {metrics['test']['accuracy']:.4f}")
            logger.info(f"  Test F1 score: {metrics['test']['f1_score']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results['training'] = {
                'status': 'failed',
                'error': str(e)
            }
            return results
        
        # Step 3: Summary
        results['completed_at'] = datetime.now().isoformat()
        results['status'] = 'success'
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETE! ✓")
        logger.info("="*80)
        logger.info(f"Training data: {results['ingestion']['records']} records")
        logger.info(f"Model version: v{metadata.get('version')}")
        logger.info(f"Test performance:")
        logger.info(f"  - Accuracy: {metrics['test']['accuracy']:.4f}")
        logger.info(f"  - Precision: {metrics['test']['precision']:.4f}")
        logger.info(f"  - Recall: {metrics['test']['recall']:.4f}")
        logger.info(f"  - F1 Score: {metrics['test']['f1_score']:.4f}")
        logger.info("="*80)
        
        return results
    
    def run_progressive_training(
        self,
        years: list = [2022, 2023, 2024, 2025],
        use_grid_search: bool = False
    ) -> Dict[str, Any]:
        """
        Run progressive training with production data.
        
        Trains multiple models showing evolution over time, but using
        fresh production data instead of static CSVs.
        
        Args:
            years: Years to train on progressively
            use_grid_search: Enable hyperparameter tuning
            
        Returns:
            Dict with results for each model version
        """
        logger.info("="*80)
        logger.info("PROGRESSIVE TRAINING WITH PRODUCTION DATA")
        logger.info("="*80)
        
        results = {}
        
        for i, year in enumerate(years, start=1):
            logger.info(f"\n{'#'*80}")
            logger.info(f"# MODEL v{i}: Training on data up to {year}")
            logger.info(f"{'#'*80}\n")
            
            # Calculate days from start of 2022 to end of this year
            from datetime import datetime
            start_date = datetime(2022, 1, 1)
            end_date = datetime(year, 12, 31)
            days = (end_date - start_date).days
            
            # Run pipeline for this period
            result = self.run_full_pipeline(
                days=min(days, 1460),  # Cap at 4 years
                include_satellite=True,
                include_tides=True,
                include_meteostat=True,
                use_grid_search=use_grid_search,
                version=i
            )
            
            results[i] = {
                'year': year,
                'years_included': years[:i],
                'result': result
            }
            
            if result.get('status') != 'success':
                logger.error(f"❌ Training for v{i} failed, stopping progressive training")
                break
        
        logger.info("\n" + "="*80)
        logger.info("PROGRESSIVE TRAINING COMPLETE")
        logger.info("="*80)
        
        # Print comparison
        for version, data in results.items():
            if data['result'].get('status') == 'success':
                metrics = data['result']['training']['metrics']['test']
                logger.info(f"v{version} ({data['year']}): Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Automated training pipeline with production data sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full production pipeline (recommended):
    python scripts/train_with_production_data.py --production
  
  Quick training with 6 months of fresh data:
    python scripts/train_with_production_data.py --days 180
  
  Progressive training showing model evolution:
    python scripts/train_with_production_data.py --progressive --grid-search
  
  Training without satellite data:
    python scripts/train_with_production_data.py --no-satellite
  
  Custom environment file:
    python scripts/train_with_production_data.py --env .env.staging
        """
    )
    
    # Data source options
    parser.add_argument('--days', type=int, default=365,
                        help='Days of historical data to fetch (default: 365)')
    parser.add_argument('--no-satellite', action='store_true',
                        help='Exclude Google Earth Engine satellite data')
    parser.add_argument('--no-tides', action='store_true',
                        help='Exclude WorldTides data')
    parser.add_argument('--no-meteostat', action='store_true',
                        help='Exclude Meteostat weather station data')
    
    # Training options
    parser.add_argument('--model-type', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'ensemble'],
                        help='Model type to train')
    parser.add_argument('--grid-search', action='store_true',
                        help='Enable hyperparameter tuning (slower but better)')
    parser.add_argument('--version', type=int,
                        help='Model version (auto-incremented if not specified)')
    
    # Pipeline modes
    parser.add_argument('--production', action='store_true',
                        help='Full production mode: grid search + all data sources')
    parser.add_argument('--progressive', action='store_true',
                        help='Progressive training: train v1-v4 showing evolution')
    
    # Environment
    parser.add_argument('--env', type=str, default=None,
                        help='Path to .env file (default: .env.production or .env)')
    
    args = parser.parse_args()
    
    # Production mode enables all features
    if args.production:
        args.grid_search = True
        args.no_satellite = False
        args.no_tides = False
        args.no_meteostat = False
        logger.info("Production mode: Enabled all data sources and grid search")
    
    # Initialize pipeline
    pipeline = ProductionTrainingPipeline(env_file=args.env)
    
    # Run appropriate mode
    if args.progressive:
        # Progressive training
        results = pipeline.run_progressive_training(
            years=[2022, 2023, 2024, 2025],
            use_grid_search=args.grid_search
        )
    else:
        # Standard single training
        results = pipeline.run_full_pipeline(
            days=args.days,
            include_satellite=not args.no_satellite,
            include_tides=not args.no_tides,
            include_meteostat=not args.no_meteostat,
            use_grid_search=args.grid_search,
            model_type=args.model_type,
            version=args.version
        )
    
    # Save pipeline results
    results_path = pipeline.data_dir / f'pipeline_results_{pipeline.timestamp}.json'
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n✓ Pipeline results saved to: {results_path}")
    
    # Exit code based on success
    if isinstance(results, dict):
        if results.get('status') == 'success':
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        # Progressive training - check if any succeeded
        success = any(r['result'].get('status') == 'success' for r in results.values())
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
