"""
Ultimate Flood Prediction Model Training - Progressive Version System
======================================================================

This script creates models progressively, showing evolution from v1 to vN:
  - v1: Official Flood Records 2022
  - v2: Official Flood Records 2022-2023
  - v3: Official Flood Records 2022-2024
  - v4: Official Flood Records 2022-2025
  - v5: PAGASA Weather Data (2020-2025) 
  - v6: ULTIMATE - All datasets combined (Official + PAGASA + Future)
  - v7+: Future datasets automatically integrated

Features:
- Progressive version training (v1 → v2 → ... → vN)
- Dynamic dataset discovery and combination
- Automatic feature engineering for multi-source data
- Future-proof: New datasets = New model version = Best model
- Full backward compatibility

Usage:
    # Train ALL versions progressively (recommended for thesis)
    python scripts/train_ultimate.py --progressive
    
    # Train only the latest/best ultimate model
    python scripts/train_ultimate.py --latest-only
    
    # With hyperparameter tuning
    python scripts/train_ultimate.py --progressive --grid-search
    
    # Production-ready with all enhancements
    python scripts/train_ultimate.py --production

Author: Floodingnaque Team
Last Updated: January 2026
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
    VotingClassifier, HistGradientBoostingClassifier
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
import joblib
import json
import logging
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
import argparse
import glob
import os

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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
MODELS_DIR = BACKEND_DIR / 'models'
REPORTS_DIR = BACKEND_DIR / 'reports'
DATA_DIR = BACKEND_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'

# Dataset discovery patterns
DATASET_PATTERNS = {
    'pagasa': [
        'Floodingnaque_CADS-S0126006_NAIA Daily Data.csv',
        'Floodingnaque_CADS-S0126006_Port Area Daily Data.csv', 
        'Floodingnaque_CADS-S0126006_Science Garden Daily Data.csv'
    ],
    'official_flood_records': [
        'Floodingnaque_Paranaque_Official_Flood_Records_*.csv'
    ],
    'processed_cumulative': [
        'processed/cumulative_up_to_*.csv'
    ],
    'processed_pagasa': [
        'processed/pagasa_training_dataset.csv'
    ],
    'synthetic': [
        'synthetic_dataset.csv'
    ]
}

# =============================================================================
# VERSION REGISTRY - Defines progressive model versions
# Add new datasets here to automatically create new model versions
# =============================================================================
VERSION_REGISTRY = [
    {
        'version': 1,
        'name': 'Official_2022',
        'description': 'Parañaque Official Flood Records 2022 (Baseline)',
        'data_file': 'data/processed/cumulative_up_to_2022.csv',
        'year_range': '2022',
        'is_cumulative': True,
    },
    {
        'version': 2,
        'name': 'Official_2022-2023',
        'description': 'Parañaque Official Flood Records 2022-2023',
        'data_file': 'data/processed/cumulative_up_to_2023.csv',
        'year_range': '2022-2023',
        'is_cumulative': True,
    },
    {
        'version': 3,
        'name': 'Official_2022-2024',
        'description': 'Parañaque Official Flood Records 2022-2024',
        'data_file': 'data/processed/cumulative_up_to_2024.csv',
        'year_range': '2022-2024',
        'is_cumulative': True,
    },
    {
        'version': 4,
        'name': 'Official_2022-2025',
        'description': 'Parañaque Official Flood Records 2022-2025',
        'data_file': 'data/processed/cumulative_up_to_2025.csv',
        'year_range': '2022-2025',
        'is_cumulative': True,
    },
    {
        'version': 5,
        'name': 'PAGASA_Enhanced',
        'description': 'DOST-PAGASA Weather Stations (NAIA, Port Area, Science Garden) 2020-2025',
        'data_file': 'data/processed/pagasa_training_dataset.csv',
        'year_range': '2020-2025',
        'is_cumulative': False,
        'requires_preprocessing': True,
        'preprocessing_script': 'preprocess_pagasa_data.py --create-training',
    },
    {
        'version': 6,
        'name': 'ULTIMATE',
        'description': 'Ultimate Combined: Official Records + PAGASA + All Available Data',
        'data_file': 'data/processed/ultimate_combined_dataset.csv',
        'year_range': '2020-2025',
        'is_cumulative': True,
        'is_ultimate': True,
    },
    # ==========================================================================
    # ADD NEW DATASETS BELOW - They will automatically become higher versions
    # Example:
    # {
    #     'version': 7,
    #     'name': 'Satellite_Radar_2026',
    #     'description': 'Satellite and Radar precipitation data 2026',
    #     'data_file': 'data/processed/satellite_radar_2026.csv',
    #     'year_range': '2020-2026',
    #     'is_cumulative': True,
    # },
    # ==========================================================================
]


class VersionedModelResult:
    """Stores results for a trained model version."""
    def __init__(self, version: int, name: str, description: str,
                 year_range: str, record_count: int, metrics: Dict,
                 feature_count: int, features: List[str],
                 model_path: str, metadata_path: str):
        self.version = version
        self.name = name
        self.description = description
        self.year_range = year_range
        self.record_count = record_count
        self.metrics = metrics
        self.feature_count = feature_count
        self.features = features
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.is_best = False
        self.created_at = datetime.now().isoformat()


class UltimateModelTrainer:
    """
    Ultimate trainer that combines all available datasets for maximum model performance.
    Automatically discovers and integrates new datasets as they become available.
    """
    
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
        
        self.model = None
        self.feature_names: List[str] = []
        self.training_metrics: Dict = {}
        self.data_info: Dict = {}
        self.discovered_datasets: Dict = {}
        self.trained_versions: Dict[int, VersionedModelResult] = {}
    
    def get_available_versions(self) -> List[Dict]:
        """Get list of versions that have available data files."""
        available = []
        for version_config in VERSION_REGISTRY:
            file_path = BACKEND_DIR / version_config['data_file']
            if file_path.exists():
                df = pd.read_csv(file_path)
                version_config['record_count'] = len(df)
                version_config['available'] = True
                available.append(version_config)
                logger.info(f"✓ v{version_config['version']} {version_config['name']}: {len(df)} records")
            else:
                version_config['available'] = False
                logger.warning(f"✗ v{version_config['version']} {version_config['name']}: File not found")
        return available
    
    def create_ultimate_combined_dataset(self) -> Path:
        """Create the ultimate combined dataset from all available sources."""
        logger.info("\n" + "="*60)
        logger.info("CREATING ULTIMATE COMBINED DATASET")
        logger.info("="*60)
        
        all_data = []
        sources_used = []
        
        # Load Official Records (latest cumulative)
        official_path = BACKEND_DIR / 'data/processed/cumulative_up_to_2025.csv'
        if official_path.exists():
            df_official = pd.read_csv(official_path)
            df_official['data_source'] = 'official_records'
            all_data.append(df_official)
            sources_used.append(f"Official Records: {len(df_official)} records")
            logger.info(f"  ✓ Added Official Records: {len(df_official)} records")
        
        # Load PAGASA data
        pagasa_path = BACKEND_DIR / 'data/processed/pagasa_training_dataset.csv'
        if pagasa_path.exists():
            df_pagasa = pd.read_csv(pagasa_path)
            df_pagasa['data_source'] = 'pagasa'
            all_data.append(df_pagasa)
            sources_used.append(f"PAGASA: {len(df_pagasa)} records")
            logger.info(f"  ✓ Added PAGASA Data: {len(df_pagasa)} records")
        else:
            # Try to create PAGASA dataset
            logger.info("  PAGASA dataset not found, attempting to create...")
            try:
                sys.path.insert(0, str(SCRIPT_DIR))
                from preprocess_pagasa_data import create_training_dataset
                create_training_dataset(use_naia_only=True, include_flood_records=True)
                if pagasa_path.exists():
                    df_pagasa = pd.read_csv(pagasa_path)
                    df_pagasa['data_source'] = 'pagasa'
                    all_data.append(df_pagasa)
                    sources_used.append(f"PAGASA: {len(df_pagasa)} records")
                    logger.info(f"  ✓ Created and added PAGASA Data: {len(df_pagasa)} records")
            except Exception as e:
                logger.warning(f"  Could not create PAGASA dataset: {e}")
        
        # Load any additional processed datasets
        for csv_file in PROCESSED_DIR.glob("*.csv"):
            if csv_file.name not in ['cumulative_up_to_2025.csv', 'pagasa_training_dataset.csv', 
                                     'ultimate_combined_dataset.csv']:
                if 'cumulative' not in csv_file.name and 'ultimate' not in csv_file.name:
                    try:
                        df_extra = pd.read_csv(csv_file)
                        if 'flood' in df_extra.columns and len(df_extra) > 100:
                            df_extra['data_source'] = csv_file.stem
                            all_data.append(df_extra)
                            sources_used.append(f"{csv_file.stem}: {len(df_extra)} records")
                            logger.info(f"  ✓ Added {csv_file.name}: {len(df_extra)} records")
                    except Exception:
                        pass
        
        if not all_data:
            raise ValueError("No datasets available for combination!")
        
        # Merge all datasets
        combined = pd.concat(all_data, ignore_index=True)
        
        # Standardize columns
        required_cols = ['temperature', 'humidity', 'precipitation', 'flood']
        for col in required_cols:
            if col not in combined.columns:
                logger.warning(f"Missing required column: {col}")
        
        # Fill missing values
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        combined[numeric_cols] = combined[numeric_cols].fillna(combined[numeric_cols].median())
        
        # Remove duplicates
        key_cols = ['temperature', 'humidity', 'precipitation', 'flood']
        key_cols_present = [c for c in key_cols if c in combined.columns]
        if key_cols_present:
            combined = combined.drop_duplicates(subset=key_cols_present, keep='first')
        
        # Save combined dataset
        output_path = PROCESSED_DIR / 'ultimate_combined_dataset.csv'
        combined.to_csv(output_path, index=False)
        
        logger.info(f"\n✓ Ultimate dataset created: {len(combined)} records")
        logger.info(f"  Sources: {', '.join(sources_used)}")
        logger.info(f"  Saved to: {output_path}")
        
        return output_path
    
    def train_single_version(
        self,
        version_config: Dict,
        use_grid_search: bool = False,
        n_folds: int = 5
    ) -> Optional[VersionedModelResult]:
        """Train a single model version."""
        version = version_config['version']
        name = version_config['name']
        
        logger.info(f"\n{'='*70}")
        logger.info(f"TRAINING MODEL v{version}: {name}")
        logger.info(f"{'='*70}")
        logger.info(f"Description: {version_config['description']}")
        logger.info(f"Year range: {version_config['year_range']}")
        
        # Check if this is the ultimate version - create dataset if needed
        if version_config.get('is_ultimate', False):
            try:
                self.create_ultimate_combined_dataset()
            except Exception as e:
                logger.error(f"Failed to create ultimate dataset: {e}")
                return None
        
        # Load data
        file_path = BACKEND_DIR / version_config['data_file']
        if not file_path.exists():
            # Try preprocessing if available
            if version_config.get('requires_preprocessing'):
                logger.info(f"Running preprocessing for {name}...")
                try:
                    script = version_config.get('preprocessing_script', '')
                    if 'pagasa' in script.lower():
                        sys.path.insert(0, str(SCRIPT_DIR))
                        from preprocess_pagasa_data import create_training_dataset
                        create_training_dataset(use_naia_only=True, include_flood_records=True)
                except Exception as e:
                    logger.error(f"Preprocessing failed: {e}")
                    return None
            
            if not file_path.exists():
                logger.error(f"Data file not found: {file_path}")
                return None
        
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Prepare features
        X, y = self.prepare_features(df)
        feature_names = list(X.columns)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        if use_grid_search:
            model = self._grid_search_train(X_train, y_train, n_folds)
        else:
            params = {
                'n_estimators': 300,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': self.random_state,
                'n_jobs': -1
            }
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
            'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
        }
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=n_folds, scoring='f1_weighted', n_jobs=-1)
        metrics['cv_f1_mean'] = float(cv_scores.mean())
        metrics['cv_f1_std'] = float(cv_scores.std())
        
        # Save model
        model_filename = f'flood_rf_model_v{version}.joblib'
        model_path = self.models_dir / model_filename
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'version': version,
            'name': name,
            'description': version_config['description'],
            'year_range': version_config['year_range'],
            'record_count': len(df),
            'feature_count': len(feature_names),
            'features': feature_names,
            'metrics': metrics,
            'model_parameters': {
                'n_estimators': getattr(model, 'n_estimators', None),
                'max_depth': getattr(model, 'max_depth', None),
                'min_samples_split': getattr(model, 'min_samples_split', None),
                'min_samples_leaf': getattr(model, 'min_samples_leaf', None),
            },
            'feature_importance': dict(zip(feature_names, [float(x) for x in model.feature_importances_])),
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create result object
        result = VersionedModelResult(
            version=version,
            name=name,
            description=version_config['description'],
            year_range=version_config['year_range'],
            record_count=len(df),
            metrics=metrics,
            feature_count=len(feature_names),
            features=feature_names,
            model_path=str(model_path),
            metadata_path=str(metadata_path)
        )
        
        self.trained_versions[version] = result
        self.model = model
        self.feature_names = feature_names
        
        # Log results
        logger.info(f"\n✓ Model v{version} trained successfully!")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"  CV F1:     {metrics['cv_f1_mean']:.4f} (+/- {metrics['cv_f1_std']*2:.4f})")
        logger.info(f"  Saved to:  {model_path}")
        
        return result
    
    def train_progressive(
        self,
        use_grid_search: bool = False,
        n_folds: int = 5,
        latest_only: bool = False
    ) -> Dict[int, VersionedModelResult]:
        """
        Train all model versions progressively (v1 → v2 → ... → vN).
        
        This showcases model evolution from baseline to ultimate.
        """
        logger.info("\n" + "="*70)
        logger.info("PROGRESSIVE MODEL TRAINING")
        logger.info("v1 → v2 → v3 → ... → ULTIMATE")
        logger.info("="*70 + "\n")
        
        # Get available versions
        available = self.get_available_versions()
        
        if not available:
            logger.error("No datasets available for training!")
            logger.info("Please run preprocessing scripts first:")
            logger.info("  python scripts/preprocess_official_flood_records.py")
            logger.info("  python scripts/preprocess_pagasa_data.py --create-training")
            return {}
        
        logger.info(f"\nFound {len(available)} versions available for training\n")
        
        if latest_only:
            # Only train the highest version
            latest = max(available, key=lambda x: x['version'])
            logger.info(f"Training only latest version: v{latest['version']} ({latest['name']})")
            result = self.train_single_version(latest, use_grid_search, n_folds)
            if result:
                self.trained_versions[latest['version']] = result
        else:
            # Train all versions progressively
            for version_config in sorted(available, key=lambda x: x['version']):
                result = self.train_single_version(version_config, use_grid_search, n_folds)
                if result:
                    self.trained_versions[version_config['version']] = result
        
        # Determine and save best model
        if self.trained_versions:
            best_version = max(
                self.trained_versions.values(),
                key=lambda x: x.metrics['f1_score']
            )
            best_version.is_best = True
            
            # Copy best model as "latest"
            best_model_path = Path(best_version.model_path)
            latest_path = self.models_dir / 'flood_rf_model_latest.joblib'
            
            best_model = joblib.load(best_model_path)
            joblib.dump(best_model, latest_path)
            
            # Generate progression report
            self._generate_progression_report()
            
            logger.info(f"\n{'='*70}")
            logger.info("BEST MODEL")
            logger.info(f"{'='*70}")
            logger.info(f"Version: v{best_version.version} ({best_version.name})")
            logger.info(f"F1 Score: {best_version.metrics['f1_score']:.4f}")
            logger.info(f"Saved as: {latest_path}")
        
        return self.trained_versions
    
    def _generate_progression_report(self):
        """Generate a comprehensive progression report."""
        if not self.trained_versions:
            return
        
        logger.info(f"\n{'='*70}")
        logger.info("MODEL PROGRESSION REPORT")
        logger.info(f"{'='*70}")
        
        # Header
        logger.info(f"\n{'Ver':<5} {'Name':<25} {'Records':<10} {'Accuracy':<10} {'F1 Score':<10} {'Δ F1':<10}")
        logger.info("-" * 70)
        
        prev_f1 = None
        sorted_versions = sorted(self.trained_versions.values(), key=lambda x: x.version)
        
        for v in sorted_versions:
            delta = ""
            if prev_f1 is not None:
                change = ((v.metrics['f1_score'] - prev_f1) / prev_f1) * 100 if prev_f1 > 0 else 0
                delta = f"{'+' if change >= 0 else ''}{change:.1f}%"
            
            best_marker = " ★" if v.is_best else ""
            logger.info(
                f"v{v.version:<4} {v.name[:24]:<25} {v.record_count:<10} "
                f"{v.metrics['accuracy']:<10.4f} {v.metrics['f1_score']:<10.4f} {delta:<10}{best_marker}"
            )
            prev_f1 = v.metrics['f1_score']
        
        # Save detailed report as JSON
        report = {
            'generated_at': datetime.now().isoformat(),
            'training_strategy': 'Progressive (v1 → v2 → ... → vN)',
            'total_versions': len(self.trained_versions),
            'best_version': next((v.version for v in self.trained_versions.values() if v.is_best), None),
            'versions': {
                v.version: {
                    'name': v.name,
                    'description': v.description,
                    'year_range': v.year_range,
                    'record_count': v.record_count,
                    'metrics': v.metrics,
                    'feature_count': v.feature_count,
                    'model_path': v.model_path,
                    'is_best': v.is_best
                }
                for v in self.trained_versions.values()
            }
        }
        
        report_path = self.reports_dir / 'progressive_training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\n✓ Report saved: {report_path}")
        
        # Generate comparison chart if matplotlib available
        if PLOT_AVAILABLE:
            self._generate_comparison_chart()
    
    def _generate_comparison_chart(self):
        """Generate a visual comparison chart."""
        if not PLOT_AVAILABLE or not self.trained_versions:
            return
        
        versions = sorted(self.trained_versions.values(), key=lambda x: x.version)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # type: ignore[name-defined]
        
        # Chart 1: Metrics comparison
        ax1 = axes[0]
        x = [f"v{v.version}" for v in versions]
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x_pos = np.arange(len(x))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            values = [v.metrics[metric] for v in versions]
            ax1.bar(x_pos + i * width, values, width, label=metric.replace('_', ' ').title())
        
        ax1.set_xlabel('Model Version')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Progression (v1 → ULTIMATE)')
        ax1.set_xticks(x_pos + width * 1.5)
        ax1.set_xticklabels([f"v{v.version}\n{v.name[:10]}" for v in versions], fontsize=8)
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.grid(axis='y', alpha=0.3)
        
        # Chart 2: F1 Score progression
        ax2 = axes[1]
        f1_scores = [v.metrics['f1_score'] for v in versions]
        records = [v.record_count for v in versions]
        
        color1 = 'tab:blue'
        ax2.set_xlabel('Model Version')
        ax2.set_ylabel('F1 Score', color=color1)
        ax2.plot([f"v{v.version}" for v in versions], f1_scores, 'o-', color=color1, linewidth=2, markersize=10)
        ax2.tick_params(axis='y', labelcolor=color1)
        
        # Mark best model
        best_idx = max(range(len(versions)), key=lambda i: versions[i].metrics['f1_score'])
        ax2.scatter([f"v{versions[best_idx].version}"], [f1_scores[best_idx]], 
                   s=200, c='gold', marker='*', zorder=5, label='Best Model')
        
        ax2_twin = ax2.twinx()
        color2 = 'tab:orange'
        ax2_twin.set_ylabel('Training Records', color=color2)
        ax2_twin.bar([f"v{v.version}" for v in versions], records, alpha=0.3, color=color2)
        ax2_twin.tick_params(axis='y', labelcolor=color2)
        
        ax2.set_title('F1 Score Progression & Data Growth')
        ax2.legend(loc='lower right')
        
        plt.tight_layout()  # type: ignore[name-defined]
        chart_path = self.reports_dir / 'model_progression_chart.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')  # type: ignore[name-defined]
        plt.close()  # type: ignore[name-defined]
        
        logger.info(f"✓ Chart saved: {chart_path}")
        """
        Dynamically discover all available datasets in the data directory.
        
        Returns:
            Dictionary mapping dataset types to lists of available file paths
        """
        logger.info("Discovering available datasets...")
        
        discovered = {}
        
        # Check for PAGASA datasets
        pagasa_files = []
        for pattern in DATASET_PATTERNS['pagasa']:
            full_path = DATA_DIR / pattern
            if full_path.exists():
                pagasa_files.append(str(full_path))
        
        if pagasa_files:
            discovered['pagasa'] = pagasa_files
            logger.info(f"✓ Found {len(pagasa_files)} PAGASA datasets")
        
        # Check for official flood records
        official_files = []
        for pattern in DATASET_PATTERNS['official_flood_records']:
            pattern_path = DATA_DIR / pattern
            for file_path in glob.glob(str(pattern_path)):
                official_files.append(file_path)
        
        if official_files:
            discovered['official_flood_records'] = official_files
            logger.info(f"✓ Found {len(official_files)} official flood record datasets")
        
        # Check for processed cumulative datasets
        processed_files = []
        for pattern in DATASET_PATTERNS['processed_cumulative']:
            pattern_path = DATA_DIR / pattern
            for file_path in glob.glob(str(pattern_path)):
                processed_files.append(file_path)
        
        if processed_files:
            discovered['processed_cumulative'] = processed_files
            logger.info(f"✓ Found {len(processed_files)} processed cumulative datasets")
        
        # Check for synthetic dataset
        synthetic_files = []
        for pattern in DATASET_PATTERNS['synthetic']:
            full_path = DATA_DIR / pattern
            if full_path.exists():
                synthetic_files.append(str(full_path))
        
        if synthetic_files:
            discovered['synthetic'] = synthetic_files
            logger.info(f"✓ Found {len(synthetic_files)} synthetic datasets")
        
        # Look for any additional CSV files in data directories
        additional_files = []
        for csv_file in DATA_DIR.rglob("*.csv"):
            if str(csv_file) not in sum(discovered.values(), []):
                # Skip if it's a duplicate or backup
                if 'backup' not in str(csv_file).lower() and 'temp' not in str(csv_file).lower():
                    additional_files.append(str(csv_file))
        
        if additional_files:
            discovered['additional'] = additional_files
            logger.info(f"✓ Found {len(additional_files)} additional datasets")
        
        self.discovered_datasets = discovered
        return discovered
    
    def load_and_merge_all_datasets(self) -> pd.DataFrame:
        """
        Load and intelligently merge all discovered datasets.
        
        Returns:
            Merged DataFrame with all available data
        """
        logger.info("Loading and merging all available datasets...")
        
        all_dataframes = []
        
        # Load PAGASA datasets
        if 'pagasa' in self.discovered_datasets:
            for pagasa_file in self.discovered_datasets['pagasa']:
                logger.info(f"Loading PAGASA dataset: {Path(pagasa_file).name}")
                
                df = pd.read_csv(pagasa_file)
                
                # Standardize column names if needed
                df = self._standardize_pagasa_columns(df)
                
                # Add source identifier
                df['data_source'] = Path(pagasa_file).stem
                
                all_dataframes.append(df)
        
        # Load official flood records
        if 'official_flood_records' in self.discovered_datasets:
            for flood_file in self.discovered_datasets['official_flood_records']:
                logger.info(f"Loading flood records: {Path(flood_file).name}")
                
                df = pd.read_csv(flood_file)
                
                # Add source identifier
                df['data_source'] = Path(flood_file).stem
                
                all_dataframes.append(df)
        
        # Load processed cumulative datasets
        if 'processed_cumulative' in self.discovered_datasets:
            for proc_file in self.discovered_datasets['processed_cumulative']:
                logger.info(f"Loading processed dataset: {Path(proc_file).name}")
                
                df = pd.read_csv(proc_file)
                
                # Add source identifier
                df['data_source'] = Path(proc_file).stem
                
                all_dataframes.append(df)
        
        # Load synthetic dataset
        if 'synthetic' in self.discovered_datasets:
            for synth_file in self.discovered_datasets['synthetic']:
                logger.info(f"Loading synthetic dataset: {Path(synth_file).name}")
                
                df = pd.read_csv(synth_file)
                
                # Add source identifier
                df['data_source'] = Path(synth_file).stem
                
                all_dataframes.append(df)
        
        # Load additional datasets
        if 'additional' in self.discovered_datasets:
            for add_file in self.discovered_datasets['additional']:
                logger.info(f"Loading additional dataset: {Path(add_file).name}")
                
                df = pd.read_csv(add_file)
                
                # Add source identifier
                df['data_source'] = Path(add_file).stem
                
                all_dataframes.append(df)
        
        if not all_dataframes:
            raise ValueError("No datasets found to merge!")
        
        logger.info(f"Merging {len(all_dataframes)} datasets...")
        
        # Concatenate all dataframes
        merged_df = pd.concat(all_dataframes, ignore_index=True, sort=False)
        
        logger.info(f"Merged dataset shape: {merged_df.shape}")
        logger.info(f"Unique data sources: {merged_df['data_source'].unique().tolist()}")
        
        # Clean and standardize the merged dataset
        merged_df = self._clean_merged_dataset(merged_df)
        
        logger.info(f"Final dataset shape after cleaning: {merged_df.shape}")
        
        # Save the merged dataset for reproducibility
        merged_path = PROCESSED_DIR / f'ultimate_merged_dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        merged_df.to_csv(merged_path, index=False)
        logger.info(f"Ultimate merged dataset saved to: {merged_path}")
        
        return merged_df
    
    def _standardize_pagasa_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize PAGASA column names to match expected format."""
        # Common PAGASA column name variations
        column_mapping = {
            # Temperature columns
            'TEMP': 'temperature',
            'Temp': 'temperature',
            'Temperature': 'temperature',
            'AIR TEMP': 'temperature',
            'AIR_TEMP': 'temperature',
            
            # Humidity columns
            'RH': 'humidity',
            'Humidity': 'humidity',
            'REL HUM': 'humidity',
            'REL_HUM': 'humidity',
            
            # Precipitation columns
            'PRCP': 'precipitation',
            'Precip': 'precipitation',
            'Rainfall': 'precipitation',
            'RAIN': 'precipitation',
            'TOTAL_RAIN': 'precipitation',
            
            # Date columns
            'DATE': 'date',
            'DATE_OBS': 'date',
            'DATE_TIME': 'datetime',
            'DATETIME': 'datetime',
            
            # Flood indicator columns
            'FLOOD': 'flood',
            'Flood': 'flood',
            'FLOOD_INDICATOR': 'flood',
            'FLOOD_EVENT': 'flood',
        }
        
        # Apply mapping
        df = df.rename(columns=column_mapping)
        
        return df
    
    def _clean_merged_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the merged dataset."""
        logger.info("Cleaning merged dataset...")
        
        # Convert date columns if they exist
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        
        # Create year/month features if dates exist
        if 'date' in df.columns:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
        
        # Handle common data quality issues
        # Replace -999, -9999, etc. with NaN (common missing value codes in weather data)
        df = df.replace([-999, -9999, -999.0, -9999.0], np.nan)
        
        # Drop rows where critical features are missing
        critical_cols = ['temperature', 'humidity', 'precipitation']
        critical_present = [col for col in critical_cols if col in df.columns]
        
        if critical_present:
            df = df.dropna(subset=critical_present)
        
        # Ensure flood column exists and is binary
        if 'flood' in df.columns:
            # Convert to binary (0/1)
            df['flood'] = df['flood'].apply(lambda x: 1 if x in [1, 'Yes', 'yes', 'YES', 'True', 'true', 'TRUE', 'Y', 'y'] else 0)
            df['flood'] = df['flood'].astype(int)
        
        logger.info(f"Dataset cleaned, shape: {df.shape}")
        return df
    
    def engineer_ultimate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive feature engineering for ultimate model."""
        logger.info("Engineering ultimate features...")
        
        df = df.copy()
        
        # Rolling precipitation windows (if date information available)
        if 'date' in df.columns and 'precipitation' in df.columns:
            df = df.sort_values('date')
            
            # 3-day, 7-day, 14-day precipitation sums
            for window in [3, 7, 14]:
                col_name = f'precip_{window}day_sum'
                df[col_name] = df['precipitation'].rolling(window=window, min_periods=1).sum()
        
        # Advanced weather interactions
        if all(col in df.columns for col in ['temperature', 'humidity']):
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100
            df['heat_index'] = df['temperature'] + 0.5 * (df['humidity'] / 100)
            df['temp_range'] = df['temperature'].rolling(3, min_periods=1).max() - df['temperature'].rolling(3, min_periods=1).min()
        
        if all(col in df.columns for col in ['humidity', 'precipitation']):
            df['humidity_precip_interaction'] = df['humidity'] * np.log1p(df['precipitation'])
        
        if all(col in df.columns for col in ['temperature', 'precipitation']):
            df['temp_precip_interaction'] = df['temperature'] * np.log1p(df['precipitation'])
        
        # Monsoon season indicator (for Philippines)
        if 'month' in df.columns:
            df['is_monsoon_season'] = df['month'].isin([5, 6, 7, 8, 9, 10]).astype(int)
            df['monsoon_precip_interaction'] = df['is_monsoon_season'] * df['precipitation']
        
        # Seasonal indicators
        if 'month' in df.columns:
            df['season'] = df['month'].map({12: 'winter', 1: 'winter', 2: 'winter',
                                         3: 'spring', 4: 'spring', 5: 'spring',
                                         6: 'summer', 7: 'summer', 8: 'summer',
                                         9: 'autumn', 10: 'autumn', 11: 'autumn'}).astype('category')
        
        # Lagged features (if date available)
        if 'date' in df.columns and 'precipitation' in df.columns:
            df = df.sort_values('date')
            df['precip_lag1'] = df['precipitation'].shift(1)
            df['precip_lag2'] = df['precipitation'].shift(2)
            df['precip_lag3'] = df['precipitation'].shift(3)
        
        # Fill any remaining NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        logger.info(f"Feature engineering complete, now have {len(df.columns)} columns")
        return df
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_column: str = 'flood'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for ultimate training.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Tuple of (X, y)
        """
        # Engineer features
        df = self.engineer_ultimate_features(df)
        
        # Identify features to use (exclude target and metadata)
        exclude_cols = {target_column, 'date', 'datetime', 'data_source'}
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in ['float64', 'int64', 'int32', 'float32']]
        
        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
        
        # Prepare X and y
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Log feature summary
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        return metrics
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        use_grid_search: bool = False,
        use_time_split: bool = True,
        n_folds: int = 5,
        custom_params: Optional[Dict] = None
    ) -> RandomForestClassifier:
        """
        Train Random Forest model on ultimate dataset.
        
        Args:
            X: Feature matrix
            y: Target vector
            use_grid_search: Perform hyperparameter tuning
            use_time_split: Use time-series split (recommended for temporal data)
            n_folds: Number of CV folds
            custom_params: Custom model parameters
            
        Returns:
            Trained model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} | Test set: {len(X_test)}")
        
        if use_grid_search:
            logger.info("Performing hyperparameter optimization...")
            model = self._grid_search_train(X_train, y_train, n_folds)
        else:
            # Use optimized parameters for ultimate training
            params = custom_params or {
                'n_estimators': 500,  # Increased for ultimate model
                'max_depth': 20,      # Deeper trees for complex patterns
                'min_samples_split': 3,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': self.random_state,
                'n_jobs': -1
            }
            logger.info(f"Training with parameters: {params}")
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        self.training_metrics = metrics
        
        # Log results
        logger.info(f"\n{'='*60}")
        logger.info("ULTIMATE TRAINING RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall:    {metrics['recall']:.4f}")
        logger.info(f"F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        logger.info(f"{'='*60}\n")
        
        # Feature importance
        self._log_feature_importance(model)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            model, X, y, cv=n_folds, scoring='f1_weighted', n_jobs=-1
        )
        logger.info(f"Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        self.model = model
        return model
    
    def _grid_search_train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_folds: int
    ) -> RandomForestClassifier:
        """Perform grid search for hyperparameter optimization."""
        param_grid = {
            'n_estimators': [300, 500, 700],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        
        base_model = RandomForestClassifier(
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=StratifiedKFold(n_folds, shuffle=True, random_state=self.random_state),
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _log_feature_importance(self, model: RandomForestClassifier, top_n: int = 15):
        """Log top feature importances."""
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop {top_n} Feature Importances:")
        for _, row in importances.head(top_n).iterrows():
            logger.info(f"  {row['feature']:30s} {row['importance']:.4f}")
    
    def generate_shap_analysis(self, X: pd.DataFrame, max_samples: int = 500):
        """Generate SHAP explainability analysis."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not installed. Run: pip install shap")
            return None
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        logger.info("Generating SHAP analysis...")
        
        # Sample data
        if len(X) > max_samples:
            X_sample = X.sample(max_samples, random_state=self.random_state)
        else:
            X_sample = X
        
        # Create explainer and calculate SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        if PLOT_AVAILABLE:
            # Summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(self.reports_dir / 'ultimate_shap_summary.png', dpi=300)
            plt.close()
            
            # Bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(self.reports_dir / 'ultimate_shap_importance.png', dpi=300)
            plt.close()
            
            logger.info(f"SHAP plots saved to {self.reports_dir}")
        
        # Return mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        return dict(zip(self.feature_names, mean_shap))
    
    def save_model(self, version: Optional[int] = None) -> Path:
        """
        Save ultimate trained model and metadata.
        
        Args:
            version: Model version number (auto-incremented if None)
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("No model to save. Call train() first.")
        
        # Auto-increment version
        if version is None:
            version = self._get_next_version()
        
        # Save model with ultimate identifier
        model_filename = f'flood_rf_model_v{version}_ultimate.joblib'
        model_path = self.models_dir / model_filename
        joblib.dump(self.model, model_path)
        logger.info(f"Ultimate model saved: {model_path}")
        
        # Also save as latest
        latest_path = self.models_dir / 'flood_rf_model_latest.joblib'
        joblib.dump(self.model, latest_path)
        logger.info(f"Ultimate model saved as latest: {latest_path}")
        
        # Save metadata
        metadata = {
            'version': version,
            'model_type': 'RandomForestClassifier',
            'model_path': str(model_path),
            'created_at': datetime.now().isoformat(),
            'data_source': 'Ultimate Model - Combined All Available Datasets',
            'datasets_combined': self.discovered_datasets,
            'data_info': self.data_info,
            'features': self.feature_names,
            'feature_count': len(self.feature_names),
            'model_parameters': {
                'n_estimators': getattr(self.model, 'n_estimators', None),
                'max_depth': getattr(self.model, 'max_depth', None),
                'min_samples_split': getattr(self.model, 'min_samples_split', None),
                'min_samples_leaf': getattr(self.model, 'min_samples_leaf', None),
                'class_weight': str(getattr(self.model, 'class_weight', None))
            },
            'metrics': self.training_metrics,
            'feature_importance': dict(zip(
                self.feature_names,
                [float(x) for x in self.model.feature_importances_]
            ))
        }
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Ultimate model metadata saved: {metadata_path}")
        
        return model_path
    
    def _get_next_version(self) -> int:
        """Get next available version number."""
        existing = list(self.models_dir.glob('flood_rf_model_v*_ultimate.joblib'))
        if not existing:
            return 1
        
        versions = []
        for f in existing:
            try:
                v = int(f.stem.split('_v')[1].split('_')[0])
                versions.append(v)
            except (ValueError, IndexError):
                continue
        
        return max(versions) + 1 if versions else 1
    
    def generate_ultimate_training_report(self) -> str:
        """Generate a comprehensive ultimate training report."""
        report_lines = [
            "=" * 70,
            "ULTIMATE FLOOD PREDICTION MODEL - TRAINING REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
            "DATA SOURCES COMBINED",
            "-" * 40,
        ]
        
        # Add dataset information
        for source_type, files in self.discovered_datasets.items():
            report_lines.append(f"{source_type.upper()}:")
            for file in files:
                report_lines.append(f"  - {Path(file).name}")
            report_lines.append("")
        
        report_lines.extend([
            "DATASET SUMMARY",
            "-" * 40,
            f"Records: {self.data_info.get('total_records', 'N/A')}",
            f"Date Range: {self.data_info.get('date_range', 'N/A')}",
            f"Features: {len(self.feature_names)}",
            "",
            "MODEL PERFORMANCE",
            "-" * 40,
            f"Accuracy:  {self.training_metrics.get('accuracy', 0):.4f}",
            f"Precision: {self.training_metrics.get('precision', 0):.4f}",
            f"Recall:    {self.training_metrics.get('recall', 0):.4f}",
            f"F1 Score:  {self.training_metrics.get('f1_score', 0):.4f}",
            f"ROC-AUC:   {self.training_metrics.get('roc_auc', 0):.4f}",
            "",
            "TOP 10 FEATURES",
            "-" * 40,
        ])
        
        if self.model is not None:
            importances = sorted(
                zip(self.feature_names, self.model.feature_importances_),
                key=lambda x: x[1],
                reverse=True
            )
            for feat, imp in importances[:10]:
                report_lines.append(f"  {feat:30s} {imp:.4f}")
        
        report_lines.extend([
            "",
            "FUTURE-PROOFING CAPABILITIES",
            "-" * 40,
            "• Automatically discovers and incorporates new datasets",
            "• Maintains backward compatibility with existing models",
            "• Preserves all historical training data provenance",
            "• Adapts to new feature patterns in incoming data",
            "",
            "=" * 70,
        ])
        
        report = "\n".join(report_lines)
        
        # Save report
        report_path = self.reports_dir / f'ultimate_training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Ultimate report saved: {report_path}")
        
        return report


def main():
    """Main entry point for ultimate model training."""
    parser = argparse.ArgumentParser(
        description='Train ultimate flood prediction model using ALL available datasets'
    )
    parser.add_argument(
        '--grid-search',
        action='store_true',
        help='Perform hyperparameter optimization with GridSearchCV'
    )
    parser.add_argument(
        '--production',
        action='store_true',
        help='Full production pipeline (progressive + grid search + SHAP + report)'
    )
    parser.add_argument(
        '--progressive',
        action='store_true',
        help='Train all versions progressively (v1 → v2 → ... → ULTIMATE)'
    )
    parser.add_argument(
        '--latest-only',
        action='store_true',
        help='Only train the latest/best model version'
    )
    parser.add_argument(
        '--version',
        type=int,
        help='Model version number (default: auto-increment)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save the trained model'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = UltimateModelTrainer()
    
    # Determine training mode
    use_progressive = args.progressive or args.production
    use_grid_search = args.grid_search or args.production
    
    if use_progressive:
        # Progressive training: v1 → v2 → ... → ULTIMATE
        logger.info("\n" + "="*70)
        logger.info("ULTIMATE PROGRESSIVE MODEL TRAINING")
        logger.info("Training models from v1 (baseline) to ULTIMATE (best)")
        logger.info("="*70)
        
        results = trainer.train_progressive(
            use_grid_search=use_grid_search,
            n_folds=args.cv_folds,
            latest_only=args.latest_only
        )
        
        if results:
            # Generate SHAP for the best model if production mode
            if args.production and SHAP_AVAILABLE:
                best_version = max(results.values(), key=lambda x: x.metrics['f1_score'])
                best_model = joblib.load(best_version.model_path)
                trainer.model = best_model
                trainer.feature_names = best_version.features
                
                # Load the best version's data for SHAP
                best_config = next((v for v in VERSION_REGISTRY if v['version'] == best_version.version), None)
                if best_config:
                    df = pd.read_csv(BACKEND_DIR / best_config['data_file'])
                    X, _ = trainer.prepare_features(df)
                    trainer.generate_shap_analysis(X)
            
            logger.info("\n" + "="*70)
            logger.info("🏆 PROGRESSIVE TRAINING COMPLETE!")
            logger.info("="*70)
            logger.info(f"\nTrained {len(results)} model versions:")
            for v in sorted(results.values(), key=lambda x: x.version):
                marker = " ★ BEST" if v.is_best else ""
                logger.info(f"  v{v.version}: {v.name} (F1={v.metrics['f1_score']:.4f}){marker}")
            logger.info(f"\nLatest model: models/flood_rf_model_latest.joblib")
            logger.info("\nTo add new datasets:")
            logger.info("  1. Add data to data/processed/")
            logger.info("  2. Add entry to VERSION_REGISTRY in train_ultimate.py")
            logger.info("  3. Run: python scripts/train_ultimate.py --progressive")
    else:
        # Legacy: Single ultimate model training
        discovered = trainer.discover_datasets()
        
        if not discovered:
            logger.error("No datasets found! Cannot train ultimate model.")
            return None
        
        # Load and merge all datasets
        df = trainer.load_and_merge_all_datasets()
        
        # Store data info
        trainer.data_info = {
            'source_files': [str(f) for files in discovered.values() for f in files],
            'total_records': len(df),
            'date_range': f"{df.get('year', pd.Series()).min()}-{df.get('year', pd.Series()).max()}" if 'year' in df.columns else 'unknown',
            'columns': list(df.columns),
            'datasets_discovered': discovered
        }
        
        # Prepare features
        X, y = trainer.prepare_features(df)
        
        # Train model
        trainer.train(X, y, use_grid_search=use_grid_search)
        
        # Generate SHAP analysis for production
        if args.production and SHAP_AVAILABLE:
            trainer.generate_shap_analysis(X)
        
        # Generate ultimate report
        if args.production:
            report = trainer.generate_ultimate_training_report()
            print(report)
        
        # Save model
        if not args.no_save:
            model_path = trainer.save_model(version=args.version)
            logger.info(f"\n🏆 ULTIMATE TRAINING COMPLETE! Model saved to: {model_path}")
    
    return trainer


if __name__ == '__main__':
    main()