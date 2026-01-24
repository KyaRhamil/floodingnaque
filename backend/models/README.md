# Floodingnaque ML Models

This directory contains trained machine learning models, model artifacts, and the training pipeline for flood prediction.

## Directory Structure

```
models/
├── README.md                    # This documentation
├── run_training_pipeline.ps1    # Main training orchestration script
├── archive/                     # Archived/deprecated model versions
│   └── .gitkeep
└── data/                        # Model-specific data cache
    └── earthengine_cache/       # Google Earth Engine data cache
```

## Model Versioning

Models follow semantic versioning: `vMAJOR.MINOR` (e.g., `v5`, `v6`)

| Version | Description | Status |
|---------|-------------|--------|
| v6 | Current production model with enhanced features | **Active** |
| v5 | Previous stable version | Archived |

### Version Naming Convention

- **Major version** (v5 → v6): Significant architecture changes, new feature sets, or breaking changes
- **Minor updates**: Incremental improvements, retraining with new data

### Model File Naming Pattern

```
flood_model_v{VERSION}_{YYYYMMDD}.joblib
```

Example: `flood_model_v6_20260124.joblib`

## Expected Artifacts

After training, the following artifacts are generated:

### Model Files
- `flood_model_v{X}.joblib` - Serialized trained model
- `scaler_v{X}.joblib` - Feature scaler/preprocessor
- `feature_names_v{X}.json` - List of features used

### Reports (in `../reports/`)
- `evaluation_report_{YYYYMMDD}.json` - Model performance metrics
- `confusion_matrix.png` - Classification confusion matrix
- `feature_importance.png` - Feature importance chart
- `roc_curve.png` - ROC curve visualization
- `precision_recall_curve.png` - PR curve
- `learning_curves.png` - Training/validation curves

### Metadata
- `model_metadata.json` - Training parameters, data version, timestamp

## Training Pipeline

### Quick Start

```powershell
# Navigate to backend directory
cd backend

# Quick training (for testing)
.\models\run_training_pipeline.ps1 -Quick

# Full training (production)
.\models\run_training_pipeline.ps1 -Full

# Progressive/Ultimate training (comprehensive)
.\models\run_training_pipeline.ps1 -Ultimate
```

### Training Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `-Quick` | Fast training with reduced hyperparameter search | Development, testing |
| `-Full` | Standard production training | Regular model updates |
| `-PAGASA` | Training with PAGASA weather data integration | Weather service integration |
| `-Ultimate` / `-Progressive` | Comprehensive training with all optimizations | Major releases |

### Data Ingestion

```powershell
# Ingest from all sources
.\models\run_training_pipeline.ps1 -Ingest

# Specific sources
.\models\run_training_pipeline.ps1 -IngestMeteostat  # Free weather data
.\models\run_training_pipeline.ps1 -IngestGoogle     # Google Earth Engine
.\models\run_training_pipeline.ps1 -IngestTides      # WorldTides API

# Custom date range
.\models\run_training_pipeline.ps1 -Ingest -IngestDays 60
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-CVFolds` | 10 | Cross-validation folds |
| `-Seed` | 42 | Random seed for reproducibility |
| `-DataDir` | data | Input data directory |
| `-ModelDir` | models | Model output directory |
| `-ReportDir` | reports | Report output directory |
| `-DryRun` | false | Preview actions without execution |
| `-JsonLogs` | false | Output logs in JSON format (CI-friendly) |

## Configuration

Training parameters are centralized in `../config/training_config.yaml`:

```yaml
# Key sections:
general:           # Project settings, random state, logging
data:              # Data paths, feature definitions, quality thresholds
model:             # Model type, hyperparameters, optimization settings
training:          # Training modes, validation settings
```

### Feature Sets

The model uses the following feature categories:

1. **Core Features**: temperature, humidity, precipitation
2. **Interaction Features**: temp_humidity_interaction, saturation_risk, etc.
3. **Rolling Features**: precip_3day_sum, precip_7day_sum, rain_streak, etc.
4. **Categorical Features**: is_monsoon_season, month, weather_type, station_id

## Model Architecture

- **Algorithm**: Random Forest Classifier
- **Class Balancing**: balanced_subsample (handles class imbalance per tree)
- **Optimization**: Optuna TPE sampler with median pruning
- **Cross-Validation**: Stratified K-Fold (default 10 folds)

### Default Hyperparameters

```yaml
n_estimators: 200
max_depth: 15
min_samples_split: 5
min_samples_leaf: 2
class_weight: balanced_subsample
max_features: sqrt
```

## MLflow Integration

When enabled (`enable_mlflow: true` in config), training logs:
- Parameters
- Metrics (accuracy, precision, recall, F1, AUC)
- Artifacts (model files, plots)
- Model registry entries

Access MLflow UI: `http://localhost:5000`

## Archiving Models

Old models should be moved to the `archive/` directory:

```powershell
# Archive old model version
Move-Item flood_model_v5.joblib archive/
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Run `pip install -r requirements.txt`
2. **Data not found**: Ensure `data/processed/` contains training data
3. **Memory issues**: Use `-Quick` mode or reduce `n_estimators`
4. **PowerShell version**: Requires PowerShell 7+

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Critical failure |
| 2 | Completed with warnings |

## Related Documentation

- [Training Configuration](../config/training_config.yaml)
- [Data Schema](../data/SCHEMA.md)
- [Reports README](../reports/README.md)
- [Centralized Documentation](../docs/CENTRALIZED_DOCUMENTATION.md)
