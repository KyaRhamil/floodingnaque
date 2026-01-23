# Floodingnaque Scripts

Utility scripts for training, validation, data processing, and maintenance.

## 🚀 Quick Start - Unified CLI

The preferred way to use these scripts is through the unified CLI:

```bash
# From the backend directory
cd backend

# Show help
python -m scripts --help

# Training
python -m scripts train                          # Basic training
python -m scripts train --mode production        # Production-ready model
python -m scripts train --mode progressive       # All model versions

# Evaluation
python -m scripts evaluate                       # Basic evaluation
python -m scripts evaluate --robustness          # Full robustness suite
python -m scripts evaluate --thesis              # Thesis defense mode

# Validation
python -m scripts validate                       # Validate current model
python -m scripts validate --all                 # Validate all models

# Data Processing
python -m scripts data preprocess                # Preprocess raw data
python -m scripts data merge                     # Merge datasets

# Database
python -m scripts db backup                      # Backup database
python -m scripts db verify-rls                  # Verify RLS policies
```

## 📦 Programmatic Usage

```python
from scripts import UnifiedTrainer, TrainingMode, UnifiedEvaluator, EvaluationMode

# Train a production model
trainer = UnifiedTrainer(mode=TrainingMode.PRODUCTION)
result = trainer.train(grid_search=True)
print(f"Model saved: {result['model_path']}")
print(f"F1 Score: {result['metrics']['f1_score']:.4f}")

# Evaluate with robustness testing
evaluator = UnifiedEvaluator()
results = evaluator.evaluate(mode=EvaluationMode.ROBUSTNESS)
evaluator.print_summary()
```

## 📁 Directory Structure

```
scripts/
├── __init__.py              # Package exports
├── __main__.py              # CLI entry point
├── train_unified.py         # 🆕 Consolidated training module
├── evaluate_unified.py      # 🆕 Consolidated evaluation module
│
├── # Legacy Training Scripts (deprecated)
├── train.py                 # Basic training
├── train_pagasa.py          # PAGASA-enhanced
├── train_production.py      # Production-ready
├── train_progressive.py     # 8-phase progressive
├── train_enhanced.py        # Multi-level classification
├── train_enterprise.py      # MLflow integration
├── train_ultimate.py        # Full pipeline
├── train_with_production_data.py
├── progressive_train.py
│
├── # Legacy Evaluation Scripts (deprecated)
├── evaluate_model.py        # Basic evaluation
├── evaluate_robustness.py   # Robustness testing
├── validate_model.py        # Model validation
├── compare_models.py        # Model comparison
│
├── # Data Processing
├── preprocess_pagasa_data.py
├── preprocess_official_flood_records.py
├── merge_datasets.py
├── ingest_training_data.py
│
├── # Database & Infrastructure
├── backup_database.py
├── backup_database.sh
├── verify_rls.py
├── verify_supabase_schema.py
├── manage_partitions.py
├── security_scan.py
│
├── # Reports
├── generate_thesis_report.py
│
└── enterprise/              # Enterprise modules
    ├── __init__.py
    ├── data_validation.py
    ├── logging_config.py
    ├── mlflow_tracking.py
    └── model_registry.py
```

## 🎯 Training Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `basic` | Simple Random Forest | Quick training, testing |
| `pagasa` | PAGASA-enhanced | Multi-station weather data |
| `production` | Calibrated model | Deployment-ready |
| `progressive` | 8-phase training | Thesis demonstration |
| `enhanced` | Multi-level classification | Risk levels (LOW/MODERATE/HIGH) |
| `enterprise` | MLflow + Registry | Full MLOps pipeline |
| `ultimate` | Combined pipeline | Best model creation |

## 📊 Evaluation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `basic` | Metrics + confusion matrix | Quick check |
| `robustness` | Full suite + noise testing | Production validation |
| `thesis` | Complete defense analysis | Academic presentation |
| `temporal` | Train past, test future | Generalization check |
| `calibration` | Probability analysis | Confidence validation |

## ⚠️ Deprecation Notice

The following scripts are deprecated and will be removed in v2.0:

- `train.py` → Use `python -m scripts train`
- `train_pagasa.py` → Use `python -m scripts train --mode pagasa`
- `train_production.py` → Use `python -m scripts train --mode production`
- `evaluate_model.py` → Use `python -m scripts evaluate`
- `evaluate_robustness.py` → Use `python -m scripts evaluate --robustness`

They still work but will emit deprecation warnings.

## 🔧 Configuration

### Environment Variables

All scripts support configuration via `FLOODINGNAQUE_*` environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `FLOODINGNAQUE_MODELS_DIR` | Models directory | `backend/models` |
| `FLOODINGNAQUE_DATA_DIR` | Data directory | `backend/data` |
| `FLOODINGNAQUE_BACKUP_DIR` | Backup directory | `backend/backups` |
| `FLOODINGNAQUE_LOG_LEVEL` | Logging level | `INFO` |
| `FLOODINGNAQUE_MAX_RETRIES` | API retry attempts | `3` |
| `FLOODINGNAQUE_RETRY_DELAY` | Initial retry delay (seconds) | `1.0` |
| `FLOODINGNAQUE_MAX_BACKUPS` | Max backup files to keep | `10` |
| `FLOODINGNAQUE_DRY_RUN` | Default dry-run mode | `false` |
| `FLOODINGNAQUE_MLFLOW_URI` | MLflow tracking URI | `mlruns` |
| `FLOODINGNAQUE_CV_FOLDS` | Cross-validation folds | `10` |
| `FLOODINGNAQUE_RANDOM_STATE` | Random seed | `42` |

### Standardized CLI Arguments

All scripts follow a consistent CLI pattern:

| Flag | Short | Description |
|------|-------|-------------|
| `--verbose` | `-v` | Enable debug logging |
| `--output` | `-o` | Output path (file or directory) |
| `--force` | `-f` | Force overwrite without confirmation |
| `--dry-run` | | Show what would be done without executing |
| `--config` | `-c` | Configuration file path |

### Optional Dependencies

For advanced features, install optional dependencies:

```bash
pip install -r scripts/requirements-scripts.txt
```

This includes:
- `shap` - SHAP explainability analysis
- `mlflow` - Experiment tracking
- `pandera` - DataFrame validation
- `structlog` - Structured logging
- `tenacity` - API retry with exponential backoff
- `optuna` - Hyperparameter optimization

### YAML Configuration

Training configuration can be set via:

1. **CLI arguments**: `--grid-search`, `--cv-folds 10`
2. **TrainingConfig dataclass**: For programmatic use
3. **YAML config file**: `config/training_config.yaml`

Example programmatic configuration:

```python
from scripts.train_unified import UnifiedTrainer, TrainingMode, TrainingConfig

config = TrainingConfig(
    data_path="data/processed/custom_data.csv",
    grid_search=True,
    cv_folds=10,
    n_estimators=300,
    max_depth=20,
)

trainer = UnifiedTrainer(mode=TrainingMode.PRODUCTION, config=config)
result = trainer.train()
```

## 📈 Version Registry

Progressive training uses the following data versions:

| Version | Name | Data File | Description |
|---------|------|-----------|-------------|
| v1 | Baseline_2022 | cumulative_up_to_2022.csv | 2022 only |
| v2 | Extended_2023 | cumulative_up_to_2023.csv | 2022-2023 |
| v3 | Extended_2024 | cumulative_up_to_2024.csv | 2022-2024 |
| v4 | Full_Official_2025 | cumulative_up_to_2025.csv | 2022-2025 |
| v5 | PAGASA_Merged | pagasa_training_dataset.csv | PAGASA data |
| v6 | Ultimate_Combined | Multiple files | Best combined |

## 🧪 Testing

```bash
# Run script tests
pytest tests/test_scripts.py -v

# Test unified modules
python -c "from scripts import UnifiedTrainer, UnifiedEvaluator; print('OK')"
```

## 📝 Author

Floodingnaque Team - January 2026
