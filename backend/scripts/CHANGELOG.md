# Scripts Changelog

All notable changes to the Floodingnaque scripts will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `preprocessing_common.py`: Extracted common preprocessing utilities from `preprocess_pagasa_data.py` and `preprocess_official_flood_records.py`
- `--interactive` flag in `generate_thesis_report.py` for Plotly interactive visualizations
- `--yes/-y` flag in `cli_utils.py` for non-interactive mode
- `confirm_overwrite()` function in `cli_utils.py` for file overwrite confirmation

### Changed
- Updated `cli_utils.py` to include `--yes/-y` in standard common arguments

## [1.5.0] - 2026-01-15

### Added
- `enterprise/` module with production-grade ML infrastructure:
  - `data_validation.py`: Pandera-based schema validation
  - `logging_config.py`: Structured JSON logging with correlation IDs
  - `mlflow_tracking.py`: MLflow experiment tracking integration
  - `model_registry.py`: Multi-stage model lifecycle management
- Sphinx-compatible docstrings for all enterprise modules
- `security_scan.py`: Comprehensive security scanning script

### Changed
- Enhanced docstrings across enterprise modules for Sphinx documentation

## [1.4.0] - 2026-01-10

### Added
- `train_enterprise.py`: Enterprise training script with MLflow integration
- `train_production.py`: Production-ready training with model registry
- `evaluate_unified.py`: Unified model evaluation script
- `train_ultimate.py`: Complete training pipeline with all features

### Changed
- Standardized CLI arguments across all training scripts

## [1.3.0] - 2025-12-15

### Added
- `preprocess_pagasa_data.py`: PAGASA weather data preprocessing
- `preprocess_official_flood_records.py`: Parañaque flood records preprocessing
- `progressive_train.py`: Progressive training with cumulative datasets
- `train_pagasa.py`: Training with PAGASA weather data

### Changed
- Updated `cli_utils.py` with environment variable support

## [1.2.0] - 2025-11-01

### Added
- `generate_thesis_report.py`: Thesis-ready visualization generator
- `compare_models.py`: Model comparison utilities
- `evaluate_robustness.py`: Robustness evaluation script
- `validate_model.py`: Model validation utilities

### Changed
- Improved error handling in training scripts
- Enhanced logging across all scripts

## [1.1.0] - 2025-09-15

### Added
- `train_enhanced.py`: Enhanced training with cross-validation
- `train_unified.py`: Unified training pipeline
- `evaluate_model.py`: Model evaluation script
- `merge_datasets.py`: Dataset merging utilities

### Changed
- Standardized output formats across scripts

## [1.0.0] - 2025-06-01

### Added
- Initial scripts release
- `train.py`: Basic Random Forest training
- `ingest_training_data.py`: Data ingestion utilities
- `backup_database.py`: Database backup script
- `cli_utils.py`: CLI utilities module

---

## Script Categories

### Training Scripts
| Script | Description | Version |
|--------|-------------|---------|
| `train.py` | Basic Random Forest training | 1.0.0 |
| `train_enhanced.py` | Training with cross-validation | 1.1.0 |
| `train_unified.py` | Unified training pipeline | 1.1.0 |
| `train_pagasa.py` | Training with PAGASA data | 1.3.0 |
| `train_progressive.py` | Progressive cumulative training | 1.3.0 |
| `train_enterprise.py` | Enterprise training with MLflow | 1.4.0 |
| `train_production.py` | Production-ready training | 1.4.0 |
| `train_ultimate.py` | Complete training pipeline | 1.4.0 |

### Preprocessing Scripts
| Script | Description | Version |
|--------|-------------|---------|
| `preprocess_pagasa_data.py` | PAGASA weather data preprocessing | 1.3.0 |
| `preprocess_official_flood_records.py` | Parañaque flood records preprocessing | 1.3.0 |
| `preprocessing_common.py` | Shared preprocessing utilities | 1.5.0 |

### Evaluation Scripts
| Script | Description | Version |
|--------|-------------|---------|
| `evaluate_model.py` | Basic model evaluation | 1.1.0 |
| `evaluate_robustness.py` | Robustness testing | 1.2.0 |
| `evaluate_unified.py` | Unified evaluation | 1.4.0 |
| `compare_models.py` | Model comparison | 1.2.0 |
| `validate_model.py` | Model validation | 1.2.0 |

### Utility Scripts
| Script | Description | Version |
|--------|-------------|---------|
| `cli_utils.py` | CLI utilities | 1.0.0 |
| `backup_database.py` | Database backup | 1.0.0 |
| `ingest_training_data.py` | Data ingestion | 1.0.0 |
| `merge_datasets.py` | Dataset merging | 1.1.0 |
| `security_scan.py` | Security scanning | 1.5.0 |
| `generate_thesis_report.py` | Thesis visualizations | 1.2.0 |

### Enterprise Modules
| Module | Description | Version |
|--------|-------------|---------|
| `enterprise/data_validation.py` | Pandera schema validation | 1.5.0 |
| `enterprise/logging_config.py` | Structured logging | 1.5.0 |
| `enterprise/mlflow_tracking.py` | MLflow integration | 1.5.0 |
| `enterprise/model_registry.py` | Model lifecycle management | 1.5.0 |

---

## Migration Notes

### From 1.4.x to 1.5.x
- Import common preprocessing functions from `preprocessing_common.py`
- Use `--yes/-y` flag for non-interactive CI/CD pipelines
- Use `--interactive` flag in `generate_thesis_report.py` for Plotly charts

### From 1.3.x to 1.4.x
- Consider using `train_enterprise.py` for MLflow tracking
- Use `evaluate_unified.py` for comprehensive evaluations

### From 1.2.x to 1.3.x
- PAGASA data now uses standardized preprocessing
- Use cumulative datasets for progressive training
