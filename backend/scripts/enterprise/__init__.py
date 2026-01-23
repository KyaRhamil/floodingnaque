"""
Enterprise Training Scripts Module
==================================

This package provides enterprise-grade ML training infrastructure for the
Floodingnaque flood prediction system.

.. module:: enterprise
   :synopsis: Enterprise ML training infrastructure.

.. moduleauthor:: Floodingnaque Team

Submodules
----------
data_validation
    Pandera-based schema validation and data quality checks.
logging_config
    Structured JSON logging with correlation IDs.
mlflow_tracking
    MLflow experiment tracking integration.
model_registry
    Multi-stage model lifecycle management.

Features
--------
- **Data Validation**: Schema validation, range checks, quality metrics
- **Experiment Tracking**: MLflow integration for reproducibility
- **Structured Logging**: JSON logs with correlation IDs
- **Model Registry**: Version management with staged promotion

Quick Start
-----------
::

    >>> from enterprise import (
    ...     FloodDataValidator,
    ...     MLflowTracker,
    ...     TrainingLogger,
    ...     ModelRegistry,
    ... )
    >>> # Validate training data
    >>> validator = FloodDataValidator()
    >>> validated_df, errors = validator.validate(df)
    >>> # Track experiments
    >>> tracker = MLflowTracker()
    >>> with tracker.start_run(run_name="experiment_001"):
    ...     tracker.log_metrics({'f1_score': 0.95})
"""

from .data_validation import (
    DataValidationError,
    FloodDataValidator,
    validate_training_data,
)
from .logging_config import (
    TrainingLogger,
    get_correlation_id,
    get_logger,
    new_correlation_id,
    set_correlation_id,
    setup_logging,
    timed,
)
from .mlflow_tracking import (
    MLflowTracker,
    create_tracker_from_config,
    log_training_run,
)
from .model_registry import (
    ModelRegistry,
    ModelStage,
    ModelVersion,
    PromotionCriteria,
    create_registry,
)

__all__ = [
    # Data validation
    "DataValidationError",
    "FloodDataValidator",
    "validate_training_data",
    # Logging
    "TrainingLogger",
    "get_correlation_id",
    "get_logger",
    "new_correlation_id",
    "set_correlation_id",
    "setup_logging",
    "timed",
    # MLflow
    "MLflowTracker",
    "create_tracker_from_config",
    "log_training_run",
    # Model registry
    "ModelRegistry",
    "ModelStage",
    "ModelVersion",
    "PromotionCriteria",
    "create_registry",
]
