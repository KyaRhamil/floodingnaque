"""
Enterprise Training Scripts Module
==================================

This package provides enterprise-grade ML training infrastructure:
- Centralized configuration management
- Experiment tracking with MLflow
- Data validation with Pandera
- Structured logging
- Model registry with staging
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
