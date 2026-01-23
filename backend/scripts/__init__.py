"""
Floodingnaque Scripts Package
=============================

Utility scripts for training, validation, data processing, and maintenance.

Unified CLI Interface:
    python -m scripts train              # Model training
    python -m scripts evaluate           # Model evaluation
    python -m scripts validate           # Model validation
    python -m scripts data preprocess    # Data processing
    python -m scripts db backup          # Database utilities

Modules:
    train_unified      - Consolidated training with multiple modes
    evaluate_unified   - Consolidated evaluation with robustness testing
    __main__           - CLI dispatcher

Legacy scripts are still available but may be deprecated in future versions.
Use the unified CLI or modules for new development.

Example:
    from scripts.train_unified import UnifiedTrainer, TrainingMode
    from scripts.evaluate_unified import UnifiedEvaluator, EvaluationMode

    # Train a production model
    trainer = UnifiedTrainer(mode=TrainingMode.PRODUCTION)
    result = trainer.train(grid_search=True)

    # Evaluate with robustness testing
    evaluator = UnifiedEvaluator()
    results = evaluator.evaluate(mode=EvaluationMode.ROBUSTNESS)
"""

from scripts.evaluate_unified import EvaluationMode, UnifiedEvaluator

# Unified modules (recommended)
from scripts.train_unified import TrainingConfig, TrainingMode, UnifiedTrainer

__all__ = [
    # Training
    "UnifiedTrainer",
    "TrainingMode",
    "TrainingConfig",
    # Evaluation
    "UnifiedEvaluator",
    "EvaluationMode",
]

__version__ = "1.0.0"
