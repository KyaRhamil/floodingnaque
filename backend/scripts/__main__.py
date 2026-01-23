#!/usr/bin/env python
"""
Floodingnaque Scripts CLI - Unified Command Line Interface
===========================================================

Central entry point for all Floodingnaque scripts.

Usage:
    python -m scripts train [OPTIONS]           # Model training
    python -m scripts evaluate [OPTIONS]        # Model evaluation
    python -m scripts validate [OPTIONS]        # Model validation
    python -m scripts data [OPTIONS]            # Data processing
    python -m scripts db [OPTIONS]              # Database utilities

Examples:
    # Training commands
    python -m scripts train                             # Basic training
    python -m scripts train --mode pagasa               # PAGASA-enhanced training
    python -m scripts train --mode production           # Production-ready model
    python -m scripts train --mode progressive          # Progressive training (all versions)
    python -m scripts train --mode enhanced             # Multi-level classification
    python -m scripts train --mode enterprise           # Enterprise with MLflow
    python -m scripts train --mode ultimate             # Full 8-phase training
    python -m scripts train --grid-search               # With hyperparameter tuning

    # Evaluation commands
    python -m scripts evaluate                          # Basic evaluation
    python -m scripts evaluate --robustness             # Full robustness tests
    python -m scripts evaluate --temporal               # Temporal validation only
    python -m scripts evaluate --model-path models/x.joblib

    # Validation commands
    python -m scripts validate                          # Validate current model
    python -m scripts validate --all                    # Validate all models

    # Data processing commands
    python -m scripts data preprocess                   # Preprocess raw data
    python -m scripts data merge                        # Merge datasets
    python -m scripts data ingest                       # Ingest training data

    # Database commands
    python -m scripts db backup                         # Backup database
    python -m scripts db verify-rls                     # Verify RLS policies
    python -m scripts db verify-schema                  # Verify Supabase schema

Author: Floodingnaque Team
Date: 2026-01-23
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Ensure the backend directory is in path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))


def create_train_parser(subparsers) -> argparse.ArgumentParser:
    """Create training subcommand parser."""
    train_parser = subparsers.add_parser(
        "train",
        help="Train flood prediction models",
        description="Unified training interface for all model types",
    )

    # Training mode
    train_parser.add_argument(
        "--mode",
        "-m",
        choices=["basic", "pagasa", "production", "progressive", "enhanced", "enterprise", "ultimate"],
        default="basic",
        help="Training mode (default: basic)",
    )

    # Data options
    train_parser.add_argument(
        "--data",
        "-d",
        type=str,
        help="Path to training data (default: auto-detect)",
    )

    train_parser.add_argument(
        "--version",
        "-v",
        type=int,
        choices=range(1, 9),
        help="Model version to train (for progressive/enterprise modes)",
    )

    # Hyperparameter options
    train_parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Enable grid search hyperparameter optimization",
    )

    train_parser.add_argument(
        "--randomized-search",
        action="store_true",
        help="Enable randomized search hyperparameter optimization",
    )

    train_parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )

    # Progressive/Ultimate options
    train_parser.add_argument(
        "--phase",
        type=int,
        choices=range(1, 9),
        help="Specific phase to train (progressive mode)",
    )

    train_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (skip Optuna optimization)",
    )

    train_parser.add_argument(
        "--with-smote",
        action="store_true",
        help="Enable SMOTENC for class imbalance",
    )

    # PAGASA options
    train_parser.add_argument(
        "--all-stations",
        action="store_true",
        help="Train with all PAGASA stations (PAGASA mode)",
    )

    # Enterprise options
    train_parser.add_argument(
        "--mlflow",
        action="store_true",
        default=True,
        help="Enable MLflow tracking (enterprise mode)",
    )

    train_parser.add_argument(
        "--promote",
        choices=["staging", "production"],
        help="Auto-promote model if criteria met",
    )

    # Output options
    train_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for models (default: models/)",
    )

    train_parser.add_argument(
        "--shap",
        action="store_true",
        help="Generate SHAP explainability analysis",
    )

    return train_parser


def create_evaluate_parser(subparsers) -> argparse.ArgumentParser:
    """Create evaluation subcommand parser."""
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate trained models",
        description="Unified evaluation interface with basic and robustness testing",
    )

    eval_parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        help="Path to model file (default: auto-detect latest)",
    )

    eval_parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        help="Path to evaluation data",
    )

    eval_parser.add_argument(
        "--robustness",
        "-r",
        action="store_true",
        help="Run full robustness evaluation suite",
    )

    eval_parser.add_argument(
        "--temporal",
        action="store_true",
        help="Run temporal validation (train 2022-2024, test 2025)",
    )

    eval_parser.add_argument(
        "--noise-test",
        action="store_true",
        help="Run input noise robustness test",
    )

    eval_parser.add_argument(
        "--calibration",
        action="store_true",
        help="Analyze probability calibration",
    )

    eval_parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for reports (default: reports/)",
    )

    eval_parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save evaluation plots",
    )

    return eval_parser


def create_validate_parser(subparsers) -> argparse.ArgumentParser:
    """Create validation subcommand parser."""
    val_parser = subparsers.add_parser(
        "validate",
        help="Validate model files and integrity",
        description="Model file validation and sanity checks",
    )

    val_parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        help="Path to model file to validate",
    )

    val_parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Validate all models in models directory",
    )

    val_parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple model versions",
    )

    val_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return val_parser


def create_data_parser(subparsers) -> argparse.ArgumentParser:
    """Create data processing subcommand parser."""
    data_parser = subparsers.add_parser(
        "data",
        help="Data processing utilities",
        description="Data preprocessing, merging, and ingestion",
    )

    data_subparsers = data_parser.add_subparsers(dest="data_command", help="Data command")

    # preprocess
    preprocess_parser = data_subparsers.add_parser(
        "preprocess",
        help="Preprocess raw data files",
    )
    preprocess_parser.add_argument(
        "--source",
        choices=["pagasa", "official", "all"],
        default="all",
        help="Data source to preprocess",
    )

    # merge
    merge_parser = data_subparsers.add_parser(
        "merge",
        help="Merge multiple datasets",
    )
    merge_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path",
    )

    # ingest
    ingest_parser = data_subparsers.add_parser(
        "ingest",
        help="Ingest training data into database",
    )

    return data_parser


def create_db_parser(subparsers) -> argparse.ArgumentParser:
    """Create database utilities subcommand parser."""
    db_parser = subparsers.add_parser(
        "db",
        help="Database utilities",
        description="Database backup, verification, and maintenance",
    )

    db_subparsers = db_parser.add_subparsers(dest="db_command", help="Database command")

    # backup
    backup_parser = db_subparsers.add_parser(
        "backup",
        help="Backup database",
    )
    backup_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Backup output path",
    )

    # verify-rls
    db_subparsers.add_parser(
        "verify-rls",
        help="Verify Row Level Security policies",
    )

    # verify-schema
    db_subparsers.add_parser(
        "verify-schema",
        help="Verify Supabase schema",
    )

    # partitions
    db_subparsers.add_parser(
        "partitions",
        help="Manage database partitions",
    )

    return db_parser


def run_train(args) -> int:
    """Execute training command."""
    print(f"🚂 Training mode: {args.mode}")

    # Route to appropriate training module
    if args.mode == "basic":
        from scripts.train import main as train_main

        # Build argv for train.py
        argv = []
        if args.data:
            argv.extend(["--data", args.data])
        if args.grid_search:
            argv.append("--grid-search")
        if args.cv_folds != 5:
            argv.extend(["--cv-folds", str(args.cv_folds)])

        sys.argv = ["train.py"] + argv
        result = train_main()
        return result if isinstance(result, int) else 0

    elif args.mode == "pagasa":
        from scripts.train_pagasa import main as pagasa_main

        argv = []
        if args.all_stations:
            argv.append("--all-stations")
        if args.grid_search:
            argv.append("--grid-search")

        sys.argv = ["train_pagasa.py"] + argv
        result = pagasa_main()
        return result if isinstance(result, int) else 0

    elif args.mode == "production":
        from scripts.train_production import main as production_main

        argv = []
        if args.data:
            argv.extend(["--data", args.data])
        if args.shap:
            argv.append("--shap")

        sys.argv = ["train_production.py", "--production"] + argv
        result = production_main()
        return result if isinstance(result, int) else 0

    elif args.mode == "progressive":
        from scripts.train_progressive import main as progressive_main

        argv = []
        if args.phase:
            argv.extend(["--phase", str(args.phase)])
        if args.quick:
            argv.append("--quick")
        if args.with_smote:
            argv.append("--with-smote")

        sys.argv = ["train_progressive.py"] + argv
        result = progressive_main()
        return result if isinstance(result, int) else 0

    elif args.mode == "enhanced":
        from scripts.train_enhanced import main as enhanced_main

        argv = []
        if args.randomized_search:
            argv.append("--randomized-search")

        sys.argv = ["train_enhanced.py"] + argv
        result = enhanced_main()
        return result if isinstance(result, int) else 0

    elif args.mode == "enterprise":
        from scripts.train_enterprise import main as enterprise_main

        argv = []
        if args.version:
            argv.extend(["--version", str(args.version)])
        if args.grid_search:
            argv.append("--grid-search")
        if args.promote:
            argv.extend(["--promote", args.promote])

        sys.argv = ["train_enterprise.py"] + argv
        result = enterprise_main()
        return result if isinstance(result, int) else 0

    elif args.mode == "ultimate":
        from scripts.train_ultimate import main as ultimate_main

        argv = []
        if args.grid_search:
            argv.append("--grid-search")

        sys.argv = ["train_ultimate.py", "--progressive"] + argv
        result = ultimate_main()
        return result if isinstance(result, int) else 0

    return 0


def run_evaluate(args) -> int:
    """Execute evaluation command."""
    print("📊 Running model evaluation...")

    if args.robustness or args.temporal or args.noise_test or args.calibration:
        # Use robustness evaluation
        from scripts.evaluate_robustness import main as robustness_main

        argv = []
        if args.model_path:
            argv.extend(["--model-path", args.model_path])

        sys.argv = ["evaluate_robustness.py"] + argv
        robustness_main()
        return 0
    else:
        # Use basic evaluation
        from scripts.evaluate_model import evaluate_model

        result = evaluate_model(
            model_path=args.model_path,
            data_path=args.data_path,
        )
        return result if isinstance(result, int) else 0


def run_validate(args) -> int:
    """Execute validation command."""
    import importlib

    print("✅ Running model validation...")

    if args.compare:
        compare_models_module = importlib.import_module("scripts.compare_models")

        if hasattr(compare_models_module, "main"):
            result = compare_models_module.main()
            return result if isinstance(result, int) else 0
        return 0
    else:
        validate_model_module = importlib.import_module("scripts.validate_model")

        argv = []
        if args.model_path:
            argv.extend(["--model-path", args.model_path])
        if args.all:
            argv.append("--all")

        sys.argv = ["validate_model.py"] + argv
        if hasattr(validate_model_module, "main"):
            result = validate_model_module.main()
            return result if isinstance(result, int) else 0
        return 0


def run_data(args) -> int:
    """Execute data processing command."""
    if not args.data_command:
        print("Error: Please specify a data command (preprocess, merge, ingest)")
        return 1

    if args.data_command == "preprocess":
        if args.source in ["pagasa", "all"]:
            from scripts.preprocess_pagasa_data import main as pagasa_main

            pagasa_main()
        if args.source in ["official", "all"]:
            from scripts.preprocess_official_flood_records import main as official_main

            official_main()
        return 0

    elif args.data_command == "merge":
        from scripts.merge_datasets import main as merge_main

        result = merge_main()
        return result if isinstance(result, int) else 0

    elif args.data_command == "ingest":
        from scripts.ingest_training_data import main as ingest_main

        result = ingest_main()
        return result if isinstance(result, int) else 0

    return 0


def run_db(args) -> int:
    """Execute database command."""
    if not args.db_command:
        print("Error: Please specify a db command (backup, verify-rls, verify-schema)")
        return 1

    if args.db_command == "backup":
        from scripts.backup_database import main as backup_main

        result = backup_main()
        return result if isinstance(result, int) else 0

    elif args.db_command == "verify-rls":
        from scripts.verify_rls import main as rls_main

        result = rls_main()
        return result if isinstance(result, int) else 0

    elif args.db_command == "verify-schema":
        from scripts.verify_supabase_schema import main as schema_main

        result = schema_main()
        return result if isinstance(result, int) else 0

    elif args.db_command == "partitions":
        from scripts.manage_partitions import main as partition_main

        result = partition_main()
        return result if isinstance(result, int) else 0

    return 0


def print_banner():
    """Print CLI banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║          🌊 FLOODINGNAQUE SCRIPTS CLI v1.0.0 🌊                ║
║         Unified Command Line Interface for Scripts             ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m scripts",
        description="Floodingnaque Scripts CLI - Unified command line interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts train                    # Basic training
  python -m scripts train --mode progressive # Progressive training
  python -m scripts evaluate --robustness    # Full evaluation suite
  python -m scripts validate --all           # Validate all models
  python -m scripts data preprocess          # Preprocess data
  python -m scripts db backup                # Backup database
        """,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Floodingnaque Scripts CLI v1.0.0",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress banner output",
    )

    # Create subparsers for main commands
    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
    )

    # Register all subcommands
    create_train_parser(subparsers)
    create_evaluate_parser(subparsers)
    create_validate_parser(subparsers)
    create_data_parser(subparsers)
    create_db_parser(subparsers)

    # Parse arguments
    args = parser.parse_args(argv)

    if not args.quiet:
        print_banner()

    # Route to appropriate command handler
    if args.command == "train":
        return run_train(args)
    elif args.command == "evaluate":
        return run_evaluate(args)
    elif args.command == "validate":
        return run_validate(args)
    elif args.command == "data":
        return run_data(args)
    elif args.command == "db":
        return run_db(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
