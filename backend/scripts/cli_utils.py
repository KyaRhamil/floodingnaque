"""
CLI Utilities for Floodingnaque Scripts
=======================================

Provides standardized CLI argument definitions and utilities.

Standardized Flags:
    --verbose, -v    Enable verbose output
    --output, -o     Output path (file or directory)
    --force, -f      Force overwrite without confirmation
    --dry-run        Show what would be done without executing
    --config, -c     Configuration file path

Environment Variables:
    FLOODINGNAQUE_MODELS_DIR    Models directory (default: backend/models)
    FLOODINGNAQUE_DATA_DIR      Data directory (default: backend/data)
    FLOODINGNAQUE_BACKUP_DIR    Backup directory (default: backend/backups)
    FLOODINGNAQUE_LOG_LEVEL     Logging level (default: INFO)
    FLOODINGNAQUE_DRY_RUN       Default dry-run mode (default: false)
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

# Resolve paths relative to backend directory
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent


def get_env_path(env_var: str, default: Path) -> Path:
    """
    Get a path from environment variable or use default.

    Args:
        env_var: Environment variable name
        default: Default path if env var not set

    Returns:
        Resolved Path object
    """
    value = os.environ.get(env_var)
    if value:
        return Path(value).resolve()
    return default


def get_env_bool(env_var: str, default: bool = False) -> bool:
    """
    Get a boolean from environment variable.

    Args:
        env_var: Environment variable name
        default: Default value if env var not set

    Returns:
        Boolean value
    """
    value = os.environ.get(env_var, "").lower()
    if value in ("true", "1", "yes", "on"):
        return True
    elif value in ("false", "0", "no", "off"):
        return False
    return default


def get_env_int(env_var: str, default: int) -> int:
    """
    Get an integer from environment variable.

    Args:
        env_var: Environment variable name
        default: Default value if env var not set

    Returns:
        Integer value
    """
    value = os.environ.get(env_var)
    if value:
        try:
            return int(value)
        except ValueError:
            pass
    return default


# Standard directory paths with environment variable support
def get_models_dir() -> Path:
    """Get models directory from env or default."""
    return get_env_path("FLOODINGNAQUE_MODELS_DIR", BACKEND_DIR / "models")


def get_data_dir() -> Path:
    """Get data directory from env or default."""
    return get_env_path("FLOODINGNAQUE_DATA_DIR", BACKEND_DIR / "data")


def get_backup_dir() -> Path:
    """Get backup directory from env or default."""
    return get_env_path("FLOODINGNAQUE_BACKUP_DIR", BACKEND_DIR / "backups")


def get_log_level() -> str:
    """Get log level from env or default."""
    return os.environ.get("FLOODINGNAQUE_LOG_LEVEL", "INFO").upper()


def get_default_dry_run() -> bool:
    """Get default dry-run mode from env."""
    return get_env_bool("FLOODINGNAQUE_DRY_RUN", False)


# Standard paths dictionary
STANDARD_PATHS = {
    "MODELS_DIR": get_models_dir,
    "DATA_DIR": get_data_dir,
    "BACKUP_DIR": get_backup_dir,
    "PROCESSED_DIR": lambda: get_data_dir() / "processed",
}


def setup_logging(verbose: bool = False, level: Optional[str] = None) -> logging.Logger:
    """
    Setup logging with consistent format.

    Args:
        verbose: Enable verbose (DEBUG) logging
        level: Override log level (e.g., "DEBUG", "INFO", "WARNING")

    Returns:
        Configured logger
    """
    if level is None:
        level = "DEBUG" if verbose else get_log_level()

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def add_common_arguments(parser: argparse.ArgumentParser, include: Optional[list] = None) -> None:
    """
    Add standardized common arguments to a parser.

    Args:
        parser: ArgumentParser to add arguments to
        include: List of arguments to include. Options:
            - "verbose": --verbose/-v flag
            - "output": --output/-o path
            - "force": --force/-f flag
            - "dry-run": --dry-run flag
            - "config": --config/-c path
            - "yes": --yes/-y flag for non-interactive mode
            If None, adds all common arguments.
    """
    if include is None:
        include = ["verbose", "output", "force", "dry-run", "config", "yes"]

    if "verbose" in include:
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Enable verbose output (debug level logging)",
        )

    if "output" in include:
        parser.add_argument(
            "--output",
            "-o",
            type=str,
            default=None,
            help="Output path (file or directory)",
        )

    if "force" in include:
        parser.add_argument(
            "--force",
            "-f",
            action="store_true",
            help="Force overwrite without confirmation",
        )

    if "yes" in include:
        parser.add_argument(
            "--yes",
            "-y",
            action="store_true",
            help="Assume 'yes' to all prompts (non-interactive mode)",
        )

    if "dry-run" in include:
        default_dry_run = get_default_dry_run()
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=default_dry_run,
            help="Show what would be done without executing"
            + (" (default: enabled via env)" if default_dry_run else ""),
        )

    if "config" in include:
        parser.add_argument(
            "--config",
            "-c",
            type=str,
            default=None,
            help="Configuration file path",
        )


def add_database_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add database-related arguments to a parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Database URL (default: from DATABASE_URL env)",
    )
    parser.add_argument(
        "--sqlite",
        action="store_true",
        help="Force SQLite mode",
    )
    parser.add_argument(
        "--postgres",
        action="store_true",
        help="Force PostgreSQL mode",
    )


def create_standard_parser(
    description: str,
    include_common: Optional[list] = None,
    include_database: bool = False,
) -> argparse.ArgumentParser:
    """
    Create a standardized argument parser.

    Args:
        description: Parser description
        include_common: List of common arguments to include
        include_database: Whether to include database arguments

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    add_common_arguments(parser, include_common)

    if include_database:
        add_database_arguments(parser)

    return parser


def confirm_action(message: str, force: bool = False, yes: bool = False) -> bool:
    """
    Prompt user for confirmation unless force or yes flag is set.

    Args:
        message: Confirmation message to display
        force: If True, skip confirmation and return True (legacy support)
        yes: If True, skip confirmation and return True (non-interactive mode)

    Returns:
        True if action confirmed, False otherwise

    Example:
        if confirm_action("Delete all files?", yes=args.yes):
            # Proceed with deletion
    """
    if force or yes:
        return True

    response = input(f"{message} [y/N]: ").strip().lower()
    return response in ("y", "yes")


def confirm_overwrite(path: Path, force: bool = False, yes: bool = False) -> bool:
    """
    Confirm overwriting an existing file or directory.

    Args:
        path: Path to the file/directory that may be overwritten
        force: If True, skip confirmation (legacy support)
        yes: If True, skip confirmation (non-interactive mode)

    Returns:
        True if overwrite confirmed or file doesn't exist, False otherwise

    Example:
        if not confirm_overwrite(output_path, yes=args.yes):
            print("Operation cancelled.")
            sys.exit(0)
    """
    if not path.exists():
        return True

    if force or yes:
        return True

    response = input(f"'{path}' already exists. Overwrite? [y/N]: ").strip().lower()
    return response in ("y", "yes")


def print_env_config() -> None:
    """Print current environment configuration for debugging."""
    print("\nFloodingnaque Environment Configuration:")
    print("-" * 40)
    print(f"  FLOODINGNAQUE_MODELS_DIR: {os.environ.get('FLOODINGNAQUE_MODELS_DIR', '(not set)')}")
    print(f"  FLOODINGNAQUE_DATA_DIR: {os.environ.get('FLOODINGNAQUE_DATA_DIR', '(not set)')}")
    print(f"  FLOODINGNAQUE_BACKUP_DIR: {os.environ.get('FLOODINGNAQUE_BACKUP_DIR', '(not set)')}")
    print(f"  FLOODINGNAQUE_LOG_LEVEL: {os.environ.get('FLOODINGNAQUE_LOG_LEVEL', '(not set)')}")
    print(f"  FLOODINGNAQUE_DRY_RUN: {os.environ.get('FLOODINGNAQUE_DRY_RUN', '(not set)')}")
    print("-" * 40)
    print("Effective paths:")
    print(f"  Models: {get_models_dir()}")
    print(f"  Data: {get_data_dir()}")
    print(f"  Backups: {get_backup_dir()}")
    print(f"  Log Level: {get_log_level()}")
    print()


def get_database_type(database_url: Optional[str] = None) -> str:
    """
    Detect database type from URL or environment.

    Args:
        database_url: Database URL to check. If None, reads from DATABASE_URL env.

    Returns:
        "sqlite", "postgresql", or "unknown"
    """
    if database_url is None:
        database_url = os.environ.get("DATABASE_URL", "")

    if not database_url:
        return "sqlite"  # Default to SQLite for development

    if database_url.startswith("sqlite"):
        return "sqlite"
    elif "postgresql" in database_url or "postgres" in database_url:
        return "postgresql"
    else:
        return "unknown"


def is_postgresql(database_url: Optional[str] = None) -> bool:
    """Check if database is PostgreSQL."""
    return get_database_type(database_url) == "postgresql"


def is_sqlite(database_url: Optional[str] = None) -> bool:
    """Check if database is SQLite."""
    return get_database_type(database_url) == "sqlite"


# Export all utilities
__all__ = [
    "BACKEND_DIR",
    "SCRIPT_DIR",
    "STANDARD_PATHS",
    "get_env_path",
    "get_env_bool",
    "get_env_int",
    "get_models_dir",
    "get_data_dir",
    "get_backup_dir",
    "get_log_level",
    "get_default_dry_run",
    "setup_logging",
    "add_common_arguments",
    "add_database_arguments",
    "create_standard_parser",
    "confirm_action",
    "confirm_overwrite",
    "print_env_config",
    "get_database_type",
    "is_postgresql",
    "is_sqlite",
]
