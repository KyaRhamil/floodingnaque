"""
Floodingnaque Configuration Module
==================================

This module provides centralized configuration management for the training pipeline.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


class Config:
    """
    Centralized configuration manager for Floodingnaque training pipeline.

    Supports:
    - YAML-based configuration
    - Environment variable overrides
    - Nested key access
    - Type-safe retrieval
    """

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._config:
            self.load()

    def load(self, config_path: Optional[Path] = None) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config file. Defaults to training_config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / "training_config.yaml"

        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

        # Apply environment variable overrides
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to config."""
        env_mappings = {
            # MLflow settings
            "FLOODINGNAQUE_MLFLOW_URI": ("mlflow", "tracking_uri"),
            "FLOODINGNAQUE_ENABLE_MLFLOW": ("general", "enable_mlflow"),
            # General settings
            "FLOODINGNAQUE_LOG_LEVEL": ("logging", "level"),
            "FLOODINGNAQUE_RANDOM_STATE": ("general", "random_state"),
            "FLOODINGNAQUE_CV_FOLDS": ("cross_validation", "folds"),
            # Directory settings
            "FLOODINGNAQUE_MODELS_DIR": ("registry", "models_dir"),
            "FLOODINGNAQUE_DATA_DIR": ("data", "raw_dir"),
            "FLOODINGNAQUE_BACKUP_DIR": ("backup", "backup_dir"),
            # API rate limiting
            "FLOODINGNAQUE_MAX_RETRIES": ("api", "max_retries"),
            "FLOODINGNAQUE_RETRY_DELAY": ("api", "retry_delay"),
            # Backup settings
            "FLOODINGNAQUE_MAX_BACKUPS": ("backup", "max_backups"),
        }

        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested(config_path, self._convert_type(value))

    def _convert_type(self, value: str) -> Any:
        """Convert string environment variable to appropriate type."""
        # Boolean
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        # String
        return value

    def _set_nested(self, path: tuple, value: Any) -> None:
        """Set a nested configuration value."""
        current = self._config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Get a configuration value by nested keys.

        Args:
            *keys: Nested keys to access (e.g., "model", "default_params", "n_estimators")
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            config.get("model", "default_params", "n_estimators")  # Returns 200
        """
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def get_model_params(self, use_defaults: bool = True) -> Dict[str, Any]:
        """Get model parameters for training."""
        if use_defaults:
            params = self.get("model", "default_params", default={}).copy()
        else:
            params = {}

        # Add random state
        params["random_state"] = self.get("general", "random_state", default=42)

        return params

    def get_grid_search_params(self) -> Dict[str, Any]:
        """Get grid search parameter grid."""
        return self.get("model", "grid_search", "param_grid", default={})

    def get_cv_config(self) -> Dict[str, Any]:
        """Get cross-validation configuration."""
        return {
            "n_splits": self.get("cross_validation", "folds", default=10),
            "shuffle": self.get("cross_validation", "shuffle", default=True),
            "random_state": self.get("general", "random_state", default=42),
        }

    def get_data_paths(self) -> Dict[str, Path]:
        """Get data directory paths."""
        backend_dir = Path(__file__).parent.parent
        return {
            "processed": backend_dir / self.get("data", "processed_dir", default="data/processed"),
            "raw": backend_dir / self.get("data", "raw_dir", default="data"),
            "models": backend_dir / self.get("registry", "models_dir", default="models"),
        }

    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow configuration."""
        return self.get("mlflow", default={})

    def get_version_config(self, version: int) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model version."""
        return self.get("versions", f"v{version}", default=None)

    def get_promotion_criteria(self, stage: str) -> Dict[str, float]:
        """Get promotion criteria for a stage."""
        return self.get("registry", "promotion_criteria", stage, default={})

    def get_feature_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Get feature validation ranges."""
        return self.get("validation", "feature_ranges", default={})

    @property
    def random_state(self) -> int:
        """Get random state for reproducibility."""
        return self.get("general", "random_state", default=42)

    @property
    def test_size(self) -> float:
        """Get test set size."""
        return self.get("data", "test_size", default=0.2)

    @property
    def cv_folds(self) -> int:
        """Get number of cross-validation folds."""
        return self.get("cross_validation", "folds", default=10)

    def __repr__(self) -> str:
        return f"Config(keys={list(self._config.keys())})"


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config(config_path: Optional[Path] = None) -> Config:
    """Reload configuration from file."""
    Config._config = {}
    config.load(config_path)
    return config
