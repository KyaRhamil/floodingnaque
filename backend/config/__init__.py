"""
Floodingnaque Configuration Module
==================================

This module provides centralized configuration management for the training pipeline.

Supports:
- YAML-based configuration with environment-specific overrides
- Pydantic schema validation on load
- Environment variable substitution
- Nested key access
- Type-safe retrieval
- Hot-reload via SIGHUP signal (Unix) or reload endpoint

Environment Configuration:
    Set FLOODINGNAQUE_ENV to one of: development, staging, production
    Default: development (when not set)

Environment Variables:
    FLOODINGNAQUE_ENV          - Environment name (development/staging/production)
    FLOODINGNAQUE_MLFLOW_URI   - MLflow tracking server URI
    FLOODINGNAQUE_MODELS_DIR   - Models directory path
    FLOODINGNAQUE_DATA_DIR     - Raw data directory path
    FLOODINGNAQUE_PROCESSED_DIR - Processed data directory path
    FLOODINGNAQUE_LOG_DIR      - Log directory path
    FLOODINGNAQUE_LOG_LEVEL    - Logging level (DEBUG/INFO/WARNING/ERROR)
    FLOODINGNAQUE_ENABLE_MLFLOW - Enable/disable MLflow tracking
    FLOODINGNAQUE_RANDOM_STATE - Random seed for reproducibility
    FLOODINGNAQUE_CV_FOLDS     - Number of cross-validation folds
    FLOODINGNAQUE_BACKUP_DIR   - Backup directory path
    FLOODINGNAQUE_MAX_BACKUPS  - Maximum number of backups to retain
    FLOODINGNAQUE_MAX_RETRIES  - Maximum API retry attempts
    FLOODINGNAQUE_RETRY_DELAY  - Delay between API retries (seconds)
    FLOODINGNAQUE_VALIDATE_CONFIG - Enable/disable schema validation (default: true)
"""

import logging
import os
import platform
import re
import signal
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

__all__ = [
    "Config",
    "ConfigurationError",
    "config",
    "get_config",
    "get_environment",
    "reload_config",
    "register_reload_callback",
    "unregister_reload_callback",
    "setup_signal_handlers",
    "SCHEMA_VALIDATION_AVAILABLE",
    "ENCRYPTION_AVAILABLE",
    "RESOURCE_DETECTION_AVAILABLE",
    "FEATURE_VALIDATION_AVAILABLE",
]

# Optional Pydantic import for schema validation
try:
    from config.schema import ConfigSchema, ConfigValidationError, validate_config

    SCHEMA_VALIDATION_AVAILABLE = True
except ImportError:
    SCHEMA_VALIDATION_AVAILABLE = False
    ConfigSchema = None
    ConfigValidationError = Exception
    validate_config = None

# Optional encryption support
try:
    from config.encryption import ConfigEncryption, get_encryptor

    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False
    ConfigEncryption = None
    get_encryptor = None

# Optional resource detection
try:
    from config.resource_detection import (
        apply_resource_detection,
        detect_resources,
        get_optimal_workers,
        get_safe_memory_limit,
    )

    RESOURCE_DETECTION_AVAILABLE = True
except ImportError:
    RESOURCE_DETECTION_AVAILABLE = False
    apply_resource_detection = None
    detect_resources = None

# Optional feature validation
try:
    from config.feature_validation import FeatureValidator, validate_config_features

    FEATURE_VALIDATION_AVAILABLE = True
except ImportError:
    FEATURE_VALIDATION_AVAILABLE = False
    FeatureValidator = None
    validate_config_features = None

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


# Reload callbacks registry
_reload_callbacks: List[Callable[["Config"], None]] = []


def register_reload_callback(callback: Callable[["Config"], None]) -> None:
    """
    Register a callback to be called when configuration is reloaded.

    Args:
        callback: Function that accepts the Config instance
    """
    _reload_callbacks.append(callback)


def unregister_reload_callback(callback: Callable[["Config"], None]) -> None:
    """Remove a registered reload callback."""
    if callback in _reload_callbacks:
        _reload_callbacks.remove(callback)


class Config:
    """
    Centralized configuration manager for Floodingnaque training pipeline.

    Supports:
    - YAML-based configuration
    - Environment-specific config files (development.yaml, staging.yaml, production.yaml)
    - Pydantic schema validation (optional, enabled by default)
    - Environment variable overrides and substitution
    - Nested key access
    - Type-safe retrieval
    - Hot-reload via SIGHUP signal or reload() method
    """

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}
    _validated_config: Optional[ConfigSchema] = None
    _environment: str = "development"
    _config_dir: Path = Path(__file__).parent
    _config_path: Optional[Path] = None
    _reload_lock: threading.Lock = threading.Lock()
    _validate_schema: bool = True

    # Pattern for environment variable substitution: ${VAR_NAME:-default_value}
    _env_pattern = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")

    def __new__(cls):
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._config:
            self.load()

    @property
    def environment(self) -> str:
        """Get current environment name."""
        return self._environment

    def load(self, config_path: Optional[Path] = None, environment: Optional[str] = None) -> None:
        """
        Load configuration from YAML file with environment-specific overrides.

        Args:
            config_path: Path to base config file. Defaults to training_config.yaml
            environment: Environment name (development/staging/production).
                        Defaults to FLOODINGNAQUE_ENV or 'development'

        Raises:
            ConfigurationError: If config file not found or validation fails
        """
        with self._reload_lock:
            # Determine environment
            self._environment = environment or os.environ.get("FLOODINGNAQUE_ENV", "development")

            # Check if schema validation is enabled
            self._validate_schema = os.environ.get("FLOODINGNAQUE_VALIDATE_CONFIG", "true").lower() == "true"

            # Load base configuration
            if config_path is None:
                config_path = self._config_dir / "training_config.yaml"

            self._config_path = config_path

            if not config_path.exists():
                raise ConfigurationError(f"Configuration file not found: {config_path}")

            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f)

            # Load environment-specific overrides
            self._load_environment_config()

            # Expand environment variables in config values
            self._expand_env_vars(self._config)

            # Apply explicit environment variable overrides
            self._apply_env_overrides()

            # Validate configuration against schema
            self._validated_config = None
            if self._validate_schema and SCHEMA_VALIDATION_AVAILABLE:
                try:
                    self._validated_config = validate_config(self._config)
                    logger.info(f"Configuration validated successfully for environment: {self._environment}")
                except ConfigValidationError as e:
                    logger.error(f"Configuration validation failed: {e}")
                    if os.environ.get("FLOODINGNAQUE_STRICT_VALIDATION", "false").lower() == "true":
                        raise ConfigurationError(f"Configuration validation failed: {e}") from e
                    logger.warning("Continuing with unvalidated configuration (strict validation disabled)")
            elif not SCHEMA_VALIDATION_AVAILABLE:
                logger.debug("Schema validation unavailable (pydantic not installed)")

            # Apply resource detection for auto-values (-1)
            if RESOURCE_DETECTION_AVAILABLE:
                self._apply_resource_detection()

            # Decrypt encrypted values if encryption is available
            if ENCRYPTION_AVAILABLE:
                self._decrypt_config_values()

            logger.info(f"Configuration loaded for environment: {self._environment}")

    def _apply_resource_detection(self) -> None:
        """Apply auto-detection for resource settings."""
        resources = self._config.get("resources", {})

        # Auto-detect max_workers
        if resources.get("max_workers") == -1:
            detected = get_optimal_workers(task_type="cpu_bound", leave_free=1)
            resources["max_workers"] = detected
            resources["_max_workers_auto"] = True
            logger.debug(f"Auto-detected max_workers: {detected}")

        # Auto-detect max_memory_gb
        if resources.get("max_memory_gb") == -1:
            detected = get_safe_memory_limit(fraction=0.8, min_free_gb=2.0)
            resources["max_memory_gb"] = detected
            resources["_max_memory_gb_auto"] = True
            logger.debug(f"Auto-detected max_memory_gb: {detected}")

        self._config["resources"] = resources

    def _decrypt_config_values(self) -> None:
        """Decrypt ENC[] wrapped values in config."""
        try:
            encryptor = get_encryptor()
            if encryptor.has_key:
                self._config = encryptor.process_config(self._config, decrypt=True)
                logger.debug("Decrypted encrypted config values")
        except Exception as e:
            logger.warning(f"Failed to decrypt config values: {e}")

    def _load_environment_config(self) -> None:
        """Load and merge environment-specific configuration."""
        env_config_path = self._config_dir / f"{self._environment}.yaml"

        if env_config_path.exists():
            with open(env_config_path, "r") as f:
                env_config = yaml.safe_load(f)
                if env_config:
                    self._deep_merge(self._config, env_config)

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Recursively merge override into base dictionary."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def _expand_env_vars(self, config: Any) -> Any:
        """Recursively expand environment variables in config values."""
        if isinstance(config, dict):
            for key, value in config.items():
                config[key] = self._expand_env_vars(value)
        elif isinstance(config, list):
            for i, item in enumerate(config):
                config[i] = self._expand_env_vars(item)
        elif isinstance(config, str):
            # Replace ${VAR:-default} patterns
            def replace_env_var(match):
                var_name = match.group(1)
                default = match.group(2) if match.group(2) is not None else ""
                return os.environ.get(var_name, default)

            config = self._env_pattern.sub(replace_env_var, config)
        return config

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
            # Processed data directory
            "FLOODINGNAQUE_PROCESSED_DIR": ("data", "processed_dir"),
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
        """Get data directory paths, resolving relative paths from backend directory."""
        backend_dir = Path(__file__).parent.parent

        processed_dir = self.get("data", "processed_dir", default="data/processed")
        raw_dir = self.get("data", "raw_dir", default="data")
        models_dir = self.get("registry", "models_dir", default="models")

        # Handle absolute vs relative paths
        def resolve_path(path_str: str) -> Path:
            path = Path(path_str)
            if path.is_absolute():
                return path
            return backend_dir / path

        return {
            "processed": resolve_path(processed_dir),
            "raw": resolve_path(raw_dir),
            "models": resolve_path(models_dir),
            "backend": backend_dir,
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

    def validate_features(self, strict: bool = False) -> Optional[Any]:
        """
        Validate feature references in the configuration.

        Args:
            strict: If True, raise error on validation failures

        Returns:
            FeatureValidationResult if validation available, None otherwise
        """
        if not FEATURE_VALIDATION_AVAILABLE:
            logger.debug("Feature validation not available")
            return None

        try:
            result = validate_config_features(self._config, strict=strict)
            if not result.valid:
                logger.warning(f"Feature validation issues: {result}")
            return result
        except Exception as e:
            logger.warning(f"Feature validation error: {e}")
            return None

    def get_resource_info(self) -> Optional[Dict[str, Any]]:
        """
        Get detected system resource information.

        Returns:
            Resource info dict if detection available, None otherwise
        """
        if not RESOURCE_DETECTION_AVAILABLE:
            return None

        try:
            resources = detect_resources()
            return resources.to_dict()
        except Exception as e:
            logger.warning(f"Resource detection error: {e}")
            return None

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

    @property
    def validated(self) -> Optional[ConfigSchema]:
        """
        Get the validated configuration schema object.

        Returns:
            ConfigSchema if validation was successful and enabled, None otherwise
        """
        return self._validated_config

    @property
    def is_validated(self) -> bool:
        """Check if configuration has been validated against schema."""
        return self._validated_config is not None

    def reload(self) -> None:
        """
        Reload configuration from disk.

        This method re-reads and validates the configuration files,
        then notifies all registered callbacks.
        """
        logger.info("Reloading configuration...")
        Config._config = {}
        Config._validated_config = None
        self.load(self._config_path, self._environment)

        # Notify all registered callbacks
        for callback in _reload_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Error in reload callback: {e}")

    def __repr__(self) -> str:
        validated_str = "validated" if self.is_validated else "unvalidated"
        return f"Config(env={self._environment}, {validated_str}, keys={list(self._config.keys())})"


# Global config instance
config = Config()


def _sighup_handler(signum: int, frame: Any) -> None:
    """Signal handler for SIGHUP to trigger config reload."""
    logger.info("Received SIGHUP signal, reloading configuration...")
    try:
        config.reload()
        logger.info("Configuration reloaded successfully via SIGHUP")
    except Exception as e:
        logger.error(f"Failed to reload configuration: {e}")


def setup_signal_handlers() -> bool:
    """
    Setup signal handlers for configuration hot-reload.

    On Unix systems, registers SIGHUP handler for config reload.
    On Windows, this is a no-op (use reload endpoint instead).

    Returns:
        bool: True if signal handlers were set up, False otherwise
    """
    if platform.system() != "Windows":
        try:
            signal.signal(signal.SIGHUP, _sighup_handler)
            logger.info("SIGHUP handler registered for config reload")
            return True
        except (ValueError, OSError) as e:
            logger.warning(f"Could not register SIGHUP handler: {e}")
            return False
    else:
        logger.debug("SIGHUP not available on Windows, use reload endpoint instead")
        return False


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def reload_config(config_path: Optional[Path] = None, environment: Optional[str] = None) -> Config:
    """
    Reload configuration from file with optional environment override.

    Args:
        config_path: Path to base config file
        environment: Environment name (development/staging/production)

    Returns:
        Reloaded Config instance
    """
    Config._config = {}
    Config._validated_config = None
    config.load(config_path, environment)

    # Notify all registered callbacks
    for callback in _reload_callbacks:
        try:
            callback(config)
        except Exception as e:
            logger.error(f"Error in reload callback: {e}")

    return config


def get_environment() -> str:
    """Get the current environment name."""
    return config.environment
