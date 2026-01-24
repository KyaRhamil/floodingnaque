"""
Floodingnaque Feature Validation
================================

Validates that feature names referenced in configuration actually exist
in the datasets they reference.

Features:
- Validates feature names against CSV column headers
- Supports checking multiple datasets
- Identifies missing and extra features
- Provides suggestions for similar feature names
- Integrates with config loading

Usage:
    from config.feature_validation import FeatureValidator, validate_config_features

    # Validate features in config
    validator = FeatureValidator(data_dir="data/processed")
    errors = validator.validate_config(config)

    # Check specific features against dataset
    missing = validator.check_features(
        features=["temperature", "tide_height"],
        dataset_file="training_data.csv"
    )

Environment Variables:
    FLOODINGNAQUE_DATA_DIR - Directory containing data files
    FLOODINGNAQUE_SKIP_FEATURE_VALIDATION - Skip validation if "true"
"""

import csv
import difflib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class FeatureValidationError(Exception):
    """Raised when feature validation fails."""

    pass


class DatasetNotFoundError(FeatureValidationError):
    """Raised when a referenced dataset file doesn't exist."""

    pass


class FeatureValidationResult:
    """Result of feature validation."""

    def __init__(
        self,
        valid: bool,
        missing_features: Dict[str, Set[str]],  # dataset -> missing features
        unknown_features: Dict[str, Set[str]],  # dataset -> unknown features
        suggestions: Dict[str, Dict[str, List[str]]],  # dataset -> feature -> suggestions
        warnings: List[str],
    ):
        self.valid = valid
        self.missing_features = missing_features
        self.unknown_features = unknown_features
        self.suggestions = suggestions
        self.warnings = warnings

    def __bool__(self) -> bool:
        return self.valid

    def __str__(self) -> str:
        if self.valid:
            return "Feature validation passed"

        lines = ["Feature validation failed:"]

        for dataset, features in self.missing_features.items():
            if features:
                lines.append(f"\n  {dataset}:")
                lines.append(f"    Missing features: {', '.join(sorted(features))}")

                # Add suggestions
                if dataset in self.suggestions:
                    for feature, similar in self.suggestions[dataset].items():
                        if similar:
                            lines.append(f"    Did you mean '{similar[0]}' instead of '{feature}'?")

        if self.warnings:
            lines.append("\n  Warnings:")
            for warning in self.warnings:
                lines.append(f"    - {warning}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "valid": self.valid,
            "missing_features": {k: list(v) for k, v in self.missing_features.items()},
            "unknown_features": {k: list(v) for k, v in self.unknown_features.items()},
            "suggestions": self.suggestions,
            "warnings": self.warnings,
        }


class FeatureValidator:
    """
    Validates feature names against dataset schemas.

    Checks that features referenced in configuration actually exist
    in the datasets they will be used with.
    """

    # Standard features that should always be available
    STANDARD_FEATURES = {
        "temperature",
        "humidity",
        "precipitation",
        "is_monsoon_season",
        "month",
    }

    # Features that are computed/engineered (don't need to exist in raw data)
    COMPUTED_FEATURES = {
        "temp_humidity_interaction",
        "humidity_precip_interaction",
        "temp_precip_interaction",
        "monsoon_precip_interaction",
        "saturation_risk",
        "precip_3day_sum",
        "precip_7day_sum",
        "precip_14day_sum",
        "precip_lag1",
        "precip_lag2",
        "rain_streak",
    }

    # Features from external APIs (may not be in all datasets)
    EXTERNAL_FEATURES = {
        "tide_height",
        "tide_level",
        "sea_level",
        "soil_moisture",
        "ndvi",
        "elevation",
    }

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        cache_schemas: bool = True,
        check_computed: bool = False,
        strict_mode: bool = False,
    ):
        """
        Initialize feature validator.

        Args:
            data_dir: Directory containing data files
            cache_schemas: Cache dataset column schemas
            check_computed: Also validate computed features (default: skip)
            strict_mode: Fail on any unknown feature (default: warn only)
        """
        self.data_dir = Path(data_dir) if data_dir else self._default_data_dir()
        self.cache_schemas = cache_schemas
        self.check_computed = check_computed
        self.strict_mode = strict_mode
        self._schema_cache: Dict[str, Set[str]] = {}

    def _default_data_dir(self) -> Path:
        """Get default data directory."""
        import os

        # Check environment variable
        env_dir = os.environ.get("FLOODINGNAQUE_DATA_DIR")
        if env_dir:
            return Path(env_dir)

        # Default to backend/data
        return Path(__file__).parent.parent / "data"

    def get_dataset_columns(self, dataset_file: Union[str, Path], subdir: str = "processed") -> Set[str]:
        """
        Get column names from a dataset file.

        Args:
            dataset_file: Name or path of the dataset file
            subdir: Subdirectory within data_dir to look in

        Returns:
            Set of column names

        Raises:
            DatasetNotFoundError: If file doesn't exist
        """
        dataset_path = self._resolve_dataset_path(dataset_file, subdir)

        # Check cache
        cache_key = str(dataset_path)
        if self.cache_schemas and cache_key in self._schema_cache:
            return self._schema_cache[cache_key]

        if not dataset_path.exists():
            # Try alternative locations
            for alt_subdir in ["raw", "cleaned", "processed", ""]:
                alt_path = self._resolve_dataset_path(dataset_file, alt_subdir)
                if alt_path.exists():
                    dataset_path = alt_path
                    break
            else:
                raise DatasetNotFoundError(f"Dataset not found: {dataset_file}")

        # Read columns from CSV
        try:
            with open(dataset_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                columns = set(next(reader))
        except StopIteration:
            raise FeatureValidationError(f"Empty dataset: {dataset_path}")
        except Exception as e:
            raise FeatureValidationError(f"Error reading dataset {dataset_path}: {e}")

        # Cache result
        if self.cache_schemas:
            self._schema_cache[cache_key] = columns

        return columns

    def _resolve_dataset_path(self, dataset_file: Union[str, Path], subdir: str = "") -> Path:
        """Resolve full path to a dataset file."""
        dataset_path = Path(dataset_file)

        # If already absolute, use as-is
        if dataset_path.is_absolute():
            return dataset_path

        # Build path from data_dir
        if subdir:
            return self.data_dir / subdir / dataset_path
        return self.data_dir / dataset_path

    def check_features(
        self,
        features: List[str],
        dataset_file: Union[str, Path],
        skip_computed: bool = True,
        skip_external: bool = True,
    ) -> Tuple[Set[str], Set[str]]:
        """
        Check if features exist in a dataset.

        Args:
            features: List of feature names to check
            dataset_file: Dataset file to check against
            skip_computed: Skip validation of computed features
            skip_external: Skip validation of external API features

        Returns:
            Tuple of (missing_features, available_features)
        """
        # Get dataset columns
        try:
            columns = self.get_dataset_columns(dataset_file)
        except DatasetNotFoundError:
            logger.warning(f"Dataset not found for validation: {dataset_file}")
            return set(), set(features)

        # Normalize column names (lowercase, strip)
        columns_normalized = {c.lower().strip() for c in columns}
        columns_map = {c.lower().strip(): c for c in columns}

        missing = set()
        available = set()

        for feature in features:
            feature_normalized = feature.lower().strip()

            # Skip computed features if configured
            if skip_computed and feature in self.COMPUTED_FEATURES:
                available.add(feature)
                continue

            # Skip external features if configured
            if skip_external and feature in self.EXTERNAL_FEATURES:
                available.add(feature)
                continue

            # Check if feature exists
            if feature_normalized in columns_normalized:
                available.add(feature)
            else:
                missing.add(feature)

        return missing, available

    def find_similar_features(
        self, feature: str, available_columns: Set[str], n: int = 3, cutoff: float = 0.6
    ) -> List[str]:
        """
        Find similar feature names for suggestions.

        Args:
            feature: Feature name to match
            available_columns: Available column names
            n: Maximum number of suggestions
            cutoff: Minimum similarity ratio

        Returns:
            List of similar feature names
        """
        return difflib.get_close_matches(feature.lower(), [c.lower() for c in available_columns], n=n, cutoff=cutoff)

    def validate_config(
        self, config: Dict[str, Any], datasets_to_check: Optional[List[str]] = None
    ) -> FeatureValidationResult:
        """
        Validate all feature references in a configuration.

        Args:
            config: Configuration dictionary
            datasets_to_check: Optional list of dataset files to check

        Returns:
            FeatureValidationResult with validation details
        """
        missing_features: Dict[str, Set[str]] = {}
        unknown_features: Dict[str, Set[str]] = {}
        suggestions: Dict[str, Dict[str, List[str]]] = {}
        warnings: List[str] = []

        # Collect all feature references from config
        feature_refs = self._collect_feature_references(config)

        # Get list of datasets to check
        if datasets_to_check is None:
            datasets_to_check = self._get_datasets_from_config(config)

        # Validate each dataset
        for dataset in datasets_to_check:
            try:
                columns = self.get_dataset_columns(dataset)
            except DatasetNotFoundError:
                warnings.append(f"Dataset not found: {dataset}")
                continue
            except FeatureValidationError as e:
                warnings.append(str(e))
                continue

            # Check features used with this dataset
            dataset_features = feature_refs.get(dataset, feature_refs.get("_global", set()))

            missing, _ = self.check_features(list(dataset_features), dataset, skip_computed=not self.check_computed)

            if missing:
                missing_features[dataset] = missing
                suggestions[dataset] = {}

                for feature in missing:
                    similar = self.find_similar_features(feature, columns)
                    if similar:
                        suggestions[dataset][feature] = similar

        # Determine validity
        valid = not missing_features or not self.strict_mode

        return FeatureValidationResult(
            valid=valid,
            missing_features=missing_features,
            unknown_features=unknown_features,
            suggestions=suggestions,
            warnings=warnings,
        )

    def _collect_feature_references(self, config: Dict[str, Any]) -> Dict[str, Set[str]]:
        """
        Collect all feature references from configuration.

        Returns dict mapping dataset -> features
        """
        refs: Dict[str, Set[str]] = {"_global": set()}

        # Global feature lists
        data_config = config.get("data", {})
        for key in ["core_features", "interaction_features", "rolling_features", "categorical_features"]:
            if key in data_config and isinstance(data_config[key], list):
                refs["_global"].update(data_config[key])

        # Phase-specific features
        phases = config.get("phases", {})
        for phase_name, phase_config in phases.items():
            if not isinstance(phase_config, dict):
                continue

            features = phase_config.get("features", [])
            if isinstance(features, list):
                # Get associated dataset
                data_file = phase_config.get("data_file")
                data_files = phase_config.get("data_files", [])

                if data_file:
                    refs.setdefault(data_file, set()).update(features)
                for df in data_files:
                    refs.setdefault(df, set()).update(features)

                if not data_file and not data_files:
                    refs["_global"].update(features)

        # Drift monitoring features
        drift_config = config.get("drift", {})
        monitored = drift_config.get("monitored_features", [])
        if monitored:
            refs["_global"].update(monitored)

        return refs

    def _get_datasets_from_config(self, config: Dict[str, Any]) -> List[str]:
        """Extract list of datasets referenced in config."""
        datasets = set()

        # From phases
        phases = config.get("phases", {})
        for phase_config in phases.values():
            if isinstance(phase_config, dict):
                if "data_file" in phase_config:
                    datasets.add(phase_config["data_file"])
                if "data_files" in phase_config:
                    datasets.update(phase_config["data_files"])

        return list(datasets)

    def clear_cache(self) -> None:
        """Clear the schema cache."""
        self._schema_cache.clear()


def validate_config_features(
    config: Dict[str, Any], data_dir: Optional[Union[str, Path]] = None, strict: bool = False
) -> FeatureValidationResult:
    """
    Convenience function to validate features in a config.

    Args:
        config: Configuration dictionary
        data_dir: Optional data directory path
        strict: Fail on any validation error

    Returns:
        FeatureValidationResult
    """
    validator = FeatureValidator(data_dir=data_dir, strict_mode=strict)
    return validator.validate_config(config)


def get_available_features(dataset_file: Union[str, Path], data_dir: Optional[Union[str, Path]] = None) -> Set[str]:
    """
    Get all available features from a dataset.

    Args:
        dataset_file: Path to dataset file
        data_dir: Optional data directory

    Returns:
        Set of available feature names
    """
    validator = FeatureValidator(data_dir=data_dir)
    return validator.get_dataset_columns(dataset_file)


def suggest_features(
    partial_name: str, dataset_file: Union[str, Path], data_dir: Optional[Union[str, Path]] = None, n: int = 5
) -> List[str]:
    """
    Suggest feature names based on partial input.

    Args:
        partial_name: Partial feature name
        dataset_file: Dataset to get features from
        data_dir: Optional data directory
        n: Maximum suggestions

    Returns:
        List of suggested feature names
    """
    validator = FeatureValidator(data_dir=data_dir)
    columns = validator.get_dataset_columns(dataset_file)
    return validator.find_similar_features(partial_name, columns, n=n)


# Feature registry for documentation and validation
FEATURE_REGISTRY = {
    # Core meteorological
    "temperature": {
        "type": "numeric",
        "unit": "°C",
        "description": "Air temperature",
        "sources": ["official", "pagasa", "meteostat"],
    },
    "humidity": {
        "type": "numeric",
        "unit": "%",
        "description": "Relative humidity",
        "sources": ["official", "pagasa", "meteostat"],
    },
    "precipitation": {
        "type": "numeric",
        "unit": "mm",
        "description": "Precipitation amount",
        "sources": ["official", "pagasa", "meteostat"],
    },
    # Temporal
    "is_monsoon_season": {
        "type": "binary",
        "description": "Whether date is in monsoon season (June-November)",
        "sources": ["computed"],
    },
    "month": {
        "type": "categorical",
        "values": list(range(1, 13)),
        "description": "Month of year",
        "sources": ["computed"],
    },
    # Rolling features
    "precip_3day_sum": {
        "type": "numeric",
        "unit": "mm",
        "description": "3-day rolling precipitation sum",
        "sources": ["computed"],
    },
    "precip_7day_sum": {
        "type": "numeric",
        "unit": "mm",
        "description": "7-day rolling precipitation sum",
        "sources": ["computed"],
    },
    "precip_14day_sum": {
        "type": "numeric",
        "unit": "mm",
        "description": "14-day rolling precipitation sum",
        "sources": ["computed"],
    },
    "rain_streak": {
        "type": "numeric",
        "unit": "days",
        "description": "Consecutive rainy days",
        "sources": ["computed"],
    },
    # Interaction features
    "temp_humidity_interaction": {
        "type": "numeric",
        "description": "Temperature × Humidity interaction",
        "sources": ["computed"],
    },
    "humidity_precip_interaction": {
        "type": "numeric",
        "description": "Humidity × Precipitation interaction",
        "sources": ["computed"],
    },
    "monsoon_precip_interaction": {
        "type": "numeric",
        "description": "Monsoon season × Precipitation interaction",
        "sources": ["computed"],
    },
    "saturation_risk": {
        "type": "numeric",
        "description": "Soil saturation risk indicator",
        "sources": ["computed"],
    },
    # External API features
    "tide_height": {
        "type": "numeric",
        "unit": "m",
        "description": "Tide height from WorldTides API",
        "sources": ["worldtides"],
        "optional": True,
    },
}
