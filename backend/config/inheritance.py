"""
Floodingnaque Configuration Inheritance
=======================================

Provides inheritance capabilities for configuration sections, particularly
for training phases that share common settings.

Features:
- Base configuration inheritance with extends/inherits keyword
- Deep merging of inherited values
- Override precedence (child values override parent)
- Multiple inheritance support
- Circular dependency detection

Usage:
    from config.inheritance import ConfigInheritance, process_inheritance

    # Process config dict with inheritance
    config = {
        "_base_phase": {
            "data_type": "official_records",
            "calibrate": false
        },
        "phases": {
            "phase_1": {
                "extends": "_base_phase",
                "name": "Baseline"
            }
        }
    }

    processed = process_inheritance(config)
    # phase_1 now has data_type and calibrate from _base_phase

Inheritance Keywords:
    extends: Single parent reference
    inherits: List of parent references (processed in order)
    _base: Convention for base configurations (not included in output)
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class InheritanceError(Exception):
    """Raised when inheritance processing fails."""

    pass


class CircularInheritanceError(InheritanceError):
    """Raised when circular inheritance is detected."""

    pass


class MissingBaseError(InheritanceError):
    """Raised when a referenced base configuration doesn't exist."""

    pass


class ConfigInheritance:
    """
    Configuration inheritance processor.

    Handles inheritance relationships in configuration dictionaries,
    allowing configurations to extend and override base configurations.
    """

    # Keywords that indicate inheritance
    EXTENDS_KEY = "extends"
    INHERITS_KEY = "inherits"

    # Prefix for base configurations (excluded from final output)
    BASE_PREFIX = "_"

    def __init__(
        self,
        base_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        remove_bases: bool = True,
        remove_inheritance_keys: bool = True,
    ):
        """
        Initialize inheritance processor.

        Args:
            base_configs: Optional dict of base configurations that can be referenced
            remove_bases: Remove base configs (starting with _) from output
            remove_inheritance_keys: Remove extends/inherits keys from output
        """
        self.base_configs = base_configs or {}
        self.remove_bases = remove_bases
        self.remove_inheritance_keys = remove_inheritance_keys
        self._processing: Set[str] = set()  # Track currently processing for cycle detection

    def process(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a configuration dictionary, resolving all inheritance.

        Args:
            config: Configuration dictionary with potential inheritance

        Returns:
            Processed configuration with inheritance resolved

        Raises:
            CircularInheritanceError: If circular inheritance detected
            MissingBaseError: If referenced base doesn't exist
        """
        result = copy.deepcopy(config)

        # First, collect all potential base configs from the config itself
        all_bases = {**self.base_configs}
        self._collect_bases(result, all_bases)

        # Process inheritance in all sections
        self._process_dict(result, all_bases, "")

        # Remove base configs from output if configured
        if self.remove_bases:
            self._remove_base_keys(result)

        return result

    def _collect_bases(self, config: Dict[str, Any], bases: Dict[str, Dict]) -> None:
        """Collect all base configurations from config."""
        for key, value in config.items():
            if isinstance(value, dict):
                if key.startswith(self.BASE_PREFIX):
                    bases[key] = value
                # Also look for nested bases
                self._collect_bases(value, bases)

    def _process_dict(self, config: Dict[str, Any], bases: Dict[str, Dict], path: str) -> None:
        """Recursively process dictionary for inheritance."""
        # Track items to update (can't modify dict during iteration)
        updates = {}

        for key, value in config.items():
            if not isinstance(value, dict):
                continue

            current_path = f"{path}.{key}" if path else key

            # Check for inheritance
            if self.EXTENDS_KEY in value or self.INHERITS_KEY in value:
                # Resolve inheritance
                resolved = self._resolve_inheritance(value, bases, current_path)
                updates[key] = resolved

            # Process nested dicts
            target = updates.get(key, value)
            if isinstance(target, dict):
                self._process_dict(target, bases, current_path)

        # Apply updates
        config.update(updates)

    def _resolve_inheritance(self, config: Dict[str, Any], bases: Dict[str, Dict], path: str) -> Dict[str, Any]:
        """
        Resolve inheritance for a single configuration.

        Args:
            config: Configuration with extends/inherits
            bases: Available base configurations
            path: Current config path (for error messages)

        Returns:
            Resolved configuration with inherited values
        """
        # Check for circular inheritance
        if path in self._processing:
            raise CircularInheritanceError(f"Circular inheritance detected at: {path}")

        self._processing.add(path)

        try:
            result = {}

            # Get parent references
            parents = []
            if self.EXTENDS_KEY in config:
                extends = config[self.EXTENDS_KEY]
                parents = [extends] if isinstance(extends, str) else extends
            elif self.INHERITS_KEY in config:
                inherits = config[self.INHERITS_KEY]
                parents = inherits if isinstance(inherits, list) else [inherits]

            # Process each parent in order
            for parent_ref in parents:
                parent_config = self._get_base(parent_ref, bases, path)

                # Parent might also have inheritance
                if self.EXTENDS_KEY in parent_config or self.INHERITS_KEY in parent_config:
                    parent_config = self._resolve_inheritance(parent_config, bases, f"{path}->parent({parent_ref})")

                # Merge parent into result
                self._deep_merge(result, parent_config)

            # Merge current config (overrides parents)
            child_config = {k: v for k, v in config.items() if k not in (self.EXTENDS_KEY, self.INHERITS_KEY)}
            self._deep_merge(result, child_config)

            # Remove inheritance keys if configured
            if not self.remove_inheritance_keys:
                if self.EXTENDS_KEY in config:
                    result[self.EXTENDS_KEY] = config[self.EXTENDS_KEY]
                if self.INHERITS_KEY in config:
                    result[self.INHERITS_KEY] = config[self.INHERITS_KEY]

            return result

        finally:
            self._processing.discard(path)

    def _get_base(self, ref: str, bases: Dict[str, Dict], path: str) -> Dict[str, Any]:
        """
        Get a base configuration by reference.

        Supports:
        - Direct key reference: "_base_phase"
        - Nested path reference: "phases._base"
        - External file reference (future): "file:base.yaml"
        """
        # Direct key reference
        if ref in bases:
            return copy.deepcopy(bases[ref])

        # Add underscore prefix if not present
        if not ref.startswith(self.BASE_PREFIX):
            prefixed = f"{self.BASE_PREFIX}{ref}"
            if prefixed in bases:
                return copy.deepcopy(bases[prefixed])

        raise MissingBaseError(f"Base configuration '{ref}' not found, referenced from: {path}")

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """
        Deep merge override into base dictionary.

        - Nested dicts are merged recursively
        - Lists are replaced (not merged)
        - All other values are replaced
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    def _remove_base_keys(self, config: Dict[str, Any]) -> None:
        """Remove base configuration keys from output."""
        keys_to_remove = [key for key in config.keys() if key.startswith(self.BASE_PREFIX)]

        for key in keys_to_remove:
            del config[key]

        # Recursively process nested dicts
        for value in config.values():
            if isinstance(value, dict):
                self._remove_base_keys(value)


def process_inheritance(
    config: Dict[str, Any], base_configs: Optional[Dict[str, Dict[str, Any]]] = None, remove_bases: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to process inheritance in a config.

    Args:
        config: Configuration dictionary
        base_configs: Optional external base configurations
        remove_bases: Remove base configs from output

    Returns:
        Processed configuration
    """
    processor = ConfigInheritance(base_configs=base_configs, remove_bases=remove_bases)
    return processor.process(config)


def create_phase_base(
    data_type: str = "official_records", model_type: Optional[str] = None, calibrate: bool = False, **kwargs
) -> Dict[str, Any]:
    """
    Create a base phase configuration.

    Args:
        data_type: Type of data for the phase
        model_type: Optional model type override
        calibrate: Whether to enable calibration
        **kwargs: Additional base settings

    Returns:
        Base phase configuration dict
    """
    base = {
        "data_type": data_type,
        "calibrate": calibrate,
    }

    if model_type:
        base["model_type"] = model_type

    base.update(kwargs)
    return base


def validate_inheritance(config: Dict[str, Any]) -> List[str]:
    """
    Validate inheritance references in a configuration.

    Args:
        config: Configuration to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Collect all base configs
    bases: Set[str] = set()
    _collect_base_names(config, bases)

    # Check all inheritance references
    _validate_references(config, bases, errors, "")

    return errors


def _collect_base_names(config: Dict[str, Any], bases: Set[str]) -> None:
    """Recursively collect base configuration names."""
    for key, value in config.items():
        if key.startswith("_") and isinstance(value, dict):
            bases.add(key)
        if isinstance(value, dict):
            _collect_base_names(value, bases)


def _validate_references(config: Dict[str, Any], bases: Set[str], errors: List[str], path: str) -> None:
    """Recursively validate inheritance references."""
    for key, value in config.items():
        if not isinstance(value, dict):
            continue

        current_path = f"{path}.{key}" if path else key

        # Check extends
        if "extends" in value:
            ref = value["extends"]
            refs = [ref] if isinstance(ref, str) else ref
            for r in refs:
                if r not in bases and f"_{r}" not in bases:
                    errors.append(f"Missing base '{r}' referenced at {current_path}")

        # Check inherits
        if "inherits" in value:
            refs = value["inherits"]
            refs = refs if isinstance(refs, list) else [refs]
            for r in refs:
                if r not in bases and f"_{r}" not in bases:
                    errors.append(f"Missing base '{r}' referenced at {current_path}")

        # Recurse
        _validate_references(value, bases, errors, current_path)


# Example base configurations for common use cases
COMMON_BASES = {
    "_official_phase": {
        "data_type": "official_records",
        "calibrate": False,
        "model_type": None,
    },
    "_merged_phase": {
        "data_type": "merged",
        "calibrate": True,
        "model_type": None,
    },
    "_ensemble_phase": {
        "data_type": "ensemble",
        "model_type": "stacking",
        "calibrate": True,
    },
    "_station_phase": {
        "data_type": "station_specific",
        "model_type": "station_specific",
        "calibrate": False,
    },
}
