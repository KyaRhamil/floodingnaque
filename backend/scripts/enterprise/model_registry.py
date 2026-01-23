"""
Model Registry Module
=====================

This module provides enterprise-grade model registry functionality:
- Multi-stage model lifecycle (development → staging → production)
- Version management and tracking
- Promotion criteria validation
- Model comparison and rollback support

.. module:: enterprise.model_registry
   :synopsis: Model registry with staged promotion and versioning.

.. moduleauthor:: Floodingnaque Team

Features
--------
- Multi-stage lifecycle management (development → staging → production)
- Automatic version tracking and indexing
- Promotion criteria validation with configurable thresholds
- Model comparison between versions
- Rollback support for production incidents
- Model archiving for storage management

Stages
------
DEVELOPMENT
    Initial stage for newly trained models.
STAGING
    Pre-production testing environment.
PRODUCTION
    Live production models serving predictions.
ARCHIVED
    Retired models kept for reference.

Example
-------
::

    >>> from enterprise.model_registry import ModelRegistry, ModelStage
    >>> registry = ModelRegistry(Path("models/registry"))
    >>> # Register new model
    >>> version = registry.register(
    ...     model=trained_model,
    ...     version=1,
    ...     name="flood_rf_model",
    ...     metrics={'f1_score': 0.92, 'roc_auc': 0.95},
    ...     parameters={'n_estimators': 100},
    ...     features=['temperature', 'humidity', 'precipitation']
    ... )
    >>> # Promote to production
    >>> success, msg = registry.promote(1, ModelStage.PRODUCTION)
"""

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """Represents a versioned model in the registry."""

    version: int
    name: str
    stage: ModelStage
    model_path: Path
    metadata_path: Path
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    features: List[str]
    created_at: datetime
    promoted_at: Optional[datetime] = None
    description: str = ""
    model_hash: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "name": self.name,
            "stage": self.stage.value,
            "model_path": str(self.model_path),
            "metadata_path": str(self.metadata_path),
            "metrics": self.metrics,
            "parameters": self.parameters,
            "features": self.features,
            "created_at": self.created_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "description": self.description,
            "model_hash": self.model_hash,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            name=data["name"],
            stage=ModelStage(data["stage"]),
            model_path=Path(data["model_path"]),
            metadata_path=Path(data["metadata_path"]),
            metrics=data["metrics"],
            parameters=data.get("parameters", {}),
            features=data.get("features", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            promoted_at=datetime.fromisoformat(data["promoted_at"]) if data.get("promoted_at") else None,
            description=data.get("description", ""),
            model_hash=data.get("model_hash", ""),
            tags=data.get("tags", {}),
        )


class PromotionCriteria:
    """Defines criteria for model promotion between stages."""

    DEFAULT_STAGING_CRITERIA = {
        "min_f1_score": 0.85,
        "min_roc_auc": 0.85,
        "max_cv_std": 0.05,
    }

    DEFAULT_PRODUCTION_CRITERIA = {
        "min_f1_score": 0.90,
        "min_roc_auc": 0.90,
        "max_cv_std": 0.03,
    }

    def __init__(self, criteria: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize promotion criteria.

        Args:
            criteria: Dictionary mapping stages to criteria
        """
        self.criteria = criteria or {
            "staging": self.DEFAULT_STAGING_CRITERIA,
            "production": self.DEFAULT_PRODUCTION_CRITERIA,
        }

    def check(self, metrics: Dict[str, float], target_stage: ModelStage) -> Tuple[bool, List[str]]:
        """
        Check if metrics meet promotion criteria.

        Args:
            metrics: Model metrics
            target_stage: Target stage for promotion

        Returns:
            Tuple of (passes, list of failure reasons)
        """
        stage_criteria = self.criteria.get(target_stage.value, {})
        failures = []

        for criterion, threshold in stage_criteria.items():
            if criterion.startswith("min_"):
                metric_name = criterion[4:]  # Remove "min_" prefix
                metric_value = metrics.get(metric_name, metrics.get(f"{metric_name}_mean", 0))
                if metric_value < threshold:
                    failures.append(f"{metric_name} ({metric_value:.4f}) < minimum ({threshold})")
            elif criterion.startswith("max_"):
                metric_name = criterion[4:]  # Remove "max_" prefix
                metric_value = metrics.get(metric_name, metrics.get(f"{metric_name}_std", float("inf")))
                if metric_value > threshold:
                    failures.append(f"{metric_name} ({metric_value:.4f}) > maximum ({threshold})")

        return len(failures) == 0, failures


class ModelRegistry:
    """
    Enterprise model registry with stage management.

    Provides:
    - Version tracking and storage
    - Stage-based lifecycle management
    - Promotion with criteria validation
    - Model comparison and rollback
    """

    def __init__(self, registry_path: Path, promotion_criteria: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize model registry.

        Args:
            registry_path: Root path for registry storage
            promotion_criteria: Criteria for stage promotion
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

        # Create stage directories
        for stage in ModelStage:
            (self.registry_path / stage.value).mkdir(exist_ok=True)

        # Registry index file
        self.index_path = self.registry_path / "registry_index.json"
        self.promotion_criteria = PromotionCriteria(promotion_criteria)

        # Load or create index
        self._load_index()

        logger.info(f"Model registry initialized at {registry_path}")

    def _load_index(self) -> None:
        """Load registry index from file."""
        if self.index_path.exists():
            with open(self.index_path, "r") as f:
                data = json.load(f)
                self.versions = {int(k): ModelVersion.from_dict(v) for k, v in data.get("versions", {}).items()}
                self.current_production = data.get("current_production")
                self.current_staging = data.get("current_staging")
        else:
            self.versions: Dict[int, ModelVersion] = {}
            self.current_production: Optional[int] = None
            self.current_staging: Optional[int] = None

    def _save_index(self) -> None:
        """Save registry index to file."""
        data = {
            "versions": {k: v.to_dict() for k, v in self.versions.items()},
            "current_production": self.current_production,
            "current_staging": self.current_staging,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)

    def _compute_model_hash(self, model_path: Path) -> str:
        """Compute SHA256 hash of model file."""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:16]

    def register(
        self,
        model: BaseEstimator,
        version: int,
        name: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        features: List[str],
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            model: Trained sklearn model
            version: Version number
            name: Model name
            metrics: Model metrics
            parameters: Model parameters
            features: Feature list
            description: Model description
            tags: Optional tags

        Returns:
            Created ModelVersion
        """
        # Create paths
        stage = ModelStage.DEVELOPMENT
        stage_dir = self.registry_path / stage.value
        model_path = stage_dir / f"model_v{version}.joblib"
        metadata_path = stage_dir / f"model_v{version}.json"

        # Save model
        joblib.dump(model, model_path)

        # Compute hash
        model_hash = self._compute_model_hash(model_path)

        # Create version object
        model_version = ModelVersion(
            version=version,
            name=name,
            stage=stage,
            model_path=model_path,
            metadata_path=metadata_path,
            metrics=metrics,
            parameters=parameters,
            features=features,
            created_at=datetime.now(),
            description=description,
            model_hash=model_hash,
            tags=tags or {},
        )

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(model_version.to_dict(), f, indent=2)

        # Update index
        self.versions[version] = model_version
        self._save_index()

        logger.info(f"Registered model v{version} in {stage.value}")

        return model_version

    def promote(self, version: int, target_stage: ModelStage, force: bool = False) -> Tuple[bool, str]:
        """
        Promote a model to a higher stage.

        Args:
            version: Version to promote
            target_stage: Target stage
            force: Skip criteria check

        Returns:
            Tuple of (success, message)
        """
        if version not in self.versions:
            return False, f"Version {version} not found in registry"

        model_version = self.versions[version]

        # Check promotion criteria
        if not force:
            passes, failures = self.promotion_criteria.check(model_version.metrics, target_stage)
            if not passes:
                return False, f"Model does not meet criteria: {'; '.join(failures)}"

        # Move model files
        old_stage_dir = self.registry_path / model_version.stage.value
        new_stage_dir = self.registry_path / target_stage.value

        new_model_path = new_stage_dir / f"model_v{version}.joblib"
        new_metadata_path = new_stage_dir / f"model_v{version}.json"

        # Copy files (keep in old stage for rollback)
        shutil.copy2(model_version.model_path, new_model_path)

        # Update version
        model_version.stage = target_stage
        model_version.model_path = new_model_path
        model_version.metadata_path = new_metadata_path
        model_version.promoted_at = datetime.now()

        # Save updated metadata
        with open(new_metadata_path, "w") as f:
            json.dump(model_version.to_dict(), f, indent=2)

        # Update current pointers
        if target_stage == ModelStage.PRODUCTION:
            self.current_production = version
        elif target_stage == ModelStage.STAGING:
            self.current_staging = version

        self._save_index()

        logger.info(f"Promoted model v{version} to {target_stage.value}")

        return True, f"Successfully promoted v{version} to {target_stage.value}"

    def get_production_model(self) -> Optional[Tuple[BaseEstimator, ModelVersion]]:
        """Get current production model."""
        if self.current_production is None:
            return None

        version = self.versions.get(self.current_production)
        if version is None:
            return None

        model = joblib.load(version.model_path)
        return model, version

    def get_staging_model(self) -> Optional[Tuple[BaseEstimator, ModelVersion]]:
        """Get current staging model."""
        if self.current_staging is None:
            return None

        version = self.versions.get(self.current_staging)
        if version is None:
            return None

        model = joblib.load(version.model_path)
        return model, version

    def get_model(self, version: int) -> Optional[Tuple[BaseEstimator, ModelVersion]]:
        """Get specific model version."""
        if version not in self.versions:
            return None

        model_version = self.versions[version]
        model = joblib.load(model_version.model_path)
        return model, model_version

    def list_versions(self, stage: Optional[ModelStage] = None) -> List[ModelVersion]:
        """List all versions, optionally filtered by stage."""
        versions = list(self.versions.values())

        if stage:
            versions = [v for v in versions if v.stage == stage]

        return sorted(versions, key=lambda v: v.version, reverse=True)

    def compare_versions(self, version_a: int, version_b: int) -> Dict[str, Any]:
        """Compare two model versions."""
        if version_a not in self.versions or version_b not in self.versions:
            return {"error": "One or both versions not found"}

        va = self.versions[version_a]
        vb = self.versions[version_b]

        comparison = {
            "version_a": version_a,
            "version_b": version_b,
            "metrics_comparison": {},
            "parameter_diff": {},
            "feature_diff": {
                "added": list(set(vb.features) - set(va.features)),
                "removed": list(set(va.features) - set(vb.features)),
            },
        }

        # Compare metrics
        all_metrics = set(va.metrics.keys()) | set(vb.metrics.keys())
        for metric in all_metrics:
            val_a = va.metrics.get(metric, 0)
            val_b = vb.metrics.get(metric, 0)
            comparison["metrics_comparison"][metric] = {
                "version_a": val_a,
                "version_b": val_b,
                "diff": val_b - val_a,
                "pct_change": ((val_b - val_a) / val_a * 100) if val_a != 0 else 0,
            }

        # Compare parameters
        all_params = set(va.parameters.keys()) | set(vb.parameters.keys())
        for param in all_params:
            val_a = va.parameters.get(param)
            val_b = vb.parameters.get(param)
            if val_a != val_b:
                comparison["parameter_diff"][param] = {
                    "version_a": val_a,
                    "version_b": val_b,
                }

        return comparison

    def rollback(self, target_version: int) -> Tuple[bool, str]:
        """Rollback production to a specific version."""
        if target_version not in self.versions:
            return False, f"Version {target_version} not found"

        model_version = self.versions[target_version]

        # Check if model file exists
        if not model_version.model_path.exists():
            return False, f"Model file not found for v{target_version}"

        # Update production pointer
        old_production = self.current_production
        self.current_production = target_version
        self._save_index()

        logger.warning(f"Rolled back production from v{old_production} to v{target_version}")

        return True, f"Rolled back to v{target_version}"

    def archive(self, version: int) -> Tuple[bool, str]:
        """Archive a model version."""
        if version not in self.versions:
            return False, f"Version {version} not found"

        if version == self.current_production:
            return False, "Cannot archive current production model"

        model_version = self.versions[version]

        # Move to archived stage
        archive_dir = self.registry_path / ModelStage.ARCHIVED.value
        new_model_path = archive_dir / f"model_v{version}.joblib"
        new_metadata_path = archive_dir / f"model_v{version}.json"

        shutil.move(str(model_version.model_path), new_model_path)

        model_version.stage = ModelStage.ARCHIVED
        model_version.model_path = new_model_path
        model_version.metadata_path = new_metadata_path

        with open(new_metadata_path, "w") as f:
            json.dump(model_version.to_dict(), f, indent=2)

        self._save_index()

        logger.info(f"Archived model v{version}")

        return True, f"Archived v{version}"

    def get_summary(self) -> Dict[str, Any]:
        """Get registry summary."""
        return {
            "total_versions": len(self.versions),
            "current_production": self.current_production,
            "current_staging": self.current_staging,
            "versions_by_stage": {
                stage.value: len([v for v in self.versions.values() if v.stage == stage]) for stage in ModelStage
            },
            "latest_version": max(self.versions.keys()) if self.versions else None,
        }


def create_registry(
    models_dir: Path, promotion_criteria: Optional[Dict[str, Dict[str, float]]] = None
) -> ModelRegistry:
    """
    Create a model registry instance.

    Args:
        models_dir: Directory for model storage
        promotion_criteria: Optional promotion criteria

    Returns:
        Configured ModelRegistry instance
    """
    registry_path = Path(models_dir) / "registry"
    return ModelRegistry(registry_path, promotion_criteria)
