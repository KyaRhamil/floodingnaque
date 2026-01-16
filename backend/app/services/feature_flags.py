"""Feature Flags Service.

Provides feature flag support for:
- Gradual rollout of new prediction models
- Emergency bypass of external API calls
- A/B testing new alert thresholds

Supports:
- Percentage-based rollouts
- User segment targeting
- Time-based activation
- Emergency kill switches
- A/B testing with experiment groups
"""

import hashlib
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeatureFlagType(Enum):
    """Types of feature flags."""

    BOOLEAN = "boolean"  # Simple on/off
    PERCENTAGE = "percentage"  # Gradual rollout
    SEGMENT = "segment"  # User segment targeting
    EXPERIMENT = "experiment"  # A/B testing
    SCHEDULE = "schedule"  # Time-based activation


class ExperimentGroup(Enum):
    """A/B testing experiment groups."""

    CONTROL = "control"
    TREATMENT_A = "treatment_a"
    TREATMENT_B = "treatment_b"


@dataclass
class FeatureFlag:
    """Feature flag configuration."""

    name: str
    description: str
    flag_type: FeatureFlagType
    enabled: bool = True

    # Percentage rollout (0-100)
    rollout_percentage: int = 100

    # Segment targeting
    allowed_segments: List[str] = field(default_factory=list)

    # A/B testing configuration
    experiment_groups: Dict[str, int] = field(default_factory=dict)  # group -> percentage
    experiment_config: Dict[str, Any] = field(default_factory=dict)  # group-specific config

    # Time-based activation
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    owner: str = ""
    tags: List[str] = field(default_factory=list)

    # Override settings
    force_value: Optional[bool] = None  # Emergency override

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["flag_type"] = self.flag_type.value
        if self.start_time:
            data["start_time"] = self.start_time.isoformat()
        if self.end_time:
            data["end_time"] = self.end_time.isoformat()
        data["created_at"] = self.created_at.isoformat()
        data["updated_at"] = self.updated_at.isoformat()
        return data


class FeatureFlagService:
    """
    Feature flag management service.

    Provides centralized feature flag management with support for:
    - Environment-based configuration
    - Runtime updates
    - User segmentation
    - A/B testing
    - Gradual rollouts
    - Emergency kill switches
    """

    _instance = None
    _lock = Lock()

    def __new__(cls):
        """Singleton pattern for feature flag service."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize feature flag service."""
        if self._initialized:
            return

        self._flags: Dict[str, FeatureFlag] = {}
        self._listeners: Dict[str, List[Callable]] = {}
        self._override_cache: Dict[str, bool] = {}
        self._stats: Dict[str, Dict[str, int]] = {}  # flag_name -> {enabled: count, disabled: count}

        # Load default flags
        self._load_default_flags()

        # Load from environment
        self._load_from_environment()

        self._initialized = True
        logger.info(f"FeatureFlagService initialized with {len(self._flags)} flags")

    def _load_default_flags(self):
        """Load default feature flags."""
        default_flags = [
            # Model rollout flags
            FeatureFlag(
                name="model_v2_rollout",
                description="Gradual rollout of prediction model v2",
                flag_type=FeatureFlagType.PERCENTAGE,
                rollout_percentage=0,  # Start at 0%
                tags=["model", "rollout"],
            ),
            FeatureFlag(
                name="model_v2_full_release",
                description="Full release of prediction model v2 (overrides rollout)",
                flag_type=FeatureFlagType.BOOLEAN,
                enabled=False,
                tags=["model", "release"],
            ),
            # Emergency bypass flags
            FeatureFlag(
                name="bypass_openweathermap",
                description="Emergency bypass for OpenWeatherMap API calls",
                flag_type=FeatureFlagType.BOOLEAN,
                enabled=False,
                tags=["emergency", "api", "weather"],
            ),
            FeatureFlag(
                name="bypass_weatherstack",
                description="Emergency bypass for Weatherstack API calls",
                flag_type=FeatureFlagType.BOOLEAN,
                enabled=False,
                tags=["emergency", "api", "weather"],
            ),
            FeatureFlag(
                name="bypass_worldtides",
                description="Emergency bypass for WorldTides API calls",
                flag_type=FeatureFlagType.BOOLEAN,
                enabled=False,
                tags=["emergency", "api", "tides"],
            ),
            FeatureFlag(
                name="bypass_all_external_apis",
                description="Emergency bypass for all external API calls (uses cached/fallback data)",
                flag_type=FeatureFlagType.BOOLEAN,
                enabled=False,
                tags=["emergency", "api", "killswitch"],
            ),
            # A/B testing flags
            FeatureFlag(
                name="alert_threshold_experiment",
                description="A/B test for new alert thresholds",
                flag_type=FeatureFlagType.EXPERIMENT,
                experiment_groups={
                    "control": 50,  # 50% - existing thresholds
                    "treatment_a": 25,  # 25% - lower thresholds (more sensitive)
                    "treatment_b": 25,  # 25% - dynamic thresholds based on historical data
                },
                experiment_config={
                    "control": {"flood_risk_low": 0.3, "flood_risk_medium": 0.5, "flood_risk_high": 0.7},
                    "treatment_a": {"flood_risk_low": 0.25, "flood_risk_medium": 0.45, "flood_risk_high": 0.65},
                    "treatment_b": {
                        "flood_risk_low": "dynamic",
                        "flood_risk_medium": "dynamic",
                        "flood_risk_high": "dynamic",
                    },
                },
                tags=["experiment", "alerts", "thresholds"],
            ),
            # Feature flags for new features
            FeatureFlag(
                name="enhanced_predictions",
                description="Enable enhanced prediction features with confidence intervals",
                flag_type=FeatureFlagType.PERCENTAGE,
                rollout_percentage=0,
                tags=["feature", "predictions"],
            ),
            FeatureFlag(
                name="satellite_data_integration",
                description="Enable satellite precipitation data integration",
                flag_type=FeatureFlagType.BOOLEAN,
                enabled=os.getenv("FEATURE_SATELLITE_DATA_ENABLED", "False").lower() == "true",
                tags=["feature", "data", "satellite"],
            ),
            FeatureFlag(
                name="realtime_alerts",
                description="Enable real-time alert notifications via SSE",
                flag_type=FeatureFlagType.BOOLEAN,
                enabled=os.getenv("FEATURE_REALTIME_ALERTS_ENABLED", "True").lower() == "true",
                tags=["feature", "alerts", "realtime"],
            ),
            # Rate limiting flags
            FeatureFlag(
                name="rate_limit_internal_bypass",
                description="Bypass rate limiting for internal services",
                flag_type=FeatureFlagType.SEGMENT,
                allowed_segments=["internal", "admin", "service_account"],
                tags=["ratelimit", "internal"],
            ),
        ]

        for flag in default_flags:
            self._flags[flag.name] = flag

    def _load_from_environment(self):
        """Load feature flag overrides from environment variables."""
        # Pattern: FEATURE_FLAG_<FLAG_NAME>=enabled|disabled|<percentage>
        prefix = "FEATURE_FLAG_"

        for key, value in os.environ.items():
            if key.startswith(prefix):
                flag_name = key[len(prefix) :].lower()

                if flag_name in self._flags:
                    flag = self._flags[flag_name]

                    if value.lower() == "enabled":
                        flag.enabled = True
                        flag.force_value = True
                    elif value.lower() == "disabled":
                        flag.enabled = False
                        flag.force_value = False
                    elif value.isdigit():
                        flag.rollout_percentage = min(100, max(0, int(value)))

                    logger.info(f"Feature flag '{flag_name}' overridden from environment: {value}")

    def register_flag(self, flag: FeatureFlag) -> None:
        """Register a new feature flag."""
        self._flags[flag.name] = flag
        logger.info(f"Registered feature flag: {flag.name}")

    def get_flag(self, flag_name: str) -> Optional[FeatureFlag]:
        """Get a feature flag by name."""
        return self._flags.get(flag_name)

    def list_flags(self, tags: Optional[List[str]] = None) -> List[FeatureFlag]:
        """List all feature flags, optionally filtered by tags."""
        flags = list(self._flags.values())

        if tags:
            flags = [f for f in flags if any(t in f.tags for t in tags)]

        return flags

    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        segment: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if a feature flag is enabled.

        Args:
            flag_name: Name of the feature flag
            user_id: Optional user identifier for percentage/experiment rollout
            segment: Optional user segment (e.g., 'internal', 'beta', 'admin')
            context: Optional additional context

        Returns:
            bool: True if the feature is enabled
        """
        flag = self._flags.get(flag_name)

        if flag is None:
            logger.warning(f"Unknown feature flag: {flag_name}")
            return False

        # Emergency override
        if flag.force_value is not None:
            self._record_evaluation(flag_name, flag.force_value)
            return flag.force_value

        # Global disable
        if not flag.enabled:
            self._record_evaluation(flag_name, False)
            return False

        # Time-based activation
        if flag.start_time or flag.end_time:
            now = datetime.now(timezone.utc)
            if flag.start_time and now < flag.start_time:
                self._record_evaluation(flag_name, False)
                return False
            if flag.end_time and now > flag.end_time:
                self._record_evaluation(flag_name, False)
                return False

        # Segment targeting
        if flag.flag_type == FeatureFlagType.SEGMENT:
            result = segment in flag.allowed_segments if segment else False
            self._record_evaluation(flag_name, result)
            return result

        # Percentage rollout
        if flag.flag_type == FeatureFlagType.PERCENTAGE:
            result = self._check_percentage_rollout(flag, user_id)
            self._record_evaluation(flag_name, result)
            return result

        # Boolean flag
        self._record_evaluation(flag_name, flag.enabled)
        return flag.enabled

    def get_experiment_group(self, flag_name: str, user_id: str) -> Optional[str]:
        """
        Get the experiment group for a user.

        Args:
            flag_name: Name of the experiment flag
            user_id: User identifier

        Returns:
            Experiment group name or None if not in experiment
        """
        flag = self._flags.get(flag_name)

        if flag is None or flag.flag_type != FeatureFlagType.EXPERIMENT:
            return None

        if not flag.enabled:
            return None

        # Consistent hashing for group assignment
        hash_value = self._get_hash_percentage(flag_name, user_id)

        cumulative = 0
        for group, percentage in flag.experiment_groups.items():
            cumulative += percentage
            if hash_value < cumulative:
                return group

        return "control"  # Default to control

    def get_experiment_config(self, flag_name: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the experiment configuration for a user.

        Args:
            flag_name: Name of the experiment flag
            user_id: User identifier

        Returns:
            Configuration for the user's experiment group
        """
        group = self.get_experiment_group(flag_name, user_id)

        if group is None:
            return None

        flag = self._flags.get(flag_name)
        return flag.experiment_config.get(group) if flag else None

    def _check_percentage_rollout(self, flag: FeatureFlag, user_id: Optional[str]) -> bool:
        """Check if user falls within rollout percentage."""
        if flag.rollout_percentage >= 100:
            return True
        if flag.rollout_percentage <= 0:
            return False

        if user_id is None:
            # Use random for anonymous users (non-sticky)
            import random

            return random.randint(1, 100) <= flag.rollout_percentage  # nosec B311

        # Consistent hash for sticky rollout
        hash_percentage = self._get_hash_percentage(flag.name, user_id)
        return hash_percentage < flag.rollout_percentage

    def _get_hash_percentage(self, flag_name: str, user_id: str) -> int:
        """Get consistent hash percentage (0-99) for a user/flag combination."""
        combined = f"{flag_name}:{user_id}"
        hash_bytes = hashlib.md5(combined.encode(), usedforsecurity=False).digest()
        return int.from_bytes(hash_bytes[:2], "big") % 100

    def _record_evaluation(self, flag_name: str, result: bool) -> None:
        """Record flag evaluation for stats."""
        if flag_name not in self._stats:
            self._stats[flag_name] = {"enabled": 0, "disabled": 0}

        key = "enabled" if result else "disabled"
        self._stats[flag_name][key] += 1

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get evaluation statistics."""
        return self._stats.copy()

    def update_flag(
        self,
        flag_name: str,
        enabled: Optional[bool] = None,
        rollout_percentage: Optional[int] = None,
        force_value: Optional[bool] = None,
    ) -> bool:
        """
        Update a feature flag at runtime.

        Args:
            flag_name: Name of the flag to update
            enabled: New enabled state
            rollout_percentage: New rollout percentage
            force_value: Force value override (emergency use)

        Returns:
            bool: True if update successful
        """
        flag = self._flags.get(flag_name)

        if flag is None:
            logger.warning(f"Cannot update unknown flag: {flag_name}")
            return False

        if enabled is not None:
            flag.enabled = enabled

        if rollout_percentage is not None:
            flag.rollout_percentage = min(100, max(0, rollout_percentage))

        if force_value is not None:
            flag.force_value = force_value

        flag.updated_at = datetime.now(timezone.utc)

        # Notify listeners
        self._notify_listeners(flag_name, flag)

        logger.info(f"Updated feature flag: {flag_name}")
        return True

    def set_emergency_bypass(self, flag_name: str, bypass: bool) -> bool:
        """
        Set emergency bypass for a flag.

        Args:
            flag_name: Name of the bypass flag
            bypass: True to enable bypass, False to disable

        Returns:
            bool: True if successful
        """
        return self.update_flag(flag_name, force_value=bypass)

    def register_listener(self, flag_name: str, callback: Callable[[str, FeatureFlag], None]) -> None:
        """Register a listener for flag changes."""
        if flag_name not in self._listeners:
            self._listeners[flag_name] = []
        self._listeners[flag_name].append(callback)

    def _notify_listeners(self, flag_name: str, flag: FeatureFlag) -> None:
        """Notify listeners of flag change."""
        for callback in self._listeners.get(flag_name, []):
            try:
                callback(flag_name, flag)
            except Exception as e:
                logger.error(f"Error notifying listener for {flag_name}: {e}")


# Singleton instance
_feature_flag_service: Optional[FeatureFlagService] = None


def get_feature_flag_service() -> FeatureFlagService:
    """Get the feature flag service singleton."""
    global _feature_flag_service
    if _feature_flag_service is None:
        _feature_flag_service = FeatureFlagService()
    return _feature_flag_service


def is_feature_enabled(
    flag_name: str,
    user_id: Optional[str] = None,
    segment: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Convenience function to check if a feature is enabled.

    Args:
        flag_name: Name of the feature flag
        user_id: Optional user identifier
        segment: Optional user segment
        context: Optional additional context

    Returns:
        bool: True if the feature is enabled
    """
    return get_feature_flag_service().is_enabled(flag_name, user_id, segment, context)


def feature_flag(flag_name: str, fallback: Optional[Callable] = None, segment_key: Optional[str] = None):
    """
    Decorator to conditionally execute function based on feature flag.

    Args:
        flag_name: Name of the feature flag
        fallback: Optional fallback function if feature is disabled
        segment_key: Optional key to extract segment from request

    Usage:
        @feature_flag("new_prediction_model", fallback=old_predict)
        def predict(data):
            return new_model.predict(data)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get user_id and segment from Flask context
            from flask import g, has_request_context

            user_id = None
            segment = None

            if has_request_context():
                user_id = getattr(g, "user_id", None)
                segment = getattr(g, "user_segment", None) or getattr(g, segment_key, None) if segment_key else None

            if is_feature_enabled(flag_name, user_id, segment):
                return func(*args, **kwargs)
            elif fallback:
                return fallback(*args, **kwargs)
            else:
                logger.debug(f"Feature {flag_name} is disabled, skipping {func.__name__}")
                return None

        return wrapper

    return decorator


def get_alert_thresholds(user_id: str) -> Dict[str, float]:
    """
    Get alert thresholds based on A/B experiment.

    Args:
        user_id: User identifier

    Returns:
        Dictionary of alert thresholds
    """
    service = get_feature_flag_service()

    config = service.get_experiment_config("alert_threshold_experiment", user_id)

    if config:
        return config

    # Default thresholds
    return {"flood_risk_low": 0.3, "flood_risk_medium": 0.5, "flood_risk_high": 0.7}


def should_use_model_v2(user_id: Optional[str] = None) -> bool:
    """
    Check if model v2 should be used for this request.

    Args:
        user_id: Optional user identifier

    Returns:
        bool: True if model v2 should be used
    """
    service = get_feature_flag_service()

    # Check full release flag first
    if service.is_enabled("model_v2_full_release"):
        return True

    # Check gradual rollout
    return service.is_enabled("model_v2_rollout", user_id=user_id)


def should_bypass_external_api(api_name: str) -> bool:
    """
    Check if external API calls should be bypassed.

    Args:
        api_name: Name of the external API (openweathermap, weatherstack, worldtides)

    Returns:
        bool: True if API should be bypassed (use fallback/cached data)
    """
    service = get_feature_flag_service()

    # Check global bypass first
    if service.is_enabled("bypass_all_external_apis"):
        logger.warning(f"Bypassing {api_name} due to global API bypass flag")
        return True

    # Check specific API bypass
    flag_name = f"bypass_{api_name}"
    if service.is_enabled(flag_name):
        logger.warning(f"Bypassing {api_name} due to specific bypass flag")
        return True

    return False


def is_internal_service(segment: Optional[str] = None) -> bool:
    """
    Check if request is from an internal service.

    Args:
        segment: User segment or service identifier

    Returns:
        bool: True if from internal service
    """
    if segment is None:
        # Try to get from Flask context
        from flask import has_request_context, request

        if has_request_context():
            # Check internal service header
            internal_token = request.headers.get("X-Internal-Token")
            expected_token = os.getenv("INTERNAL_API_TOKEN")

            if internal_token and expected_token and internal_token == expected_token:
                return True

    return get_feature_flag_service().is_enabled("rate_limit_internal_bypass", segment=segment)
