"""Environment Variable Validation.

Validates required and optional environment variables on application startup.
Provides early detection of configuration issues with helpful error messages.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any, Dict, Set
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity level for validation issues."""
    CRITICAL = "critical"   # Application cannot start
    ERROR = "error"         # Significant issue, may cause failures
    WARNING = "warning"     # Potential issue, application can start
    INFO = "info"           # Informational message


@dataclass
class ValidationResult:
    """Result of validating an environment variable."""
    name: str
    valid: bool
    severity: ValidationSeverity
    message: str
    value_preview: Optional[str] = None  # Partial value for debugging (redacted)


@dataclass
class EnvVarSpec:
    """Specification for an environment variable."""
    name: str
    required: bool = False
    required_in_prod: bool = False
    default: Optional[str] = None
    description: str = ""
    validator: Optional[Callable[[str], bool]] = None
    validator_message: str = ""
    sensitive: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[str]] = None
    type_hint: str = "string"  # string, int, float, bool, url


# Built-in validators
def is_url(value: str) -> bool:
    """Validate URL format."""
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, value, re.IGNORECASE))


def is_email(value: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, value))


def is_positive_int(value: str) -> bool:
    """Validate positive integer."""
    try:
        return int(value) > 0
    except ValueError:
        return False


def is_non_negative_int(value: str) -> bool:
    """Validate non-negative integer."""
    try:
        return int(value) >= 0
    except ValueError:
        return False


def is_percentage(value: str) -> bool:
    """Validate percentage (0-100)."""
    try:
        val = float(value)
        return 0 <= val <= 100
    except ValueError:
        return False


def is_boolean(value: str) -> bool:
    """Validate boolean string."""
    return value.lower() in ('true', 'false', '1', '0', 'yes', 'no')


def is_hex_key(value: str, min_length: int = 32) -> bool:
    """Validate hexadecimal key."""
    return len(value) >= min_length and all(c in '0123456789abcdefABCDEF' for c in value)


def is_postgres_url(value: str) -> bool:
    """Validate PostgreSQL connection URL."""
    return value.startswith(('postgresql://', 'postgres://'))


def is_redis_url(value: str) -> bool:
    """Validate Redis connection URL."""
    return value.startswith(('redis://', 'rediss://'))


# Environment variable specifications
ENV_VAR_SPECS: List[EnvVarSpec] = [
    # Required in all environments
    EnvVarSpec(
        name="SECRET_KEY",
        required_in_prod=True,
        sensitive=True,
        min_length=32,
        description="Flask secret key for session security"
    ),
    EnvVarSpec(
        name="JWT_SECRET_KEY",
        required_in_prod=True,
        sensitive=True,
        min_length=32,
        description="JWT signing key"
    ),
    
    # Database
    EnvVarSpec(
        name="DATABASE_URL",
        required_in_prod=True,
        sensitive=True,
        validator=is_postgres_url,
        validator_message="Must be a PostgreSQL connection URL",
        description="Database connection string"
    ),
    EnvVarSpec(
        name="DB_POOL_SIZE",
        default="20",
        validator=is_positive_int,
        validator_message="Must be a positive integer",
        type_hint="int",
        description="Database connection pool size"
    ),
    EnvVarSpec(
        name="DB_MAX_OVERFLOW",
        default="10",
        validator=is_non_negative_int,
        validator_message="Must be a non-negative integer",
        type_hint="int",
        description="Maximum pool overflow connections"
    ),
    
    # API Keys
    EnvVarSpec(
        name="OWM_API_KEY",
        required_in_prod=True,
        sensitive=True,
        min_length=16,
        description="OpenWeatherMap API key"
    ),
    EnvVarSpec(
        name="WEATHERSTACK_API_KEY",
        sensitive=True,
        min_length=16,
        description="Weatherstack API key (fallback weather source)"
    ),
    EnvVarSpec(
        name="WORLDTIDES_API_KEY",
        sensitive=True,
        description="WorldTides API key for tidal data"
    ),
    EnvVarSpec(
        name="INTERNAL_API_TOKEN",
        sensitive=True,
        min_length=32,
        description="Internal service authentication token"
    ),
    
    # Environment and App Settings
    EnvVarSpec(
        name="APP_ENV",
        default="development",
        allowed_values=["development", "staging", "production", "test", "dev", "prod", "stage"],
        description="Application environment"
    ),
    EnvVarSpec(
        name="FLASK_DEBUG",
        default="False",
        validator=is_boolean,
        validator_message="Must be True or False",
        type_hint="bool",
        description="Flask debug mode"
    ),
    EnvVarSpec(
        name="PORT",
        default="5000",
        validator=is_positive_int,
        validator_message="Must be a positive integer",
        type_hint="int",
        description="Server port"
    ),
    EnvVarSpec(
        name="HOST",
        default="0.0.0.0",
        description="Server host"
    ),
    
    # CORS
    EnvVarSpec(
        name="CORS_ORIGINS",
        required_in_prod=True,
        description="Allowed CORS origins (comma-separated)"
    ),
    
    # Rate Limiting
    EnvVarSpec(
        name="RATE_LIMIT_ENABLED",
        default="True",
        validator=is_boolean,
        validator_message="Must be True or False",
        type_hint="bool",
        description="Enable rate limiting"
    ),
    EnvVarSpec(
        name="RATE_LIMIT_DEFAULT",
        default="100",
        validator=is_positive_int,
        validator_message="Must be a positive integer",
        type_hint="int",
        description="Default rate limit per window"
    ),
    EnvVarSpec(
        name="RATE_LIMIT_STORAGE_URL",
        default="memory://",
        description="Rate limit storage backend URL"
    ),
    
    # Redis (Optional but recommended for production)
    EnvVarSpec(
        name="REDIS_URL",
        sensitive=True,
        validator=is_redis_url,
        validator_message="Must be a Redis URL (redis:// or rediss://)",
        description="Redis connection URL for caching and sessions"
    ),
    
    # Logging
    EnvVarSpec(
        name="LOG_LEVEL",
        default="INFO",
        allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        description="Logging level"
    ),
    EnvVarSpec(
        name="LOG_FORMAT",
        default="json",
        allowed_values=["json", "ecs", "text"],
        description="Log output format"
    ),
    
    # Sentry
    EnvVarSpec(
        name="SENTRY_DSN",
        sensitive=True,
        validator=is_url,
        validator_message="Must be a valid Sentry DSN URL",
        description="Sentry error tracking DSN"
    ),
    
    # Security Headers
    EnvVarSpec(
        name="ENABLE_HTTPS",
        default="True",
        validator=is_boolean,
        validator_message="Must be True or False",
        type_hint="bool",
        description="Enable HTTPS enforcement"
    ),
    EnvVarSpec(
        name="HSTS_MAX_AGE",
        default="31536000",
        validator=is_non_negative_int,
        validator_message="Must be a non-negative integer",
        type_hint="int",
        description="HSTS max-age in seconds"
    ),
    
    # Feature Flags
    EnvVarSpec(
        name="FEATURE_FLAG_MODEL_V2_ROLLOUT",
        default="0",
        validator=is_percentage,
        validator_message="Must be a percentage (0-100)",
        type_hint="int",
        description="Model v2 rollout percentage"
    ),
    
    # Model Configuration
    EnvVarSpec(
        name="MODEL_DIR",
        default="models",
        description="Directory containing ML models"
    ),
    EnvVarSpec(
        name="MODEL_NAME",
        default="flood_rf_model",
        description="Name of the prediction model"
    ),
    
    # Health Check SLA
    EnvVarSpec(
        name="HEALTH_CHECK_RESPONSE_TIME_SLA_MS",
        default="500",
        validator=is_positive_int,
        validator_message="Must be a positive integer",
        type_hint="int",
        description="Health check response time SLA in milliseconds"
    ),
    EnvVarSpec(
        name="HEALTH_CHECK_DB_TIMEOUT_SECONDS",
        default="5",
        validator=is_positive_int,
        validator_message="Must be a positive integer",
        type_hint="int",
        description="Database health check timeout"
    ),
]


@dataclass
class ValidationReport:
    """Summary of environment variable validation."""
    results: List[ValidationResult] = field(default_factory=list)
    critical_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    
    @property
    def is_valid(self) -> bool:
        """Check if all critical validations passed."""
        return self.critical_count == 0
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return self.error_count > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return self.warning_count > 0
    
    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)
        
        if result.severity == ValidationSeverity.CRITICAL:
            self.critical_count += 1
        elif result.severity == ValidationSeverity.ERROR:
            self.error_count += 1
        elif result.severity == ValidationSeverity.WARNING:
            self.warning_count += 1
        else:
            self.info_count += 1
    
    def get_summary(self) -> str:
        """Get a summary string."""
        parts = []
        if self.critical_count:
            parts.append(f"{self.critical_count} critical")
        if self.error_count:
            parts.append(f"{self.error_count} errors")
        if self.warning_count:
            parts.append(f"{self.warning_count} warnings")
        if self.info_count:
            parts.append(f"{self.info_count} info")
        
        return ", ".join(parts) if parts else "All validations passed"


def _redact_value(value: str, sensitive: bool, preview_length: int = 8) -> str:
    """Create a redacted preview of a value."""
    if sensitive:
        if len(value) <= preview_length:
            return "***"
        return value[:4] + "..." + value[-4:]
    
    if len(value) > 50:
        return value[:47] + "..."
    return value


def validate_env_var(spec: EnvVarSpec, is_production: bool) -> ValidationResult:
    """Validate a single environment variable."""
    value = os.getenv(spec.name, "")
    
    # Check if variable is set
    if not value:
        if spec.required or (spec.required_in_prod and is_production):
            return ValidationResult(
                name=spec.name,
                valid=False,
                severity=ValidationSeverity.CRITICAL if is_production else ValidationSeverity.ERROR,
                message=f"Required environment variable '{spec.name}' is not set. {spec.description}"
            )
        elif spec.default is not None:
            return ValidationResult(
                name=spec.name,
                valid=True,
                severity=ValidationSeverity.INFO,
                message=f"Using default value for '{spec.name}'",
                value_preview=spec.default
            )
        else:
            return ValidationResult(
                name=spec.name,
                valid=True,
                severity=ValidationSeverity.INFO,
                message=f"Optional variable '{spec.name}' is not set"
            )
    
    # Check minimum length
    if spec.min_length and len(value) < spec.min_length:
        severity = ValidationSeverity.ERROR if is_production else ValidationSeverity.WARNING
        return ValidationResult(
            name=spec.name,
            valid=False,
            severity=severity,
            message=f"'{spec.name}' is too short (min {spec.min_length} characters). {spec.description}",
            value_preview=_redact_value(value, spec.sensitive)
        )
    
    # Check maximum length
    if spec.max_length and len(value) > spec.max_length:
        return ValidationResult(
            name=spec.name,
            valid=False,
            severity=ValidationSeverity.WARNING,
            message=f"'{spec.name}' is too long (max {spec.max_length} characters)",
            value_preview=_redact_value(value, spec.sensitive)
        )
    
    # Check pattern
    if spec.pattern and not re.match(spec.pattern, value):
        return ValidationResult(
            name=spec.name,
            valid=False,
            severity=ValidationSeverity.ERROR,
            message=f"'{spec.name}' does not match required pattern",
            value_preview=_redact_value(value, spec.sensitive)
        )
    
    # Check allowed values
    if spec.allowed_values and value not in spec.allowed_values:
        return ValidationResult(
            name=spec.name,
            valid=False,
            severity=ValidationSeverity.ERROR,
            message=f"'{spec.name}' has invalid value. Allowed: {', '.join(spec.allowed_values)}",
            value_preview=value
        )
    
    # Run custom validator
    if spec.validator and not spec.validator(value):
        return ValidationResult(
            name=spec.name,
            valid=False,
            severity=ValidationSeverity.ERROR,
            message=f"'{spec.name}' validation failed: {spec.validator_message}",
            value_preview=_redact_value(value, spec.sensitive)
        )
    
    # Validation passed
    return ValidationResult(
        name=spec.name,
        valid=True,
        severity=ValidationSeverity.INFO,
        message=f"'{spec.name}' is valid",
        value_preview=_redact_value(value, spec.sensitive)
    )


def validate_all_env_vars(
    additional_specs: Optional[List[EnvVarSpec]] = None,
    raise_on_critical: bool = True,
    log_results: bool = True
) -> ValidationReport:
    """
    Validate all environment variables.
    
    Args:
        additional_specs: Additional variable specifications to validate
        raise_on_critical: Raise exception if critical validation fails
        log_results: Log validation results
    
    Returns:
        ValidationReport with all results
    
    Raises:
        ValueError: If critical validation fails and raise_on_critical is True
    """
    app_env = os.getenv('APP_ENV', 'development').lower()
    is_production = app_env in ('production', 'prod', 'staging', 'stage')
    
    specs = ENV_VAR_SPECS.copy()
    if additional_specs:
        specs.extend(additional_specs)
    
    report = ValidationReport()
    
    for spec in specs:
        result = validate_env_var(spec, is_production)
        report.add_result(result)
        
        if log_results:
            if result.severity == ValidationSeverity.CRITICAL:
                logger.critical(result.message)
            elif result.severity == ValidationSeverity.ERROR:
                logger.error(result.message)
            elif result.severity == ValidationSeverity.WARNING:
                logger.warning(result.message)
            elif not result.valid:
                logger.info(result.message)
    
    # Log summary
    if log_results:
        if report.is_valid:
            if report.has_warnings:
                logger.warning(f"Environment validation completed with warnings: {report.get_summary()}")
            else:
                logger.info(f"Environment validation passed: {report.get_summary()}")
        else:
            logger.critical(f"Environment validation FAILED: {report.get_summary()}")
    
    # Raise exception for critical failures in production
    if raise_on_critical and not report.is_valid and is_production:
        failed_vars = [r.name for r in report.results if r.severity == ValidationSeverity.CRITICAL]
        raise ValueError(
            f"Critical environment validation failed. Missing or invalid: {', '.join(failed_vars)}. "
            "Please configure these environment variables before starting in production."
        )
    
    return report


def get_missing_required_vars() -> List[str]:
    """Get list of missing required environment variables."""
    app_env = os.getenv('APP_ENV', 'development').lower()
    is_production = app_env in ('production', 'prod', 'staging', 'stage')
    
    missing = []
    for spec in ENV_VAR_SPECS:
        if spec.required or (spec.required_in_prod and is_production):
            value = os.getenv(spec.name, "")
            if not value:
                missing.append(spec.name)
    
    return missing


def get_env_var_documentation() -> Dict[str, Dict[str, Any]]:
    """Get documentation for all environment variables."""
    docs = {}
    
    for spec in ENV_VAR_SPECS:
        docs[spec.name] = {
            'description': spec.description,
            'required': spec.required,
            'required_in_prod': spec.required_in_prod,
            'default': spec.default,
            'type': spec.type_hint,
            'sensitive': spec.sensitive,
            'allowed_values': spec.allowed_values
        }
    
    return docs
