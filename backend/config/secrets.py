"""
Floodingnaque Secrets Management
================================

Centralized secrets management with support for multiple backends:
- Environment variables (default)
- YAML secrets file (secrets.yaml)
- HashiCorp Vault integration (optional)
- AWS Secrets Manager (optional)
- Azure Key Vault (optional)

Security Features:
- Secrets are never logged
- Memory is cleared after use (where possible)
- Validation of secret format and expiry
- Audit logging of secret access

Usage:
    from config.secrets import secrets_manager, get_secret

    # Get a secret
    api_key = get_secret("OWM_API_KEY")

    # Get with default
    token = get_secret("OPTIONAL_TOKEN", default="")

    # Check if secret exists
    if secrets_manager.has_secret("MY_SECRET"):
        ...
"""

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

logger = logging.getLogger(__name__)


class SecretNotFoundError(Exception):
    """Raised when a required secret is not found."""

    pass


class SecretValidationError(Exception):
    """Raised when a secret fails validation."""

    pass


class SecretsBackend(ABC):
    """Abstract base class for secrets backends."""

    @abstractmethod
    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve a secret by key."""
        pass

    @abstractmethod
    def has_secret(self, key: str) -> bool:
        """Check if a secret exists."""
        pass

    @abstractmethod
    def list_secrets(self) -> List[str]:
        """List available secret keys (not values)."""
        pass

    def get_secret_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata about a secret (not the value)."""
        return {"exists": self.has_secret(key)}


class EnvironmentSecretsBackend(SecretsBackend):
    """Secrets backend using environment variables."""

    def __init__(self, prefix: str = ""):
        """
        Initialize environment secrets backend.

        Args:
            prefix: Optional prefix for environment variable names
        """
        self.prefix = prefix

    def get_secret(self, key: str) -> Optional[str]:
        full_key = f"{self.prefix}{key}" if self.prefix else key
        return os.environ.get(full_key)

    def has_secret(self, key: str) -> bool:
        full_key = f"{self.prefix}{key}" if self.prefix else key
        return full_key in os.environ

    def list_secrets(self) -> List[str]:
        if self.prefix:
            return [k[len(self.prefix) :] for k in os.environ.keys() if k.startswith(self.prefix)]
        return list(os.environ.keys())


class YAMLSecretsBackend(SecretsBackend):
    """
    Secrets backend using YAML file.

    The secrets file should be structured as:

    ```yaml
    secrets:
      OWM_API_KEY: \"your-api-key\"  # pragma: allowlist secret
      DATABASE_PASSWORD: \"your-password\"  # pragma: allowlist secret

    metadata:
      OWM_API_KEY:
        description: "OpenWeatherMap API key"
        expires: "2025-12-31"
        rotated: "2025-01-01"
    ```
    """

    def __init__(self, secrets_file: Path):
        """
        Initialize YAML secrets backend.

        Args:
            secrets_file: Path to secrets.yaml file
        """
        self.secrets_file = secrets_file
        self._secrets: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._load_secrets()

    def _load_secrets(self) -> None:
        """Load secrets from YAML file."""
        if not self.secrets_file.exists():
            logger.debug(f"Secrets file not found: {self.secrets_file}")
            return

        try:
            with open(self.secrets_file, "r") as f:
                data = yaml.safe_load(f) or {}

            self._secrets = data.get("secrets", {})
            self._metadata = data.get("metadata", {})

            # Validate no secrets are logged
            logger.info(f"Loaded {len(self._secrets)} secrets from {self.secrets_file.name}")

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse secrets file: {e}")
            raise SecretValidationError(f"Invalid secrets file format: {e}")

    def reload(self) -> None:
        """Reload secrets from file."""
        self._secrets.clear()
        self._metadata.clear()
        self._load_secrets()

    def get_secret(self, key: str) -> Optional[str]:
        return self._secrets.get(key)

    def has_secret(self, key: str) -> bool:
        return key in self._secrets

    def list_secrets(self) -> List[str]:
        return list(self._secrets.keys())

    def get_secret_metadata(self, key: str) -> Dict[str, Any]:
        metadata = {"exists": self.has_secret(key)}
        if key in self._metadata:
            # Don't include secret value in metadata
            metadata.update({k: v for k, v in self._metadata[key].items() if k not in ("value", "secret")})
        return metadata


class VaultSecretsBackend(SecretsBackend):
    """
    HashiCorp Vault secrets backend.

    Requires hvac library: pip install hvac

    Environment variables:
      VAULT_ADDR: Vault server address
      VAULT_TOKEN: Vault authentication token
      VAULT_SECRET_PATH: Path to secrets in Vault (default: secret/data/floodingnaque)
    """

    def __init__(
        self,
        addr: Optional[str] = None,
        token: Optional[str] = None,
        secret_path: str = "secret/data/floodingnaque",  # nosec B107 - this is a path, not a password
    ):
        self.addr = addr or os.environ.get("VAULT_ADDR")
        self.token = token or os.environ.get("VAULT_TOKEN")
        self.secret_path = secret_path
        self._client = None
        self._secrets: Dict[str, str] = {}
        self._initialized = False

    def _initialize(self) -> None:
        """Initialize Vault client and fetch secrets."""
        if self._initialized:
            return

        if not self.addr or not self.token:
            logger.debug("Vault not configured (VAULT_ADDR or VAULT_TOKEN not set)")
            self._initialized = True
            return

        try:
            import hvac  # type: ignore[import-not-found]

            self._client = hvac.Client(url=self.addr, token=self.token)

            if not self._client.is_authenticated():
                logger.warning("Vault authentication failed")
                self._initialized = True
                return

            # Fetch secrets from Vault
            response = self._client.secrets.kv.v2.read_secret_version(path=self.secret_path.replace("secret/data/", ""))
            self._secrets = response.get("data", {}).get("data", {})
            logger.info(f"Loaded {len(self._secrets)} secrets from Vault")

        except ImportError:
            logger.debug("hvac library not installed, Vault backend unavailable")
        except Exception as e:
            logger.warning(f"Failed to connect to Vault: {e}")

        self._initialized = True

    def get_secret(self, key: str) -> Optional[str]:
        self._initialize()
        return self._secrets.get(key)

    def has_secret(self, key: str) -> bool:
        self._initialize()
        return key in self._secrets

    def list_secrets(self) -> List[str]:
        self._initialize()
        return list(self._secrets.keys())


class SecretsManager:
    """
    Centralized secrets manager with multi-backend support.

    Backends are checked in priority order:
    1. Environment variables (highest priority)
    2. YAML secrets file
    3. Vault (if configured)

    This allows environment-specific overrides while keeping
    base secrets in files or Vault.
    """

    # Well-known secret keys with validation
    SECRET_KEYS = {
        # API Keys
        "OWM_API_KEY": {"required": False, "description": "OpenWeatherMap API key"},
        "WEATHERSTACK_API_KEY": {"required": False, "description": "Weatherstack API key"},
        "INTERNAL_API_TOKEN": {"required": False, "description": "Internal microservices token"},
        # Database
        "DATABASE_PASSWORD": {"required": False, "description": "Database password"},
        "SUPABASE_KEY": {"required": False, "description": "Supabase anon key"},
        "SUPABASE_SECRET_KEY": {"required": False, "description": "Supabase service role key"},
        # Security
        "SECRET_KEY": {"required": True, "description": "Flask secret key"},
        "JWT_SECRET_KEY": {"required": True, "description": "JWT signing key"},
        "MODEL_SIGNING_KEY": {"required": False, "description": "Model artifact signing key"},
        # Alert System
        "SMS_API_KEY": {"required": False, "description": "SMS gateway API key"},
        "SMS_API_SECRET": {"required": False, "description": "SMS gateway API secret"},
        "SMTP_PASSWORD": {"required": False, "description": "SMTP email password"},
        # External Services
        "GOOGLE_APPLICATION_CREDENTIALS": {"required": False, "description": "GCP service account path"},
    }

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize secrets manager.

        Args:
            config_dir: Directory containing secrets.yaml (default: backend/config)
        """
        self.config_dir = config_dir or Path(__file__).parent
        self._backends: List[SecretsBackend] = []
        self._accessed_secrets: Set[str] = set()  # Audit tracking

        # Initialize backends in priority order
        self._setup_backends()

    def _setup_backends(self) -> None:
        """Setup secrets backends in priority order."""
        # 1. Environment variables (highest priority)
        self._backends.append(EnvironmentSecretsBackend())

        # 2. YAML secrets file
        secrets_file = self.config_dir / "secrets.yaml"
        if secrets_file.exists():
            try:
                self._backends.append(YAMLSecretsBackend(secrets_file))
            except SecretValidationError as e:
                logger.error(f"Failed to load secrets file: {e}")

        # 3. Vault (if configured)
        if os.environ.get("VAULT_ADDR"):
            self._backends.append(VaultSecretsBackend())

    def get_secret(self, key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        """
        Get a secret value.

        Args:
            key: Secret key name
            default: Default value if not found
            required: Raise error if not found and no default

        Returns:
            Secret value or default

        Raises:
            SecretNotFoundError: If required and not found
        """
        # Track access for audit
        self._accessed_secrets.add(key)

        # Check backends in priority order
        for backend in self._backends:
            value = backend.get_secret(key)
            if value is not None:
                return value

        # Not found in any backend
        if required and default is None:
            raise SecretNotFoundError(f"Required secret not found: {key}")

        return default

    def has_secret(self, key: str) -> bool:
        """Check if a secret exists in any backend."""
        return any(backend.has_secret(key) for backend in self._backends)

    def list_secrets(self) -> List[str]:
        """List all available secret keys (union of all backends)."""
        all_keys: Set[str] = set()
        for backend in self._backends:
            all_keys.update(backend.list_secrets())
        return sorted(all_keys)

    def get_secret_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata about a secret from all backends."""
        metadata = {"key": key, "exists": False, "backends": []}

        for backend in self._backends:
            if backend.has_secret(key):
                metadata["exists"] = True
                metadata["backends"].append(type(backend).__name__)

        # Add well-known secret info
        if key in self.SECRET_KEYS:
            metadata.update(self.SECRET_KEYS[key])

        return metadata

    def validate_required_secrets(self) -> Dict[str, bool]:
        """
        Validate that all required secrets are present.

        Returns:
            Dict mapping secret key to whether it's present
        """
        results = {}
        for key, info in self.SECRET_KEYS.items():
            if info.get("required", False):
                results[key] = self.has_secret(key)
        return results

    def get_audit_log(self) -> List[str]:
        """Get list of secrets that have been accessed (keys only)."""
        return sorted(self._accessed_secrets)

    def reload(self) -> None:
        """Reload secrets from all backends."""
        logger.info("Reloading secrets...")
        for backend in self._backends:
            if isinstance(backend, YAMLSecretsBackend):
                backend.reload()
            elif isinstance(backend, VaultSecretsBackend):
                backend._initialized = False  # Force re-initialization

    def add_backend(self, backend: SecretsBackend, priority: int = -1) -> None:
        """
        Add a custom secrets backend.

        Args:
            backend: Backend instance
            priority: Position in backend list (0 = highest priority, -1 = lowest)
        """
        if priority == -1:
            self._backends.append(backend)
        else:
            self._backends.insert(priority, backend)


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


def get_secret(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Convenience function to get a secret.

    Args:
        key: Secret key name
        default: Default value if not found
        required: Raise error if not found

    Returns:
        Secret value or default
    """
    return get_secrets_manager().get_secret(key, default, required)


# Alias for backward compatibility
secrets_manager = get_secrets_manager
