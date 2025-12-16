"""Configuration management for Floodingnaque API."""

from dotenv import load_dotenv
import os
import secrets
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)


def is_debug_mode() -> bool:
    """
    Check if application is running in debug mode.
    
    Centralized check for consistent debug mode detection across the app.
    Uses FLASK_DEBUG environment variable.
    
    Returns:
        bool: True if debug mode is enabled
    """
    return os.getenv('FLASK_DEBUG', 'False').lower() == 'true'


def _get_secret_key() -> str:
    """
    Get SECRET_KEY from environment with security validation.
    
    In production (FLASK_DEBUG=False), fails if not explicitly set.
    In development, generates a temporary key with warning.
    """
    key = os.getenv('SECRET_KEY')
    is_debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    if not key or key in ('change-me-in-production', 'change_this_to_a_random_secret_key_in_production'):
        if not is_debug:
            raise ValueError(
                "CRITICAL: SECRET_KEY must be set in production! "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        # Development mode - generate temporary key
        key = secrets.token_hex(32)
        logger.warning(
            "SECRET_KEY not configured - using temporary key. "
            "Set SECRET_KEY in .env for persistent sessions."
        )
    return key


def _get_jwt_secret_key() -> str:
    """Get JWT_SECRET_KEY with security validation."""
    key = os.getenv('JWT_SECRET_KEY')
    is_debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    if not key or key in ('change_this_to_another_random_secret_key',):
        if not is_debug:
            raise ValueError(
                "CRITICAL: JWT_SECRET_KEY must be set in production! "
                "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        key = secrets.token_hex(32)
        logger.warning("JWT_SECRET_KEY not configured - using temporary key.")
    return key


def load_env():
    """Load environment variables from .env file."""
    load_dotenv()


@dataclass
class Config:
    """
    Application configuration with validation.
    
    All values are loaded from environment variables with sensible defaults.
    """
    
    # Flask Settings - SECRET_KEY fails in production if not set
    SECRET_KEY: str = field(default_factory=_get_secret_key)
    JWT_SECRET_KEY: str = field(default_factory=_get_jwt_secret_key)
    DEBUG: bool = field(default_factory=lambda: os.getenv('FLASK_DEBUG', 'False').lower() == 'true')
    HOST: str = field(default_factory=lambda: os.getenv('HOST', '0.0.0.0'))
    PORT: int = field(default_factory=lambda: int(os.getenv('PORT', '5000')))
    
    # Database
    DATABASE_URL: str = field(default_factory=lambda: os.getenv('DATABASE_URL', 'sqlite:///data/floodingnaque.db'))
    DB_POOL_SIZE: int = field(default_factory=lambda: int(os.getenv('DB_POOL_SIZE', '20')))
    DB_MAX_OVERFLOW: int = field(default_factory=lambda: int(os.getenv('DB_MAX_OVERFLOW', '10')))
    DB_POOL_RECYCLE: int = field(default_factory=lambda: int(os.getenv('DB_POOL_RECYCLE', '3600')))
    
    # Security
    RATE_LIMIT_ENABLED: bool = field(default_factory=lambda: os.getenv('RATE_LIMIT_ENABLED', 'True').lower() == 'true')
    RATE_LIMIT_DEFAULT: int = field(default_factory=lambda: int(os.getenv('RATE_LIMIT_DEFAULT', '100')))
    ENABLE_HTTPS: bool = field(default_factory=lambda: os.getenv('ENABLE_HTTPS', 'False').lower() == 'true')
    
    # JWT Token Settings
    JWT_ACCESS_TOKEN_EXPIRES_MINUTES: int = field(
        default_factory=lambda: int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES_MINUTES', '30'))
    )
    JWT_REFRESH_TOKEN_EXPIRES_DAYS: int = field(
        default_factory=lambda: int(os.getenv('JWT_REFRESH_TOKEN_EXPIRES_DAYS', '7'))
    )
    JWT_TOKEN_LOCATION: List[str] = field(
        default_factory=lambda: os.getenv('JWT_TOKEN_LOCATION', 'headers').split(',')
    )
    JWT_ALGORITHM: str = field(default_factory=lambda: os.getenv('JWT_ALGORITHM', 'HS256'))
    
    # Model Security
    MODEL_SIGNING_KEY: str = field(default_factory=lambda: os.getenv('MODEL_SIGNING_KEY', ''))
    REQUIRE_MODEL_SIGNATURE: bool = field(
        default_factory=lambda: os.getenv('REQUIRE_MODEL_SIGNATURE', 'false').lower() == 'true'
    )
    VERIFY_MODEL_INTEGRITY: bool = field(
        default_factory=lambda: os.getenv('VERIFY_MODEL_INTEGRITY', 'true').lower() == 'true'
    )
    
    # CORS
    CORS_ORIGINS: str = field(default_factory=lambda: os.getenv('CORS_ORIGINS', 'https://floodingnaque.vercel.app'))
    
    # External APIs
    OWM_API_KEY: str = field(default_factory=lambda: os.getenv('OWM_API_KEY', ''))
    WEATHERSTACK_API_KEY: str = field(default_factory=lambda: os.getenv('WEATHERSTACK_API_KEY', ''))
    
    # Model Configuration
    MODEL_DIR: str = field(default_factory=lambda: os.getenv('MODEL_DIR', 'models'))
    MODEL_NAME: str = field(default_factory=lambda: os.getenv('MODEL_NAME', 'flood_rf_model'))
    
    # Logging
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    LOG_FORMAT: str = field(default_factory=lambda: os.getenv('LOG_FORMAT', 'json'))
    
    # Default Location (Parañaque City)
    DEFAULT_LATITUDE: float = field(default_factory=lambda: float(os.getenv('DEFAULT_LATITUDE', '14.4793')))
    DEFAULT_LONGITUDE: float = field(default_factory=lambda: float(os.getenv('DEFAULT_LONGITUDE', '121.0198')))
    
    def get_cors_origins_list(self) -> List[str]:
        """Get CORS origins as a list."""
        if not self.CORS_ORIGINS:
            return []
        return [origin.strip() for origin in self.CORS_ORIGINS.split(',') if origin.strip()]
    
    def get_jwt_access_token_expires(self) -> timedelta:
        """Get JWT access token expiration as timedelta."""
        return timedelta(minutes=self.JWT_ACCESS_TOKEN_EXPIRES_MINUTES)
    
    def get_jwt_refresh_token_expires(self) -> timedelta:
        """Get JWT refresh token expiration as timedelta."""
        return timedelta(days=self.JWT_REFRESH_TOKEN_EXPIRES_DAYS)
    
    @classmethod
    def validate(cls) -> List[str]:
        """
        Validate configuration and return list of warnings/errors.
        
        Returns:
            List of warning/error messages (empty if all valid)
        """
        warnings = []
        config = cls()
        
        # Check required API keys in production
        if not config.DEBUG:
            if not config.OWM_API_KEY or config.OWM_API_KEY == 'your_openweathermap_api_key_here':
                warnings.append("OWM_API_KEY is not configured")
            
            if config.SECRET_KEY == 'change-me-in-production':
                warnings.append("SECRET_KEY should be changed in production")
            
            if not config.CORS_ORIGINS:
                warnings.append("CORS_ORIGINS should be configured in production")
            
            if not config.ENABLE_HTTPS:
                warnings.append("HTTPS should be enabled in production")
            
            # JWT security checks
            if config.JWT_ACCESS_TOKEN_EXPIRES_MINUTES > 60:
                warnings.append(
                    f"JWT_ACCESS_TOKEN_EXPIRES_MINUTES is {config.JWT_ACCESS_TOKEN_EXPIRES_MINUTES}. "
                    "Consider shorter expiration (15-30 minutes) for production."
                )
            
            # Model security checks
            if config.REQUIRE_MODEL_SIGNATURE and not config.MODEL_SIGNING_KEY:
                warnings.append(
                    "REQUIRE_MODEL_SIGNATURE is enabled but MODEL_SIGNING_KEY is not set"
                )
        
        return warnings
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Create Config instance from environment variables."""
        load_env()
        return cls()


# Global config instance (lazy initialization)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config
