"""Configuration management for Floodingnaque API."""

from dotenv import load_dotenv
import os
from dataclasses import dataclass, field
from typing import List, Optional


def load_env():
    """Load environment variables from .env file."""
    load_dotenv()


@dataclass
class Config:
    """
    Application configuration with validation.
    
    All values are loaded from environment variables with sensible defaults.
    """
    
    # Flask Settings
    SECRET_KEY: str = field(default_factory=lambda: os.getenv('SECRET_KEY', 'change-me-in-production'))
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
