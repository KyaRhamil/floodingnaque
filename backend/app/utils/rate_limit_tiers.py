"""
Rate Limiting Tiers Configuration.

Defines rate limiting tiers based on API keys for different service levels.
"""

import os
from typing import Dict, Optional
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RateLimitTier:
    """Rate limiting tier configuration."""
    
    def __init__(self, name: str, requests_per_minute: int, requests_per_hour: int, 
                 requests_per_day: int, burst_capacity: int = 10):
        self.name = name
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.requests_per_day = requests_per_day
        self.burst_capacity = burst_capacity
    
    def get_limits(self) -> Dict[str, str]:
        """Get rate limit strings for Flask-Limiter."""
        return {
            'per_minute': f"{self.requests_per_minute}/minute",
            'per_hour': f"{self.requests_per_hour}/hour", 
            'per_day': f"{self.requests_per_day}/day"
        }


# Define rate limiting tiers
RATE_LIMIT_TIERS = {
    'free': RateLimitTier(
        name='Free Tier',
        requests_per_minute=5,
        requests_per_hour=100,
        requests_per_day=1000,
        burst_capacity=10
    ),
    'basic': RateLimitTier(
        name='Basic Tier',
        requests_per_minute=20,
        requests_per_hour=500,
        requests_per_day=10000,
        burst_capacity=50
    ),
    'pro': RateLimitTier(
        name='Pro Tier',
        requests_per_minute=100,
        requests_per_hour=2000,
        requests_per_day=50000,
        burst_capacity=200
    ),
    'enterprise': RateLimitTier(
        name='Enterprise Tier',
        requests_per_minute=500,
        requests_per_hour=10000,
        requests_per_day=250000,
        burst_capacity=1000
    ),
    'unlimited': RateLimitTier(
        name='Unlimited Tier',
        requests_per_minute=10000,
        requests_per_hour=100000,
        requests_per_day=1000000,
        burst_capacity=5000
    )
}


def get_api_key_tier(api_key_hash: str) -> str:
    """
    Get the rate limiting tier for an API key hash.
    
    In a real implementation, this would:
    1. Look up the API key in a database
    2. Return the associated tier based on subscription plan
    3. Cache the result for performance
    
    For now, we use environment variables to configure tiers.
    
    Args:
        api_key_hash: Hashed API key
        
    Returns:
        str: Tier name ('free', 'basic', 'pro', 'enterprise', 'unlimited')
    """
    # Check environment variable for API key tiers
    tier_config = os.getenv('API_KEY_TIERS', '')
    
    if tier_config:
        # Format: "key_hash1:tier1,key_hash2:tier2"
        try:
            for mapping in tier_config.split(','):
                key_hash, tier = mapping.strip().split(':')
                if key_hash == api_key_hash:
                    return tier.lower()
        except (ValueError, AttributeError):
            logger.warning(f"Invalid API_KEY_TIERS format: {tier_config}")
    
    # Default tier based on API key prefix (for demo purposes)
    if api_key_hash.startswith('a'):
        return 'free'
    elif api_key_hash.startswith('b'):
        return 'basic'
    elif api_key_hash.startswith('c'):
        return 'pro'
    elif api_key_hash.startswith('d'):
        return 'enterprise'
    else:
        return 'free'  # Default to free tier


def get_tier_limits(tier_name: str) -> RateLimitTier:
    """
    Get rate limiting configuration for a tier.
    
    Args:
        tier_name: Name of the tier
        
    Returns:
        RateLimitTier: Tier configuration
    """
    tier = RATE_LIMIT_TIERS.get(tier_name.lower())
    if not tier:
        logger.warning(f"Unknown tier '{tier_name}', defaulting to free tier")
        return RATE_LIMIT_TIERS['free']
    
    return tier


def get_rate_limit_for_key(api_key_hash: str, limit_type: str = 'per_minute') -> str:
    """
    Get rate limit string for a specific API key.
    
    Args:
        api_key_hash: Hashed API key
        limit_type: Type of limit ('per_minute', 'per_hour', 'per_day')
        
    Returns:
        str: Rate limit string for Flask-Limiter (e.g., "100/minute")
    """
    tier_name = get_api_key_tier(api_key_hash)
    tier = get_tier_limits(tier_name)
    limits = tier.get_limits()
    
    return limits.get(limit_type, "10/minute")


def get_anonymous_limits() -> str:
    """
    Get rate limits for unauthenticated requests.
    
    Returns:
        str: Rate limit string for anonymous users
    """
    # Anonymous users get very restrictive limits
    return "2/minute;20/hour;100/day"


def check_rate_limit_status(api_key_hash: Optional[str] = None) -> Dict:
    """
    Check current rate limiting status for an API key.
    
    Args:
        api_key_hash: Hashed API key (None for anonymous)
        
    Returns:
        dict: Rate limiting status information
    """
    if api_key_hash:
        tier_name = get_api_key_tier(api_key_hash)
        tier = get_tier_limits(tier_name)
        
        return {
            'authenticated': True,
            'tier': tier_name,
            'tier_name': tier.name,
            'limits': {
                'per_minute': tier.requests_per_minute,
                'per_hour': tier.requests_per_hour,
                'per_day': tier.requests_per_day,
                'burst_capacity': tier.burst_capacity
            }
        }
    else:
        return {
            'authenticated': False,
            'tier': 'anonymous',
            'tier_name': 'Anonymous Users',
            'limits': {
                'per_minute': 2,
                'per_hour': 20,
                'per_day': 100,
                'burst_capacity': 5
            }
        }


# Environment-based configuration helpers
def load_api_key_tiers_from_env() -> Dict[str, str]:
    """
    Load API key to tier mappings from environment variables.
    
    Returns:
        dict: Mapping of API key hashes to tier names
    """
    mappings = {}
    tier_config = os.getenv('API_KEY_TIERS', '')
    
    if tier_config:
        try:
            for mapping in tier_config.split(','):
                key_hash, tier = mapping.strip().split(':')
                mappings[key_hash.strip()] = tier.strip().lower()
        except (ValueError, AttributeError):
            logger.warning(f"Invalid API_KEY_TIERS format: {tier_config}")
    
    return mappings


def validate_tier_configuration() -> bool:
    """
    Validate the rate limiting tier configuration.
    
    Returns:
        bool: True if configuration is valid
    """
    try:
        # Check if all required tiers exist
        required_tiers = ['free', 'basic', 'pro', 'enterprise']
        for tier in required_tiers:
            if tier not in RATE_LIMIT_TIERS:
                logger.error(f"Missing required tier: {tier}")
                return False
        
        # Validate tier configurations
        for tier_name, tier in RATE_LIMIT_TIERS.items():
            if (tier.requests_per_minute <= 0 or 
                tier.requests_per_hour <= 0 or 
                tier.requests_per_day <= 0):
                logger.error(f"Invalid limits for tier {tier_name}")
                return False
        
        logger.info("Rate limiting tier configuration is valid")
        return True
        
    except Exception as e:
        logger.error(f"Error validating tier configuration: {str(e)}")
        return False


# Initialize and validate configuration on import
if not validate_tier_configuration():
    logger.error("Rate limiting tier configuration validation failed")
