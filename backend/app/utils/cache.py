"""
Redis Caching Module for Floodingnaque API.

Provides caching utilities for frequently accessed data to improve performance.
Supports both Redis (production) and simple in-memory caching (development).
"""

import os
import json
import hashlib
import logging
from functools import wraps
from typing import Any, Callable, Optional, Union
from datetime import timedelta

logger = logging.getLogger(__name__)

# Redis client singleton
_redis_client = None
_cache_enabled = None


def get_redis_client():
    """
    Get or create a Redis client connection.
    
    Returns:
        Redis client or None if Redis is not configured/available
    """
    global _redis_client, _cache_enabled
    
    if _cache_enabled is False:
        return None
    
    if _redis_client is not None:
        return _redis_client
    
    redis_url = os.getenv('REDIS_URL') or os.getenv('RATE_LIMIT_STORAGE_URL', '')
    
    if not redis_url or 'redis' not in redis_url.lower():
        _cache_enabled = False
        logger.info("Redis caching not configured (REDIS_URL not set)")
        return None
    
    try:
        import redis
        from urllib.parse import urlparse
        
        parsed = urlparse(redis_url)
        _redis_client = redis.Redis(
            host=parsed.hostname or 'localhost',
            port=parsed.port or 6379,
            password=parsed.password,
            db=int(parsed.path[1:]) if parsed.path and len(parsed.path) > 1 else 0,
            socket_timeout=5,
            socket_connect_timeout=5,
            decode_responses=True  # Return strings instead of bytes
        )
        
        # Test connection
        _redis_client.ping()
        _cache_enabled = True
        logger.info(f"Redis caching enabled: {parsed.hostname}:{parsed.port}")
        return _redis_client
        
    except ImportError:
        _cache_enabled = False
        logger.warning("redis package not installed - caching disabled")
        return None
    except Exception as e:
        _cache_enabled = False
        logger.warning(f"Redis connection failed - caching disabled: {e}")
        return None


def is_cache_enabled() -> bool:
    """Check if caching is enabled."""
    get_redis_client()  # Initialize
    return _cache_enabled is True


def _make_cache_key(prefix: str, *args, **kwargs) -> str:
    """
    Generate a cache key from prefix and arguments.
    
    Args:
        prefix: Cache key prefix (e.g., 'weather', 'prediction')
        *args: Positional arguments to include in key
        **kwargs: Keyword arguments to include in key
        
    Returns:
        str: Unique cache key
    """
    key_parts = [prefix]
    
    if args:
        key_parts.extend([str(arg) for arg in args])
    
    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        key_parts.extend([f"{k}={v}" for k, v in sorted_kwargs])
    
    key_string = ':'.join(key_parts)
    
    # Hash if key is too long
    if len(key_string) > 200:
        key_hash = hashlib.md5(key_string.encode()).hexdigest()
        return f"{prefix}:{key_hash}"
    
    return key_string


def cache_get(key: str) -> Optional[Any]:
    """
    Get a value from cache.
    
    Args:
        key: Cache key
        
    Returns:
        Cached value or None if not found/expired
    """
    client = get_redis_client()
    if not client:
        return None
    
    try:
        value = client.get(f"floodingnaque:{key}")
        if value:
            return json.loads(value)
        return None
    except Exception as e:
        logger.debug(f"Cache get error: {e}")
        return None


def cache_set(
    key: str,
    value: Any,
    ttl: Union[int, timedelta] = 300
) -> bool:
    """
    Set a value in cache.
    
    Args:
        key: Cache key
        value: Value to cache (must be JSON serializable)
        ttl: Time to live in seconds or timedelta
        
    Returns:
        bool: True if successful
    """
    client = get_redis_client()
    if not client:
        return False
    
    try:
        if isinstance(ttl, timedelta):
            ttl = int(ttl.total_seconds())
        
        serialized = json.dumps(value)
        client.setex(f"floodingnaque:{key}", ttl, serialized)
        return True
    except Exception as e:
        logger.debug(f"Cache set error: {e}")
        return False


def cache_delete(key: str) -> bool:
    """
    Delete a value from cache.
    
    Args:
        key: Cache key
        
    Returns:
        bool: True if successful
    """
    client = get_redis_client()
    if not client:
        return False
    
    try:
        client.delete(f"floodingnaque:{key}")
        return True
    except Exception as e:
        logger.debug(f"Cache delete error: {e}")
        return False


def cache_clear_pattern(pattern: str) -> int:
    """
    Clear all cache keys matching a pattern.
    
    Args:
        pattern: Key pattern (e.g., 'weather:*')
        
    Returns:
        int: Number of keys deleted
    """
    client = get_redis_client()
    if not client:
        return 0
    
    try:
        keys = list(client.scan_iter(f"floodingnaque:{pattern}"))
        if keys:
            return client.delete(*keys)
        return 0
    except Exception as e:
        logger.debug(f"Cache clear pattern error: {e}")
        return 0


def cached(
    prefix: str,
    ttl: Union[int, timedelta] = 300,
    key_builder: Optional[Callable] = None
):
    """
    Decorator to cache function results.
    
    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds or timedelta
        key_builder: Optional custom key builder function
        
    Usage:
        @cached('weather', ttl=300)
        def get_weather(lat, lon):
            ...
            
        @cached('prediction', ttl=timedelta(minutes=5))
        def predict_flood(features):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = _make_cache_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_value = cache_get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_value
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                cache_set(cache_key, result, ttl)
                logger.debug(f"Cache set: {cache_key}")
            
            return result
        return wrapper
    return decorator


def get_cache_stats() -> dict:
    """
    Get cache statistics.
    
    Returns:
        dict: Cache statistics including connection status, memory usage, etc.
    """
    client = get_redis_client()
    if not client:
        return {
            'enabled': False,
            'connected': False,
            'reason': 'Redis not configured'
        }
    
    try:
        info = client.info('memory')
        keys_count = len(list(client.scan_iter('floodingnaque:*', count=1000)))
        
        return {
            'enabled': True,
            'connected': True,
            'used_memory_human': info.get('used_memory_human', 'unknown'),
            'used_memory_peak_human': info.get('used_memory_peak_human', 'unknown'),
            'keys_count': keys_count
        }
    except Exception as e:
        return {
            'enabled': True,
            'connected': False,
            'error': str(e)
        }
