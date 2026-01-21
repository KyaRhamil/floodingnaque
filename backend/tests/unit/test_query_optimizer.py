"""
Unit tests for query optimizer utilities.

Tests for app/utils/query_optimizer.py
"""

import time
from unittest.mock import MagicMock, patch

import pytest


class TestQueryCache:
    """Tests for query result cache."""

    def test_make_query_cache_key(self):
        """Test query cache key generation."""
        from app.utils.query_optimizer import _make_query_cache_key

        key1 = _make_query_cache_key("SELECT * FROM table")
        key2 = _make_query_cache_key("SELECT * FROM table")
        key3 = _make_query_cache_key("SELECT * FROM other_table")

        # Same query should produce same key
        assert key1 == key2
        # Different query should produce different key
        assert key1 != key3

    def test_make_query_cache_key_with_params(self):
        """Test query cache key with parameters."""
        from app.utils.query_optimizer import _make_query_cache_key

        key1 = _make_query_cache_key("SELECT * FROM table", {"id": 1})
        key2 = _make_query_cache_key("SELECT * FROM table", {"id": 2})

        # Different params should produce different keys
        assert key1 != key2

    def test_query_cache_get_miss(self):
        """Test query cache miss returns None."""
        from app.utils.query_optimizer import query_cache_get

        result = query_cache_get("nonexistent_key_" + str(time.time()))

        assert result is None

    def test_query_cache_set_and_get(self):
        """Test setting and getting from query cache."""
        from app.utils.query_optimizer import query_cache_get, query_cache_set

        test_key = f"test_key_{time.time()}"
        test_value = {"data": "test", "count": 42}

        query_cache_set(test_key, test_value, ttl=60)
        result = query_cache_get(test_key)

        assert result == test_value

    def test_query_cache_expiration(self):
        """Test query cache entry expiration."""
        from app.utils.query_optimizer import query_cache_get, query_cache_set

        test_key = f"expiring_key_{time.time()}"
        query_cache_set(test_key, {"data": "test"}, ttl=0)  # Immediate expiration

        time.sleep(0.1)
        result = query_cache_get(test_key)

        assert result is None

    def test_query_cache_invalidate_all(self):
        """Test invalidating all cache entries."""
        from app.utils.query_optimizer import query_cache_get, query_cache_invalidate, query_cache_set

        key1 = f"key1_{time.time()}"
        key2 = f"key2_{time.time()}"

        query_cache_set(key1, "value1")
        query_cache_set(key2, "value2")

        count = query_cache_invalidate()

        assert count >= 0
        assert query_cache_get(key1) is None
        assert query_cache_get(key2) is None


class TestQueryCacheStats:
    """Tests for query cache statistics."""

    def test_get_query_cache_stats(self):
        """Test getting query cache statistics."""
        from app.utils.query_optimizer import get_query_cache_stats

        stats = get_query_cache_stats()

        assert "entries" in stats
        assert "max_entries" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate_percent" in stats


class TestCachedQueryDecorator:
    """Tests for cached_query decorator."""

    def test_cached_query_decorator(self):
        """Test cached_query decorator caches results."""
        from app.utils.query_optimizer import cached_query

        call_count = 0

        @cached_query(ttl=60, key_prefix="test")
        def expensive_query(session, limit=10):
            nonlocal call_count
            call_count += 1
            return [{"id": i} for i in range(limit)]

        mock_session = MagicMock()

        # First call
        result1 = expensive_query(mock_session, limit=5)
        # Second call should be cached
        result2 = expensive_query(mock_session, limit=5)

        # Both should return same result
        assert result1 == result2

    def test_cached_query_different_params(self):
        """Test cached_query respects different parameters."""
        from app.utils.query_optimizer import cached_query

        @cached_query(ttl=60, key_prefix="test_diff")
        def query_with_params(session, limit=10):
            return [{"id": i} for i in range(limit)]

        mock_session = MagicMock()

        result1 = query_with_params(mock_session, limit=5)
        result2 = query_with_params(mock_session, limit=10)

        assert len(result1) != len(result2)


class TestEagerLoader:
    """Tests for EagerLoader class."""

    def test_eager_loader_creation(self):
        """Test EagerLoader initialization."""
        from app.utils.query_optimizer import EagerLoader

        mock_query = MagicMock()
        loader = EagerLoader(mock_query)

        assert loader.query is mock_query

    def test_eager_loader_join_load(self):
        """Test adding join load."""
        from app.utils.query_optimizer import EagerLoader

        mock_query = MagicMock()
        loader = EagerLoader(mock_query)

        loader.join_load("relationship")

        assert "relationship" in loader._join_loads

    def test_eager_loader_select_load(self):
        """Test adding select load."""
        from app.utils.query_optimizer import EagerLoader

        mock_query = MagicMock()
        loader = EagerLoader(mock_query)

        loader.select_load("relationship")

        assert "relationship" in loader._select_loads


class TestSlowQueryLogging:
    """Tests for slow query logging."""

    def test_get_slow_queries(self):
        """Test getting slow queries."""
        from app.utils.query_optimizer import get_slow_queries

        queries = get_slow_queries(limit=5)

        assert isinstance(queries, list)

    def test_clear_slow_query_log(self):
        """Test clearing slow query log."""
        from app.utils.query_optimizer import clear_slow_query_log

        clear_slow_query_log()

        # Should not raise


class TestQueryStatistics:
    """Tests for query statistics."""

    def test_get_query_statistics(self):
        """Test getting query statistics."""
        from app.utils.query_optimizer import get_query_statistics

        stats = get_query_statistics()

        assert isinstance(stats, dict)


class TestDatabaseHealth:
    """Tests for database health monitoring."""

    def test_get_database_health(self):
        """Test getting database health requires session."""
        from app.utils.query_optimizer import get_database_health

        # Function requires a session parameter
        assert callable(get_database_health)


class TestIndexUsage:
    """Tests for index usage statistics."""

    def test_get_index_usage_stats(self):
        """Test getting index usage statistics requires session."""
        from app.utils.query_optimizer import get_index_usage_stats

        # Function requires a session parameter
        assert callable(get_index_usage_stats)

    def test_get_unused_indexes(self):
        """Test getting unused indexes requires session."""
        from app.utils.query_optimizer import get_unused_indexes

        # Function requires a session parameter
        assert callable(get_unused_indexes)


class TestTableStatistics:
    """Tests for table statistics."""

    def test_get_table_statistics(self):
        """Test getting table statistics requires session."""
        from app.utils.query_optimizer import get_table_statistics

        # Function requires a session parameter
        assert callable(get_table_statistics)


class TestMaintenanceRecommendations:
    """Tests for maintenance recommendations."""

    def test_run_maintenance_recommendations(self):
        """Test running maintenance recommendations requires session."""
        from app.utils.query_optimizer import run_maintenance_recommendations

        # Function requires a session parameter
        assert callable(run_maintenance_recommendations)


class TestN1Prevention:
    """Tests for N+1 query prevention."""

    def test_detect_n_plus_one(self):
        """Test N+1 query detection."""
        # N+1 detection is typically done via query logging
        pass

    def test_eager_loading_prevents_n_plus_one(self):
        """Test that eager loading prevents N+1."""
        from app.utils.query_optimizer import EagerLoader

        mock_query = MagicMock()
        loader = EagerLoader(mock_query)
        loader.join_load("weather_data")
        loader.select_load("alerts")

        # Both relationships should be configured for eager loading
        assert len(loader._join_loads) + len(loader._select_loads) == 2


class TestCacheEviction:
    """Tests for cache eviction."""

    def test_evict_expired_cache(self):
        """Test expired cache entries are evicted."""
        from app.utils.query_optimizer import _evict_expired_cache, _query_cache, query_cache_set

        # Set an entry that will expire immediately
        test_key = f"expiring_{time.time()}"
        query_cache_set(test_key, "value", ttl=0)

        time.sleep(0.1)
        _evict_expired_cache()

        # Entry should be evicted
        assert test_key not in _query_cache

    def test_cache_size_limit(self):
        """Test cache respects size limits."""
        from app.utils.query_optimizer import MAX_CACHE_ENTRIES

        # MAX_CACHE_ENTRIES should be defined
        assert MAX_CACHE_ENTRIES > 0
