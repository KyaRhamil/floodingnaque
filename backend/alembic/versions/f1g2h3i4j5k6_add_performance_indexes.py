"""Add performance optimization indexes

Revision ID: f1g2h3i4j5k6
Revises: e9f0g1h2i3j4
Create Date: 2026-01-09 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f1g2h3i4j5k6'
down_revision = 'e9f0g1h2i3j4'
branch_labels = None
depends_on = None


def upgrade():
    """
    Add performance optimization indexes for common query patterns.
    
    These indexes are designed to optimize:
    1. WeatherData queries filtering by is_deleted + timestamp
    2. Prediction queries filtering by risk_level and created_at
    3. AlertHistory queries by status and creation time
    4. APIRequest queries for performance monitoring
    """
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == 'sqlite'
    
    # ==========================================
    # 1. WeatherData Performance Indexes
    # ==========================================
    
    # Index for common filter: active records by timestamp
    op.create_index(
        'idx_weather_active_timestamp',
        'weather_data',
        ['is_deleted', 'timestamp'],
        unique=False
    )
    
    # Index for common filter: active records by creation time
    op.create_index(
        'idx_weather_active_created',
        'weather_data',
        ['is_deleted', 'created_at'],
        unique=False
    )
    
    # Index for filtering by source and time
    op.create_index(
        'idx_weather_source_timestamp',
        'weather_data',
        ['source', 'timestamp'],
        unique=False
    )
    
    # ==========================================
    # 2. Prediction Performance Indexes
    # ==========================================
    
    # Index for active predictions by creation time
    op.create_index(
        'idx_prediction_active_created',
        'predictions',
        ['is_deleted', 'created_at'],
        unique=False
    )
    
    # Index for risk level filtering with time
    op.create_index(
        'idx_prediction_risk_created',
        'predictions',
        ['risk_level', 'created_at'],
        unique=False
    )
    
    # Composite index for full filter combination
    op.create_index(
        'idx_prediction_active_risk',
        'predictions',
        ['is_deleted', 'risk_level', 'created_at'],
        unique=False
    )
    
    # ==========================================
    # 3. AlertHistory Performance Indexes
    # ==========================================
    
    # Index for active alerts by creation time
    op.create_index(
        'idx_alert_active_created',
        'alert_history',
        ['is_deleted', 'created_at'],
        unique=False
    )
    
    # Index for filtering by status and time
    op.create_index(
        'idx_alert_status_created',
        'alert_history',
        ['delivery_status', 'created_at'],
        unique=False
    )
    
    # Index for filtering by risk level and status
    op.create_index(
        'idx_alert_risk_status',
        'alert_history',
        ['risk_level', 'delivery_status'],
        unique=False
    )
    
    # ==========================================
    # 4. APIRequest Performance Indexes
    # ==========================================
    
    # Index for slow query analysis
    op.create_index(
        'idx_api_request_response_time',
        'api_requests',
        ['response_time_ms'],
        unique=False
    )
    
    # Index for endpoint performance over time
    op.create_index(
        'idx_api_request_endpoint_time',
        'api_requests',
        ['endpoint', 'created_at'],
        unique=False
    )
    
    # Index for active requests by time
    op.create_index(
        'idx_api_request_active_created',
        'api_requests',
        ['is_deleted', 'created_at'],
        unique=False
    )
    
    # ==========================================
    # 5. Cache Table Indexes (if tables exist)
    # ==========================================
    
    # Add covering index for satellite weather cache lookups
    try:
        op.create_index(
            'idx_swc_location_valid_time',
            'satellite_weather_cache',
            ['latitude', 'longitude', 'is_valid', 'timestamp'],
            unique=False
        )
    except Exception:
        pass  # Table may not exist
    
    # Add covering index for tide data cache lookups
    try:
        op.create_index(
            'idx_tdc_location_valid_time',
            'tide_data_cache',
            ['latitude', 'longitude', 'is_valid', 'timestamp'],
            unique=False
        )
    except Exception:
        pass  # Table may not exist
    
    # ==========================================
    # 6. Partial Indexes (PostgreSQL only)
    # ==========================================
    
    if not is_sqlite:
        # Partial index for active weather data only
        try:
            op.execute("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_weather_active_only 
                ON weather_data (timestamp DESC) 
                WHERE is_deleted = false
            """)
        except Exception:
            pass  # May fail in transaction, that's ok
        
        # Partial index for high-risk predictions
        try:
            op.execute("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_prediction_high_risk 
                ON predictions (created_at DESC) 
                WHERE is_deleted = false AND risk_level >= 1
            """)
        except Exception:
            pass
        
        # Partial index for failed alerts
        try:
            op.execute("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alert_failed 
                ON alert_history (created_at DESC) 
                WHERE delivery_status = 'failed'
            """)
        except Exception:
            pass


def downgrade():
    """Remove performance optimization indexes."""
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == 'sqlite'
    
    # Drop partial indexes (PostgreSQL only)
    if not is_sqlite:
        try:
            op.execute("DROP INDEX IF EXISTS idx_alert_failed")
            op.execute("DROP INDEX IF EXISTS idx_prediction_high_risk")
            op.execute("DROP INDEX IF EXISTS idx_weather_active_only")
        except Exception:
            pass
    
    # Drop cache table indexes
    try:
        op.drop_index('idx_tdc_location_valid_time', table_name='tide_data_cache')
    except Exception:
        pass
    
    try:
        op.drop_index('idx_swc_location_valid_time', table_name='satellite_weather_cache')
    except Exception:
        pass
    
    # Drop APIRequest indexes
    op.drop_index('idx_api_request_active_created', table_name='api_requests')
    op.drop_index('idx_api_request_endpoint_time', table_name='api_requests')
    op.drop_index('idx_api_request_response_time', table_name='api_requests')
    
    # Drop AlertHistory indexes
    op.drop_index('idx_alert_risk_status', table_name='alert_history')
    op.drop_index('idx_alert_status_created', table_name='alert_history')
    op.drop_index('idx_alert_active_created', table_name='alert_history')
    
    # Drop Prediction indexes
    op.drop_index('idx_prediction_active_risk', table_name='predictions')
    op.drop_index('idx_prediction_risk_created', table_name='predictions')
    op.drop_index('idx_prediction_active_created', table_name='predictions')
    
    # Drop WeatherData indexes
    op.drop_index('idx_weather_source_timestamp', table_name='weather_data')
    op.drop_index('idx_weather_active_created', table_name='weather_data')
    op.drop_index('idx_weather_active_timestamp', table_name='weather_data')
