"""Add satellite_weather_cache and tide_data_cache tables

Revision ID: d8e9f0g1h2i3
Revises: c7f8d9e0a1b2
Create Date: 2025-12-22 20:40:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd8e9f0g1h2i3'
down_revision = 'c7f8d9e0a1b2'
branch_labels = None
depends_on = None


def upgrade():
    """
    Create satellite_weather_cache and tide_data_cache tables.
    
    satellite_weather_cache:
    - Caches GPM, CHIRPS, and ERA5 satellite weather data
    - Reduces Google Earth Engine API calls
    - Improves prediction response times
    
    tide_data_cache:
    - Caches WorldTides API tidal data
    - Minimizes API credit usage
    - Supports coastal flood prediction
    """
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == 'sqlite'
    
    # ==========================================
    # Create satellite_weather_cache table
    # ==========================================
    op.create_table(
        'satellite_weather_cache',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        
        # Location
        sa.Column('latitude', sa.Float(), nullable=False, index=True),
        sa.Column('longitude', sa.Float(), nullable=False, index=True),
        
        # Timestamp for the weather observation
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, index=True),
        
        # GPM Precipitation data
        sa.Column('precipitation_rate', sa.Float(), nullable=True),
        sa.Column('precipitation_1h', sa.Float(), nullable=True),
        sa.Column('precipitation_3h', sa.Float(), nullable=True),
        sa.Column('precipitation_24h', sa.Float(), nullable=True),
        
        # Data quality and source
        sa.Column('data_quality', sa.Float(), nullable=True),
        sa.Column('dataset', sa.String(50), nullable=True, server_default='GPM'),
        sa.Column('source', sa.String(50), nullable=True, server_default='earth_engine'),
        
        # ERA5 reanalysis data
        sa.Column('era5_temperature', sa.Float(), nullable=True),
        sa.Column('era5_humidity', sa.Float(), nullable=True),
        sa.Column('era5_precipitation', sa.Float(), nullable=True),
        sa.Column('era5_wind_speed', sa.Float(), nullable=True),
        sa.Column('era5_pressure', sa.Float(), nullable=True),
        
        # Cache metadata
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('is_valid', sa.Boolean(), nullable=False, server_default='true', index=True),
    )
    
    # Create composite indexes for satellite_weather_cache
    op.create_index('idx_swc_location', 'satellite_weather_cache', ['latitude', 'longitude'])
    op.create_index('idx_swc_location_time', 'satellite_weather_cache', ['latitude', 'longitude', 'timestamp'])
    
    # Add check constraints for PostgreSQL
    if not is_sqlite:
        op.create_check_constraint(
            'swc_valid_latitude',
            'satellite_weather_cache',
            'latitude >= -90 AND latitude <= 90'
        )
        op.create_check_constraint(
            'swc_valid_longitude',
            'satellite_weather_cache',
            'longitude >= -180 AND longitude <= 180'
        )
        op.create_check_constraint(
            'swc_valid_quality',
            'satellite_weather_cache',
            'data_quality IS NULL OR (data_quality >= 0 AND data_quality <= 1)'
        )
        op.create_check_constraint(
            'swc_valid_humidity',
            'satellite_weather_cache',
            'era5_humidity IS NULL OR (era5_humidity >= 0 AND era5_humidity <= 100)'
        )
    
    # ==========================================
    # Create tide_data_cache table
    # ==========================================
    op.create_table(
        'tide_data_cache',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        
        # Location
        sa.Column('latitude', sa.Float(), nullable=False, index=True),
        sa.Column('longitude', sa.Float(), nullable=False, index=True),
        
        # Timestamp for the tide observation/prediction
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, index=True),
        
        # Tide measurements
        sa.Column('tide_height', sa.Float(), nullable=False),
        sa.Column('tide_type', sa.String(20), nullable=True),
        sa.Column('datum', sa.String(20), nullable=True, server_default='MSL'),
        
        # Calculated fields for prediction
        sa.Column('tide_trend', sa.String(20), nullable=True),
        sa.Column('tide_risk_factor', sa.Float(), nullable=True),
        sa.Column('hours_until_high_tide', sa.Float(), nullable=True),
        sa.Column('next_high_tide_height', sa.Float(), nullable=True),
        
        # Cache metadata
        sa.Column('source', sa.String(50), nullable=True, server_default='worldtides'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False, index=True),
        sa.Column('is_valid', sa.Boolean(), nullable=False, server_default='true', index=True),
    )
    
    # Create composite indexes for tide_data_cache
    op.create_index('idx_tdc_location', 'tide_data_cache', ['latitude', 'longitude'])
    op.create_index('idx_tdc_location_time', 'tide_data_cache', ['latitude', 'longitude', 'timestamp'])
    
    # Add check constraints for PostgreSQL
    if not is_sqlite:
        op.create_check_constraint(
            'tdc_valid_latitude',
            'tide_data_cache',
            'latitude >= -90 AND latitude <= 90'
        )
        op.create_check_constraint(
            'tdc_valid_longitude',
            'tide_data_cache',
            'longitude >= -180 AND longitude <= 180'
        )
        op.create_check_constraint(
            'tdc_valid_risk_factor',
            'tide_data_cache',
            'tide_risk_factor IS NULL OR (tide_risk_factor >= 0 AND tide_risk_factor <= 1)'
        )
        op.create_check_constraint(
            'tdc_valid_trend',
            'tide_data_cache',
            "tide_trend IS NULL OR tide_trend IN ('rising', 'falling')"
        )


def downgrade():
    """Drop satellite_weather_cache and tide_data_cache tables."""
    bind = op.get_bind()
    is_sqlite = bind.dialect.name == 'sqlite'
    
    # Drop tide_data_cache constraints and indexes
    if not is_sqlite:
        op.drop_constraint('tdc_valid_trend', 'tide_data_cache', type_='check')
        op.drop_constraint('tdc_valid_risk_factor', 'tide_data_cache', type_='check')
        op.drop_constraint('tdc_valid_longitude', 'tide_data_cache', type_='check')
        op.drop_constraint('tdc_valid_latitude', 'tide_data_cache', type_='check')
    
    op.drop_index('idx_tdc_location_time', table_name='tide_data_cache')
    op.drop_index('idx_tdc_location', table_name='tide_data_cache')
    op.drop_table('tide_data_cache')
    
    # Drop satellite_weather_cache constraints and indexes
    if not is_sqlite:
        op.drop_constraint('swc_valid_humidity', 'satellite_weather_cache', type_='check')
        op.drop_constraint('swc_valid_quality', 'satellite_weather_cache', type_='check')
        op.drop_constraint('swc_valid_longitude', 'satellite_weather_cache', type_='check')
        op.drop_constraint('swc_valid_latitude', 'satellite_weather_cache', type_='check')
    
    op.drop_index('idx_swc_location_time', table_name='satellite_weather_cache')
    op.drop_index('idx_swc_location', table_name='satellite_weather_cache')
    op.drop_table('satellite_weather_cache')
