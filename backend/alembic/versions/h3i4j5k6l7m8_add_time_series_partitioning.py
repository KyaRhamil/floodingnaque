"""Add time-series partitioning for historical flood data

Revision ID: h3i4j5k6l7m8
Revises: g2h3i4j5k6l7
Create Date: 2026-01-13

This migration adds time-series partitioning to:
- weather_data: Partitioned by created_at (monthly)
- predictions: Partitioned by created_at (monthly)

Benefits:
- Faster queries on time-bounded data
- Efficient data retention (drop old partitions)
- Improved vacuum/analyze performance
- Better I/O distribution

Note: This migration requires PostgreSQL 11+
"""

from alembic import op
import sqlalchemy as sa
from datetime import datetime, timedelta

# revision identifiers, used by Alembic.
revision = 'h3i4j5k6l7m8'
down_revision = 'g2h3i4j5k6l7'
branch_labels = None
depends_on = None


def upgrade():
    """
    Convert weather_data and predictions tables to partitioned tables.
    
    Strategy:
    1. Create new partitioned table with same schema
    2. Create partitions for existing data range
    3. Migrate data from old table to new partitioned table
    4. Rename tables (old -> _backup, new -> original)
    5. Create future partitions
    """
    
    # Get database URL to check if PostgreSQL
    connection = op.get_bind()
    dialect = connection.dialect.name
    
    if dialect != 'postgresql':
        print(f"Skipping partitioning migration: {dialect} does not support table partitioning")
        return
    
    # =====================================================
    # WEATHER_DATA PARTITIONING
    # =====================================================
    
    print("Creating partitioned weather_data table...")
    
    # Create the partitioned table
    op.execute("""
        CREATE TABLE IF NOT EXISTS weather_data_partitioned (
            id SERIAL,
            temperature FLOAT NOT NULL,
            humidity FLOAT NOT NULL,
            precipitation FLOAT NOT NULL,
            wind_speed FLOAT,
            pressure FLOAT,
            tide_height FLOAT,
            tide_trend VARCHAR(20),
            tide_risk_factor FLOAT,
            hours_until_high_tide FLOAT,
            satellite_precipitation_rate FLOAT,
            precipitation_1h FLOAT,
            precipitation_3h FLOAT,
            precipitation_24h FLOAT,
            data_quality FLOAT,
            dataset VARCHAR(50) DEFAULT 'OWM',
            location_lat FLOAT,
            location_lon FLOAT,
            source VARCHAR(50) DEFAULT 'OWM',
            station_id VARCHAR(50),
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP WITH TIME ZONE,
            PRIMARY KEY (id, created_at),
            CONSTRAINT valid_temperature CHECK (temperature >= 173.15 AND temperature <= 333.15),
            CONSTRAINT valid_humidity CHECK (humidity >= 0 AND humidity <= 100),
            CONSTRAINT valid_precipitation CHECK (precipitation >= 0)
        ) PARTITION BY RANGE (created_at);
    """)
    
    # Create default partition for data without valid created_at
    op.execute("""
        CREATE TABLE IF NOT EXISTS weather_data_default
        PARTITION OF weather_data_partitioned DEFAULT;
    """)
    
    # Create partitions for historical data (last 2 years + future year)
    current_year = datetime.now().year
    for year in range(current_year - 2, current_year + 2):
        for month in range(1, 13):
            partition_name = f"weather_data_{year}_{month:02d}"
            start_date = f"{year}-{month:02d}-01"
            
            if month == 12:
                end_date = f"{year + 1}-01-01"
            else:
                end_date = f"{year}-{month + 1:02d}-01"
            
            op.execute(f"""
                CREATE TABLE IF NOT EXISTS {partition_name}
                PARTITION OF weather_data_partitioned
                FOR VALUES FROM ('{start_date}') TO ('{end_date}');
            """)
            print(f"  Created partition: {partition_name}")
    
    # Migrate data from old table to partitioned table
    print("Migrating weather_data to partitioned table...")
    op.execute("""
        INSERT INTO weather_data_partitioned (
            id, temperature, humidity, precipitation, wind_speed, pressure,
            tide_height, tide_trend, tide_risk_factor, hours_until_high_tide,
            satellite_precipitation_rate, precipitation_1h, precipitation_3h, precipitation_24h,
            data_quality, dataset, location_lat, location_lon, source, station_id,
            timestamp, created_at, updated_at, is_deleted, deleted_at
        )
        SELECT 
            id, temperature, humidity, precipitation, wind_speed, pressure,
            tide_height, tide_trend, tide_risk_factor, hours_until_high_tide,
            satellite_precipitation_rate, precipitation_1h, precipitation_3h, precipitation_24h,
            data_quality, dataset, location_lat, location_lon, source, station_id,
            timestamp, 
            COALESCE(created_at, timestamp, NOW()) as created_at,
            updated_at, is_deleted, deleted_at
        FROM weather_data
        ON CONFLICT DO NOTHING;
    """)
    
    # Create indexes on partitioned table
    print("Creating indexes on weather_data_partitioned...")
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_weather_part_timestamp 
        ON weather_data_partitioned (timestamp);
        
        CREATE INDEX IF NOT EXISTS idx_weather_part_created 
        ON weather_data_partitioned (created_at);
        
        CREATE INDEX IF NOT EXISTS idx_weather_part_location 
        ON weather_data_partitioned (location_lat, location_lon);
        
        CREATE INDEX IF NOT EXISTS idx_weather_part_source 
        ON weather_data_partitioned (source);
        
        CREATE INDEX IF NOT EXISTS idx_weather_part_active 
        ON weather_data_partitioned (is_deleted);
        
        CREATE INDEX IF NOT EXISTS idx_weather_part_active_timestamp 
        ON weather_data_partitioned (is_deleted, timestamp);
    """)
    
    # Rename tables
    print("Swapping weather_data tables...")
    op.execute("ALTER TABLE IF EXISTS weather_data RENAME TO weather_data_old;")
    op.execute("ALTER TABLE weather_data_partitioned RENAME TO weather_data;")
    
    # Reset sequence
    op.execute("""
        SELECT setval('weather_data_id_seq', COALESCE((SELECT MAX(id) FROM weather_data), 1));
    """)
    
    # =====================================================
    # PREDICTIONS PARTITIONING
    # =====================================================
    
    print("Creating partitioned predictions table...")
    
    op.execute("""
        CREATE TABLE IF NOT EXISTS predictions_partitioned (
            id SERIAL,
            weather_data_id INTEGER,
            prediction INTEGER NOT NULL,
            risk_level INTEGER,
            risk_label VARCHAR(50),
            confidence FLOAT,
            model_version INTEGER,
            model_name VARCHAR(100) DEFAULT 'flood_rf_model',
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
            deleted_at TIMESTAMP WITH TIME ZONE,
            PRIMARY KEY (id, created_at),
            CONSTRAINT valid_prediction CHECK (prediction IN (0, 1)),
            CONSTRAINT valid_risk_level CHECK (risk_level IS NULL OR risk_level IN (0, 1, 2)),
            CONSTRAINT valid_confidence CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1))
        ) PARTITION BY RANGE (created_at);
    """)
    
    # Create default partition
    op.execute("""
        CREATE TABLE IF NOT EXISTS predictions_default
        PARTITION OF predictions_partitioned DEFAULT;
    """)
    
    # Create partitions
    for year in range(current_year - 2, current_year + 2):
        for month in range(1, 13):
            partition_name = f"predictions_{year}_{month:02d}"
            start_date = f"{year}-{month:02d}-01"
            
            if month == 12:
                end_date = f"{year + 1}-01-01"
            else:
                end_date = f"{year}-{month + 1:02d}-01"
            
            op.execute(f"""
                CREATE TABLE IF NOT EXISTS {partition_name}
                PARTITION OF predictions_partitioned
                FOR VALUES FROM ('{start_date}') TO ('{end_date}');
            """)
            print(f"  Created partition: {partition_name}")
    
    # Migrate data
    print("Migrating predictions to partitioned table...")
    op.execute("""
        INSERT INTO predictions_partitioned (
            id, weather_data_id, prediction, risk_level, risk_label,
            confidence, model_version, model_name, created_at, is_deleted, deleted_at
        )
        SELECT 
            id, weather_data_id, prediction, risk_level, risk_label,
            confidence, model_version, model_name, 
            COALESCE(created_at, NOW()) as created_at,
            is_deleted, deleted_at
        FROM predictions
        ON CONFLICT DO NOTHING;
    """)
    
    # Create indexes
    print("Creating indexes on predictions_partitioned...")
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_pred_part_weather 
        ON predictions_partitioned (weather_data_id);
        
        CREATE INDEX IF NOT EXISTS idx_pred_part_created 
        ON predictions_partitioned (created_at);
        
        CREATE INDEX IF NOT EXISTS idx_pred_part_risk 
        ON predictions_partitioned (risk_level);
        
        CREATE INDEX IF NOT EXISTS idx_pred_part_active 
        ON predictions_partitioned (is_deleted);
        
        CREATE INDEX IF NOT EXISTS idx_pred_part_active_created 
        ON predictions_partitioned (is_deleted, created_at);
        
        CREATE INDEX IF NOT EXISTS idx_pred_part_risk_created 
        ON predictions_partitioned (risk_level, created_at);
    """)
    
    # Rename tables
    print("Swapping predictions tables...")
    op.execute("ALTER TABLE IF EXISTS predictions RENAME TO predictions_old;")
    op.execute("ALTER TABLE predictions_partitioned RENAME TO predictions;")
    
    # Reset sequence
    op.execute("""
        SELECT setval('predictions_id_seq', COALESCE((SELECT MAX(id) FROM predictions), 1));
    """)
    
    # =====================================================
    # CREATE PARTITION MANAGEMENT FUNCTION
    # =====================================================
    
    print("Creating partition management functions...")
    
    # Function to create future partitions automatically
    op.execute("""
        CREATE OR REPLACE FUNCTION create_monthly_partitions(
            table_name TEXT,
            months_ahead INTEGER DEFAULT 3
        ) RETURNS VOID AS $$
        DECLARE
            partition_date DATE;
            partition_name TEXT;
            start_date TEXT;
            end_date TEXT;
        BEGIN
            FOR i IN 0..months_ahead LOOP
                partition_date := DATE_TRUNC('month', CURRENT_DATE + (i || ' months')::INTERVAL);
                partition_name := table_name || '_' || TO_CHAR(partition_date, 'YYYY_MM');
                start_date := TO_CHAR(partition_date, 'YYYY-MM-DD');
                end_date := TO_CHAR(partition_date + '1 month'::INTERVAL, 'YYYY-MM-DD');
                
                BEGIN
                    EXECUTE format(
                        'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                        partition_name, table_name, start_date, end_date
                    );
                    RAISE NOTICE 'Created partition: %', partition_name;
                EXCEPTION WHEN duplicate_table THEN
                    -- Partition already exists, skip
                    NULL;
                END;
            END LOOP;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Function to drop old partitions (for data retention)
    op.execute("""
        CREATE OR REPLACE FUNCTION drop_old_partitions(
            table_name TEXT,
            months_to_keep INTEGER DEFAULT 24
        ) RETURNS INTEGER AS $$
        DECLARE
            cutoff_date DATE;
            partition_record RECORD;
            dropped_count INTEGER := 0;
        BEGIN
            cutoff_date := DATE_TRUNC('month', CURRENT_DATE - (months_to_keep || ' months')::INTERVAL);
            
            FOR partition_record IN
                SELECT inhrelid::regclass::text AS partition_name
                FROM pg_inherits
                WHERE inhparent = table_name::regclass
                AND inhrelid::regclass::text ~ (table_name || '_\d{4}_\d{2}$')
            LOOP
                -- Extract date from partition name and check if older than cutoff
                IF TO_DATE(
                    SUBSTRING(partition_record.partition_name FROM '\d{4}_\d{2}$'),
                    'YYYY_MM'
                ) < cutoff_date THEN
                    EXECUTE format('DROP TABLE IF EXISTS %I', partition_record.partition_name);
                    RAISE NOTICE 'Dropped partition: %', partition_record.partition_name;
                    dropped_count := dropped_count + 1;
                END IF;
            END LOOP;
            
            RETURN dropped_count;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    # Function to get partition statistics
    op.execute("""
        CREATE OR REPLACE FUNCTION get_partition_stats(table_name TEXT)
        RETURNS TABLE (
            partition_name TEXT,
            row_count BIGINT,
            table_size TEXT,
            index_size TEXT,
            total_size TEXT
        ) AS $$
        BEGIN
            RETURN QUERY
            SELECT 
                inhrelid::regclass::text,
                (xpath('/row/cnt/text()', query_to_xml(
                    format('SELECT COUNT(*) AS cnt FROM %I', inhrelid::regclass::text), 
                    true, true, '')))[1]::text::bigint,
                pg_size_pretty(pg_table_size(inhrelid)),
                pg_size_pretty(pg_indexes_size(inhrelid)),
                pg_size_pretty(pg_total_relation_size(inhrelid))
            FROM pg_inherits
            WHERE inhparent = table_name::regclass
            ORDER BY inhrelid::regclass::text;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
    print("Time-series partitioning migration complete!")


def downgrade():
    """Revert to non-partitioned tables."""
    
    connection = op.get_bind()
    dialect = connection.dialect.name
    
    if dialect != 'postgresql':
        return
    
    print("Reverting weather_data to non-partitioned table...")
    
    # Check if backup tables exist
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'weather_data_old') THEN
                DROP TABLE IF EXISTS weather_data CASCADE;
                ALTER TABLE weather_data_old RENAME TO weather_data;
            END IF;
        END $$;
    """)
    
    print("Reverting predictions to non-partitioned table...")
    
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'predictions_old') THEN
                DROP TABLE IF EXISTS predictions CASCADE;
                ALTER TABLE predictions_old RENAME TO predictions;
            END IF;
        END $$;
    """)
    
    # Drop partition management functions
    op.execute("DROP FUNCTION IF EXISTS create_monthly_partitions(TEXT, INTEGER);")
    op.execute("DROP FUNCTION IF EXISTS drop_old_partitions(TEXT, INTEGER);")
    op.execute("DROP FUNCTION IF EXISTS get_partition_stats(TEXT);")
    
    print("Downgrade complete!")
