"""Fix Supabase RLS & Function Security Lints

Revision ID: i4j5k6l7m8n9
Revises: h3i4j5k6l7m8
Create Date: 2026-01-15

This migration fixes Supabase security lints by:
1. Enabling Row Level Security (RLS) on all tables (98+ including partitions)
2. Creating appropriate RLS policies:
   - weather_data: Public read access, service_role full access
   - predictions: Authenticated read access, service_role full access
   - users: Own row access for authenticated, service_role full access
3. Fixing 3 functions with mutable search_path security issues:
   - create_monthly_partitions
   - drop_old_partitions
   - get_partition_stats

Security Model:
- weather_data: Publicly readable (environmental data for transparency)
- predictions: Requires authentication (user-generated predictions)
- users: Row-level access (users can only see their own data)
- service_role: Bypasses RLS for backend operations
"""

from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers, used by Alembic.
revision = 'i4j5k6l7m8n9'
down_revision = 'h3i4j5k6l7m8'
branch_labels = None
depends_on = None


def get_partition_names():
    """Generate all partition table names for weather_data and predictions."""
    current_year = datetime.now().year
    weather_partitions = ['weather_data_default']
    predictions_partitions = ['predictions_default']
    
    # Generate partitions from 2024 through 2027 (as created in partitioning migration)
    for year in range(2024, 2028):
        for month in range(1, 13):
            weather_partitions.append(f"weather_data_{year}_{month:02d}")
            predictions_partitions.append(f"predictions_{year}_{month:02d}")
    
    return weather_partitions, predictions_partitions


def upgrade():
    """
    Enable RLS on all tables and create appropriate security policies.
    
    Strategy:
    1. Enable RLS on parent tables (weather_data, predictions, users)
    2. Enable RLS on all partition tables
    3. Create permissive/restrictive policies based on table purpose
    4. Recreate functions with fixed search_path
    """
    
    connection = op.get_bind()
    dialect = connection.dialect.name
    
    if dialect != 'postgresql':
        print(f"Skipping RLS migration: {dialect} does not support Row Level Security")
        return
    
    weather_partitions, predictions_partitions = get_partition_names()
    
    # =====================================================
    # ENABLE RLS ON ALL TABLES
    # =====================================================
    
    print("Enabling Row Level Security on all tables...")
    
    # Parent tables
    parent_tables = ['weather_data', 'predictions', 'users']
    for table in parent_tables:
        op.execute(f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table}') THEN
                    ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;
                    ALTER TABLE {table} FORCE ROW LEVEL SECURITY;
                    RAISE NOTICE 'Enabled RLS on: {table}';
                END IF;
            END $$;
        """)
    
    # Weather data partitions
    print("Enabling RLS on weather_data partitions...")
    for partition in weather_partitions:
        op.execute(f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{partition}') THEN
                    ALTER TABLE {partition} ENABLE ROW LEVEL SECURITY;
                    ALTER TABLE {partition} FORCE ROW LEVEL SECURITY;
                END IF;
            END $$;
        """)
    print(f"  Enabled RLS on {len(weather_partitions)} weather_data partitions")
    
    # Predictions partitions
    print("Enabling RLS on predictions partitions...")
    for partition in predictions_partitions:
        op.execute(f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{partition}') THEN
                    ALTER TABLE {partition} ENABLE ROW LEVEL SECURITY;
                    ALTER TABLE {partition} FORCE ROW LEVEL SECURITY;
                END IF;
            END $$;
        """)
    print(f"  Enabled RLS on {len(predictions_partitions)} predictions partitions")
    
    # =====================================================
    # RLS POLICIES FOR WEATHER_DATA (Public Read Access)
    # =====================================================
    
    print("Creating RLS policies for weather_data...")
    
    # Policy for public read access on parent table
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'weather_data') THEN
                -- Drop existing policies first
                DROP POLICY IF EXISTS weather_data_public_select ON weather_data;
                DROP POLICY IF EXISTS weather_data_service_role_all ON weather_data;
                
                -- Public read access (anyone can read weather data)
                CREATE POLICY weather_data_public_select ON weather_data
                    FOR SELECT
                    USING (true);
                
                -- Service role and postgres have full access
                CREATE POLICY weather_data_service_role_all ON weather_data
                    FOR ALL
                    TO service_role, postgres
                    USING (true)
                    WITH CHECK (true);
                
                RAISE NOTICE 'Created RLS policies for weather_data';
            END IF;
        END $$;
    """)
    
    # Create policies on weather_data partitions
    for partition in weather_partitions:
        op.execute(f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{partition}') THEN
                    DROP POLICY IF EXISTS {partition}_public_select ON {partition};
                    DROP POLICY IF EXISTS {partition}_service_role_all ON {partition};
                    
                    CREATE POLICY {partition}_public_select ON {partition}
                        FOR SELECT
                        USING (true);
                    
                    CREATE POLICY {partition}_service_role_all ON {partition}
                        FOR ALL
                        TO service_role, postgres
                        USING (true)
                        WITH CHECK (true);
                END IF;
            END $$;
        """)
    print(f"  Created policies on {len(weather_partitions)} weather_data partitions")
    
    # =====================================================
    # RLS POLICIES FOR PREDICTIONS (Authenticated Access)
    # =====================================================
    
    print("Creating RLS policies for predictions...")
    
    # Policy for authenticated read access on parent table
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'predictions') THEN
                -- Drop existing policies first
                DROP POLICY IF EXISTS predictions_authenticated_select ON predictions;
                DROP POLICY IF EXISTS predictions_service_role_all ON predictions;
                
                -- Authenticated users can read predictions
                CREATE POLICY predictions_authenticated_select ON predictions
                    FOR SELECT
                    TO authenticated
                    USING (true);
                
                -- Service role and postgres have full access
                CREATE POLICY predictions_service_role_all ON predictions
                    FOR ALL
                    TO service_role, postgres
                    USING (true)
                    WITH CHECK (true);
                
                RAISE NOTICE 'Created RLS policies for predictions';
            END IF;
        END $$;
    """)
    
    # Create policies on predictions partitions
    for partition in predictions_partitions:
        op.execute(f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{partition}') THEN
                    DROP POLICY IF EXISTS {partition}_authenticated_select ON {partition};
                    DROP POLICY IF EXISTS {partition}_service_role_all ON {partition};
                    
                    CREATE POLICY {partition}_authenticated_select ON {partition}
                        FOR SELECT
                        TO authenticated
                        USING (true);
                    
                    CREATE POLICY {partition}_service_role_all ON {partition}
                        FOR ALL
                        TO service_role, postgres
                        USING (true)
                        WITH CHECK (true);
                END IF;
            END $$;
        """)
    print(f"  Created policies on {len(predictions_partitions)} predictions partitions")
    
    # =====================================================
    # RLS POLICIES FOR USERS (Own Row Access)
    # =====================================================
    
    print("Creating RLS policies for users...")
    
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users') THEN
                -- Drop existing policies first
                DROP POLICY IF EXISTS users_own_row_select ON users;
                DROP POLICY IF EXISTS users_service_role_all ON users;
                
                -- Users can only see their own row (use (select auth.uid()) for performance)
                CREATE POLICY users_own_row_select ON users
                    FOR SELECT
                    TO authenticated
                    USING ((select auth.uid())::text = id::text);
                
                -- Service role and postgres have full access
                CREATE POLICY users_service_role_all ON users
                    FOR ALL
                    TO service_role, postgres
                    USING (true)
                    WITH CHECK (true);
                
                RAISE NOTICE 'Created RLS policies for users';
            END IF;
        END $$;
    """)

    # Remove duplicate permissive SELECT policies for alert_history, model_registry, predictions_old, weather_data_old
    for table, policies in [
        ("alert_history", ["alert_history_select_policy", "authenticated_alert_history_select"]),
        ("model_registry", ["authenticated_model_registry_select", "model_registry_select_policy"]),
        ("predictions_old", ["authenticated_predictions_select", "predictions_select_policy"]),
        ("weather_data_old", ["authenticated_weather_data_select", "weather_data_select_policy"]),
    ]:
        op.execute(f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table}') THEN
                    -- Drop all but the first policy in the list
                    DROP POLICY IF EXISTS {policies[1]} ON {table};
                END IF;
            END $$;
        """)
    
    # =====================================================
    # RECREATE FUNCTIONS WITH FIXED SEARCH_PATH
    # =====================================================
    
    print("Recreating partition management functions with fixed search_path...")
    
    # Drop existing functions first
    op.execute("DROP FUNCTION IF EXISTS create_monthly_partitions(TEXT, INTEGER);")
    op.execute("DROP FUNCTION IF EXISTS drop_old_partitions(TEXT, INTEGER);")
    op.execute("DROP FUNCTION IF EXISTS get_partition_stats(TEXT);")
    
    # Recreate create_monthly_partitions with fixed search_path and RLS enablement
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
            policy_name_select TEXT;
            policy_name_all TEXT;
            is_weather_table BOOLEAN;
        BEGIN
            is_weather_table := table_name = 'weather_data';
            
            FOR i IN 0..months_ahead LOOP
                partition_date := DATE_TRUNC('month', CURRENT_DATE + (i || ' months')::INTERVAL);
                partition_name := table_name || '_' || TO_CHAR(partition_date, 'YYYY_MM');
                start_date := TO_CHAR(partition_date, 'YYYY-MM-DD');
                end_date := TO_CHAR(partition_date + '1 month'::INTERVAL, 'YYYY-MM-DD');
                
                BEGIN
                    -- Create the partition
                    EXECUTE format(
                        'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                        partition_name, table_name, start_date, end_date
                    );
                    
                    -- Enable RLS on the new partition
                    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', partition_name);
                    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', partition_name);
                    
                    -- Create RLS policies for the new partition
                    policy_name_select := partition_name || '_' || CASE WHEN is_weather_table THEN 'public_select' ELSE 'authenticated_select' END;
                    policy_name_all := partition_name || '_service_role_all';
                    
                    -- Drop existing policies if they exist
                    EXECUTE format('DROP POLICY IF EXISTS %I ON %I', policy_name_select, partition_name);
                    EXECUTE format('DROP POLICY IF EXISTS %I ON %I', policy_name_all, partition_name);
                    
                    IF is_weather_table THEN
                        -- Public read access for weather data
                        EXECUTE format(
                            'CREATE POLICY %I ON %I FOR SELECT USING (true)',
                            policy_name_select, partition_name
                        );
                    ELSE
                        -- Authenticated read access for predictions
                        EXECUTE format(
                            'CREATE POLICY %I ON %I FOR SELECT TO authenticated USING (true)',
                            policy_name_select, partition_name
                        );
                    END IF;
                    
                    -- Service role full access
                    EXECUTE format(
                        'CREATE POLICY %I ON %I FOR ALL TO service_role, postgres USING (true) WITH CHECK (true)',
                        policy_name_all, partition_name
                    );
                    
                    RAISE NOTICE 'Created partition with RLS: %', partition_name;
                EXCEPTION WHEN duplicate_table THEN
                    -- Partition already exists, skip
                    NULL;
                END;
            END LOOP;
        END;
        $$ LANGUAGE plpgsql
        SET search_path = public;
    """)
    
    # Recreate drop_old_partitions with fixed search_path
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
                AND inhrelid::regclass::text ~ (table_name || '_\\\\d{4}_\\\\d{2}$')
            LOOP
                -- Extract date from partition name and check if older than cutoff
                IF TO_DATE(
                    SUBSTRING(partition_record.partition_name FROM '\\d{4}_\\d{2}$'),
                    'YYYY_MM'
                ) < cutoff_date THEN
                    EXECUTE format('DROP TABLE IF EXISTS %I', partition_record.partition_name);
                    RAISE NOTICE 'Dropped partition: %', partition_record.partition_name;
                    dropped_count := dropped_count + 1;
                END IF;
            END LOOP;
            
            RETURN dropped_count;
        END;
        $$ LANGUAGE plpgsql
        SET search_path = public;
    """)
    
    # Recreate get_partition_stats with fixed search_path
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
        $$ LANGUAGE plpgsql
        SET search_path = public;
    """)
    
    print("Supabase RLS & Function Security Lints migration complete!")
    print(f"  - Enabled RLS on 3 parent tables")
    print(f"  - Enabled RLS on {len(weather_partitions)} weather_data partitions")
    print(f"  - Enabled RLS on {len(predictions_partitions)} predictions partitions")
    print(f"  - Fixed search_path on 3 functions")


def downgrade():
    """Remove RLS policies and disable RLS on all tables."""
    
    connection = op.get_bind()
    dialect = connection.dialect.name
    
    if dialect != 'postgresql':
        return
    
    weather_partitions, predictions_partitions = get_partition_names()
    
    print("Removing RLS policies and disabling Row Level Security...")
    
    # =====================================================
    # DROP POLICIES AND DISABLE RLS ON PARENT TABLES
    # =====================================================
    
    # Weather data
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'weather_data') THEN
                DROP POLICY IF EXISTS weather_data_public_select ON weather_data;
                DROP POLICY IF EXISTS weather_data_service_role_all ON weather_data;
                ALTER TABLE weather_data DISABLE ROW LEVEL SECURITY;
            END IF;
        END $$;
    """)
    
    # Predictions
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'predictions') THEN
                DROP POLICY IF EXISTS predictions_authenticated_select ON predictions;
                DROP POLICY IF EXISTS predictions_service_role_all ON predictions;
                ALTER TABLE predictions DISABLE ROW LEVEL SECURITY;
            END IF;
        END $$;
    """)
    
    # Users
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users') THEN
                DROP POLICY IF EXISTS users_own_row_select ON users;
                DROP POLICY IF EXISTS users_service_role_all ON users;
                ALTER TABLE users DISABLE ROW LEVEL SECURITY;
            END IF;
        END $$;
    """)
    
    # =====================================================
    # DROP POLICIES AND DISABLE RLS ON PARTITIONS
    # =====================================================
    
    # Weather data partitions
    for partition in weather_partitions:
        op.execute(f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{partition}') THEN
                    DROP POLICY IF EXISTS {partition}_public_select ON {partition};
                    DROP POLICY IF EXISTS {partition}_service_role_all ON {partition};
                    ALTER TABLE {partition} DISABLE ROW LEVEL SECURITY;
                END IF;
            END $$;
        """)
    
    # Predictions partitions
    for partition in predictions_partitions:
        op.execute(f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{partition}') THEN
                    DROP POLICY IF EXISTS {partition}_authenticated_select ON {partition};
                    DROP POLICY IF EXISTS {partition}_service_role_all ON {partition};
                    ALTER TABLE {partition} DISABLE ROW LEVEL SECURITY;
                END IF;
            END $$;
        """)
    
    # =====================================================
    # RESTORE FUNCTIONS WITHOUT search_path FIX
    # =====================================================
    
    # Drop and recreate functions without SET search_path
    op.execute("DROP FUNCTION IF EXISTS create_monthly_partitions(TEXT, INTEGER);")
    op.execute("DROP FUNCTION IF EXISTS drop_old_partitions(TEXT, INTEGER);")
    op.execute("DROP FUNCTION IF EXISTS get_partition_stats(TEXT);")
    
    # Recreate original create_monthly_partitions without RLS and search_path
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
                    NULL;
                END;
            END LOOP;
        END;
        $$ LANGUAGE plpgsql;
    """)
    
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
                AND inhrelid::regclass::text ~ (table_name || '_\\d{4}_\\d{2}$')
            LOOP
                IF TO_DATE(
                    SUBSTRING(partition_record.partition_name FROM '\\d{4}_\\d{2}$'),
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
    
    print("Downgrade complete - RLS disabled and policies removed")
