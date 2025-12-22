-- =====================================================
-- Supabase Migration: Complete Schema Sync
-- Updated: 2025-12-22
-- Description: Complete schema including satellite cache, tide cache,
--              earth engine requests, webhooks, and missing columns
-- =====================================================

-- =====================================================
-- 0. ADD MISSING COLUMNS TO EXISTING TABLES
-- =====================================================

-- Add missing soft delete columns to predictions table
ALTER TABLE public.predictions ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE public.predictions ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_prediction_active ON public.predictions (is_deleted);

-- Add missing soft delete columns to alert_history table
ALTER TABLE public.alert_history ADD COLUMN IF NOT EXISTS is_deleted BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE public.alert_history ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_alert_active ON public.alert_history (is_deleted);

-- Add missing tide columns to weather_data table
ALTER TABLE public.weather_data ADD COLUMN IF NOT EXISTS tide_height DOUBLE PRECISION;
ALTER TABLE public.weather_data ADD COLUMN IF NOT EXISTS tide_trend VARCHAR(20);
ALTER TABLE public.weather_data ADD COLUMN IF NOT EXISTS tide_risk_factor DOUBLE PRECISION;
ALTER TABLE public.weather_data ADD COLUMN IF NOT EXISTS hours_until_high_tide DOUBLE PRECISION;

-- Add check constraints for tide columns (if not exist)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'valid_tide_risk_factor') THEN
        ALTER TABLE public.weather_data ADD CONSTRAINT valid_tide_risk_factor 
            CHECK (tide_risk_factor IS NULL OR (tide_risk_factor >= 0 AND tide_risk_factor <= 1));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'valid_tide_trend') THEN
        ALTER TABLE public.weather_data ADD CONSTRAINT valid_tide_trend 
            CHECK (tide_trend IS NULL OR tide_trend IN ('rising', 'falling'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'valid_hours_until_high_tide') THEN
        ALTER TABLE public.weather_data ADD CONSTRAINT valid_hours_until_high_tide 
            CHECK (hours_until_high_tide IS NULL OR hours_until_high_tide >= 0);
    END IF;
END $$;


-- =====================================================
-- 1. SATELLITE WEATHER CACHE TABLE
-- =====================================================
-- Caches GPM, CHIRPS, and ERA5 satellite weather data from Google Earth Engine
-- Purpose: Reduce API calls and improve prediction response times

CREATE TABLE IF NOT EXISTS public.satellite_weather_cache (
    id SERIAL PRIMARY KEY,
    
    -- Location coordinates
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    
    -- Observation timestamp
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- GPM Precipitation data
    precipitation_rate DOUBLE PRECISION,           -- Current precipitation rate (mm/hour)
    precipitation_1h DOUBLE PRECISION,             -- 1-hour accumulated precipitation (mm)
    precipitation_3h DOUBLE PRECISION,             -- 3-hour accumulated precipitation (mm)
    precipitation_24h DOUBLE PRECISION,            -- 24-hour accumulated precipitation (mm)
    
    -- Data quality and source
    data_quality DOUBLE PRECISION,                 -- Quality score 0-1
    dataset VARCHAR(50) DEFAULT 'GPM',             -- Dataset: GPM, CHIRPS, ERA5
    source VARCHAR(50) DEFAULT 'earth_engine',     -- Data source identifier
    
    -- ERA5 reanalysis data
    era5_temperature DOUBLE PRECISION,             -- ERA5 temperature (Kelvin)
    era5_humidity DOUBLE PRECISION,                -- ERA5 relative humidity (%)
    era5_precipitation DOUBLE PRECISION,           -- ERA5 precipitation (mm)
    era5_wind_speed DOUBLE PRECISION,              -- ERA5 wind speed (m/s)
    era5_pressure DOUBLE PRECISION,                -- ERA5 pressure (hPa)
    
    -- Cache metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    is_valid BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Constraints
    CONSTRAINT swc_valid_latitude CHECK (latitude >= -90 AND latitude <= 90),
    CONSTRAINT swc_valid_longitude CHECK (longitude >= -180 AND longitude <= 180),
    CONSTRAINT swc_valid_quality CHECK (data_quality IS NULL OR (data_quality >= 0 AND data_quality <= 1)),
    CONSTRAINT swc_valid_humidity CHECK (era5_humidity IS NULL OR (era5_humidity >= 0 AND era5_humidity <= 100))
);

-- Add table comment
COMMENT ON TABLE public.satellite_weather_cache IS 'Satellite weather data cache for Earth Engine (GPM, CHIRPS, ERA5)';

-- Create indexes for satellite_weather_cache
CREATE INDEX IF NOT EXISTS idx_swc_latitude ON public.satellite_weather_cache (latitude);
CREATE INDEX IF NOT EXISTS idx_swc_longitude ON public.satellite_weather_cache (longitude);
CREATE INDEX IF NOT EXISTS idx_swc_timestamp ON public.satellite_weather_cache (timestamp);
CREATE INDEX IF NOT EXISTS idx_swc_location ON public.satellite_weather_cache (latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_swc_location_time ON public.satellite_weather_cache (latitude, longitude, timestamp);
CREATE INDEX IF NOT EXISTS idx_swc_expires ON public.satellite_weather_cache (expires_at);
CREATE INDEX IF NOT EXISTS idx_swc_valid ON public.satellite_weather_cache (is_valid);


-- =====================================================
-- 2. TIDE DATA CACHE TABLE
-- =====================================================
-- Caches WorldTides API tidal data
-- Purpose: Minimize API credit usage while supporting coastal flood prediction

CREATE TABLE IF NOT EXISTS public.tide_data_cache (
    id SERIAL PRIMARY KEY,
    
    -- Location coordinates
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    
    -- Tide observation/prediction timestamp
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Tide measurements
    tide_height DOUBLE PRECISION NOT NULL,         -- Tide height in meters (relative to datum)
    tide_type VARCHAR(20),                         -- Type: 'high', 'low', or NULL
    datum VARCHAR(20) DEFAULT 'MSL',               -- Vertical reference: MSL, LAT, etc.
    
    -- Calculated fields for prediction
    tide_trend VARCHAR(20),                        -- 'rising' or 'falling'
    tide_risk_factor DOUBLE PRECISION,             -- Risk factor 0-1 for flood prediction
    hours_until_high_tide DOUBLE PRECISION,        -- Hours until next high tide
    next_high_tide_height DOUBLE PRECISION,        -- Predicted height of next high tide (m)
    
    -- Cache metadata
    source VARCHAR(50) DEFAULT 'worldtides',       -- Data source identifier
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    is_valid BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Constraints
    CONSTRAINT tdc_valid_latitude CHECK (latitude >= -90 AND latitude <= 90),
    CONSTRAINT tdc_valid_longitude CHECK (longitude >= -180 AND longitude <= 180),
    CONSTRAINT tdc_valid_risk_factor CHECK (tide_risk_factor IS NULL OR (tide_risk_factor >= 0 AND tide_risk_factor <= 1)),
    CONSTRAINT tdc_valid_trend CHECK (tide_trend IS NULL OR tide_trend IN ('rising', 'falling'))
);

-- Add table comment
COMMENT ON TABLE public.tide_data_cache IS 'Tide data cache for WorldTides API';

-- Create indexes for tide_data_cache
CREATE INDEX IF NOT EXISTS idx_tdc_latitude ON public.tide_data_cache (latitude);
CREATE INDEX IF NOT EXISTS idx_tdc_longitude ON public.tide_data_cache (longitude);
CREATE INDEX IF NOT EXISTS idx_tdc_timestamp ON public.tide_data_cache (timestamp);
CREATE INDEX IF NOT EXISTS idx_tdc_location ON public.tide_data_cache (latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_tdc_location_time ON public.tide_data_cache (latitude, longitude, timestamp);
CREATE INDEX IF NOT EXISTS idx_tdc_expires ON public.tide_data_cache (expires_at);
CREATE INDEX IF NOT EXISTS idx_tdc_valid ON public.tide_data_cache (is_valid);


-- =====================================================
-- 3. ROW LEVEL SECURITY (RLS) POLICIES
-- =====================================================
-- Enable RLS for production security

-- Enable RLS on satellite_weather_cache
ALTER TABLE public.satellite_weather_cache ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist, then recreate
DROP POLICY IF EXISTS "satellite_weather_cache_select_policy" ON public.satellite_weather_cache;
DROP POLICY IF EXISTS "satellite_weather_cache_service_policy" ON public.satellite_weather_cache;

-- Policy: Allow authenticated users to read cache data
CREATE POLICY "satellite_weather_cache_select_policy" ON public.satellite_weather_cache
    FOR SELECT
    USING (true);

-- Policy: Allow service role to insert/update/delete
CREATE POLICY "satellite_weather_cache_service_policy" ON public.satellite_weather_cache
    FOR ALL
    USING (auth.role() = 'service_role');


-- Enable RLS on tide_data_cache
ALTER TABLE public.tide_data_cache ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist, then recreate
DROP POLICY IF EXISTS "tide_data_cache_select_policy" ON public.tide_data_cache;
DROP POLICY IF EXISTS "tide_data_cache_service_policy" ON public.tide_data_cache;

-- Policy: Allow authenticated users to read cache data
CREATE POLICY "tide_data_cache_select_policy" ON public.tide_data_cache
    FOR SELECT
    USING (true);

-- Policy: Allow service role to insert/update/delete
CREATE POLICY "tide_data_cache_service_policy" ON public.tide_data_cache
    FOR ALL
    USING (auth.role() = 'service_role');


-- =====================================================
-- 4. HELPER FUNCTIONS
-- =====================================================

-- Function to clean up expired satellite weather cache entries
CREATE OR REPLACE FUNCTION public.cleanup_expired_satellite_weather_cache()
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM public.satellite_weather_cache
    WHERE expires_at < NOW() OR is_valid = FALSE;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;

COMMENT ON FUNCTION public.cleanup_expired_satellite_weather_cache() IS 'Removes expired or invalid satellite weather cache entries';


-- Function to clean up expired tide data cache entries
CREATE OR REPLACE FUNCTION public.cleanup_expired_tide_data_cache()
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM public.tide_data_cache
    WHERE expires_at < NOW() OR is_valid = FALSE;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;

COMMENT ON FUNCTION public.cleanup_expired_tide_data_cache() IS 'Removes expired or invalid tide data cache entries';


-- Function to get valid satellite weather cache entry
CREATE OR REPLACE FUNCTION public.get_satellite_weather_cache(
    p_latitude DOUBLE PRECISION,
    p_longitude DOUBLE PRECISION,
    p_tolerance DOUBLE PRECISION DEFAULT 0.01
)
RETURNS SETOF public.satellite_weather_cache
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
BEGIN
    RETURN QUERY
    SELECT *
    FROM public.satellite_weather_cache
    WHERE is_valid = TRUE
      AND expires_at > NOW()
      AND ABS(latitude - p_latitude) <= p_tolerance
      AND ABS(longitude - p_longitude) <= p_tolerance
    ORDER BY timestamp DESC
    LIMIT 1;
END;
$$;

COMMENT ON FUNCTION public.get_satellite_weather_cache(DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION) IS 'Get valid cached satellite weather data for a location';


-- Function to get valid tide data cache entry
CREATE OR REPLACE FUNCTION public.get_tide_data_cache(
    p_latitude DOUBLE PRECISION,
    p_longitude DOUBLE PRECISION,
    p_tolerance DOUBLE PRECISION DEFAULT 0.01
)
RETURNS SETOF public.tide_data_cache
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
BEGIN
    RETURN QUERY
    SELECT *
    FROM public.tide_data_cache
    WHERE is_valid = TRUE
      AND expires_at > NOW()
      AND ABS(latitude - p_latitude) <= p_tolerance
      AND ABS(longitude - p_longitude) <= p_tolerance
    ORDER BY timestamp DESC
    LIMIT 1;
END;
$$;

COMMENT ON FUNCTION public.get_tide_data_cache(DOUBLE PRECISION, DOUBLE PRECISION, DOUBLE PRECISION) IS 'Get valid cached tide data for a location';


-- =====================================================
-- 5. GRANT PERMISSIONS
-- =====================================================

-- Grant permissions to authenticated users
GRANT SELECT ON public.satellite_weather_cache TO authenticated;
GRANT SELECT ON public.tide_data_cache TO authenticated;

-- Grant all permissions to service role
GRANT ALL ON public.satellite_weather_cache TO service_role;
GRANT ALL ON public.tide_data_cache TO service_role;

-- Grant sequence usage
GRANT USAGE ON SEQUENCE public.satellite_weather_cache_id_seq TO service_role;
GRANT USAGE ON SEQUENCE public.tide_data_cache_id_seq TO service_role;


-- =====================================================
-- MIGRATION COMPLETE
-- =====================================================
-- Run this SQL in Supabase SQL Editor to create the tables
-- Or use: psql -d your_database -f supabase_satellite_tide_cache_migration.sql


-- =====================================================
-- 6. API REQUESTS TABLE
-- =====================================================
-- Logs all API requests for analytics and monitoring

CREATE TABLE IF NOT EXISTS public.api_requests (
    id SERIAL PRIMARY KEY,
    
    -- Request identification
    request_id VARCHAR(36) NOT NULL UNIQUE,
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    
    -- Response info
    status_code INTEGER NOT NULL,
    response_time_ms DOUBLE PRECISION NOT NULL,
    
    -- Request metadata
    user_agent VARCHAR(500),
    ip_address VARCHAR(45),
    api_version VARCHAR(10) DEFAULT 'v1',
    error_message TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Soft delete
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    deleted_at TIMESTAMPTZ
);

-- Add table comment
COMMENT ON TABLE public.api_requests IS 'API request logs for analytics and monitoring';

-- Create indexes for api_requests
CREATE INDEX IF NOT EXISTS idx_api_request_id ON public.api_requests (request_id);
CREATE INDEX IF NOT EXISTS idx_api_request_endpoint ON public.api_requests (endpoint);
CREATE INDEX IF NOT EXISTS idx_api_request_status ON public.api_requests (status_code);
CREATE INDEX IF NOT EXISTS idx_api_request_ip ON public.api_requests (ip_address);
CREATE INDEX IF NOT EXISTS idx_api_request_created ON public.api_requests (created_at);
CREATE INDEX IF NOT EXISTS idx_api_request_endpoint_status ON public.api_requests (endpoint, status_code);
CREATE INDEX IF NOT EXISTS idx_api_request_active ON public.api_requests (is_deleted);

-- Enable RLS on api_requests
ALTER TABLE public.api_requests ENABLE ROW LEVEL SECURITY;

-- Policy: Allow service role full access
CREATE POLICY "api_requests_service_policy" ON public.api_requests
    FOR ALL
    USING (auth.role() = 'service_role');

-- Grant permissions
GRANT ALL ON public.api_requests TO service_role;
GRANT USAGE ON SEQUENCE public.api_requests_id_seq TO service_role;


-- =====================================================
-- 7. WEBHOOKS TABLE
-- =====================================================
-- Webhook configurations for external notifications

CREATE TABLE IF NOT EXISTS public.webhooks (
    id SERIAL PRIMARY KEY,
    
    -- Webhook configuration
    url VARCHAR(500) NOT NULL,
    events TEXT NOT NULL,                          -- JSON array of event types
    secret VARCHAR(255) NOT NULL,                  -- Webhook secret for verification
    
    -- Status
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    last_triggered_at TIMESTAMPTZ,
    failure_count INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ,
    
    -- Soft delete
    is_deleted BOOLEAN NOT NULL DEFAULT FALSE,
    deleted_at TIMESTAMPTZ
);

-- Add table comment
COMMENT ON TABLE public.webhooks IS 'Webhook configurations for external notifications';

-- Create indexes for webhooks
CREATE INDEX IF NOT EXISTS idx_webhook_active ON public.webhooks (is_active, is_deleted);
CREATE INDEX IF NOT EXISTS idx_webhook_is_active ON public.webhooks (is_active);
CREATE INDEX IF NOT EXISTS idx_webhook_is_deleted ON public.webhooks (is_deleted);

-- Enable RLS on webhooks
ALTER TABLE public.webhooks ENABLE ROW LEVEL SECURITY;

-- Policy: Allow service role full access
CREATE POLICY "webhooks_service_policy" ON public.webhooks
    FOR ALL
    USING (auth.role() = 'service_role');

-- Grant permissions
GRANT ALL ON public.webhooks TO service_role;
GRANT USAGE ON SEQUENCE public.webhooks_id_seq TO service_role;


-- =====================================================
-- COMPLETE - ALL TABLES SYNCED
-- =====================================================
