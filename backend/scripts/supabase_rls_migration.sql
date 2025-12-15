-- ============================================================
-- SUPABASE ROW-LEVEL SECURITY (RLS) MIGRATION
-- ============================================================
-- This script enables RLS on all tables to secure against
-- unauthorized access via PostgREST/Supabase client
-- 
-- Run this in: Supabase Dashboard > SQL Editor
-- Or via: psql connection
-- ============================================================

-- ============================================================
-- STEP 1: ENABLE RLS ON ALL TABLES
-- ============================================================
-- Once enabled, all access is denied until policies are created

ALTER TABLE public.weather_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alert_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.model_registry ENABLE ROW LEVEL SECURITY;

-- ============================================================
-- STEP 2: REVOKE PUBLIC GRANTS
-- ============================================================
-- Remove any default grants that might allow anonymous access

REVOKE ALL ON public.weather_data FROM PUBLIC;
REVOKE ALL ON public.weather_data FROM anon;

REVOKE ALL ON public.predictions FROM PUBLIC;
REVOKE ALL ON public.predictions FROM anon;

REVOKE ALL ON public.alert_history FROM PUBLIC;
REVOKE ALL ON public.alert_history FROM anon;

REVOKE ALL ON public.model_registry FROM PUBLIC;
REVOKE ALL ON public.model_registry FROM anon;

-- ============================================================
-- STEP 3: CREATE SERVICE ROLE POLICIES
-- ============================================================
-- These policies allow the backend service (using service_role)
-- to have full access to all tables.
-- The postgres user and service_role bypass RLS by default,
-- but we create explicit policies for defense-in-depth.

-- Weather Data Policies (service role full access)
CREATE POLICY "service_role_weather_data_all" ON public.weather_data
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Predictions Policies (service role full access)
CREATE POLICY "service_role_predictions_all" ON public.predictions
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Alert History Policies (service role full access)
CREATE POLICY "service_role_alert_history_all" ON public.alert_history
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Model Registry Policies (service role full access)
CREATE POLICY "service_role_model_registry_all" ON public.model_registry
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- ============================================================
-- STEP 4: AUTHENTICATED USER POLICIES (READ-ONLY)
-- ============================================================
-- If you want authenticated users (via Supabase Auth) to have
-- read-only access to certain data, enable these policies.

-- Allow authenticated users to read weather data (public data)
CREATE POLICY "authenticated_weather_data_select" ON public.weather_data
    FOR SELECT
    TO authenticated
    USING (true);

-- Allow authenticated users to read predictions (public data)
CREATE POLICY "authenticated_predictions_select" ON public.predictions
    FOR SELECT
    TO authenticated
    USING (true);

-- Allow authenticated users to read alerts (public data)
CREATE POLICY "authenticated_alert_history_select" ON public.alert_history
    FOR SELECT
    TO authenticated
    USING (true);

-- Model registry - read only for authenticated users
CREATE POLICY "authenticated_model_registry_select" ON public.model_registry
    FOR SELECT
    TO authenticated
    USING (true);

-- ============================================================
-- STEP 5: ANONYMOUS USER POLICIES (OPTIONAL - READ ONLY)
-- ============================================================
-- Uncomment these if you want anonymous/public read access
-- to weather and prediction data via the frontend.
-- 
-- WARNING: Only enable if your frontend needs direct DB access
-- without authentication. Consider using backend API instead.

-- Allow anonymous users to read weather data
-- CREATE POLICY "anon_weather_data_select" ON public.weather_data
--     FOR SELECT
--     TO anon
--     USING (true);

-- Allow anonymous users to read predictions
-- CREATE POLICY "anon_predictions_select" ON public.predictions
--     FOR SELECT
--     TO anon
--     USING (true);

-- Allow anonymous users to read alerts
-- CREATE POLICY "anon_alert_history_select" ON public.alert_history
--     FOR SELECT
--     TO anon
--     USING (true);

-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================
-- Run these to verify RLS is enabled correctly

-- Check RLS status on all tables
SELECT 
    schemaname,
    tablename,
    rowsecurity
FROM pg_tables 
WHERE schemaname = 'public' 
AND tablename IN ('weather_data', 'predictions', 'alert_history', 'model_registry');

-- Check policies on all tables
SELECT 
    schemaname,
    tablename,
    policyname,
    permissive,
    roles,
    cmd
FROM pg_policies 
WHERE schemaname = 'public'
ORDER BY tablename, policyname;

-- ============================================================
-- ROLLBACK SCRIPT (if needed)
-- ============================================================
-- To undo these changes, run:
-- 
-- DROP POLICY IF EXISTS "service_role_weather_data_all" ON public.weather_data;
-- DROP POLICY IF EXISTS "service_role_predictions_all" ON public.predictions;
-- DROP POLICY IF EXISTS "service_role_alert_history_all" ON public.alert_history;
-- DROP POLICY IF EXISTS "service_role_model_registry_all" ON public.model_registry;
-- DROP POLICY IF EXISTS "authenticated_weather_data_select" ON public.weather_data;
-- DROP POLICY IF EXISTS "authenticated_predictions_select" ON public.predictions;
-- DROP POLICY IF EXISTS "authenticated_alert_history_select" ON public.alert_history;
-- DROP POLICY IF EXISTS "authenticated_model_registry_select" ON public.model_registry;
-- 
-- ALTER TABLE public.weather_data DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.predictions DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.alert_history DISABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.model_registry DISABLE ROW LEVEL SECURITY;
