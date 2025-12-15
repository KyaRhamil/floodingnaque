#!/usr/bin/env python3
"""
Execute Supabase RLS (Row-Level Security) Migration.

This script enables RLS on all tables and creates appropriate policies
to secure the database against unauthorized PostgREST access.

Usage:
    python scripts/apply_rls_migration.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text

def apply_rls_migration():
    """Apply RLS migration to Supabase database."""
    print("=" * 60)
    print("SUPABASE RLS MIGRATION")
    print("=" * 60)
    
    from app.models.db import engine
    
    # Define migration steps
    migration_steps = [
        # Step 1: Enable RLS on all tables
        ("Enable RLS on weather_data", 
         "ALTER TABLE public.weather_data ENABLE ROW LEVEL SECURITY"),
        ("Enable RLS on predictions", 
         "ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY"),
        ("Enable RLS on alert_history", 
         "ALTER TABLE public.alert_history ENABLE ROW LEVEL SECURITY"),
        ("Enable RLS on model_registry", 
         "ALTER TABLE public.model_registry ENABLE ROW LEVEL SECURITY"),
        
        # Step 2: Revoke public grants
        ("Revoke PUBLIC on weather_data", 
         "REVOKE ALL ON public.weather_data FROM PUBLIC"),
        ("Revoke anon on weather_data", 
         "REVOKE ALL ON public.weather_data FROM anon"),
        ("Revoke PUBLIC on predictions", 
         "REVOKE ALL ON public.predictions FROM PUBLIC"),
        ("Revoke anon on predictions", 
         "REVOKE ALL ON public.predictions FROM anon"),
        ("Revoke PUBLIC on alert_history", 
         "REVOKE ALL ON public.alert_history FROM PUBLIC"),
        ("Revoke anon on alert_history", 
         "REVOKE ALL ON public.alert_history FROM anon"),
        ("Revoke PUBLIC on model_registry", 
         "REVOKE ALL ON public.model_registry FROM PUBLIC"),
        ("Revoke anon on model_registry", 
         "REVOKE ALL ON public.model_registry FROM anon"),
        
        # Step 3: Create service role policies (full access for backend)
        ("Create service_role policy for weather_data",
         """CREATE POLICY "service_role_weather_data_all" ON public.weather_data
            FOR ALL TO service_role USING (true) WITH CHECK (true)"""),
        ("Create service_role policy for predictions",
         """CREATE POLICY "service_role_predictions_all" ON public.predictions
            FOR ALL TO service_role USING (true) WITH CHECK (true)"""),
        ("Create service_role policy for alert_history",
         """CREATE POLICY "service_role_alert_history_all" ON public.alert_history
            FOR ALL TO service_role USING (true) WITH CHECK (true)"""),
        ("Create service_role policy for model_registry",
         """CREATE POLICY "service_role_model_registry_all" ON public.model_registry
            FOR ALL TO service_role USING (true) WITH CHECK (true)"""),
        
        # Step 4: Create authenticated user policies (read-only)
        ("Create authenticated SELECT policy for weather_data",
         """CREATE POLICY "authenticated_weather_data_select" ON public.weather_data
            FOR SELECT TO authenticated USING (true)"""),
        ("Create authenticated SELECT policy for predictions",
         """CREATE POLICY "authenticated_predictions_select" ON public.predictions
            FOR SELECT TO authenticated USING (true)"""),
        ("Create authenticated SELECT policy for alert_history",
         """CREATE POLICY "authenticated_alert_history_select" ON public.alert_history
            FOR SELECT TO authenticated USING (true)"""),
        ("Create authenticated SELECT policy for model_registry",
         """CREATE POLICY "authenticated_model_registry_select" ON public.model_registry
            FOR SELECT TO authenticated USING (true)"""),
    ]
    
    print(f"\nExecuting {len(migration_steps)} migration steps...\n")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    with engine.begin() as conn:
        for step_name, sql in migration_steps:
            try:
                conn.execute(text(sql))
                print(f"  ✅ {step_name}")
                success_count += 1
            except Exception as e:
                error_msg = str(e)
                if "already exists" in error_msg.lower():
                    print(f"  ⏭️  {step_name} (already exists)")
                    skip_count += 1
                elif "does not exist" in error_msg.lower() and "role" in error_msg.lower():
                    print(f"  ⏭️  {step_name} (role not available)")
                    skip_count += 1
                else:
                    print(f"  ❌ {step_name}: {error_msg[:80]}")
                    error_count += 1
    
    # Verify RLS status
    print("\n" + "-" * 60)
    print("VERIFICATION: RLS Status")
    print("-" * 60)
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT tablename, rowsecurity 
            FROM pg_tables 
            WHERE schemaname = 'public' 
            AND tablename IN ('weather_data', 'predictions', 'alert_history', 'model_registry')
        """))
        
        for row in result:
            status = "✅ Enabled" if row[1] else "❌ Disabled"
            print(f"  {row[0]}: {status}")
    
    # Summary
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"  ✅ Successful: {success_count}")
    print(f"  ⏭️  Skipped: {skip_count}")
    print(f"  ❌ Errors: {error_count}")
    
    if error_count == 0:
        print("\n🎉 RLS migration completed successfully!")
        print("\nYour database is now secured with Row-Level Security.")
        print("Backend service (using postgres user) will bypass RLS.")
        print("Anonymous access via PostgREST is blocked.")
        return True
    else:
        print("\n⚠️  Migration completed with some errors.")
        print("Review the errors above and run the SQL manually if needed.")
        return False

if __name__ == "__main__":
    success = apply_rls_migration()
    sys.exit(0 if success else 1)
