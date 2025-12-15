#!/usr/bin/env python3
"""
Supabase Database Setup Script.

Creates all required tables in Supabase PostgreSQL database.
Run this ONCE before deploying to production.

Usage:
    python scripts/setup_supabase_tables.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_database_url():
    """Verify DATABASE_URL is configured for Supabase."""
    db_url = os.getenv('DATABASE_URL', '')
    
    if not db_url:
        print("ERROR: DATABASE_URL not set in .env file")
        return False
    
    if 'sqlite' in db_url.lower():
        print("ERROR: DATABASE_URL points to SQLite, not Supabase")
        print(f"Current: {db_url}")
        print("\nPlease set DATABASE_URL to your Supabase PostgreSQL connection string")
        return False
    
    if 'supabase' in db_url.lower():
        print(f"✓ Supabase database detected")
        # Mask password for display
        masked = db_url.split('@')[0].rsplit(':', 1)[0] + ':****@' + db_url.split('@')[1] if '@' in db_url else db_url
        print(f"  Connection: {masked}")
        return True
    
    print(f"Database URL configured (non-Supabase PostgreSQL)")
    return True


def create_tables():
    """Create all database tables using SQLAlchemy models."""
    from app.models.db import Base, engine, WeatherData, Prediction, AlertHistory, ModelRegistry
    
    print("\nCreating database tables...")
    print("-" * 50)
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        # List created tables
        tables = list(Base.metadata.tables.keys())
        print(f"\n✓ Successfully created/verified {len(tables)} tables:")
        for table in tables:
            print(f"  - {table}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error creating tables: {str(e)}")
        
        # Provide helpful error messages
        error_str = str(e).lower()
        if 'password authentication failed' in error_str:
            print("\n→ Check your database password in DATABASE_URL")
        elif 'could not connect' in error_str or 'connection refused' in error_str:
            print("\n→ Cannot connect to database. Check:")
            print("  1. Your internet connection")
            print("  2. Supabase project is active")
            print("  3. DATABASE_URL is correct")
        elif 'ssl' in error_str:
            print("\n→ SSL connection issue. Supabase requires SSL.")
        elif 'permission denied' in error_str:
            print("\n→ Permission denied. Check your database user permissions.")
        
        return False


def verify_tables():
    """Verify tables exist and show their structure."""
    from app.models.db import engine
    from sqlalchemy import inspect
    
    print("\nVerifying table structure...")
    print("-" * 50)
    
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if not tables:
            print("✗ No tables found in database")
            return False
        
        print(f"\n✓ Found {len(tables)} tables in database:\n")
        
        for table_name in sorted(tables):
            columns = inspector.get_columns(table_name)
            print(f"  {table_name}:")
            for col in columns:
                nullable = "NULL" if col.get('nullable', True) else "NOT NULL"
                print(f"    - {col['name']}: {col['type']} {nullable}")
            print()
        
        # Check for required tables
        required_tables = ['weather_data', 'predictions', 'alert_history', 'model_registry']
        missing = [t for t in required_tables if t not in tables]
        
        if missing:
            print(f"\n⚠ Missing required tables: {', '.join(missing)}")
            return False
        
        print("✓ All required tables present")
        return True
        
    except Exception as e:
        print(f"✗ Error verifying tables: {str(e)}")
        return False


def test_connection():
    """Test basic database operations."""
    from app.models.db import get_db_session, WeatherData
    from datetime import datetime, timezone
    
    print("\nTesting database connection...")
    print("-" * 50)
    
    try:
        with get_db_session() as session:
            # Test read
            count = session.query(WeatherData).count()
            print(f"✓ Connection successful")
            print(f"  Current weather_data records: {count}")
        
        return True
        
    except Exception as e:
        print(f"✗ Connection test failed: {str(e)}")
        return False


def main():
    """Main setup routine."""
    print("=" * 60)
    print("FLOODINGNAQUE - Supabase Database Setup")
    print("=" * 60)
    
    # Step 1: Check configuration
    print("\n[1/4] Checking database configuration...")
    if not check_database_url():
        print("\n❌ Setup aborted. Please fix DATABASE_URL in .env")
        sys.exit(1)
    
    # Step 2: Create tables
    print("\n[2/4] Creating database tables...")
    if not create_tables():
        print("\n❌ Setup failed at table creation")
        sys.exit(1)
    
    # Step 3: Verify tables
    print("\n[3/4] Verifying tables...")
    if not verify_tables():
        print("\n⚠ Verification had warnings, but continuing...")
    
    # Step 4: Test connection
    print("\n[4/4] Testing connection...")
    if not test_connection():
        print("\n⚠ Connection test failed, but tables may still be created")
    
    # Summary
    print("\n" + "=" * 60)
    print("✓ SUPABASE DATABASE SETUP COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Verify tables in Supabase Dashboard → Table Editor")
    print("  2. (Optional) Run RLS migration: python scripts/apply_rls_migration.py")
    print("  3. Start the server: waitress-serve --host=0.0.0.0 --port=5000 --threads=4 main:app")
    print()


if __name__ == '__main__':
    main()
