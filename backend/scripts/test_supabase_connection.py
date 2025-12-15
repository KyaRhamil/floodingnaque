#!/usr/bin/env python3
"""
Test Supabase database connection.
Run this script to verify your Supabase PostgreSQL connection is working.

Usage:
    python scripts/test_supabase_connection.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

def test_connection():
    """Test the database connection."""
    print("=" * 60)
    print("SUPABASE CONNECTION TEST")
    print("=" * 60)
    
    # Get database URL
    db_url = os.getenv('DATABASE_URL', 'NOT SET')
    
    # Mask password for display
    if 'YOUR_DATABASE_PASSWORD' in db_url:
        print("\n❌ ERROR: Database password not configured!")
        print("\nPlease update your .env file:")
        print("1. Go to: https://supabase.com/dashboard/project/REDACTED_PROJECT_REF/settings/database")
        print("2. Copy your database password")
        print("3. Replace 'YOUR_DATABASE_PASSWORD' in the DATABASE_URL in .env")
        print("\nExample DATABASE_URL format:")
        print("postgresql://postgres.REDACTED_PROJECT_REF:YOUR_ACTUAL_PASSWORD@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres")
        return False
    
    # Mask password for logging
    masked_url = db_url
    if '@' in db_url:
        parts = db_url.split('@')
        creds = parts[0].rsplit(':', 1)
        if len(creds) == 2:
            masked_url = f"{creds[0]}:****@{parts[1]}"
    
    print(f"\nDatabase URL: {masked_url}")
    print(f"\nSupabase URL: {os.getenv('SUPABASE_URL', 'NOT SET')}")
    print(f"Supabase Key: {'SET' if os.getenv('SUPABASE_KEY') else 'NOT SET'}")
    
    print("\n" + "-" * 60)
    print("Testing database connection...")
    print("-" * 60)
    
    try:
        from app.models.db import engine, Base, init_db
        
        # Test connection
        with engine.connect() as conn:
            from sqlalchemy import text
            print("\n✅ Database connection successful!")
            
            # Get PostgreSQL version
            version_result = conn.execute(text("SELECT version()"))
            version = version_result.fetchone()[0]
            print(f"PostgreSQL Version: {version[:80]}...")
            
        # Initialize tables
        print("\n" + "-" * 60)
        print("Initializing database tables...")
        print("-" * 60)
        
        init_db()
        print("\n✅ Database tables created/verified successfully!")
        
        # List tables
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"\nTables in database: {', '.join(tables)}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Connection failed: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your database password is correct")
        print("2. Ensure your IP is not blocked by Supabase firewall")
        print("3. Verify the connection string format")
        print("4. Check if the database is running on Supabase dashboard")
        return False

if __name__ == "__main__":
    success = test_connection()
    print("\n" + "=" * 60)
    sys.exit(0 if success else 1)
