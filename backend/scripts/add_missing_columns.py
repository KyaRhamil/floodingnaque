#!/usr/bin/env python3
"""
Migration script to add missing columns (created_at, updated_at) to weather_data table.

This fixes the sqlite3.OperationalError: no such column: weather_data.created_at error.
"""

import sqlite3
import os
from datetime import datetime, timezone

# Path to the SQLite database
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'floodingnaque.db')


def get_db_path():
    """Get the absolute path to the database file."""
    db_path = os.path.abspath(DB_PATH)
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        return None
    return db_path


def check_column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns


def add_missing_columns():
    """Add created_at and updated_at columns to weather_data table if missing."""
    db_path = get_db_path()
    if not db_path:
        print("Cannot proceed without database file.")
        return False
    
    print(f"Connecting to database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check and add created_at column
        if not check_column_exists(cursor, 'weather_data', 'created_at'):
            print("Adding 'created_at' column to weather_data table...")
            # Use current timestamp as default for existing rows
            default_time = datetime.now(timezone.utc).isoformat()
            cursor.execute(f"""
                ALTER TABLE weather_data 
                ADD COLUMN created_at DATETIME DEFAULT '{default_time}'
            """)
            print("  -> created_at column added successfully")
        else:
            print("  -> created_at column already exists")
        
        # Check and add updated_at column
        if not check_column_exists(cursor, 'weather_data', 'updated_at'):
            print("Adding 'updated_at' column to weather_data table...")
            default_time = datetime.now(timezone.utc).isoformat()
            cursor.execute(f"""
                ALTER TABLE weather_data 
                ADD COLUMN updated_at DATETIME DEFAULT '{default_time}'
            """)
            print("  -> updated_at column added successfully")
        else:
            print("  -> updated_at column already exists")
        
        # Update existing rows that have NULL values
        print("Updating NULL values in existing rows...")
        current_time = datetime.now(timezone.utc).isoformat()
        cursor.execute(f"""
            UPDATE weather_data 
            SET created_at = '{current_time}' 
            WHERE created_at IS NULL
        """)
        cursor.execute(f"""
            UPDATE weather_data 
            SET updated_at = '{current_time}' 
            WHERE updated_at IS NULL
        """)
        
        conn.commit()
        print("\nMigration completed successfully!")
        
        # Verify the columns
        print("\nVerifying table structure:")
        cursor.execute("PRAGMA table_info(weather_data)")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        return True
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")
        conn.rollback()
        return False
    finally:
        conn.close()


if __name__ == '__main__':
    print("=" * 60)
    print("Weather Data Table Migration")
    print("Adding missing columns: created_at, updated_at")
    print("=" * 60)
    print()
    
    success = add_missing_columns()
    
    if success:
        print("\n" + "=" * 60)
        print("Migration successful! You can now restart the server.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Migration failed. Please check the error messages above.")
        print("=" * 60)
        exit(1)
