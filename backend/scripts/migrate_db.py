#!/usr/bin/env python3
"""
Database Migration Script
Safely migrates existing database to new enhanced schema.
"""

import sqlite3
import os
import sys
from pathlib import Path
from datetime import datetime
import logging
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.db import init_db, engine, Base
from sqlalchemy import text


def backup_database(db_path):
    """Create a backup of the existing database."""
    if not os.path.exists(db_path):
        logger.warning(f"Database not found at {db_path}, no backup needed")
        return None
    
    backup_path = f"{db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to backup database: {str(e)}")
        raise


def check_existing_schema(conn):
    """Check existing database schema."""
    cursor = conn.cursor()
    
    # Get existing tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    logger.info(f"Existing tables: {tables}")
    
    # Get weather_data columns if table exists
    if 'weather_data' in tables:
        cursor.execute("PRAGMA table_info(weather_data)")
        columns = cursor.fetchall()
        logger.info(f"Existing weather_data columns: {[col[1] for col in columns]}")
        return columns
    
    return []


def migrate_data(old_conn, new_conn):
    """Migrate data from old schema to new schema."""
    old_cursor = old_conn.cursor()
    new_cursor = new_conn.cursor()
    
    try:
        # Get existing data from old weather_data table
        old_cursor.execute("SELECT id, temperature, humidity, precipitation, timestamp FROM weather_data")
        old_data = old_cursor.fetchall()
        
        if not old_data:
            logger.info("No existing data to migrate")
            return
        
        logger.info(f"Migrating {len(old_data)} weather records...")
        
        # Insert into new schema with additional default values
        for row in old_data:
            new_cursor.execute("""
                INSERT INTO weather_data 
                (id, temperature, humidity, precipitation, timestamp, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'Migrated', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, row)
        
        new_conn.commit()
        logger.info(f"Successfully migrated {len(old_data)} records")
        
    except Exception as e:
        logger.error(f"Error during data migration: {str(e)}")
        new_conn.rollback()
        raise


def add_new_columns_if_missing(conn):
    """Add new columns to existing weather_data table if they don't exist."""
    cursor = conn.cursor()
    
    # Get current columns
    cursor.execute("PRAGMA table_info(weather_data)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    
    # Define new columns to add
    new_columns = {
        'wind_speed': 'FLOAT',
        'pressure': 'FLOAT',
        'location_lat': 'FLOAT',
        'location_lon': 'FLOAT',
        'source': "VARCHAR(50) DEFAULT 'OWM'",
        'created_at': "DATETIME DEFAULT CURRENT_TIMESTAMP",
        'updated_at': "DATETIME DEFAULT CURRENT_TIMESTAMP"
    }
    
    # Add missing columns
    for col_name, col_type in new_columns.items():
        if col_name not in existing_columns:
            try:
                cursor.execute(f"ALTER TABLE weather_data ADD COLUMN {col_name} {col_type}")
                logger.info(f"Added column: {col_name}")
            except sqlite3.OperationalError as e:
                if 'duplicate column name' not in str(e).lower():
                    logger.warning(f"Could not add column {col_name}: {str(e)}")
    
    conn.commit()


def create_new_tables(conn):
    """Create new tables if they don't exist."""
    cursor = conn.cursor()
    
    # Check existing tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cursor.fetchall()}
    
    # Predictions table
    if 'predictions' not in existing_tables:
        cursor.execute("""
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                weather_data_id INTEGER,
                prediction INTEGER NOT NULL CHECK (prediction IN (0, 1)),
                risk_level INTEGER CHECK (risk_level IS NULL OR risk_level IN (0, 1, 2)),
                risk_label VARCHAR(50),
                confidence FLOAT CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
                model_version INTEGER,
                model_name VARCHAR(100) DEFAULT 'flood_rf_model',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
                FOREIGN KEY (weather_data_id) REFERENCES weather_data(id) ON DELETE SET NULL
            )
        """)
        cursor.execute("CREATE INDEX idx_prediction_risk ON predictions(risk_level)")
        cursor.execute("CREATE INDEX idx_prediction_model ON predictions(model_version)")
        cursor.execute("CREATE INDEX idx_prediction_created ON predictions(created_at)")
        logger.info("Created predictions table")
    
    # Alert history table
    if 'alert_history' not in existing_tables:
        cursor.execute("""
            CREATE TABLE alert_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                risk_level INTEGER NOT NULL,
                risk_label VARCHAR(50) NOT NULL,
                location VARCHAR(255),
                recipients TEXT,
                message TEXT,
                delivery_status VARCHAR(50),
                delivery_channel VARCHAR(50),
                error_message TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
                delivered_at DATETIME,
                FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
            )
        """)
        cursor.execute("CREATE INDEX idx_alert_risk ON alert_history(risk_level)")
        cursor.execute("CREATE INDEX idx_alert_status ON alert_history(delivery_status)")
        cursor.execute("CREATE INDEX idx_alert_created ON alert_history(created_at)")
        logger.info("Created alert_history table")
    
    # Model registry table
    if 'model_registry' not in existing_tables:
        cursor.execute("""
            CREATE TABLE model_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version INTEGER UNIQUE NOT NULL,
                file_path VARCHAR(500) NOT NULL,
                algorithm VARCHAR(100) DEFAULT 'RandomForest',
                accuracy FLOAT CHECK (accuracy IS NULL OR (accuracy >= 0 AND accuracy <= 1)),
                precision_score FLOAT CHECK (precision_score IS NULL OR (precision_score >= 0 AND precision_score <= 1)),
                recall_score FLOAT CHECK (recall_score IS NULL OR (recall_score >= 0 AND recall_score <= 1)),
                f1_score FLOAT CHECK (f1_score IS NULL OR (f1_score >= 0 AND f1_score <= 1)),
                roc_auc FLOAT,
                training_date DATETIME,
                dataset_size INTEGER,
                dataset_path VARCHAR(500),
                parameters TEXT,
                feature_importance TEXT,
                is_active BOOLEAN DEFAULT 0,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
                created_by VARCHAR(100)
            )
        """)
        cursor.execute("CREATE INDEX idx_model_version ON model_registry(version)")
        cursor.execute("CREATE INDEX idx_model_active ON model_registry(is_active)")
        logger.info("Created model_registry table")
    
    conn.commit()


def create_indexes(conn):
    """Create additional indexes for performance."""
    cursor = conn.cursor()
    
    indexes = [
        ("idx_weather_timestamp", "weather_data", "timestamp"),
        ("idx_weather_created", "weather_data", "created_at"),
    ]
    
    for idx_name, table, column in indexes:
        try:
            cursor.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})")
            logger.info(f"Created index: {idx_name}")
        except sqlite3.OperationalError as e:
            logger.warning(f"Index {idx_name} may already exist: {str(e)}")
    
    conn.commit()


def run_migration():
    """Run the complete migration process."""
    logger.info("="*60)
    logger.info("Starting Database Migration")
    logger.info("="*60)
    
    # Determine database path
    db_url = os.getenv('DATABASE_URL', 'sqlite:///data/floodingnaque.db')
    
    # Extract path from SQLite URL
    if db_url.startswith('sqlite:///'):
        db_path = db_url.replace('sqlite:///', '')
    else:
        logger.error("Only SQLite databases are supported for migration")
        return False
    
    # Make path relative to backend directory
    backend_dir = Path(__file__).parent.parent
    db_path = backend_dir / db_path
    
    logger.info(f"Database path: {db_path}")
    
    # Check if database exists
    db_exists = os.path.exists(db_path)
    
    if db_exists:
        # Backup existing database
        logger.info("Step 1: Backing up existing database...")
        backup_path = backup_database(db_path)
        
        # Open connection to existing database
        logger.info("Step 2: Analyzing existing schema...")
        conn = sqlite3.connect(db_path)
        check_existing_schema(conn)
        
        # Add new columns to existing table
        logger.info("Step 3: Adding new columns to weather_data...")
        add_new_columns_if_missing(conn)
        
        # Create new tables
        logger.info("Step 4: Creating new tables...")
        create_new_tables(conn)
        
        # Create indexes
        logger.info("Step 5: Creating performance indexes...")
        create_indexes(conn)
        
        conn.close()
        logger.info("Migration completed successfully!")
        logger.info(f"Backup saved at: {backup_path}")
    else:
        # Create fresh database with new schema
        logger.info("No existing database found. Creating fresh database...")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize with new schema
        init_db()
        logger.info("Fresh database created successfully!")
    
    # Verify final schema
    logger.info("\nStep 6: Verifying final schema...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    logger.info(f"Final tables: {', '.join(tables)}")
    
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        logger.info(f"  - {table}: {count} records")
    
    conn.close()
    
    logger.info("="*60)
    logger.info("Migration completed successfully! ✅")
    logger.info("="*60)
    
    return True


def analyze_migration_plan(db_path: Path) -> dict:
    """
    Analyze what changes would be made during migration without executing them.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        dict: Migration plan with changes to be made
    """
    plan = {
        'database_path': str(db_path),
        'database_exists': os.path.exists(db_path),
        'actions': [],
        'new_tables': [],
        'new_columns': [],
        'new_indexes': [],
    }
    
    if not plan['database_exists']:
        plan['actions'].append('CREATE new database with full schema')
        plan['new_tables'] = ['weather_data', 'predictions', 'alert_history', 'model_registry', 'webhooks', 'api_requests']
        return plan
    
    # Analyze existing database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get existing tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cursor.fetchall()}
    
    plan['actions'].append('BACKUP existing database')
    
    # Check for missing tables
    required_tables = ['predictions', 'alert_history', 'model_registry']
    for table in required_tables:
        if table not in existing_tables:
            plan['new_tables'].append(table)
            plan['actions'].append(f'CREATE TABLE {table}')
    
    # Check for missing columns in weather_data
    if 'weather_data' in existing_tables:
        cursor.execute("PRAGMA table_info(weather_data)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        new_columns = {
            'wind_speed': 'FLOAT',
            'pressure': 'FLOAT',
            'location_lat': 'FLOAT',
            'location_lon': 'FLOAT',
            'source': "VARCHAR(50) DEFAULT 'OWM'",
            'created_at': "DATETIME DEFAULT CURRENT_TIMESTAMP",
            'updated_at': "DATETIME DEFAULT CURRENT_TIMESTAMP"
        }
        
        for col_name, col_type in new_columns.items():
            if col_name not in existing_columns:
                plan['new_columns'].append({'table': 'weather_data', 'column': col_name, 'type': col_type})
                plan['actions'].append(f'ADD COLUMN weather_data.{col_name} ({col_type})')
    
    # Check for missing indexes
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    existing_indexes = {row[0] for row in cursor.fetchall()}
    
    required_indexes = [
        ('idx_weather_timestamp', 'weather_data', 'timestamp'),
        ('idx_weather_created', 'weather_data', 'created_at'),
        ('idx_prediction_risk', 'predictions', 'risk_level'),
        ('idx_prediction_model', 'predictions', 'model_version'),
    ]
    
    for idx_name, table, column in required_indexes:
        if idx_name not in existing_indexes and table in existing_tables:
            plan['new_indexes'].append({'name': idx_name, 'table': table, 'column': column})
            plan['actions'].append(f'CREATE INDEX {idx_name} ON {table}({column})')
    
    conn.close()
    return plan


def print_dry_run_report(plan: dict):
    """Print a formatted dry-run report."""
    logger.info("="*60)
    logger.info("DRY RUN MIGRATION REPORT")
    logger.info("="*60)
    logger.info(f"Database path: {plan['database_path']}")
    logger.info(f"Database exists: {plan['database_exists']}")
    logger.info("")
    
    if not plan['actions']:
        logger.info("No changes needed - database is up to date!")
        return
    
    logger.info(f"Actions to be performed ({len(plan['actions'])}):\n")
    for i, action in enumerate(plan['actions'], 1):
        logger.info(f"  {i}. {action}")
    
    if plan['new_tables']:
        logger.info(f"\nNew tables to create: {', '.join(plan['new_tables'])}")
    
    if plan['new_columns']:
        logger.info("\nNew columns to add:")
        for col in plan['new_columns']:
            logger.info(f"  - {col['table']}.{col['column']}: {col['type']}")
    
    if plan['new_indexes']:
        logger.info("\nNew indexes to create:")
        for idx in plan['new_indexes']:
            logger.info(f"  - {idx['name']} on {idx['table']}({idx['column']})")
    
    logger.info("")
    logger.info("="*60)
    logger.info("To execute these changes, run without --dry-run flag")
    logger.info("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate database to enhanced schema')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    parser.add_argument('--force', action='store_true', help='Force migration even if backup fails')
    
    args = parser.parse_args()
    
    # Determine database path
    db_url = os.getenv('DATABASE_URL', 'sqlite:///data/floodingnaque.db')
    if db_url.startswith('sqlite:///'):
        db_path = db_url.replace('sqlite:///', '')
    else:
        logger.error("Only SQLite databases are supported for migration")
        sys.exit(1)
    
    backend_dir = Path(__file__).parent.parent
    db_path = backend_dir / db_path
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
        plan = analyze_migration_plan(db_path)
        print_dry_run_report(plan)
        sys.exit(0)
    
    try:
        success = run_migration()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
