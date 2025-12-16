"""Add station_id column to weather_data table for Meteostat integration."""
from sqlalchemy import create_engine, text
import os
import sys

# Load environment from .env.production if APP_ENV is production
if os.getenv('APP_ENV') == 'production':
    from dotenv import load_dotenv
    from pathlib import Path
    env_file = Path(__file__).parent.parent / '.env.production'
    if env_file.exists():
        load_dotenv(env_file, override=True)
        print(f"Loaded environment from: {env_file.name}")

def migrate():
    """Run migration to add station_id column."""
    db_url = os.getenv('DATABASE_URL', 'sqlite:///data/floodingnaque.db')
    
    print(f"Connecting to database...")
    
    # Detect database type
    is_postgres = 'postgresql' in db_url or 'postgres' in db_url
    
    if is_postgres:
        print("Database type: PostgreSQL (Supabase)")
    else:
        print(f"Database type: SQLite ({db_url})")
    
    # Handle postgres URL format
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql+pg8000://', 1)
    elif db_url.startswith('postgresql://') and '+' not in db_url.split('://')[0]:
        db_url = db_url.replace('postgresql://', 'postgresql+pg8000://', 1)
    
    engine = create_engine(db_url)
    
    with engine.begin() as conn:
        # Check if column exists first
        if 'sqlite' in db_url:
            result = conn.execute(text("PRAGMA table_info(weather_data)"))
            columns = [row[1] for row in result.fetchall()]
        else:
            result = conn.execute(text("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'weather_data' AND column_name = 'station_id'
            """))
            columns = [row[0] for row in result.fetchall()]
        
        if 'station_id' not in columns:
            conn.execute(text('ALTER TABLE weather_data ADD COLUMN station_id VARCHAR(50)'))
            print("Added 'station_id' column to weather_data table")
        else:
            print("'station_id' column already exists")
    
    # Create index on source column if not exists
    with engine.begin() as conn:
        try:
            if 'sqlite' in db_url:
                conn.execute(text('CREATE INDEX IF NOT EXISTS idx_weather_source ON weather_data(source)'))
            else:
                conn.execute(text('CREATE INDEX IF NOT EXISTS idx_weather_source ON weather_data(source)'))
            print("Created index on 'source' column")
        except Exception as e:
            print(f"Index may already exist: {e}")
    
    print("Migration completed successfully!")

if __name__ == '__main__':
    migrate()
