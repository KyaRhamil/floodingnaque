"""Verify Supabase schema has the station_id column."""
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine, text
import os

# Load production environment
load_dotenv(Path(__file__).parent.parent / '.env.production', override=True)

db_url = os.getenv('DATABASE_URL')
if not db_url:
    print("ERROR: DATABASE_URL not set")
    exit(1)

# Use pg8000 driver
db_url = db_url.replace('postgresql://', 'postgresql+pg8000://')

print("Connecting to Supabase...")
engine = create_engine(db_url)

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'weather_data' 
        ORDER BY ordinal_position
    """))
    
    print("\nColumns in weather_data table:")
    print("-" * 40)
    for row in result.fetchall():
        marker = " <-- NEW" if row[0] == 'station_id' else ""
        print(f"  {row[0]}: {row[1]}{marker}")

print("\nVerification complete!")
