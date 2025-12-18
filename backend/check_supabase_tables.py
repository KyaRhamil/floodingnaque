import os
os.environ['APP_ENV'] = 'production'

from app.core.config import load_env
load_env()

from app.models.db import engine
from sqlalchemy import text

print("Checking Supabase tables...")
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_name
    """))
    print('Tables:')
    for row in result.fetchall():
        print(f'  - {row[0]}')
        
    # Check structure of each table
    tables = [row[0] for row in result.fetchall()]
    for table_name in ['weather_data', 'predictions', 'alert_history', 'model_registry']:
        print(f'\nColumns in {table_name}:')
        result = conn.execute(text(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table_name}' 
            ORDER BY ordinal_position
        """))
        for row in result.fetchall():
            print(f'  - {row[0]}: {row[1]}')