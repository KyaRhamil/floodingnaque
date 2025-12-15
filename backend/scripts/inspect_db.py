import sqlite3
import os

# Use the correct database path
db_path = os.getenv('DATABASE_URL', 'sqlite:///data/floodingnaque.db')
if db_path.startswith('sqlite:///'):
    db_path = db_path.replace('sqlite:///', '')

print(f"Connecting to database: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print('\nTables:', tables)

for table in tables:
    table_name = table[0]
    # Get columns
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f'\nColumns in {table_name}:')
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    # Get record count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"Records: {count}")

print("\n" + "="*60)
print("Database inspection complete!")
print("="*60)

conn.close()
