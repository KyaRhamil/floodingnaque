import sqlite3

conn = sqlite3.connect('floodingnaque.db')
cursor = conn.cursor()

# Get all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print('Tables:', tables)

for table in tables:
    table_name = table[0]
    # Get columns
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    print(f'Columns in {table_name}:', columns)
    # Get record count
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f'Records in {table_name}:', count)

conn.close()
