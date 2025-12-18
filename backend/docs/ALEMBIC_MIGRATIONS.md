# Database Migrations with Alembic

This guide explains how to use Alembic for database schema migrations in the Floodingnaque backend.

## What is Alembic?

Alembic is a database migration tool for SQLAlchemy that allows you to:
- **Version control your database schema**
- **Safely update production databases**
- **Rollback changes if needed**
- **Track schema changes over time**

## Quick Start

### 1. Create Your First Migration

```powershell
# From the backend directory
cd c:\floodingnaque\backend

# Generate migration from current models
alembic revision --autogenerate -m "Initial schema"
```

**Note:** The correct flag is `--autogenerate` (not `--autoregenerate`)

### 2. Review the Migration

Check the generated file in `alembic/versions/`:

```python
# Example: alembic/versions/abc123_initial_schema.py
def upgrade() -> None:
    # Creates tables
    op.create_table('weather_data',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('temperature', sa.Float(), nullable=True),
        # ...
    )

def downgrade() -> None:
    # Drops tables
    op.drop_table('weather_data')
```

### 3. Apply the Migration

```powershell
# Apply all pending migrations
alembic upgrade head

# Or apply one migration at a time
alembic upgrade +1
```

### 4. Verify Migration Status

```powershell
# Check current migration version
alembic current

# View migration history
alembic history --verbose
```

---

## Common Commands

### Create Migrations

```powershell
# Auto-generate from model changes
alembic revision --autogenerate -m "Add user_id column"

# Create empty migration (manual)
alembic revision -m "Custom data migration"
```

### Apply Migrations

```powershell
# Upgrade to latest
alembic upgrade head

# Upgrade one version
alembic upgrade +1

# Upgrade to specific version
alembic upgrade abc123
```

### Rollback Migrations

```powershell
# Downgrade one version
alembic downgrade -1

# Downgrade to specific version
alembic downgrade abc123

# Downgrade all (back to empty database)
alembic downgrade base
```

### View Information

```powershell
# Current version
alembic current

# Migration history
alembic history

# Show SQL without executing
alembic upgrade head --sql
```

---

## Configuration

### Database URL

Alembic automatically uses your `DATABASE_URL` environment variable:

```bash
# .env file
DATABASE_URL=sqlite:///data/floodingnaque.db

# Or for Supabase PostgreSQL
DATABASE_URL=postgresql://user:pass@host:5432/database
```

### Models Location

Models are imported from `app.models.db`:

```python
# alembic/env.py (already configured)
from app.models.db import Base
target_metadata = Base.metadata
```

---

## Workflow Examples

### Example 1: Add New Column

1. **Update your model:**

```python
# app/models/db.py
class WeatherData(Base):
    # ... existing columns ...
    wind_speed = Column(Float, nullable=True)  # NEW
```

2. **Generate migration:**

```powershell
alembic revision --autogenerate -m "Add wind_speed column"
```

3. **Review generated migration:**

```python
# alembic/versions/xxx_add_wind_speed_column.py
def upgrade():
    op.add_column('weather_data', 
        sa.Column('wind_speed', sa.Float(), nullable=True))

def downgrade():
    op.drop_column('weather_data', 'wind_speed')
```

4. **Apply migration:**

```powershell
alembic upgrade head
```

### Example 2: Create New Table

1. **Add new model:**

```python
# app/models/db.py
class UserPreferences(Base):
    __tablename__ = 'user_preferences'
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), unique=True, nullable=False)
    alert_threshold = Column(Float, default=0.7)
```

2. **Generate and apply:**

```powershell
alembic revision --autogenerate -m "Add user_preferences table"
alembic upgrade head
```

### Example 3: Data Migration

For data transformations, create a manual migration:

```powershell
alembic revision -m "Migrate old risk levels"
```

Edit the generated file:

```python
def upgrade():
    # Custom SQL or Python code
    op.execute("""
        UPDATE predictions 
        SET risk_level = risk_level * 100 
        WHERE risk_level < 1
    """)

def downgrade():
    op.execute("""
        UPDATE predictions 
        SET risk_level = risk_level / 100 
        WHERE risk_level > 1
    """)
```

---

## Best Practices

### 1. Always Review Auto-Generated Migrations

Alembic might not detect:
- Column renames (appears as drop + add)
- Table renames
- Complex constraints

### 2. Test Migrations Locally First

```powershell
# Test upgrade
alembic upgrade head

# Test downgrade
alembic downgrade -1

# Test upgrade again
alembic upgrade head
```

### 3. Never Edit Applied Migrations

Once a migration is applied in production, **never edit it**. Instead:
- Create a new migration to fix issues
- Or rollback and create a corrected version

### 4. Use Descriptive Messages

```powershell
# Good
alembic revision --autogenerate -m "Add email_verified column to users"

# Bad
alembic revision --autogenerate -m "Update"
```

### 5. Backup Before Production Migrations

```powershell
# PostgreSQL backup
pg_dump database_name > backup_$(date +%Y%m%d).sql

# SQLite backup
cp data/floodingnaque.db data/floodingnaque_backup_$(date +%Y%m%d).db
```

---

## Production Deployment

### Option 1: Manual Migration

```powershell
# SSH into production server
ssh production-server

# Navigate to backend
cd /app/backend

# Apply migrations
alembic upgrade head
```

### Option 2: Automated in CI/CD

```yaml
# .github/workflows/deploy.yml
- name: Run database migrations
  run: |
    cd backend
    alembic upgrade head
```

### Option 3: Docker Entrypoint

```bash
#!/bin/bash
# docker-entrypoint.sh

# Run migrations on startup
cd /app/backend
alembic upgrade head

# Start application
python main.py
```

---

## Troubleshooting

### Error: "Can't locate revision identified by 'xxx'"

**Cause:** Migration file is missing or database is out of sync

**Solution:**
```powershell
# Check current version
alembic current

# Stamp database with current version
alembic stamp head
```

### Error: "Target database is not up to date"

**Cause:** Database has unapplied migrations

**Solution:**
```powershell
# Apply pending migrations
alembic upgrade head
```

### Error: "Table already exists"

**Cause:** Database has tables but no migration history

**Solution:**
```powershell
# Mark database as having the initial migration
alembic stamp head
```

### Autogenerate Detects No Changes

**Cause:** Models not imported or metadata not set

**Solution:**
- Verify `alembic/env.py` imports `Base` from `app.models.db`
- Ensure all models inherit from `Base`
- Check that `target_metadata = Base.metadata`

---

## Migration File Structure

```
backend/
├── alembic/
│   ├── versions/
│   │   ├── abc123_initial_schema.py
│   │   ├── def456_add_user_id.py
│   │   └── ghi789_add_indexes.py
│   ├── env.py          # Configuration
│   ├── README
│   └── script.py.mako  # Template for new migrations
├── alembic.ini         # Alembic settings
└── app/
    └── models/
        └── db.py       # Your SQLAlchemy models
```

---

## Integration with Floodingnaque

### Current Models

The following tables will be created:

- `weather_data` - Historical weather records
- `predictions` - Flood predictions
- `alert_history` - Alert delivery logs
- `model_metadata` - ML model versions
- `api_keys` - API authentication (if implemented)

### Running Migrations

```powershell
# Development (SQLite)
cd backend
alembic upgrade head

# Production (Supabase PostgreSQL)
# Set DATABASE_URL in .env first
alembic upgrade head
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `alembic revision --autogenerate -m "msg"` | Create migration from models |
| `alembic upgrade head` | Apply all pending migrations |
| `alembic downgrade -1` | Rollback one migration |
| `alembic current` | Show current version |
| `alembic history` | Show migration history |
| `alembic stamp head` | Mark database as up-to-date |

---

## Next Steps

1. ✅ Alembic is configured and ready to use
2. Create your first migration: `alembic revision --autogenerate -m "Initial schema"`
3. Review the generated migration file
4. Apply it: `alembic upgrade head`
5. Commit migrations to version control

---

## Support

- **Alembic Docs:** https://alembic.sqlalchemy.org/
- **SQLAlchemy Docs:** https://docs.sqlalchemy.org/
- **Floodingnaque Backend:** `docs/BACKEND_COMPLETE.md`

Happy migrating! 🚀
