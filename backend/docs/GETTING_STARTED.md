# Getting Started Guide

**Version**: 2.0 | **Estimated Setup Time**: 5-10 minutes

Complete guide to setting up and running the Floodingnaque backend API for flood prediction.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start (5 Minutes)](#2-quick-start-5-minutes)
3. [Installation](#3-installation)
4. [Database Setup](#4-database-setup)
5. [Train the ML Model](#5-train-the-ml-model)
6. [Start the Server](#6-start-the-server)
7. [Verify Installation](#7-verify-installation)
8. [Test API Endpoints](#8-test-api-endpoints)
9. [Common Operations](#9-common-operations)
10. [Troubleshooting](#10-troubleshooting)
11. [Next Steps](#11-next-steps)

---

## 1. Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.9+ | Runtime environment |
| pip | Latest | Package manager |
| Git | Latest | Version control |

### Optional Software

| Software | Purpose |
|----------|---------|
| PostgreSQL | Production database |
| Docker | Containerized deployment |
| Redis | Caching (optional) |

### API Keys Required

| Service | Purpose | Required |
|---------|---------|----------|
| OpenWeatherMap | Temperature, humidity data | **Yes** |
| Weatherstack | Precipitation data | Optional |

Get your free API key at:
- **OpenWeatherMap**: https://openweathermap.org/api
- **Weatherstack**: https://weatherstack.com/

---

## 2. Quick Start (5 Minutes)

### Step 1: Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Configure Environment
```bash
# Copy example configuration
cp .env.example .env

# Edit .env and add your API keys
# OWM_API_KEY=your_openweathermap_key
# METEOSTAT_API_KEY=your_weatherstack_key
```

### Step 3: Database is Ready!
```
Migration already completed
All tables created
Indexes applied
Existing data preserved
```

Verify database:
```bash
python scripts/inspect_db.py
```

### Step 4: Start the Server
```bash
python main.py
```

Server running at: **http://localhost:5000**

### Step 5: Quick Test
```bash
# Health check
curl http://localhost:5000/health

# Test prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 298.15, "humidity": 75, "precipitation": 15}'
```

---

## 3. Installation

### 3.1 Clone Repository

```bash
git clone <repository-url>
cd floodingnaque
```

### 3.2 Virtual Environment

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
# Create virtual environment
python -m venv venv

# Activate
source venv/bin/activate
```

### 3.3 Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Verify installation:**
```bash
pip list | grep Flask
# Should show: Flask==3.0.0
```

### 3.4 Environment Configuration

Create a `.env` file in the `backend/` directory:

```env
# Database Configuration
DATABASE_URL=sqlite:///data/floodingnaque.db

# API Keys (Required for /ingest endpoint)
OWM_API_KEY=your_openweathermap_api_key_here

# Optional: Better precipitation data
METEOSTAT_API_KEY=your_weatherstack_api_key_here

# Flask Configuration
PORT=5000
HOST=0.0.0.0
FLASK_DEBUG=False
LOG_LEVEL=INFO

# Optional: Sentry for error tracking
# SENTRY_DSN=your_sentry_dsn_here
```

**Important:**
- Never commit `.env` file to version control
- Use `.env.example` as a template
- Store production keys securely

---

## 4. Database Setup

### Automatic Setup

The database is automatically initialized when the Flask application starts. Tables are created if they don't exist.

### Manual Setup

```bash
# Initialize database
python -c "from app.models.db import init_db; init_db()"

# Or run migration
python scripts/migrate_db.py
```

### Verify Database

```bash
python scripts/inspect_db.py
```

Expected output:
```
Tables: weather_data, predictions, alert_history, model_registry

weather_data columns (12):
id, temperature, humidity, precipitation, wind_speed
pressure, location_lat, location_lon, source, timestamp
created_at, updated_at
```

### Using PostgreSQL (Production)

Update your `.env` file:
```env
DATABASE_URL=postgresql://user:password@host:5432/database
```

Then run migrations:
```bash
alembic upgrade head
```

---

## 5. Train the ML Model

### First Time Training

```bash
cd backend
python scripts/train.py
```

The script will:
1. Load data from `data/synthetic_dataset.csv`
2. Train a Random Forest classifier
3. Evaluate the model
4. Save to `models/flood_rf_model.joblib`

### Training with Options

```bash
# With hyperparameter tuning (recommended for thesis)
python scripts/train.py --grid-search --cv-folds 10

# With custom dataset
python scripts/train.py --data data/your_dataset.csv

# Merge and train multiple datasets
python scripts/train.py --data "data/*.csv" --merge-datasets
```

### Verify Model

```bash
python scripts/validate_model.py
```

---

## 6. Start the Server

### 6.1 Development Mode

```bash
cd backend
python main.py
```

Output:
```
* Running on http://0.0.0.0:5000
* Restarting with stat
* Debugger is active!
```

### 6.2 Production Mode (Windows)

**Using Waitress:**
```powershell
waitress-serve --host=0.0.0.0 --port=5000 --threads=4 main:app
```

**Using PowerShell Script:**
```powershell
.\start_server.ps1
```

### 6.3 Production Mode (Linux/Docker)

**Using Gunicorn:**
```bash
gunicorn --bind 0.0.0.0:$PORT --workers 4 --threads 2 --timeout 120 main:app
```

**Using Docker:**
```bash
docker build -t floodingnaque-backend .
docker run -p 5000:5000 --env-file .env floodingnaque-backend
```

---

## 7. Verify Installation

### Health Check

```bash
curl http://localhost:5000/health
```

Expected response:
```json
{
  "status": "healthy",
  "database": "connected",
  "model_available": true,
  "scheduler_running": true
}
```

### Status Check

```bash
curl http://localhost:5000/status
```

Expected response:
```json
{
  "status": "running",
  "database": "connected",
  "model": "loaded"
}
```

### API Documentation

Open in browser: http://localhost:5000/api/docs

---

## 8. Test API Endpoints

### 8.1 Using cURL

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Ingest Weather Data:**
```bash
curl -X POST http://localhost:5000/ingest \
  -H "Content-Type: application/json" \
  -d '{"lat": 14.6, "lon": 120.98}'
```

**Get Historical Data:**
```bash
curl "http://localhost:5000/data?limit=10"
```

**Predict Flood Risk:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 298.15, "humidity": 65.0, "precipitation": 5.0}'
```

### 8.2 Using PowerShell

**Health Check:**
```powershell
Invoke-RestMethod -Uri http://localhost:5000/health
```

**Ingest Weather Data:**
```powershell
$body = @{
    lat = 14.6
    lon = 120.98
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:5000/ingest `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

**Predict Flood Risk:**
```powershell
$body = @{
    temperature = 298.15
    humidity = 65.0
    precipitation = 5.0
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:5000/predict `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

**Batch Prediction:**
```powershell
$body = @{
    predictions = @(
        @{ temperature = 298; humidity = 65; precipitation = 5 },
        @{ temperature = 300; humidity = 70; precipitation = 10 }
    )
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:5000/batch/predict `
  -Method POST `
  -ContentType "application/json" `
  -Body $body
```

---

## 9. Common Operations

### Database Management

```bash
# Run migration
python scripts/migrate_db.py

# Inspect database
python scripts/inspect_db.py

# Check migration status
alembic current

# Apply pending migrations
alembic upgrade head
```

### Model Training

```bash
# Basic training
python scripts/train.py

# With grid search (best for thesis)
python scripts/train.py --grid-search --cv-folds 10

# Validate model
python scripts/validate_model.py

# Compare model versions
python scripts/compare_models.py
```

### Server Management

```bash
# Start development server
python main.py

# Start production server (Windows)
waitress-serve --host=0.0.0.0 --port=5000 main:app

# Start production server (Linux)
gunicorn main:app
```

### Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov

# Specific test file
pytest tests/unit/test_predict.py -v
```

### Documentation

```bash
# Generate thesis report
python scripts/generate_thesis_report.py

# Compare model versions
python scripts/compare_models.py
```

---

## 10. Troubleshooting

### Import Error

**Error:** `ModuleNotFoundError: No module named 'flask'`

**Solution:**
```bash
# Ensure virtual environment is activated
.\venv\Scripts\Activate.ps1  # Windows
source venv/bin/activate      # Linux/macOS

# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Database Error

**Error:** `sqlite3.OperationalError: no such table`

**Solution:**
```bash
# Check database exists
ls data/floodingnaque.db

# Reinitialize if needed
python scripts/migrate_db.py
```

### API Key Error

**Error:** `OWM_API_KEY not set`

**Solution:**
```bash
# Make sure .env file exists and has your keys
cat .env | grep API_KEY

# Verify format
OWM_API_KEY=your_actual_key_here
```

### Port Already in Use

**Error:** `Address already in use: ('0.0.0.0', 5000)`

**Solution (Windows):**
```powershell
# Find process using port 5000
netstat -ano | findstr :5000

# Kill the process
taskkill /PID <PID> /F
```

**Solution (Linux/macOS):**
```bash
# Find and kill process
lsof -ti:5000 | xargs kill -9
```

### Model Not Found

**Error:** `Model file not found`

**Solution:**
```bash
# Train the model
python scripts/train.py

# Verify model exists
ls models/flood_rf_model.joblib
```

### Virtual Environment Not Activated

**Symptom:** Commands fail or use system Python

**Solution (Windows):**
```powershell
# Check if venv is active (should see (venv) in prompt)
# If not, activate:
.\venv\Scripts\Activate.ps1
```

### Diagnostic Commands

```bash
# Test import
python -c "from app.api.app import app; print('App imports OK')"

# Test database
python scripts/inspect_db.py

# Check Python version
python --version

# Check installed packages
pip list
```

---

## 11. Next Steps

### For Development

1. Set up API keys in `.env` file
2. Train the model: `python scripts/train.py`
3. Start the server: `python main.py`
4. Test all endpoints
5. Review API documentation at `/api/docs`

### For Production

1. [ ] Install dependencies: `pip install -r requirements.txt --upgrade`
2. [ ] Configure PostgreSQL database
3. [ ] Set up SSL/TLS certificates
4. [ ] Configure monitoring (Sentry)
5. [ ] Set up CI/CD pipeline
6. [ ] Configure load balancer

### For Thesis

1. [ ] Collect sufficient training data (500+ samples)
2. [ ] Merge all datasets: `python scripts/merge_datasets.py`
3. [ ] Train with grid search: `python scripts/train.py --grid-search`
4. [ ] Generate thesis report: `python scripts/generate_thesis_report.py`
5. [ ] Compare model versions: `python scripts/compare_models.py`

---

## Pro Tips

1. **Never commit .env file** - It contains your API keys
2. **Use validation** - Import from `app.utils.validation`
3. **Check logs** - All errors are logged with context in `logs/app.log`
4. **Test inputs** - Use validators before database insert
5. **Monitor performance** - Check slow query logs

---

## Key Documentation

| Document | Purpose |
|----------|---------|
| [BACKEND_ARCHITECTURE.md](BACKEND_ARCHITECTURE.md) | Complete system architecture |
| [DATABASE_GUIDE.md](DATABASE_GUIDE.md) | Database reference |
| [ALEMBIC_MIGRATIONS.md](ALEMBIC_MIGRATIONS.md) | Migration system |
| [MODEL_MANAGEMENT.md](MODEL_MANAGEMENT.md) | ML model management |
| [POWERSHELL_API_EXAMPLES.md](POWERSHELL_API_EXAMPLES.md) | PowerShell API examples |

---

## You're Ready!

Your backend is now running with:
- Enhanced database schema
- Production-grade security
- Optimized performance
- Complete documentation

**Happy coding! 🚀**

---

**Last Updated**: December 2025  
**Version**: 2.0
