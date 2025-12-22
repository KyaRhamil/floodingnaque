# Floodingnaque Backend - Startup Instructions

## ✅ All Issues Fixed!

All errors have been resolved. The application is ready to start.

---

## 🚀 Quick Start

### Option 1: Using PowerShell Script (Recommended)
```powershell
cd backend
.\start_server.ps1
```

### Option 2: Manual Commands
```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. Navigate to backend
cd backend

# 3. Start server
python main.py
```

---

## 📊 What's Available

### Core Endpoints (Legacy - No Prefix)
- `GET /health` - Health check
- `POST /predict` - Flood prediction
- `POST /ingest` - Data ingestion
- `GET /data` - Historical data
- `GET /models` - Model info

### New Feature Endpoints (V1 Only)
- `POST /webhooks/register` - Register webhook
- `GET /webhooks/list` - List webhooks
- `DELETE /webhooks/{id}` - Delete webhook
- `POST /webhooks/{id}/toggle` - Toggle webhook
- `POST /batch/predict` - Batch predictions
- `GET /export/weather` - Export weather data
- `GET /export/predictions` - Export predictions

---

## 🧪 Testing Commands

Once server is running, open a **new terminal** and test:

### 1. Health Check
```powershell
Invoke-RestMethod -Uri http://localhost:5000/health
```

### 2. Batch Prediction
```powershell
$body = @{
    predictions = @(
        @{
            temperature = 298
            humidity = 65
            precipitation = 5
        },
        @{
            temperature = 300
            humidity = 70
            precipitation = 10
        }
    )
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:5000/batch/predict `
  -Method POST `
  -Headers @{"X-API-Key"="your_api_key_here"} `
  -ContentType "application/json" `
  -Body $body
```

### 3. Register Webhook
```powershell
$body = @{
    url = "https://example.com/webhook"
    events = @("flood_detected", "high_risk")
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:5000/webhooks/register `
  -Method POST `
  -Headers @{"X-API-Key"="your_api_key_here"} `
  -ContentType "application/json" `
  -Body $body
```

### 4. Export Data
```powershell
Invoke-RestMethod -Uri "http://localhost:5000/export/weather?format=json&limit=10" `
  -Headers @{"X-API-Key"="your_api_key_here"}
```

---

## 📝 What Was Fixed

### 1. Import Error in batch.py ✅
**Issue:** `predict_flood_risk` doesn't exist  
**Fix:** Changed to `predict_flood(input_data=dict)`

### 2. Blueprint Registration Conflict ✅
**Issue:** Blueprints registered twice with same name  
**Fix:** Simplified registration - each blueprint registered once

### 3. Database Tables Created ✅
**Tables Added:**
- `api_requests` - Request/response logging
- `webhooks` - Webhook configurations

---

## 📁 New Files Created

1. ✅ `app/api/middleware/request_logger.py`
2. ✅ `app/api/routes/webhooks.py`
3. ✅ `app/api/routes/batch.py`
4. ✅ `app/api/routes/export.py`
5. ✅ `app/api/routes/v1/__init__.py`
6. ✅ `start_server.ps1`
7. ✅ `docs/NEW_FEATURES_IMPLEMENTATION.md`
8. ✅ `docs/ALEMBIC_MIGRATIONS.md`
9. ✅ `STARTUP_INSTRUCTIONS.md` (this file)

---

## 🎯 Features Implemented

- ✅ Request/Response Logging to Database
- ✅ Proper Log Rotation (10MB, 5 backups)
- ✅ Webhook Support for Alerts
- ✅ Data Export API (CSV/JSON)
- ✅ Batch Prediction Endpoint
- ✅ Database Migrations (Alembic)

---

## 🔍 Troubleshooting

### Server Won't Start
```powershell
# Make sure you're in the backend directory
cd backend

# Make sure venv is activated (you should see (venv) in prompt)
.\venv\Scripts\Activate.ps1

# Try starting again
python main.py
```

### "No module named 'flask'" Error
```powershell
# Virtual environment not activated
.\venv\Scripts\Activate.ps1
```

### Port Already in Use
```powershell
# Check what's using port 5000
netstat -ano | findstr :5000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

---

## 📚 Documentation

- **New Features:** `docs/NEW_FEATURES_IMPLEMENTATION.md`
- **Database Migrations:** `docs/ALEMBIC_MIGRATIONS.md`
- **Sentry Setup:** `docs/SENTRY_SETUP.md`
- **Backend Complete:** `docs/BACKEND_COMPLETE.md`

---

## ✨ Summary

**Status:** ✅ Ready to start  
**Errors Fixed:** 2  
**New Features:** 6  
**New Endpoints:** 8  
**Database Tables:** 2 new tables  

**Next Step:** Run `.\start_server.ps1` or `python main.py` 🚀
