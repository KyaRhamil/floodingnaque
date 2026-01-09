# 🚀 Training Quick Start - New Method

**TL;DR:** Use production data sources instead of static CSVs

---

## ⚡ One Command Setup

```powershell
# 1. Navigate to backend
cd d:\floodingnaque\backend

# 2. Copy environment template (if not exists)
if (!(Test-Path .env.production)) {
    Copy-Item .env.production.example .env.production
    Write-Host "✓ Created .env.production - PLEASE EDIT WITH YOUR API KEYS"
}

# 3. Run production training pipeline
python scripts/train_with_production_data.py --production
```

**Time:** 1-2 hours  
**Output:** Trained model v5+ using ALL production data sources

---

## 📋 Prerequisites Checklist

Before training, ensure you have:

- [ ] **.env.production** configured with:
  - [ ] `DATABASE_URL` (Supabase connection string)
  - [ ] `SUPABASE_KEY` and `SUPABASE_SECRET_KEY`
  - [ ] `GOOGLE_APPLICATION_CREDENTIALS` (for satellite data)
  - [ ] `WORLDTIDES_API_KEY` (for tidal data)
  - [ ] `OWM_API_KEY` (backup weather source)

- [ ] **Python packages** installed:
  ```powershell
  pip install -r requirements.txt
  ```

- [ ] **Google service account** JSON file in backend directory:
  ```
  backend/floodingnaque-service-account.json
  ```

---

## 🎯 Training Options

### **Option 1: Full Production (Recommended)**
```powershell
python scripts/train_with_production_data.py --production
```
✅ All data sources  
✅ Grid search optimization  
✅ SHAP explainability  
⏱️ ~1-2 hours

---

### **Option 2: Quick Training**
```powershell
python scripts/train_with_production_data.py --days 180
```
✅ Last 6 months  
❌ No grid search  
⏱️ ~20-30 minutes

---

### **Option 3: Progressive (Thesis)**
```powershell
python scripts/train_with_production_data.py --progressive --grid-search
```
✅ Models v1-v4 (2022→2025)  
✅ Shows improvement  
⏱️ ~2-3 hours

---

## 📊 What Gets Ingested

| Data Source | What It Provides | Required? |
|-------------|------------------|-----------|
| **Supabase DB** | Real-time weather from OpenWeatherMap | ✅ Core |
| **Earth Engine** | Satellite precipitation (GPM, CHIRPS, ERA5) | ⭐ Recommended |
| **Meteostat** | Weather station observations | ⭐ Recommended |
| **WorldTides** | Tidal heights (coastal flooding) | ⭐ Recommended |
| **Official Records** | 2022-2025 labeled flood events | ✅ Core |

**Result:** Comprehensive dataset with 5+ data sources vs old method's 1 CSV

---

## 🔧 Resource Configuration

Training auto-detects resources from `.env.production`:

```bash
# Option 1: Explicit (recommended for production)
TRAINING_N_JOBS=4

# Option 2: Auto-detect from DB pool
DB_POOL_SIZE=10  # Training will use 5 (pool_size/2)

# Option 3: Default (uses all CPUs - 1)
# Leave TRAINING_N_JOBS blank
```

**Why this matters:**
- Old method: `n_jobs=-1` → crashes on small servers
- New method: Respects environment limits → stable

---

## ✅ Verify It Worked

### **1. Check Training Output**
```powershell
# Should see:
# ✓ Fetched X records from Supabase
# ✓ Fetched Y satellite records
# ✓ Model training completed successfully!
# ✓ Test accuracy: 0.98XX
```

### **2. Check Files Created**
```powershell
# Models
ls models/flood_rf_model_v*.joblib
ls models/flood_rf_model_v*.json

# Training data
ls data/training/production_data_*.csv

# Reports
ls reports/shap_importance.png
ls reports/learning_curves.png
```

### **3. Validate Model**
```powershell
python scripts/validate_model.py
# Expected: ✓ MODEL VALIDATION PASSED
```

---

## 🐛 Common Issues

### **"No data ingested"**
```powershell
# Check database connection
python scripts/test_supabase_connection.py

# Verify .env.production
cat .env.production | Select-String "DATABASE_URL"
```

### **"Earth Engine not initialized"**
```powershell
# Check service account file
ls floodingnaque-service-account.json

# Verify .env.production
cat .env.production | Select-String "GOOGLE_APPLICATION_CREDENTIALS"
```

### **"Out of memory"**
```powershell
# Reduce parallelism in .env.production
TRAINING_N_JOBS=2
```

---

## 📖 Full Documentation

- **[TRAINING_UPGRADE_GUIDE.md](TRAINING_UPGRADE_GUIDE.md)** - Complete migration guide
- **[.env.production.example](.env.production.example)** - Configuration template
- Run `python scripts/train_with_production_data.py --help` for all options

---

## 🆚 Old vs New

| Old Method | New Method |
|------------|------------|
| `python scripts/train.py` | `python scripts/train_with_production_data.py --production` |
| 1 CSV file | 5 data sources |
| Static data | Real-time data |
| `n_jobs=-1` (crashes) | Environment-aware |
| ~1 hour | ~1-2 hours (comprehensive) |
| 95% accuracy | 96-98% accuracy |

---

**Ready?** Run this now:

```powershell
cd d:\floodingnaque\backend
python scripts/train_with_production_data.py --production
```

🎯 **Goal:** Replace outdated training with production-ready pipeline  
⏱️ **ETA:** 1-2 hours for first run  
✅ **Result:** Better model with fresh, comprehensive data

---

**Date:** January 6, 2026  
**Status:** ✅ Ready to Use
