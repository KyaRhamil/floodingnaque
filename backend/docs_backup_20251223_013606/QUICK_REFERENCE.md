# 🚀 QUICK REFERENCE - Random Forest Model Training

## Common Workflows

### 1️⃣ Train with New CSV File

```powershell
cd backend
python scripts/train.py --data data/your_new_file.csv
```

### 2️⃣ Merge Multiple Datasets & Train

```powershell
# Step 1: Merge
python scripts/merge_datasets.py --input "data/flood_*.csv"

# Step 2: Train
python scripts/train.py --data data/merged_dataset.csv
```

### 3️⃣ Best Model for Thesis (RECOMMENDED)

```powershell
# Complete optimization pipeline
python scripts/train.py --data "data/*.csv" --merge-datasets --grid-search --cv-folds 10
```

### 4️⃣ Generate Thesis Report

```powershell
python scripts/generate_thesis_report.py
```

---

## File Requirements

### Your CSV Must Have:
- `temperature` (float)
- `humidity` (float)  
- `precipitation` (float)
- `flood` (0 or 1)

### Optional Columns:
- `wind_speed` (float)
- Any other weather features

---

## Model Versioning

### Automatic Versioning
```
Training #1 → flood_rf_model_v1.joblib
Training #2 → flood_rf_model_v2.joblib
Training #3 → flood_rf_model_v3.joblib
```

### Version Metadata
Each version saves:
- Training date/time
- Dataset used
- Model parameters
- Performance metrics
- Feature importance

### Check Available Models
```powershell
python -c "from app.services.predict import list_available_models; import json; print(json.dumps(list_available_models(), indent=2))"
```

---

## Training Options

### Basic
```powershell
python scripts/train.py
```

### With Specific Dataset
```powershell
python scripts/train.py --data data/my_data.csv
```

### With Hyperparameter Tuning
```powershell
python scripts/train.py --grid-search
```

### With Cross-Validation
```powershell
python scripts/train.py --cv-folds 10
```

### Merge Multiple Files
```powershell
python scripts/train.py --data "data/*.csv" --merge-datasets
```

### Everything Combined (BEST)
```powershell
python scripts/train.py --data "data/*.csv" --merge-datasets --grid-search --cv-folds 10
```

---

## Dataset Management

### Merge All CSVs in Folder
```powershell
python scripts/merge_datasets.py
```

### Merge Specific Pattern
```powershell
python scripts/merge_datasets.py --input "data/flood_*.csv" --output data/combined.csv
```

### Keep Duplicates
```powershell
python scripts/merge_datasets.py --keep-duplicates
```

---

## Performance Reports

### Generate All Visualizations
```powershell
python scripts/generate_thesis_report.py
```

### Custom Report
```powershell
python scripts/generate_thesis_report.py --model models/flood_rf_model_v3.joblib --output my_report
```

### What You Get:
- Feature importance chart
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Metrics comparison
- Learning curves
- Comprehensive text report

---

## Validation

### Validate Current Model
```powershell
python scripts/validate_model.py
```

### Validate Specific Version
```powershell
python scripts/validate_model.py --model models/flood_rf_model_v3.joblib
```

---

## API Usage

### Start Server
```powershell
cd backend
python main.py
```

### Test Prediction
```powershell
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"temperature\": 25.0, \"humidity\": 80.0, \"precipitation\": 15.0}"
```

### Use Specific Model Version
```powershell
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"temperature\": 25.0, \"humidity\": 80.0, \"precipitation\": 15.0, \"model_version\": 3}"
```

### List All Models
```powershell
curl http://localhost:5000/api/models
```

### Check Status
```powershell
curl http://localhost:5000/status
```

---

## Expected Performance

### Good Model
- Accuracy: 85-95%
- Precision: 80-95%
- Recall: 80-95%
- F1 Score: 80-95%

### Excellent Model
- Accuracy: 95%+
- Precision: 95%+
- Recall: 95%+
- F1 Score: 95%+

---

## Troubleshooting

### Issue: "Data file not found"
**Solution:** Check file path and ensure CSV exists
```powershell
ls data/*.csv
```

### Issue: "Missing required columns"
**Solution:** Ensure CSV has temperature, humidity, precipitation, flood columns

### Issue: Low accuracy
**Solutions:**
1. Use `--grid-search` for better parameters
2. Collect more training data
3. Merge multiple datasets
4. Add more features to CSV

### Issue: Model not loading
**Solution:** Retrain the model
```powershell
python scripts/train.py
```

---

## File Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── app.py               # Flask application factory
│   │   ├── routes/              # API route blueprints
│   │   │   ├── data.py
│   │   │   ├── health.py
│   │   │   ├── ingest.py
│   │   │   ├── models.py
│   │   │   └── predict.py
│   │   ├── middleware/          # Request middleware
│   │   │   ├── auth.py
│   │   │   ├── logging.py
│   │   │   ├── rate_limit.py
│   │   │   └── security.py
│   │   └── schemas/             # Request/response schemas
│   ├── core/                    # Config, exceptions
│   ├── services/                # Business logic
│   ├── models/                  # Database models
│   └── utils/                   # Utilities
├── scripts/
│   ├── train.py                 # ← Main training
│   ├── progressive_train.py     # ← Progressive training (v1-v4)
│   ├── generate_thesis_report.py # ← Generate reports
│   ├── merge_datasets.py        # ← Merge CSVs
│   ├── compare_models.py
│   ├── validate_model.py
│   └── evaluate_model.py
├── models/
│   ├── flood_rf_model.joblib    # ← Latest model
│   ├── flood_rf_model_v*.joblib # ← Versioned models
│   └── *.json                   # ← Metadata
├── data/
│   └── *.csv                    # ← Your datasets
├── tests/
│   ├── unit/
│   ├── integration/
│   └── security/
└── reports/
    └── *.png, *.txt             # ← Generated reports
```

---

## Key Points for Thesis

1. **Random Forest = Ensemble Learning**
   - Multiple decision trees voting together
   - Robust and accurate
   - Provides feature importance

2. **3-Level Risk Classification**
   - Safe (Green)
   - Alert (Yellow)
   - Critical (Red)

3. **Automatic Versioning**
   - Every training creates new version
   - Compare models over time
   - Track improvements

4. **Easy Dataset Integration**
   - Just add CSV to data/ folder
   - Run training command
   - New model ready!

---

## For Your Thesis Defense

### Best Workflow:
```powershell
# 1. Prepare data
python scripts/merge_datasets.py --input "data/*.csv"

# 2. Train optimal model
python scripts/train.py --data data/merged_dataset.csv --grid-search --cv-folds 10

# 3. Generate presentation materials
python scripts/generate_thesis_report.py

# 4. Validate
python scripts/validate_model.py
```

### Show These Charts:
- ✅ Feature importance
- ✅ Confusion matrix  
- ✅ ROC curve
- ✅ Metrics comparison
- ✅ Learning curves

### Be Ready to Explain:
- ✅ Why Random Forest?
- ✅ What is cross-validation?
- ✅ What do the metrics mean?
- ✅ Which features matter most?
- ✅ How versioning works?

---

## Need Help?

See detailed guides:
- `THESIS_GUIDE.md` - Complete thesis preparation guide
- `MODEL_MANAGEMENT.md` - Detailed model management
- `BACKEND_COMPLETE.md` - Full system documentation

---

**Quick Tip:** For thesis, ALWAYS use `--grid-search` for your final model! ⚡
