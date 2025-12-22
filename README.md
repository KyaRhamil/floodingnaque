# 🌊 Floodingnaque - Flood Prediction System for Parañaque City

**Random Forest-Based Flood Detection and Alert System**

**🆕 Now with Official Flood Records Training!** Train models with 3,700+ real flood events from 2022-2025!

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/github/license/KyaRhamil/floodingnaque)](LICENSE)
[![Last Updated](https://img.shields.io/github/last-commit/KyaRhamil/floodingnaque)](#)

---
## 🎓 Thesis Defense Ready!

This project implements a **Random Forest machine learning model** to predict flood risks in Parañaque City with a **3-level risk classification system** (Safe/Alert/Critical).

### 🆕 NEW: Official Flood Records Training (2022-2025)

**Train your models with 3,700+ real flood events** from the Parañaque City Disaster Risk Reduction and Management Office (DRRMO):

- **Real-world data** from official government sources
- **4 years of historical records** (2022-2025)
- **Progressive training** showing model evolution over time
- **Comprehensive preprocessing** of diverse CSV formats
- **Publication-ready visualizations** for thesis presentations

This enhancement makes your thesis significantly stronger compared to projects using synthetic data!

### ⚡ Quick Start Guide
**See: [OFFICIAL_RECORDS_QUICK_START.md](OFFICIAL_RECORDS_QUICK_START.md)** for training with real flood data!

**See: [RANDOM_FOREST_THESIS_READY.md](RANDOM_FOREST_THESIS_READY.md)** for complete thesis preparation guide!

---

## 🆕 Latest Enhancements

### ✨ New Features for Thesis Defense

1. **Official Flood Records Training** ⭐ NEW!
   - Use 3,700+ real flood events from Parañaque City (2022-2025)
   - Progressive training shows model evolution
   - Automated preprocessing of official CSVs
   - Year-specific training options
2. **Enhanced Training Script** - Hyperparameter tuning with GridSearchCV
3. **Thesis Report Generator** - Publication-ready visualizations (300 DPI)
4. **Dataset Merger Tool** - Combine multiple CSV files easily
5. **Model Comparison** - Compare performance across versions
6. **Automatic Versioning** - Track all model improvements
7. **Comprehensive Documentation** - Complete guides and references

### 📈 Model Evolution Visualization

Track your model's improvement over time with:
- **Metrics Evolution Charts** - See accuracy, precision, recall improvement
- **Parameters Evolution** - Track hyperparameter changes
- **Feature Importance Comparison** - Understand what drives predictions
## 📋 Your Questions - Answered

### Q1: Can I add new CSV files for training?

**✅ YES! Very Easy!**

```powershell
cd backend
python scripts/train.py --data data/your_new_file.csv
```

Or merge multiple files:

```powershell
python scripts/merge_datasets.py --input "data/*.csv"
python scripts/train.py --data data/merged_dataset.csv
```

### Q2: How does model versioning work?

**✅ Automatic Version Control!**

With Official Records (Progressive Training):
```
Model v1: Trained on 2022 data only          (~100 records)
Model v2: Trained on 2022 + 2023 data       (~270 records)
Model v3: Trained on 2022 + 2023 + 2024     (~1,100 records)
Model v4: Trained on ALL data (2022-2025)   (~3,700 records) ← BEST!
```

With Custom Data:
```
Training #1 → flood_rf_model_v1.joblib + metadata
Training #2 → flood_rf_model_v2.joblib + metadata
Training #3 → flood_rf_model_v3.joblib + metadata
```

Each version stores:
- Model file (.joblib)
- Metadata (.json) with training date, dataset, parameters, metrics, feature importance
---

## 🚀 Best Workflow for Thesis

### **Option A: Train with Official Flood Records** ⭐ RECOMMENDED

Use **real flood data** from Parañaque City (2022-2025):

```powershell
cd backend

# Step 1: Preprocess official records
python scripts/preprocess_official_flood_records.py

# Step 2: Progressive training (shows model evolution)
python scripts/progressive_train.py --grid-search --cv-folds 10

# Step 3: Generate thesis materials
python scripts/generate_thesis_report.py
python scripts/compare_models.py

# Step 4: Validate
python scripts/validate_model.py
```

**What you get:**
- ✅ 4 models trained on real data (v1, v2, v3, v4)
- ✅ 3,700+ real flood events from official records
- ✅ Model evolution showing improvement over time
- ✅ Publication-ready charts and reports

### **Option B: Complete Custom Training Pipeline**

Use your own CSV files:

```powershell
cd backend

# Step 1: Merge all datasets
python scripts/merge_datasets.py --input "data/*.csv"

# Step 2: Train optimal model (with hyperparameter tuning)
python scripts/train.py --data data/merged_dataset.csv --grid-search --cv-folds 10

# Step 3: Generate thesis presentation materials
python scripts/generate_thesis_report.py

# Step 4: Compare model versions
python scripts/compare_models.py

# Step 5: Validate
python scripts/validate_model.py
```

**In `reports/` folder (publication quality):**
- Feature importance chart
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Metrics comparison
- Learning curves
- Comprehensive text report
- Version comparison charts

**All ready for PowerPoint and thesis document!**
---

## 📊 System Architecture

```
Official Flood Records (2022-2025) → Preprocessing → Progressive Training → Random Forest Models
                                           ↓              (v1, v2, v3, v4)
                                    Custom CSV Files → Data Merger → Training Script  
                                          ↓
                                   Model Versions
                                   (v1, v2, v3, v4...)
                                          ↓
                                    Flask API
                                          ↓
                              3-Level Risk Classification
                              (Safe / Alert / Critical)
                                          ↓
                                  Alert Delivery
                                  (SMS / Email)
```

------

## 🎯 Random Forest Model Features

### Why Random Forest?

- ✅ **Ensemble Learning** - Multiple decision trees voting together
- ✅ **Robust** - Less prone to overfitting
- ✅ **Feature Importance** - Shows which weather factors matter most
- ✅ **No Scaling Needed** - Works with raw weather data
- ✅ **Interpretable** - Easy to explain
- ✅ **Industry Standard** - Widely used in production

### Model Capabilities

- **Hyperparameter Tuning** - Automatic optimization with GridSearchCV
- **Cross-Validation** - Robust k-fold validation
- **Multi-Dataset Training** - Merge multiple CSV files
- **Automatic Versioning** - Track improvements over time
- **Comprehensive Metrics** - Accuracy, Precision, Recall, F1, ROC-AUC
- **Feature Importance Analysis** - Understand model decisions

---

## 💻 Quick Commands

### Training

```powershell
# Basic training
python scripts/train.py

# With new dataset
python scripts/train.py --data data/my_data.csv

# With hyperparameter tuning (RECOMMENDED)
python scripts/train.py --grid-search --cv-folds 10

# Merge multiple datasets during training
python scripts/train.py --data "data/*.csv" --merge-datasets

# Progressive training with official records (RECOMMENDED for thesis)
python scripts/progressive_train.py --grid-search --cv-folds 10

# Year-specific training
python scripts/progressive_train.py --year-specific
```

### Analysis

```powershell
# Generate thesis report
python scripts/generate_thesis_report.py

# Compare model versions
python scripts/compare_models.py

# Merge datasets
python scripts/merge_datasets.py

# Preprocess official flood records
python scripts/preprocess_official_flood_records.py
```

### API

```powershell
# Start server
python main.py

# Test prediction
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"temperature\": 25.0, \"humidity\": 80.0, \"precipitation\": 15.0}"

# List models
curl http://localhost:5000/api/models
```
## 📚 Documentation

### Quick References

- **[RANDOM_FOREST_THESIS_READY.md](RANDOM_FOREST_THESIS_READY.md)** - Quick start for thesis
- **[OFFICIAL_RECORDS_QUICK_START.md](OFFICIAL_RECORDS_QUICK_START.md)** - Training with official flood records
- **[backend/docs/QUICK_REFERENCE.md](backend/docs/QUICK_REFERENCE.md)** - Command cheat sheet
- **[backend/docs/THESIS_GUIDE.md](backend/docs/THESIS_GUIDE.md)** - Complete thesis guide

### Detailed Guides

- **[backend/docs/OFFICIAL_FLOOD_RECORDS_GUIDE.md](backend/docs/OFFICIAL_FLOOD_RECORDS_GUIDE.md)** - Complete guide for official records training
- **[backend/docs/IMPROVEMENTS_SUMMARY.md](backend/docs/IMPROVEMENTS_SUMMARY.md)** - All improvements explained
- **[backend/docs/SYSTEM_OVERVIEW.md](backend/docs/SYSTEM_OVERVIEW.md)** - System architecture
- **[backend/docs/MODEL_MANAGEMENT.md](backend/docs/MODEL_MANAGEMENT.md)** - Model versioning
- **[backend/docs/BACKEND_COMPLETE.md](backend/docs/BACKEND_COMPLETE.md)** - Full documentation

---
## 🎓 For Thesis Defense

### Key Talking Points

**About Random Forest:**
- Ensemble of 200 decision trees
- Each tree votes on prediction
- Majority decision wins
- Feature importance shows which factors matter most

**About Your System:**
- Automatic model versioning
- Easy dataset integration
- Hyperparameter optimization
- 3-level risk classification (Safe/Alert/Critical)
- Real-time predictions via API
- **Progressive training with 3,700+ real flood events**
- **Model evolution demonstrating improvement over time**

### Presentation Materials

Generated automatically in `reports/`:
- ✅ Feature importance (which weather factors matter)
- ✅ Confusion matrix (prediction accuracy)
- ✅ ROC curve (model performance)
- ✅ Learning curves (no overfitting proof)
- ✅ Metrics evolution (improvement over time)
- ✅ Parameters evolution (hyperparameter changes)
- ✅ Model comparison charts (side-by-side performance)
---

## 🔧 Installation

### Requirements

- Python 3.8+
- pip

### Setup

```powershell
# Clone repository
git clone https://github.com/KyaRhamil/floodingnaque.git
cd floodingnaque/backend

# Install dependencies
pip install -r requirements.txt

# Train model
python scripts/train.py

# Start API
python main.py
```

---

## 📊 Sample Results

### Expected Performance

**With grid search optimization:**
- Accuracy: 95%+
- Precision: 95%+
- Recall: 95%+
- F1 Score: 95%+
- ROC-AUC: 0.98+

### Feature Importance (Example)

- Precipitation: 45%
- Humidity: 30%
- Temperature: 20%
- Wind Speed: 5%

------

## 🌟 Key Features

### Data Management
- ✅ Easy CSV integration
- ✅ Multi-dataset merging
- ✅ Duplicate removal
- ✅ Column validation
- ✅ **Official flood records preprocessing (2022-2025)**

### Model Training
- ✅ Random Forest Classifier
- ✅ Hyperparameter tuning (GridSearchCV)
- ✅ Cross-validation (k-fold)
- ✅ Automatic versioning
- ✅ **Progressive training with model evolution**
- ✅ **Year-specific training options**

### Evaluation
- ✅ Comprehensive metrics
- ✅ Publication-quality charts
- ✅ Feature importance analysis
- ✅ Model comparison tools
- ✅ **Metrics evolution visualization**

### Deployment
- ✅ Flask REST API
- ✅ 3-level risk classification
- ✅ Real-time predictions
- ✅ Alert delivery system
## 📞 Support

For detailed instructions, see the documentation in `backend/docs/`.

---

## 📄 License

See [LICENSE](LICENSE) file for details.

---

## 🎉 Ready for Thesis Defense!

Your Random Forest flood prediction model is now fully equipped with:
- ✅ Hyperparameter optimization
- ✅ Publication-ready visualizations
- ✅ Model versioning and comparison
- ✅ Comprehensive documentation
- ✅ Easy dataset integration
- ✅ **3,700+ real flood events from official records**
- ✅ **Progressive training showing model evolution**
- ✅ **Professional ML development practices****Good luck with your thesis defense! 🚀🎓**