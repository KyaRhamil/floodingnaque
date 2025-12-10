# Model Management Guide

This guide explains how to use the enhanced model training, versioning, validation, and evaluation features.

## Table of Contents

1. [Training Models](#training-models)
2. [Model Versioning](#model-versioning)
3. [Model Validation](#model-validation)
4. [Model Metadata](#model-metadata)
5. [Evaluation Metrics](#evaluation-metrics)
6. [API Integration](#api-integration)

## Training Models

### Basic Training

Train a new model with automatic versioning:

```bash
cd backend
python train.py
```

This will:
- Load data from `data/synthetic_dataset.csv`
- Train a Random Forest classifier
- Automatically assign the next version number
- Save the model as `models/flood_rf_model_v{N}.joblib`
- Also save as `models/flood_rf_model.joblib` (latest)
- Generate comprehensive evaluation metrics
- Create metadata JSON file

### Training with Custom Options

```bash
# Specify a version number
python train.py --version 5

# Use a different data file
python train.py --data data/custom_dataset.csv

# Specify models directory
python train.py --models-dir models/production
```

### Training Output

The training script generates:

1. **Model File**: `flood_rf_model_v{N}.joblib` - The trained model
2. **Latest Model**: `flood_rf_model.joblib` - Symlink to latest version
3. **Metadata File**: `flood_rf_model_v{N}.json` - Model metadata and metrics

## Model Versioning

### Version Numbering

Models are automatically versioned:
- First model: `flood_rf_model_v1.joblib`
- Second model: `flood_rf_model_v2.joblib`
- And so on...

The latest model is always saved as `flood_rf_model.joblib` for backward compatibility.

### Listing Available Models

Via API:
```bash
curl http://localhost:5000/api/models
```

Response:
```json
{
  "models": [
    {
      "version": 3,
      "path": "models/flood_rf_model_v3.joblib",
      "is_current": true,
      "created_at": "2025-01-15T10:30:00",
      "metrics": {
        "accuracy": 0.95,
        "precision": 0.94,
        "recall": 0.96,
        "f1_score": 0.95
      }
    }
  ],
  "current_version": 3,
  "total_versions": 3
}
```

### Using Specific Model Versions

In Python:
```python
from predict import load_model_version, predict_flood

# Load specific version
model = load_model_version(version=2)

# Make prediction with specific version
result = predict_flood(
    {'temperature': 298.15, 'humidity': 65.0, 'precipitation': 5.0},
    model_version=2
)
```

Via API:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 298.15,
    "humidity": 65.0,
    "precipitation": 5.0,
    "model_version": 2
  }'
```

## Model Validation

### Running Validation

Validate the latest model:
```bash
python validate_model.py
```

Validate a specific model version:
```bash
python validate_model.py --model models/flood_rf_model_v2.joblib
```

Validate with custom test data:
```bash
python validate_model.py --data data/test_dataset.csv
```

Get JSON output:
```bash
python validate_model.py --json
```

### Validation Checks

The validation script performs:

1. **Model Integrity Check**
   - Verifies model file exists
   - Tests model loading
   - Validates model type

2. **Metadata Check**
   - Verifies metadata file exists
   - Validates metadata structure

3. **Feature Validation**
   - Checks expected features match
   - Validates feature names

4. **Prediction Test**
   - Tests predictions with sample data
   - Verifies prediction format

5. **Performance Evaluation**
   - Calculates metrics on test data
   - Compares with training metrics

### Validation Output

```
============================================================
MODEL VALIDATION
============================================================

[1/4] Model Integrity Check
✓ Model loaded successfully
✓ Model type: RandomForestClassifier

[2/4] Metadata Check
✓ Metadata file found
  Version: 3
  Created: 2025-01-15T10:30:00
  Accuracy: 0.95

[3/4] Feature Validation
✓ Features match: ['temperature', 'humidity', 'precipitation']

[4/4] Prediction Test
  Test 1: {'temperature': 298.15, 'humidity': 65.0, 'precipitation': 0.0} -> Prediction: 0
  Test 2: {'temperature': 298.15, 'humidity': 90.0, 'precipitation': 50.0} -> Prediction: 1
  Test 3: {'temperature': 293.15, 'humidity': 50.0, 'precipitation': 5.0} -> Prediction: 0
✓ All test predictions successful

[5/5] Performance Evaluation
Performance Metrics:
  Accuracy:  0.9500
  Precision: 0.9400
  Recall:    0.9600
  F1 Score:  0.9500
  ROC-AUC:   0.9800

============================================================
✓ MODEL VALIDATION PASSED
============================================================
```

## Model Metadata

### Metadata Structure

Each model has a corresponding JSON metadata file:

```json
{
  "version": 3,
  "model_type": "RandomForestClassifier",
  "model_path": "models/flood_rf_model_v3.joblib",
  "created_at": "2025-01-15T10:30:00",
  "training_data": {
    "file": "data/synthetic_dataset.csv",
    "shape": [1000, 4],
    "features": ["temperature", "humidity", "precipitation"],
    "target_distribution": {
      "0": 600,
      "1": 400
    }
  },
  "model_parameters": {
    "n_estimators": 100,
    "random_state": 42,
    "max_depth": null,
    "min_samples_split": 2,
    "min_samples_leaf": 1
  },
  "metrics": {
    "accuracy": 0.95,
    "precision": 0.94,
    "recall": 0.96,
    "f1_score": 0.95,
    "roc_auc": 0.98,
    "precision_per_class": {
      "0": 0.96,
      "1": 0.92
    },
    "recall_per_class": {
      "0": 0.98,
      "1": 0.90
    },
    "f1_per_class": {
      "0": 0.97,
      "1": 0.91
    },
    "confusion_matrix": [[196, 4], [8, 192]]
  },
  "feature_importance": {
    "temperature": 0.25,
    "humidity": 0.30,
    "precipitation": 0.45
  }
}
```

### Accessing Metadata

In Python:
```python
from predict import get_model_metadata

metadata = get_model_metadata('models/flood_rf_model_v3.joblib')
print(f"Version: {metadata['version']}")
print(f"Accuracy: {metadata['metrics']['accuracy']}")
```

Via API:
```bash
# Check status (includes model info)
curl http://localhost:5000/status

# Detailed health check (includes full model info)
curl http://localhost:5000/health
```

## Evaluation Metrics

### Available Metrics

The training script calculates comprehensive metrics:

1. **Accuracy**: Overall prediction accuracy
2. **Precision**: Weighted and per-class precision
3. **Recall**: Weighted and per-class recall
4. **F1 Score**: Weighted and per-class F1 score
5. **ROC-AUC**: Area under ROC curve (if applicable)
6. **Confusion Matrix**: True/False positives and negatives

### Metric Interpretation

- **Accuracy**: Overall correctness (0-1, higher is better)
- **Precision**: Of predicted floods, how many were correct (0-1, higher is better)
- **Recall**: Of actual floods, how many were detected (0-1, higher is better)
- **F1 Score**: Harmonic mean of precision and recall (0-1, higher is better)
- **ROC-AUC**: Model's ability to distinguish classes (0-1, higher is better)

### Viewing Metrics

During training, metrics are logged to console:
```
==================================================
MODEL EVALUATION METRICS
==================================================
Accuracy:  0.9500
Precision: 0.9400
Recall:    0.9600
F1 Score:  0.9500
ROC-AUC:   0.9800

Per-class Metrics:
  Class 0:
    Precision: 0.9600
    Recall:    0.9800
    F1:        0.9700
  Class 1:
    Precision: 0.9200
    Recall:    0.9000
    F1:        0.9100
```

Metrics are also saved in the metadata JSON file.

## API Integration

### Status Endpoints

**Basic Status** (`/status`):
```bash
curl http://localhost:5000/status
```

Response:
```json
{
  "status": "running",
  "database": "connected",
  "model": "loaded",
  "model_version": 3,
  "model_accuracy": 0.95
}
```

**Detailed Health** (`/health`):
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "database": "connected",
  "model_available": true,
  "scheduler_running": true,
  "model": {
    "loaded": true,
    "type": "RandomForestClassifier",
    "path": "models/flood_rf_model.joblib",
    "features": ["temperature", "humidity", "precipitation"],
    "version": 3,
    "created_at": "2025-01-15T10:30:00",
    "metrics": {
      "accuracy": 0.95,
      "precision": 0.94,
      "recall": 0.96,
      "f1_score": 0.95
    }
  }
}
```

### Prediction with Probabilities

Get prediction probabilities:
```bash
curl -X POST "http://localhost:5000/predict?return_proba=true" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 298.15,
    "humidity": 90.0,
    "precipitation": 50.0
  }'
```

Response:
```json
{
  "prediction": 1,
  "flood_risk": "high",
  "probability": {
    "no_flood": 0.15,
    "flood": 0.85
  },
  "model_version": 3,
  "request_id": "uuid-string"
}
```

### List Models

```bash
curl http://localhost:5000/api/models
```

## Best Practices

1. **Version Control**: Always train new models with versioning enabled
2. **Validation**: Run validation after training new models
3. **Metadata**: Review metadata before deploying models
4. **Metrics**: Compare metrics across model versions
5. **Testing**: Test predictions with known cases
6. **Backup**: Keep old model versions for rollback

## Troubleshooting

### Model Not Found

If you get "Model file not found":
1. Train a model: `python train.py`
2. Check models directory: `ls models/`
3. Verify model path in code

### Version Conflicts

If version numbering is incorrect:
- Delete old metadata files
- Retrain with explicit version: `python train.py --version 1`

### Validation Failures

If validation fails:
1. Check model file integrity
2. Verify feature names match
3. Ensure test data format is correct
4. Review error messages in validation output

## Examples

### Complete Workflow

```bash
# 1. Train a new model
python train.py

# 2. Validate the model
python validate_model.py

# 3. Check model info via API
curl http://localhost:5000/health

# 4. Test prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"temperature": 298.15, "humidity": 65.0, "precipitation": 5.0}'

# 5. List all models
curl http://localhost:5000/api/models
```

### Comparing Model Versions

```python
from predict import list_available_models, get_model_metadata

models = list_available_models()
for model in models:
    metadata = get_model_metadata(model['path'])
    print(f"Version {model['version']}: Accuracy = {metadata['metrics']['accuracy']}")
```

