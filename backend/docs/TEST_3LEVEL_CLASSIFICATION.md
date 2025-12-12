# Testing 3-Level Risk Classification

## Quick Test

### Test Safe Level (Low Risk)
```powershell
$body = @{
    temperature = 298.15
    humidity = 50.0
    precipitation = 2.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -ContentType "application/json" -Body $body
```

**Expected Response:**
```json
{
  "prediction": 0,
  "flood_risk": "low",
  "risk_level": 0,
  "risk_label": "Safe",
  "risk_color": "#28a745",
  "risk_description": "No immediate flood risk. Normal weather conditions.",
  "confidence": 0.95
}
```

### Test Alert Level (Moderate Risk)
```powershell
$body = @{
    temperature = 298.15
    humidity = 85.0
    precipitation = 15.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -ContentType "application/json" -Body $body
```

**Expected Response:**
```json
{
  "prediction": 0,
  "flood_risk": "low",
  "risk_level": 1,
  "risk_label": "Alert",
  "risk_color": "#ffc107",
  "risk_description": "Moderate flood risk. Monitor conditions closely. Prepare for possible flooding.",
  "confidence": 0.65
}
```

### Test Critical Level (High Risk)
```powershell
$body = @{
    temperature = 298.15
    humidity = 90.0
    precipitation = 50.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -ContentType "application/json" -Body $body
```

**Expected Response:**
```json
{
  "prediction": 1,
  "flood_risk": "high",
  "risk_level": 2,
  "risk_label": "Critical",
  "risk_color": "#dc3545",
  "risk_description": "High flood risk. Immediate action required. Evacuate if necessary.",
  "confidence": 0.92
}
```

## Risk Level Thresholds

- **Safe (0)**: Flood probability < 30%, Precipitation < 10mm
- **Alert (1)**: Flood probability 30-75%, OR Precipitation 10-30mm, OR High humidity (>85%) with precipitation
- **Critical (2)**: Flood probability ≥ 75%, OR Binary prediction = 1 with high confidence

## Testing with Probabilities

```powershell
$body = @{
    temperature = 298.15
    humidity = 90.0
    precipitation = 50.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/predict?return_proba=true" -Method POST -ContentType "application/json" -Body $body
```

This will include probability breakdown:
```json
{
  "probability": {
    "no_flood": 0.08,
    "flood": 0.92
  },
  "risk_level": 2,
  "risk_label": "Critical",
  ...
}
```

