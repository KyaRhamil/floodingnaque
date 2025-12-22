# PowerShell API Examples

This guide provides PowerShell syntax for testing the Floodingnaque API.

## PowerShell vs Bash curl

In PowerShell, `curl` is an alias for `Invoke-WebRequest`. For REST API calls, use `Invoke-RestMethod` instead, which automatically parses JSON responses.

## Basic Examples

### Health Check

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://localhost:5000/health" -Method GET
```

**Or using curl alias:**
```powershell
curl http://localhost:5000/health
```

### Status Check

```powershell
Invoke-RestMethod -Uri "http://localhost:5000/status" -Method GET
```

### List Models

```powershell
Invoke-RestMethod -Uri "http://localhost:5000/api/models" -Method GET
```

## POST Requests

### Predict Flood Risk (Basic)

```powershell
$body = @{
    temperature = 298.15
    humidity = 90.0
    precipitation = 50.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -ContentType "application/json" -Body $body
```

### Predict with Probabilities

```powershell
$body = @{
    temperature = 298.15
    humidity = 90.0
    precipitation = 50.0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/predict?return_proba=true" -Method POST -ContentType "application/json" -Body $body
```

### Predict with Specific Model Version

```powershell
$body = @{
    temperature = 298.15
    humidity = 90.0
    precipitation = 50.0
    model_version = 1
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -ContentType "application/json" -Body $body
```

### Ingest Weather Data

```powershell
$body = @{
    lat = 14.6
    lon = 120.98
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/ingest" -Method POST -ContentType "application/json" -Body $body
```

## Using Variables

### Store Response in Variable

```powershell
$response = Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -ContentType "application/json" -Body $body
$response.prediction
$response.flood_risk
```

### Pretty Print JSON Response

```powershell
$response = Invoke-RestMethod -Uri "http://localhost:5000/health" -Method GET
$response | ConvertTo-Json -Depth 10
```

## Error Handling

### Try-Catch Block

```powershell
try {
    $body = @{
        temperature = 298.15
        humidity = 90.0
        precipitation = 50.0
    } | ConvertTo-Json
    
    $response = Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -ContentType "application/json" -Body $body
    Write-Host "Prediction: $($response.prediction)"
    Write-Host "Flood Risk: $($response.flood_risk)"
} catch {
    Write-Host "Error: $($_.Exception.Message)"
    if ($_.ErrorDetails.Message) {
        Write-Host "Details: $($_.ErrorDetails.Message)"
    }
}
```

## Advanced Examples

### Get Historical Data with Pagination

```powershell
$params = @{
    limit = 10
    offset = 0
}
$queryString = ($params.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join '&'
Invoke-RestMethod -Uri "http://localhost:5000/data?$queryString" -Method GET
```

### Get Data with Date Range

```powershell
$startDate = "2025-01-01T00:00:00"
$endDate = "2025-12-31T23:59:59"
Invoke-RestMethod -Uri "http://localhost:5000/data?start_date=$startDate&end_date=$endDate&limit=100" -Method GET
```

## Using curl.exe (Windows)

If you prefer the Unix-style curl syntax, use `curl.exe` explicitly:

```powershell
curl.exe -X POST "http://localhost:5000/predict" `
  -H "Content-Type: application/json" `
  -d '{\"temperature\": 298.15, \"humidity\": 90.0, \"precipitation\": 50.0}'
```

Note: In PowerShell, you need to escape quotes in JSON strings.

## Quick Reference

| Task | PowerShell Command |
|------|-------------------|
| Health check | `Invoke-RestMethod -Uri "http://localhost:5000/health"` |
| Predict | `$body = @{temp=298.15; humidity=90.0; precip=50.0} \| ConvertTo-Json; Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -ContentType "application/json" -Body $body` |
| List models | `Invoke-RestMethod -Uri "http://localhost:5000/api/models"` |
| Ingest data | `$body = @{lat=14.6; lon=120.98} \| ConvertTo-Json; Invoke-RestMethod -Uri "http://localhost:5000/ingest" -Method POST -ContentType "application/json" -Body $body` |

## Troubleshooting

### "No input data provided" Error

This usually means the JSON wasn't parsed correctly. Try:

```powershell
# Explicitly convert to JSON
$body = @{
    temperature = 298.15
    humidity = 90.0
    precipitation = 50.0
}
$jsonBody = $body | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method POST -ContentType "application/json" -Body $jsonBody
```

### Connection Refused

Make sure the Flask server is running:
```powershell
cd backend
python main.py
```

### Check Server Status

```powershell
try {
    Invoke-RestMethod -Uri "http://localhost:5000/status"
    Write-Host "Server is running"
} catch {
    Write-Host "Server is not responding"
}
```

