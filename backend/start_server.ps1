# Floodingnaque Backend Startup Script
# This script activates the virtual environment and starts the server

Write-Host "Starting Floodingnaque Backend..." -ForegroundColor Green

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ..\venv\Scripts\Activate.ps1

# Check if activation was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "Virtual environment activated successfully" -ForegroundColor Green
} else {
    Write-Host "Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}

# Start the server
Write-Host "Starting Flask application..." -ForegroundColor Yellow
python main.py
