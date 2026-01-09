# Floodingnaque Model Training Pipeline
# Automated training script for all models
# Estimated time: 2-3 hours

param(
    [switch]$Quick,          # Quick training (30 min)
    [switch]$Full,           # Full training (2-3 hours) - DEFAULT
    [switch]$SkipMultiLevel, # Skip multi-level classifier
    [switch]$SkipValidation  # Skip final validation
)

$ErrorActionPreference = "Stop"
$StartTime = Get-Date

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "FLOODINGNAQUE MODEL TRAINING PIPELINE" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if in backend directory
if (-not (Test-Path "scripts/train.py")) {
    Write-Host "ERROR: Must run from backend directory!" -ForegroundColor Red
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
    Write-Host "Expected: d:\floodingnaque\backend" -ForegroundColor Yellow
    exit 1
}

# Check Python installation
Write-Host "[CHECK] Verifying Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found! Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Check required packages
Write-Host "[CHECK] Verifying required packages..." -ForegroundColor Yellow
$requiredPackages = @("pandas", "numpy", "scikit-learn", "joblib")
foreach ($package in $requiredPackages) {
    python -c "import $package" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Package '$package' not found!" -ForegroundColor Red
        Write-Host "Run: pip install -r requirements.txt" -ForegroundColor Yellow
        exit 1
    }
}
Write-Host "✓ All required packages found" -ForegroundColor Green

# Check data availability
Write-Host "[CHECK] Verifying training data..." -ForegroundColor Yellow
$dataFile = "data/processed/cumulative_up_to_2025.csv"
if (-not (Test-Path $dataFile)) {
    Write-Host "✗ Training data not found: $dataFile" -ForegroundColor Red
    Write-Host "Run preprocessing script first:" -ForegroundColor Yellow
    Write-Host "  python scripts/preprocess_official_flood_records.py" -ForegroundColor Cyan
    exit 1
}
$dataSize = (Get-Item $dataFile).Length / 1KB
Write-Host "✓ Training data found: $dataFile ($([math]::Round($dataSize, 1)) KB)" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

# Determine training mode
if ($Quick) {
    Write-Host "MODE: Quick Training (30 minutes)" -ForegroundColor Magenta
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    # Quick training
    Write-Host "[STEP 1/3] Training basic model..." -ForegroundColor Yellow
    python scripts/train.py --data $dataFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Training failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Basic model trained" -ForegroundColor Green
    
    Write-Host "[STEP 2/3] Validating model..." -ForegroundColor Yellow
    python scripts/validate_model.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠ Validation warnings (non-critical)" -ForegroundColor Yellow
    } else {
        Write-Host "✓ Model validated" -ForegroundColor Green
    }
    
    Write-Host "[STEP 3/3] Quick evaluation..." -ForegroundColor Yellow
    python scripts/evaluate_model.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠ Evaluation warnings" -ForegroundColor Yellow
    } else {
        Write-Host "✓ Evaluation complete" -ForegroundColor Green
    }
    
} else {
    Write-Host "MODE: Full Training Pipeline (2-3 hours)" -ForegroundColor Magenta
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    $step = 1
    $totalSteps = 6
    if ($SkipMultiLevel) { $totalSteps-- }
    if ($SkipValidation) { $totalSteps-- }
    
    # Step 1: Progressive Training (MOST IMPORTANT)
    Write-Host "[STEP $step/$totalSteps] Progressive Training (2022 → 2025)" -ForegroundColor Yellow
    Write-Host "  This will train 4 models showing evolution over time" -ForegroundColor Gray
    Write-Host "  Estimated time: 45 minutes" -ForegroundColor Gray
    Write-Host "  Starting: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    $step1Start = Get-Date
    
    python scripts/progressive_train.py --grid-search --cv-folds 10
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Progressive training failed!" -ForegroundColor Red
        Write-Host "Check the error messages above" -ForegroundColor Yellow
        exit 1
    }
    
    $step1Duration = [math]::Round(((Get-Date) - $step1Start).TotalMinutes, 1)
    Write-Host "✓ Progressive training complete ($step1Duration min)" -ForegroundColor Green
    Write-Host ""
    $step++
    
    # Step 2: Production Model
    Write-Host "[STEP $step/$totalSteps] Production Model Training" -ForegroundColor Yellow
    Write-Host "  Training production-ready model with full validation" -ForegroundColor Gray
    Write-Host "  Estimated time: 45 minutes" -ForegroundColor Gray
    Write-Host "  Starting: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    $step2Start = Get-Date
    
    python scripts/train_production.py --production
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Production training failed!" -ForegroundColor Red
        Write-Host "Check the error messages above" -ForegroundColor Yellow
        exit 1
    }
    
    $step2Duration = [math]::Round(((Get-Date) - $step2Start).TotalMinutes, 1)
    Write-Host "✓ Production model complete ($step2Duration min)" -ForegroundColor Green
    Write-Host ""
    $step++
    
    # Step 3: Multi-Level Model (Optional)
    if (-not $SkipMultiLevel) {
        Write-Host "[STEP $step/$totalSteps] Multi-Level Risk Classifier" -ForegroundColor Yellow
        Write-Host "  Training 3-level classifier (LOW/MODERATE/HIGH)" -ForegroundColor Gray
        Write-Host "  Estimated time: 20 minutes" -ForegroundColor Gray
        Write-Host "  Starting: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
        $step3Start = Get-Date
        
        python scripts/train_enhanced.py --multi-level --randomized-search
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "⚠ Multi-level training failed (non-critical)" -ForegroundColor Yellow
        } else {
            $step3Duration = [math]::Round(((Get-Date) - $step3Start).TotalMinutes, 1)
            Write-Host "✓ Multi-level model complete ($step3Duration min)" -ForegroundColor Green
        }
        Write-Host ""
        $step++
    }
    
    # Step 4: Generate Comparisons
    Write-Host "[STEP $step/$totalSteps] Generating Model Comparisons" -ForegroundColor Yellow
    Write-Host "  Creating comparison charts and reports" -ForegroundColor Gray
    Write-Host "  Estimated time: 1 minute" -ForegroundColor Gray
    
    New-Item -ItemType Directory -Force -Path "reports/thesis_comparison" | Out-Null
    python scripts/compare_models.py --output reports/thesis_comparison
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠ Comparison generation failed (non-critical)" -ForegroundColor Yellow
    } else {
        Write-Host "✓ Comparison charts generated" -ForegroundColor Green
    }
    Write-Host ""
    $step++
    
    # Step 5: Robustness Evaluation
    Write-Host "[STEP $step/$totalSteps] Robustness Evaluation" -ForegroundColor Yellow
    Write-Host "  Testing model stability and generalization" -ForegroundColor Gray
    Write-Host "  Estimated time: 5 minutes" -ForegroundColor Gray
    
    python scripts/evaluate_robustness.py --output thesis_robustness.json
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠ Robustness evaluation failed (non-critical)" -ForegroundColor Yellow
    } else {
        Write-Host "✓ Robustness evaluation complete" -ForegroundColor Green
    }
    Write-Host ""
    $step++
    
    # Step 6: Final Validation
    if (-not $SkipValidation) {
        Write-Host "[STEP $step/$totalSteps] Final Model Validation" -ForegroundColor Yellow
        Write-Host "  Validating production model" -ForegroundColor Gray
        
        python scripts/validate_model.py --json | Out-File "reports/validation_results.json"
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "⚠ Validation warnings (check validation_results.json)" -ForegroundColor Yellow
        } else {
            Write-Host "✓ Model validation passed" -ForegroundColor Green
        }
        Write-Host ""
    }
}

# Training complete
$TotalDuration = [math]::Round(((Get-Date) - $StartTime).TotalMinutes, 1)

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✓ TRAINING PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Total time: $TotalDuration minutes" -ForegroundColor Cyan
Write-Host "Started:    $($StartTime.ToString('HH:mm:ss'))" -ForegroundColor Gray
Write-Host "Completed:  $((Get-Date).ToString('HH:mm:ss'))" -ForegroundColor Gray
Write-Host ""

# List generated files
Write-Host "Generated Files:" -ForegroundColor Cyan
Write-Host "----------------" -ForegroundColor Cyan

Write-Host ""
Write-Host "Models:" -ForegroundColor Yellow
if (Test-Path "models") {
    Get-ChildItem "models" -Filter "*.joblib" | ForEach-Object {
        $size = [math]::Round($_.Length / 1MB, 2)
        Write-Host "  ✓ $($_.Name) ($size MB)" -ForegroundColor Green
    }
} else {
    Write-Host "  ⚠ No models found" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Reports:" -ForegroundColor Yellow
if (Test-Path "reports") {
    # Check for comparison charts
    if (Test-Path "reports/thesis_comparison") {
        Write-Host "  ✓ reports/thesis_comparison/" -ForegroundColor Green
        Get-ChildItem "reports/thesis_comparison" -Filter "*.png" | ForEach-Object {
            Write-Host "    - $($_.Name)" -ForegroundColor Gray
        }
    }
    
    # Check for other reports
    Get-ChildItem "reports" -Filter "*.json" | ForEach-Object {
        Write-Host "  ✓ $($_.Name)" -ForegroundColor Green
    }
    
    Get-ChildItem "reports" -Filter "*.png" | ForEach-Object {
        Write-Host "  ✓ $($_.Name)" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Review model performance:" -ForegroundColor White
Write-Host "   - Open reports/thesis_comparison/metrics_evolution.png" -ForegroundColor Gray
Write-Host "   - Review reports/thesis_comparison/comparison_report.txt" -ForegroundColor Gray
Write-Host "   - Check models/flood_rf_model_v*.json for metrics" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Test the API:" -ForegroundColor White
Write-Host "   python main.py" -ForegroundColor Cyan
Write-Host "   # In another terminal:" -ForegroundColor Gray
Write-Host "   curl -X POST http://localhost:5000/predict \" -ForegroundColor Cyan
Write-Host "     -H 'Content-Type: application/json' \" -ForegroundColor Cyan
Write-Host "     -d '{\"temperature\": 298.15, \"humidity\": 80, \"precipitation\": 35}'" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Prepare thesis defense:" -ForegroundColor White
Write-Host "   - Use charts from reports/thesis_comparison/" -ForegroundColor Gray
Write-Host "   - Review robustness results in reports/thesis_robustness.json" -ForegroundColor Gray
Write-Host "   - Prepare explanations for high accuracy" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Backup your models:" -ForegroundColor White
Write-Host "   Copy-Item models\ -Destination models_backup_$(Get-Date -Format 'yyyyMMdd') -Recurse" -ForegroundColor Cyan
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Training Complete! 🎉" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
