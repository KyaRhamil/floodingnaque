# Floodingnaque Model Training Pipeline
# Automated training script for all models
# Estimated time: 2-3 hours

param(
    [switch]$Quick,          # Quick training (30 min)
    [switch]$Full,           # Full training (2-3 hours) - DEFAULT
    [switch]$PAGASA,         # Use PAGASA weather data (recommended)
    [switch]$Ultimate,       # Ultimate progressive training (v1 → v2 → ... → ULTIMATE)
    [switch]$Progressive,    # Same as Ultimate - train all versions progressively
    [switch]$LatestOnly,     # Only train the latest/best model
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

# Check data availability - Support both PAGASA and legacy data
Write-Host "[CHECK] Verifying training data..." -ForegroundColor Yellow

if ($PAGASA) {
    # Check for PAGASA weather data files
    $pagasaFiles = @(
        "data/Floodingnaque_CADS-S0126006_NAIA Daily Data.csv",
        "data/Floodingnaque_CADS-S0126006_Port Area Daily Data.csv",
        "data/Floodingnaque_CADS-S0126006_Science Garden Daily Data.csv"
    )
    $pagasaFound = $true
    foreach ($file in $pagasaFiles) {
        if (-not (Test-Path $file)) {
            Write-Host "✗ PAGASA data not found: $file" -ForegroundColor Red
            $pagasaFound = $false
        }
    }
    if (-not $pagasaFound) {
        Write-Host "Please ensure PAGASA data files are in the data/ directory" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "✓ PAGASA weather data found (3 stations)" -ForegroundColor Green
} else {
    $dataFile = "data/processed/cumulative_up_to_2025.csv"
    if (-not (Test-Path $dataFile)) {
        Write-Host "✗ Training data not found: $dataFile" -ForegroundColor Red
        Write-Host "Run preprocessing script first:" -ForegroundColor Yellow
        Write-Host "  python scripts/preprocess_official_flood_records.py" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "Or use PAGASA data with: .\run_training_pipeline.ps1 -PAGASA" -ForegroundColor Cyan
        exit 1
    }
    $dataSize = (Get-Item $dataFile).Length / 1KB
    Write-Host "✓ Training data found: $dataFile ($([math]::Round($dataSize, 1)) KB)" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

# PAGASA Training Mode
if ($PAGASA) {
    Write-Host "MODE: PAGASA Weather Data Training" -ForegroundColor Magenta
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "Data Sources:" -ForegroundColor Gray
    Write-Host "  - NAIA Station (closest to Parañaque)" -ForegroundColor Gray
    Write-Host "  - Port Area Station" -ForegroundColor Gray
    Write-Host "  - Science Garden Station" -ForegroundColor Gray
    Write-Host "  - DOST-PAGASA Climate Data (2020-2025)" -ForegroundColor Gray
    Write-Host ""
    
    $step = 1
    $totalSteps = 4
    
    # Step 1: Preprocess PAGASA Data
    Write-Host "[STEP $step/$totalSteps] Preprocessing PAGASA Data" -ForegroundColor Yellow
    Write-Host "  Creating training dataset from weather stations" -ForegroundColor Gray
    Write-Host "  Starting: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
    $step1Start = Get-Date
    
    python scripts/preprocess_pagasa_data.py --create-training
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ PAGASA preprocessing failed!" -ForegroundColor Red
        exit 1
    }
    
    $step1Duration = [math]::Round(((Get-Date) - $step1Start).TotalMinutes, 1)
    Write-Host "✓ PAGASA data preprocessed ($step1Duration min)" -ForegroundColor Green
    Write-Host ""
    $step++
    
    # Step 2: Train Model with PAGASA Data
    Write-Host "[STEP $step/$totalSteps] Training Model with PAGASA Data" -ForegroundColor Yellow
    Write-Host "  Using enhanced features: rolling precipitation, monsoon indicators, etc." -ForegroundColor Gray
    $step2Start = Get-Date
    
    if ($Quick) {
        Write-Host "  Mode: Quick training" -ForegroundColor Gray
        python scripts/train_pagasa.py
    } else {
        Write-Host "  Mode: Full training with grid search" -ForegroundColor Gray
        python scripts/train_pagasa.py --production
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ PAGASA model training failed!" -ForegroundColor Red
        exit 1
    }
    
    $step2Duration = [math]::Round(((Get-Date) - $step2Start).TotalMinutes, 1)
    Write-Host "✓ PAGASA model trained ($step2Duration min)" -ForegroundColor Green
    Write-Host ""
    $step++
    
    # Step 3: Validate Model
    Write-Host "[STEP $step/$totalSteps] Validating Model" -ForegroundColor Yellow
    python scripts/validate_model.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠ Validation warnings (non-critical)" -ForegroundColor Yellow
    } else {
        Write-Host "✓ Model validated" -ForegroundColor Green
    }
    Write-Host ""
    $step++
    
    # Step 4: Model Evaluation
    Write-Host "[STEP $step/$totalSteps] Model Evaluation" -ForegroundColor Yellow
    python scripts/evaluate_model.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠ Evaluation warnings" -ForegroundColor Yellow
    } else {
        Write-Host "✓ Evaluation complete" -ForegroundColor Green
    }
    
    # Summary
    $TotalDuration = [math]::Round(((Get-Date) - $StartTime).TotalMinutes, 1)
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "PAGASA TRAINING COMPLETE!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Total Time: $TotalDuration minutes" -ForegroundColor White
    Write-Host ""
    Write-Host "Model files saved to: models/" -ForegroundColor White
    Write-Host "  - flood_rf_model_latest.joblib" -ForegroundColor Gray
    Write-Host "  - flood_rf_model_v*_pagasa.joblib" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Reports saved to: reports/" -ForegroundColor White
    Write-Host ""
    exit 0
}

# Ultimate/Progressive Training Mode (v1 → v2 → ... → ULTIMATE)
if ($Ultimate -or $Progressive) {
    Write-Host "MODE: ULTIMATE PROGRESSIVE TRAINING" -ForegroundColor Magenta
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Training all model versions progressively:" -ForegroundColor Gray
    Write-Host "  v1: Official Records 2022 (Baseline)" -ForegroundColor Gray
    Write-Host "  v2: Official Records 2022-2023" -ForegroundColor Gray
    Write-Host "  v3: Official Records 2022-2024" -ForegroundColor Gray
    Write-Host "  v4: Official Records 2022-2025" -ForegroundColor Gray
    Write-Host "  v5: PAGASA Weather Data (2020-2025)" -ForegroundColor Gray
    Write-Host "  v6: ULTIMATE - All datasets combined" -ForegroundColor Gray
    Write-Host ""
    Write-Host "This showcases model evolution from baseline to best!" -ForegroundColor Yellow
    Write-Host ""
    
    $step = 1
    $totalSteps = 3
    
    # Step 1: Preprocessing (if needed)
    Write-Host "[STEP $step/$totalSteps] Checking/Creating Required Datasets" -ForegroundColor Yellow
    
    # Check for PAGASA dataset
    if (-not (Test-Path "data/processed/pagasa_training_dataset.csv")) {
        Write-Host "  Creating PAGASA training dataset..." -ForegroundColor Gray
        python scripts/preprocess_pagasa_data.py --create-training
        if ($LASTEXITCODE -ne 0) {
            Write-Host "⚠ PAGASA preprocessing failed (will skip v5)" -ForegroundColor Yellow
        }
    }
    
    # Check for cumulative datasets
    if (-not (Test-Path "data/processed/cumulative_up_to_2025.csv")) {
        Write-Host "  Creating cumulative flood records..." -ForegroundColor Gray
        python scripts/preprocess_official_flood_records.py
        if ($LASTEXITCODE -ne 0) {
            Write-Host "⚠ Flood records preprocessing failed" -ForegroundColor Yellow
        }
    }
    
    Write-Host "✓ Dataset check complete" -ForegroundColor Green
    Write-Host ""
    $step++
    
    # Step 2: Progressive Training
    Write-Host "[STEP $step/$totalSteps] Progressive Model Training (v1 → ULTIMATE)" -ForegroundColor Yellow
    Write-Host "  Training all available model versions..." -ForegroundColor Gray
    $step2Start = Get-Date
    
    if ($Quick) {
        if ($LatestOnly) {
            Write-Host "  Mode: Quick (latest only)" -ForegroundColor Gray
            python scripts/train_ultimate.py --progressive --latest-only
        } else {
            Write-Host "  Mode: Quick (all versions)" -ForegroundColor Gray
            python scripts/train_ultimate.py --progressive
        }
    } else {
        if ($LatestOnly) {
            Write-Host "  Mode: Full with grid search (latest only)" -ForegroundColor Gray
            python scripts/train_ultimate.py --production --latest-only
        } else {
            Write-Host "  Mode: Full with grid search (all versions)" -ForegroundColor Gray
            python scripts/train_ultimate.py --production
        }
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Progressive training failed!" -ForegroundColor Red
        exit 1
    }
    
    $step2Duration = [math]::Round(((Get-Date) - $step2Start).TotalMinutes, 1)
    Write-Host "✓ Progressive training complete ($step2Duration min)" -ForegroundColor Green
    Write-Host ""
    $step++
    
    # Step 3: Validation & Evaluation
    Write-Host "[STEP $step/$totalSteps] Final Validation & Evaluation" -ForegroundColor Yellow
    
    python scripts/validate_model.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠ Validation warnings (non-critical)" -ForegroundColor Yellow
    } else {
        Write-Host "✓ Model validated" -ForegroundColor Green
    }
    
    python scripts/evaluate_model.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠ Evaluation warnings" -ForegroundColor Yellow
    } else {
        Write-Host "✓ Evaluation complete" -ForegroundColor Green
    }
    
    # Summary
    $TotalDuration = [math]::Round(((Get-Date) - $StartTime).TotalMinutes, 1)
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "PROGRESSIVE TRAINING COMPLETE!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "Total Time: $TotalDuration minutes" -ForegroundColor White
    Write-Host ""
    Write-Host "Model versions saved to: models/" -ForegroundColor White
    Write-Host "  - flood_rf_model_v1.joblib (Baseline)" -ForegroundColor Gray
    Write-Host "  - flood_rf_model_v2.joblib" -ForegroundColor Gray
    Write-Host "  - flood_rf_model_v3.joblib" -ForegroundColor Gray
    Write-Host "  - flood_rf_model_v4.joblib" -ForegroundColor Gray
    Write-Host "  - flood_rf_model_v5.joblib (PAGASA)" -ForegroundColor Gray
    Write-Host "  - flood_rf_model_v6.joblib (ULTIMATE)" -ForegroundColor Gray
    Write-Host "  - flood_rf_model_latest.joblib (Best)" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Reports saved to: reports/" -ForegroundColor White
    Write-Host "  - progressive_training_report.json" -ForegroundColor Gray
    Write-Host "  - model_progression_chart.png" -ForegroundColor Gray
    Write-Host ""
    Write-Host "To add new datasets in the future:" -ForegroundColor Yellow
    Write-Host "  1. Add your dataset to data/processed/" -ForegroundColor Gray
    Write-Host "  2. Add entry to VERSION_REGISTRY in train_ultimate.py" -ForegroundColor Gray
    Write-Host "  3. Run: .\run_training_pipeline.ps1 -Progressive" -ForegroundColor Gray
    Write-Host ""
    exit 0
}

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
