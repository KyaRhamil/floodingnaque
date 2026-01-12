# ============================================================================
# CREDENTIAL REMOVAL SCRIPT
# ============================================================================
# This script removes sensitive credential files from Git history
# 
# WARNING: This rewrites Git history! All team members must re-clone after this.
# 
# FILES TO REMOVE:
#   - google_oauth_credentials.json
#   - floodingnaque-service-account.json
#   - floodingnaque-googe-cloud-482008-g2-4a707b57bfa3.json
#   - floodingnaque-google-cloud-482008-g2-4a707b57bfa3.json
#
# PREREQUISITES:
#   1. Install git-filter-repo: pip install git-filter-repo
#   2. OR download BFG Repo-Cleaner: https://rtyley.github.io/bfg-repo-cleaner/
#   3. Backup your repository first!
#
# ============================================================================

param(
    [switch]$DryRun = $false,
    [switch]$UseBFG = $false
)

$ErrorActionPreference = "Stop"

Write-Host "=" -NoNewline; Write-Host ("=" * 70) -ForegroundColor Yellow
Write-Host "  CREDENTIAL REMOVAL FROM GIT HISTORY" -ForegroundColor Red
Write-Host "=" -NoNewline; Write-Host ("=" * 70) -ForegroundColor Yellow
Write-Host ""

# Files to remove from history
$filesToRemove = @(
    "google_oauth_credentials.json",
    "floodingnaque-service-account.json",
    "floodingnaque-googe-cloud-482008-g2-4a707b57bfa3.json",
    "floodingnaque-google-cloud-482008-g2-4a707b57bfa3.json",
    "backend/google_oauth_credentials.json",
    "backend/floodingnaque-service-account.json",
    "backend/floodingnaque-googe-cloud-482008-g2-4a707b57bfa3.json",
    "backend/floodingnaque-google-cloud-482008-g2-4a707b57bfa3.json"
)

Write-Host "Files to be removed from Git history:" -ForegroundColor Cyan
foreach ($file in $filesToRemove) {
    Write-Host "  - $file" -ForegroundColor White
}
Write-Host ""

if ($DryRun) {
    Write-Host "[DRY RUN] No changes will be made" -ForegroundColor Magenta
    Write-Host ""
}

# Check if we're in a git repository
if (-not (Test-Path ".git")) {
    Write-Host "ERROR: Not in a Git repository root. Please run from project root." -ForegroundColor Red
    exit 1
}

# Create backup branch
Write-Host "Step 1: Creating backup branch..." -ForegroundColor Green
if (-not $DryRun) {
    git branch backup-before-credential-removal 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Backup branch already exists, continuing..." -ForegroundColor Yellow
    } else {
        Write-Host "  Created backup branch: backup-before-credential-removal" -ForegroundColor White
    }
}

# Method 1: Using git-filter-repo (recommended)
if (-not $UseBFG) {
    Write-Host ""
    Write-Host "Step 2: Removing files using git-filter-repo..." -ForegroundColor Green
    
    # Check if git-filter-repo is installed
    $filterRepoInstalled = $null
    try {
        $filterRepoInstalled = Get-Command git-filter-repo -ErrorAction SilentlyContinue
    } catch {}
    
    if (-not $filterRepoInstalled) {
        Write-Host "  git-filter-repo not found. Installing..." -ForegroundColor Yellow
        if (-not $DryRun) {
            pip install git-filter-repo
        }
    }
    
    # Create paths file for filter-repo
    $pathsFile = "paths-to-remove.txt"
    if (-not $DryRun) {
        $filesToRemove | Out-File -FilePath $pathsFile -Encoding UTF8
    }
    
    Write-Host "  Running git-filter-repo..." -ForegroundColor White
    if (-not $DryRun) {
        # Remove each file from history
        foreach ($file in $filesToRemove) {
            Write-Host "    Removing: $file" -ForegroundColor Gray
            git filter-repo --invert-paths --path $file --force 2>$null
        }
        
        # Clean up
        Remove-Item $pathsFile -ErrorAction SilentlyContinue
    }
}
# Method 2: Using BFG Repo-Cleaner
else {
    Write-Host ""
    Write-Host "Step 2: Removing files using BFG Repo-Cleaner..." -ForegroundColor Green
    Write-Host "  NOTE: Download BFG from https://rtyley.github.io/bfg-repo-cleaner/" -ForegroundColor Yellow
    
    # Create blob IDs file
    $blobFile = "blobs-to-remove.txt"
    
    if (-not $DryRun) {
        foreach ($file in $filesToRemove) {
            Write-Host "    Processing: $file" -ForegroundColor Gray
            # BFG command (requires bfg.jar)
            # java -jar bfg.jar --delete-files $file
        }
    }
    
    Write-Host ""
    Write-Host "  Manual BFG commands to run:" -ForegroundColor Cyan
    Write-Host "    java -jar bfg.jar --delete-files google_oauth_credentials.json" -ForegroundColor White
    Write-Host "    java -jar bfg.jar --delete-files floodingnaque-service-account.json" -ForegroundColor White
    Write-Host "    java -jar bfg.jar --delete-files 'floodingnaque-googe-cloud-*.json'" -ForegroundColor White
    Write-Host "    git reflog expire --expire=now --all && git gc --prune=now --aggressive" -ForegroundColor White
}

# Step 3: Delete the actual files from working directory
Write-Host ""
Write-Host "Step 3: Deleting credential files from working directory..." -ForegroundColor Green
foreach ($file in $filesToRemove) {
    $fullPath = Join-Path (Get-Location) $file
    if (Test-Path $fullPath) {
        if (-not $DryRun) {
            Remove-Item $fullPath -Force
            Write-Host "  Deleted: $file" -ForegroundColor White
        } else {
            Write-Host "  Would delete: $file" -ForegroundColor Magenta
        }
    }
}

# Step 4: Clean up Git
Write-Host ""
Write-Host "Step 4: Cleaning up Git repository..." -ForegroundColor Green
if (-not $DryRun) {
    git reflog expire --expire=now --all
    git gc --prune=now --aggressive
    Write-Host "  Git cleanup complete" -ForegroundColor White
}

# Final instructions
Write-Host ""
Write-Host "=" -NoNewline; Write-Host ("=" * 70) -ForegroundColor Yellow
Write-Host "  NEXT STEPS" -ForegroundColor Cyan
Write-Host "=" -NoNewline; Write-Host ("=" * 70) -ForegroundColor Yellow
Write-Host ""
Write-Host "1. ROTATE ALL CREDENTIALS IMMEDIATELY:" -ForegroundColor Red
Write-Host "   - Google Cloud Console: https://console.cloud.google.com/iam-admin/serviceaccounts" -ForegroundColor White
Write-Host "   - Delete old service account keys and create new ones" -ForegroundColor White
Write-Host "   - Google OAuth: https://console.cloud.google.com/apis/credentials" -ForegroundColor White
Write-Host "   - Reset client secret for OAuth 2.0 Client IDs" -ForegroundColor White
Write-Host ""
Write-Host "2. UPDATE ENVIRONMENT VARIABLES:" -ForegroundColor Yellow
Write-Host "   - Set GOOGLE_APPLICATION_CREDENTIALS to path of new service account JSON" -ForegroundColor White
Write-Host "   - Set GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET in .env" -ForegroundColor White
Write-Host ""
Write-Host "3. FORCE PUSH TO REMOTE (DESTRUCTIVE!):" -ForegroundColor Red
Write-Host "   git push origin --force --all" -ForegroundColor White
Write-Host "   git push origin --force --tags" -ForegroundColor White
Write-Host ""
Write-Host "4. NOTIFY ALL TEAM MEMBERS to re-clone the repository" -ForegroundColor Yellow
Write-Host ""
Write-Host "5. REVOKE ANY TOKENS that may have been exposed" -ForegroundColor Red
Write-Host ""

if ($DryRun) {
    Write-Host "[DRY RUN COMPLETE] Run without -DryRun to execute" -ForegroundColor Magenta
}
