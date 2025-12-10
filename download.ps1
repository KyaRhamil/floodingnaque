##############################################################################
# Blackbox CLI v2 Install Script for Windows PowerShell (Extension Service Version)
#
# This script downloads and installs the Blackbox CLI v2 (Node.js-based)
# from the Extension Upload Service.
#
# Supported OS: Windows
# Supported Architectures: x86_64
#
# Usage:
#   Invoke-WebRequest -Uri "https://releases.blackbox.ai/api/scripts/blackbox-cli-v2/download.ps1" -OutFile "download.ps1"; .\download.ps1
#
# Environment variables:
#   $env:BLACKBOX_INSTALL_DIR - Directory to install Blackbox CLI v2 (default: $env:USERPROFILE\.blackbox-cli-v2)
#   $env:BLACKBOX_BIN_DIR     - Directory for the executable wrapper (default: $env:USERPROFILE\.local\bin)
#   $env:EXTENSION_SERVICE_URL - Extension service URL (default: https://releases.blackbox.ai)
#   $env:CONFIGURE           - Optional: if set to "false", disables running blackbox configure interactively
##############################################################################

$ErrorActionPreference = "Stop"

# --- 1) Check for Node.js and npm ---
Write-Host "Checking for Node.js..." -ForegroundColor Gray
try {
    $nodeVersion = node --version
    $nodeMajorVersion = [int]($nodeVersion -replace 'v(\d+)\..*', '$1')
    
    if ($nodeMajorVersion -lt 20) {
        Write-Error "Node.js version 20 or higher is required. Current version: $nodeVersion"
        Write-Host "Please upgrade Node.js from https://nodejs.org/ and try again." -ForegroundColor Red
        exit 1
    }
    Write-Host "Found Node.js $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "Node.js is not installed or not in PATH." -ForegroundColor Red
    Write-Host ""
    Write-Host "To install Node.js:" -ForegroundColor Yellow
    Write-Host "1. Visit https://nodejs.org/" -ForegroundColor Yellow
    Write-Host "2. Download the LTS version for Windows" -ForegroundColor Yellow
    Write-Host "3. Run the installer and follow the prompts" -ForegroundColor Yellow
    Write-Host "4. Restart your terminal and run this script again" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Or use a package manager:" -ForegroundColor Yellow
    Write-Host "  winget install OpenJS.NodeJS.LTS" -ForegroundColor Cyan
    Write-Host "  choco install nodejs-lts" -ForegroundColor Cyan
    exit 1
}

# Check for npm
try {
    $npmVersion = npm --version
    Write-Host "Found npm $npmVersion" -ForegroundColor Green
} catch {
    Write-Host "npm is not installed or not in PATH." -ForegroundColor Red
    Write-Host "npm should be installed with Node.js. Please reinstall Node.js from https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

# --- 2) Variables ---
$PRODUCT_SLUG = "blackbox-cli-v2"
$EXTENSION_SERVICE_URL = if ($env:EXTENSION_SERVICE_URL) { $env:EXTENSION_SERVICE_URL } else { "https://releases.blackbox.ai" }

# Set default directories
if (-not $env:BLACKBOX_INSTALL_DIR) {
    $env:BLACKBOX_INSTALL_DIR = Join-Path $env:USERPROFILE ".blackbox-cli-v2"
}
if (-not $env:BLACKBOX_BIN_DIR) {
    $env:BLACKBOX_BIN_DIR = Join-Path $env:USERPROFILE ".local\bin"
}

$CONFIGURE = if ($env:CONFIGURE -eq "false") { "false" } else { "true" }

# --- 3) Detect Architecture ---
$ARCH = $env:PROCESSOR_ARCHITECTURE
if ($ARCH -eq "AMD64") {
    $PLATFORM = "win-x64"
} elseif ($ARCH -eq "ARM64") {
    Write-Error "Windows ARM64 is not currently supported."
    exit 1
} else {
    Write-Error "Unsupported architecture '$ARCH'. Only x86_64 is supported on Windows."
    exit 1
}

# --- 4) Get latest release information ---
Write-Host "Fetching latest release information for platform: $PLATFORM..." -ForegroundColor Gray

$RELEASE_API_URL = "$EXTENSION_SERVICE_URL/api/v0/latest?product=$PRODUCT_SLUG&platform=$PLATFORM"

try {
    $RELEASE_INFO = Invoke-WebRequest -Uri $RELEASE_API_URL -UseBasicParsing | ConvertFrom-Json
    Write-Host "Successfully fetched release information." -ForegroundColor Green
} catch {
    Write-Error "Failed to fetch release information from $RELEASE_API_URL. Error: $($_.Exception.Message)"
    Write-Host "Please check that the extension service is available and the product '$PRODUCT_SLUG' exists." -ForegroundColor Red
    exit 1
}

# Extract download URL and version
$DOWNLOAD_URL = $RELEASE_INFO.url
$VERSION = $RELEASE_INFO.version

if (-not $DOWNLOAD_URL) {
    Write-Error "Could not parse download URL from release information"
    Write-Host "Release info: $($RELEASE_INFO | ConvertTo-Json)" -ForegroundColor Red
    exit 1
}

# Ensure download URL is absolute
if (-not $DOWNLOAD_URL.StartsWith("http")) {
    $DOWNLOAD_URL = "$EXTENSION_SERVICE_URL$DOWNLOAD_URL"
}

Write-Host "Downloading Blackbox CLI v2 version $VERSION from: $DOWNLOAD_URL" -ForegroundColor Blue

# --- 5) Download the file ---
$FILENAME = Split-Path $DOWNLOAD_URL -Leaf
if (-not $FILENAME -or $FILENAME -eq "") {
    $FILENAME = "blackbox-cli-v2-$PLATFORM.zip"
}

try {
    Invoke-WebRequest -Uri $DOWNLOAD_URL -OutFile $FILENAME -UseBasicParsing
    Write-Host "Download completed successfully." -ForegroundColor Green
} catch {
    Write-Error "Failed to download $DOWNLOAD_URL. Error: $($_.Exception.Message)"
    exit 1
}

# --- 6) Remove existing installation if present ---
if (Test-Path $env:BLACKBOX_INSTALL_DIR) {
    Write-Host "Removing existing installation at $env:BLACKBOX_INSTALL_DIR..." -ForegroundColor DarkYellow
    Remove-Item -Path $env:BLACKBOX_INSTALL_DIR -Recurse -Force
}

# Remove old blackbox executables from bin directory
if (Test-Path $env:BLACKBOX_BIN_DIR) {
    Write-Host "Removing old blackbox executables from $env:BLACKBOX_BIN_DIR..." -ForegroundColor DarkYellow
    Remove-Item -Path (Join-Path $env:BLACKBOX_BIN_DIR "blackbox.cmd") -Force -ErrorAction SilentlyContinue
    Remove-Item -Path (Join-Path $env:BLACKBOX_BIN_DIR "blackbox.mjs") -Force -ErrorAction SilentlyContinue
    Remove-Item -Path (Join-Path $env:BLACKBOX_BIN_DIR "blackbox") -Force -ErrorAction SilentlyContinue
}

# --- 7) Create installation directory and extract ---
Write-Host "Creating installation directory: $env:BLACKBOX_INSTALL_DIR" -ForegroundColor DarkYellow
New-Item -ItemType Directory -Path $env:BLACKBOX_INSTALL_DIR -Force | Out-Null

Write-Host "Extracting $FILENAME..." -ForegroundColor Gray
try {
    Expand-Archive -Path $FILENAME -DestinationPath $env:BLACKBOX_INSTALL_DIR -Force
    Write-Host "Extraction completed successfully." -ForegroundColor Green
} catch {
    Write-Error "Failed to extract $FILENAME. Error: $($_.Exception.Message)"
    Remove-Item -Path $FILENAME -Force -ErrorAction SilentlyContinue
    exit 1
}

Remove-Item -Path $FILENAME -Force

# npm global install
Remove-Item -Path $env:BLACKBOX_INSTALL_DIR -Recurse -Force
npm i -g @blackbox_ai/blackbox-cli

blackbox --version

Write-Host "Installation completed successfully. You can now use the blackbox command." -ForegroundColor Green


exit 0;
# TODO: execution stopped for now. will continue back when avaialable.

# --- 8) Verify installation structure ---
$CLI_DIST_PATH = Join-Path $env:BLACKBOX_INSTALL_DIR "packages\cli\dist"
if (-not (Test-Path $CLI_DIST_PATH)) {
    Write-Error "Invalid package structure. Expected packages\cli\dist directory not found."
    Write-Host "Contents of $env:BLACKBOX_INSTALL_DIR:" -ForegroundColor Red
    Get-ChildItem -Path $env:BLACKBOX_INSTALL_DIR -Recurse | Select-Object FullName
    exit 1
}

# --- 8.5) Install npm dependencies ---
Write-Host "Installing dependencies..." -ForegroundColor Gray

$CLI_DIST = Join-Path $env:BLACKBOX_INSTALL_DIR "packages\cli\dist"
if (Test-Path (Join-Path $CLI_DIST "package.json")) {
    Write-Host "  Installing npm packages (this may take a minute)..." -ForegroundColor Gray
    
    Push-Location $CLI_DIST
    try {
        # Install all dependencies (not just production) to ensure dev dependencies like react are available
        $npmOutput = npm install 2>&1 | Out-String
        if ($npmOutput -match "added.*packages") {
            Write-Host "  ✓ Dependencies installed successfully" -ForegroundColor Green
        } elseif ($npmOutput -match "error") {
            Write-Host "  ✗ npm install encountered errors:" -ForegroundColor Red
            $npmOutput -split "`n" | Where-Object { $_ -match "error" } | Select-Object -First 5 | ForEach-Object {
                Write-Host "    $_" -ForegroundColor Red
            }
            Write-Warning "The CLI may not work correctly."
        } else {
            Write-Host "  ✓ Dependencies installed" -ForegroundColor Green
        }
    } catch {
        Write-Warning "Failed to install npm dependencies: $($_.Exception.Message)"
        Write-Warning "You may need to run 'npm install' manually in $CLI_DIST"
    }
    Pop-Location
}

# --- 8.6) Create package.json for packages that don't have one ---
Write-Host "Setting up package metadata..." -ForegroundColor Gray

# Map package directory names to their expected import names
$PKG_MAP = @{
    "cli" = "blackbox-cli"
    "core" = "blackbox-cli-core"
    "test-utils" = "blackbox-cli-test-utils"
    "vscode-ide-companion" = "blackbox-cli-vscode-ide-companion"
}

# Create package.json for packages that don't have one
foreach ($pkg_dir in @("core", "test-utils", "vscode-ide-companion")) {
    $PKG_DIST_PATH = Join-Path $env:BLACKBOX_INSTALL_DIR "packages\$pkg_dir\dist"
    $PKG_JSON_PATH = Join-Path $PKG_DIST_PATH "package.json"
    
    if ((Test-Path $PKG_DIST_PATH) -and (-not (Test-Path $PKG_JSON_PATH))) {
        $pkg_name = $PKG_MAP[$pkg_dir]
        $packageJson = @{
            name = "@blackbox_ai/$pkg_name"
            version = "0.0.9"
            type = "module"
            main = "index.js"
        } | ConvertTo-Json
        
        Set-Content -Path $PKG_JSON_PATH -Value $packageJson -Encoding UTF8
        Write-Host "  Created package.json for @blackbox_ai/$pkg_name" -ForegroundColor Gray
    }
}

# --- 8.7) Set up node_modules with symlinks for local packages ---
Write-Host "Setting up package resolution..." -ForegroundColor Gray

# Create node_modules/@blackbox_ai directory structure in the CLI package
$NODE_MODULES_DIR = Join-Path $CLI_DIST "node_modules\@blackbox_ai"
New-Item -ItemType Directory -Path $NODE_MODULES_DIR -Force | Out-Null

# Create symlinks for each package in the CLI's node_modules
foreach ($pkg_dir in @("core", "test-utils", "vscode-ide-companion")) {
    $PKG_DIST_PATH = Join-Path $env:BLACKBOX_INSTALL_DIR "packages\$pkg_dir\dist"
    if (Test-Path $PKG_DIST_PATH) {
        $pkg_name = $PKG_MAP[$pkg_dir]
        $LINK_PATH = Join-Path $NODE_MODULES_DIR $pkg_name
        
        # Remove existing file, directory, or symlink
        if (Test-Path $LINK_PATH) {
            Remove-Item -Path $LINK_PATH -Recurse -Force -ErrorAction SilentlyContinue
        }
        
        # Create symlink (requires admin on older Windows, but works on Windows 10+ with developer mode)
        try {
            New-Item -ItemType SymbolicLink -Path $LINK_PATH -Target $PKG_DIST_PATH -Force -ErrorAction Stop | Out-Null
            Write-Host "  Linked @blackbox_ai/$pkg_name" -ForegroundColor Gray
        } catch {
            # If symlink fails, try junction (works without admin)
            try {
                cmd /c mklink /J "$LINK_PATH" "$PKG_DIST_PATH" 2>&1 | Out-Null
                Write-Host "  Linked @blackbox_ai/$pkg_name (junction)" -ForegroundColor Gray
            } catch {
                Write-Warning "Failed to create link for @blackbox_ai/$pkg_name. The CLI may not work correctly."
            }
        }
        
        # Also create a symlink from each package to the CLI's node_modules
        # so they can resolve shared dependencies
        $PKG_NODE_MODULES = Join-Path $PKG_DIST_PATH "node_modules"
        $CLI_NODE_MODULES = Join-Path $CLI_DIST "node_modules"
        
        if (-not (Test-Path $PKG_NODE_MODULES)) {
            try {
                New-Item -ItemType SymbolicLink -Path $PKG_NODE_MODULES -Target $CLI_NODE_MODULES -Force -ErrorAction Stop | Out-Null
                Write-Host "  Linked node_modules for $pkg_name" -ForegroundColor Gray
            } catch {
                # Try junction as fallback
                try {
                    cmd /c mklink /J "$PKG_NODE_MODULES" "$CLI_NODE_MODULES" 2>&1 | Out-Null
                    Write-Host "  Linked node_modules for $pkg_name (junction)" -ForegroundColor Gray
                } catch {
                    Write-Warning "Failed to create node_modules link for $pkg_name"
                }
            }
        }
    }
}

# --- 9) Create bin directory if needed ---
if (-not (Test-Path $env:BLACKBOX_BIN_DIR)) {
    Write-Host "Creating bin directory: $env:BLACKBOX_BIN_DIR" -ForegroundColor DarkYellow
    New-Item -ItemType Directory -Path $env:BLACKBOX_BIN_DIR -Force | Out-Null
}

# --- 10) Create executable wrapper scripts ---
$WRAPPER_CMD = Join-Path $env:BLACKBOX_BIN_DIR "blackbox.cmd"
$WRAPPER_MJS = Join-Path $env:BLACKBOX_BIN_DIR "blackbox.mjs"

Write-Host "Creating executable wrappers at $env:BLACKBOX_BIN_DIR" -ForegroundColor Gray

# Create CMD wrapper that calls the .mjs file
$cmdContent = @"
@echo off
setlocal

REM Blackbox CLI v2 Wrapper (CMD)
set "BLACKBOX_INSTALL_DIR=$env:BLACKBOX_INSTALL_DIR"
set "BLACKBOX_CLI_V2_ROOT=%BLACKBOX_INSTALL_DIR%"

node "%~dp0blackbox.mjs" %*
"@

Set-Content -Path $WRAPPER_CMD -Value $cmdContent -Encoding ASCII

# Create ES Module wrapper
$mjsContent = @"
#!/usr/bin/env node

// Blackbox CLI v2 Wrapper
import { fileURLToPath, pathToFileURL } from 'url';
import { dirname, join } from 'path';
import { existsSync } from 'fs';

// Determine installation directory
const installDir = process.env.BLACKBOX_INSTALL_DIR || join(process.env.USERPROFILE || process.env.HOME, '.blackbox-cli-v2');
const cliEntry = join(installDir, 'packages', 'cli', 'dist', 'index.js');

// Verify CLI entry point exists
if (!existsSync(cliEntry)) {
  console.error('Error: Blackbox CLI v2 installation not found at:', installDir);
  console.error('Expected entry point:', cliEntry);
  console.error('');
  console.error('Please reinstall Blackbox CLI v2.');
  process.exit(1);
}

// Set up environment
process.env.BLACKBOX_CLI_V2_ROOT = installDir;

// Load and execute the CLI using dynamic import
// Convert Windows path to file:// URL for ESM loader compatibility
try {
  await import(pathToFileURL(cliEntry).href);
} catch (error) {
  console.error('Error running Blackbox CLI v2:', error.message);
  process.exit(1);
}
"@

Set-Content -Path $WRAPPER_MJS -Value $mjsContent -Encoding UTF8

# --- 11) Verify installation ---
Write-Host "Verifying installation..." -ForegroundColor Gray
try {
    & $WRAPPER_CMD --version 2>&1 | Out-Null
    Write-Host "Installation verified successfully." -ForegroundColor Green
} catch {
    Write-Warning "Could not verify installation with --version command."
    Write-Warning "The CLI may still work, but please test it manually."
}

# --- 12) Configure Blackbox (Optional) ---
# Skip interactive configuration during installation since it may not work in all contexts
# Users can run 'blackbox configure' manually after installation
if ($CONFIGURE -eq "true") {
    Write-Host ""
    Write-Host "Skipping interactive configuration during installation." -ForegroundColor Gray
    Write-Host "You can run 'blackbox configure' manually after installation." -ForegroundColor Gray
}

# --- 13) Check PATH and add to environment if needed ---
$CURRENT_PATH = $env:PATH
if ($CURRENT_PATH -notlike "*$env:BLACKBOX_BIN_DIR*") {
    Write-Host ""
    Write-Host "Adding $env:BLACKBOX_BIN_DIR to your PATH..." -ForegroundColor Green
    
    try {
        # Get current user PATH
        $currentUserPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
        
        # Check if the path is already in user PATH
        if ($currentUserPath -notlike "*$env:BLACKBOX_BIN_DIR*") {
            Write-Host "Adding Blackbox CLI v2 to user PATH..." -ForegroundColor Green
            
            # Add to user PATH
            $newUserPath = if ($currentUserPath) { "$currentUserPath;$env:BLACKBOX_BIN_DIR" } else { "$env:BLACKBOX_BIN_DIR" }
            [Environment]::SetEnvironmentVariable('PATH', $newUserPath, 'User')
            
            # Add BLACKBOX_INSTALL_DIR to user environment
            [Environment]::SetEnvironmentVariable('BLACKBOX_INSTALL_DIR', $env:BLACKBOX_INSTALL_DIR, 'User')
            
            # Update current session PATH
            $env:PATH += ";$env:BLACKBOX_BIN_DIR"
            
            Write-Host "PATH successfully updated!" -ForegroundColor Green
            Write-Host "- Added to user environment variables (permanent)" -ForegroundColor Green
            Write-Host "- Updated current session PATH" -ForegroundColor Green
            Write-Host ""
            Write-Host "Note: New terminal sessions will automatically have the updated PATH." -ForegroundColor DarkYellow
        } else {
            Write-Host "PATH entry already exists in user environment variables." -ForegroundColor DarkYellow
            # Still update current session if needed
            if ($env:PATH -notlike "*$env:BLACKBOX_BIN_DIR*") {
                $env:PATH += ";$env:BLACKBOX_BIN_DIR"
                Write-Host "Updated current session PATH." -ForegroundColor Green
            }
        }
    } catch {
        Write-Host "Error: Failed to update PATH automatically. $($_.Exception.Message)" -ForegroundColor Red
        Write-Host ""
        Write-Host "Please add manually using one of these methods:" -ForegroundColor DarkYellow
        Write-Host "For user PATH (no admin required):" -ForegroundColor DarkYellow
        Write-Host "    [Environment]::SetEnvironmentVariable('PATH', `$env:PATH + ';$env:BLACKBOX_BIN_DIR', 'User')" -ForegroundColor Cyan
        Write-Host ""
        Write-Host "For this session only:" -ForegroundColor DarkYellow
        Write-Host "    `$env:PATH += ';$env:BLACKBOX_BIN_DIR'" -ForegroundColor Cyan
    }
    Write-Host ""
}

Write-Host "✓ Blackbox CLI v2 version $VERSION installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Installation directory: $env:BLACKBOX_INSTALL_DIR" -ForegroundColor Gray
Write-Host "Executable wrapper: $WRAPPER_CMD" -ForegroundColor Gray
Write-Host ""
Write-Host "You can now use the ``'blackbox``' command (after updating PATH or restarting terminal)" -ForegroundColor Green"
