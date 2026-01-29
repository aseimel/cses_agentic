# CSES Assistant - One-line installer for Windows
# Usage: irm https://raw.githubusercontent.com/aseimel/cses_agentic/main/install.ps1 | iex
# Or: powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "           CSES Assistant                                              " -ForegroundColor Cyan
Write-Host "           Windows Installation                                       " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""

# Installation paths (no admin needed - user directory)
$InstallDir = "$env:USERPROFILE\.cses-agent"
$RepoUrl = "https://github.com/aseimel/cses_agentic/archive/refs/heads/main.zip"

# Check Python (3.10-3.14 all work - 3.14 requires latest package versions)
function Test-Python {
    $pythonCmds = @("python", "python3", "py -3.14", "py -3.13", "py -3.12", "py -3.11", "py -3", "py")

    foreach ($cmd in $pythonCmds) {
        try {
            $cmdParts = $cmd.Split()
            $version = & $cmdParts[0] $cmdParts[1..99] --version 2>&1
            if ($version -match "Python 3\.(\d+)") {
                $minor = [int]$Matches[1]
                if ($minor -ge 10 -and $minor -le 14) {
                    Write-Host "[OK] $version" -ForegroundColor Green
                    $script:PythonCmd = $cmdParts[0]
                    if ($cmdParts.Length -gt 1) {
                        $script:PythonArgs = $cmdParts[1..99]
                    } else {
                        $script:PythonArgs = @()
                    }
                    $script:PythonMinor = $minor
                    return $true
                }
            }
        } catch {}
    }

    Write-Host "[X] Python 3.10+ required" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python from:" -ForegroundColor Yellow
    Write-Host "https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "IMPORTANT: Make sure 'Add Python to PATH' is enabled!" -ForegroundColor Yellow
    return $false
}

$script:PythonCmd = "python"
$script:PythonArgs = @()

Write-Host "Checking requirements..." -ForegroundColor Cyan
if (-not (Test-Python)) {
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# Check for existing installation and preserve .env
$ExistingEnv = $null
$IsUpdate = $false
if (Test-Path $InstallDir) {
    $IsUpdate = $true
    Write-Host "Existing installation found - updating..." -ForegroundColor Yellow

    # Preserve .env file
    if (Test-Path "$InstallDir\.env") {
        Write-Host "Preserving your configuration..." -ForegroundColor Cyan
        $ExistingEnv = Get-Content "$InstallDir\.env" -Raw
    }

    # Stop any Python processes that might lock the venv
    Write-Host "Stopping Python processes..." -ForegroundColor Cyan
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2

    # Try to remove old installation
    try {
        Remove-Item -Recurse -Force $InstallDir -ErrorAction Stop
    } catch {
        Write-Host "Cannot remove old installation (files locked)." -ForegroundColor Yellow
        Write-Host "Moving to backup location..." -ForegroundColor Cyan

        # Move to backup instead
        $BackupDir = "$InstallDir.old"
        if (Test-Path $BackupDir) {
            Remove-Item -Recurse -Force $BackupDir -ErrorAction SilentlyContinue
        }

        try {
            Rename-Item $InstallDir $BackupDir -ErrorAction Stop
            Write-Host "Old installation moved to $BackupDir" -ForegroundColor Gray
        } catch {
            Write-Host "[X] Cannot move or delete old installation." -ForegroundColor Red
            Write-Host "    Please close all terminals and try again." -ForegroundColor Yellow
            Write-Host "    Or manually delete: $InstallDir" -ForegroundColor Yellow
            Read-Host "Press Enter to exit"
            exit 1
        }
    }
}

# Download and extract
Write-Host "Downloading CSES Assistant..." -ForegroundColor Cyan
$ZipFile = "$env:TEMP\cses_agentic.zip"
try {
    Invoke-WebRequest -Uri $RepoUrl -OutFile $ZipFile -UseBasicParsing
} catch {
    Write-Host "[X] Download failed: $_" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Extracting..." -ForegroundColor Cyan
Expand-Archive -Path $ZipFile -DestinationPath $env:TEMP -Force
Move-Item "$env:TEMP\cses_agentic-main" $InstallDir
Remove-Item $ZipFile

# Restore .env if it existed
if ($ExistingEnv) {
    Write-Host "Restoring your configuration..." -ForegroundColor Cyan
    Set-Content -Path "$InstallDir\.env" -Value $ExistingEnv
}

# Create virtual environment
Write-Host "Setting up Python environment..." -ForegroundColor Cyan
Set-Location $InstallDir

if ($script:PythonArgs.Length -gt 0) {
    & $script:PythonCmd $script:PythonArgs -m venv .venv
} else {
    & $script:PythonCmd -m venv .venv
}

# Check venv was created
$pipExe = "$InstallDir\.venv\Scripts\pip.exe"
$pythonExe = "$InstallDir\.venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "[X] Failed to create Python virtual environment" -ForegroundColor Red
    Write-Host "    Try running: $script:PythonCmd -m venv $InstallDir\.venv" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[OK] Virtual environment created" -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Cyan
Write-Host ""

# Upgrade pip first
Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $pythonExe -m pip install --upgrade pip setuptools wheel 2>&1 | ForEach-Object {
    if ($_ -match "Successfully") { Write-Host $_ -ForegroundColor Green }
}

# Install packages one by one to show progress and handle failures
Write-Host ""
Write-Host "Installing required packages..." -ForegroundColor Cyan

$corePackages = @(
    "python-dotenv",
    "pyyaml",
    "lxml",
    "reportlab",
    "pypdf",
    "python-docx",
    "openpyxl",
    "pandas",
    "pyreadstat",
    "litellm",
    "anthropic",
    "openai",
    "rich",
    "tqdm"
)

$installedCount = 0
$failedPackages = @()

foreach ($pkg in $corePackages) {
    Write-Host "  Installing $pkg..." -ForegroundColor Cyan -NoNewline
    $output = & $pipExe install --upgrade $pkg 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host " [OK]" -ForegroundColor Green
        $installedCount++
    } else {
        Write-Host " [FAILED]" -ForegroundColor Red
        $failedPackages += $pkg
    }
}

Write-Host ""
Write-Host "Installed $installedCount of $($corePackages.Count) packages" -ForegroundColor Cyan

if ($failedPackages.Count -gt 0) {
    Write-Host "Failed packages: $($failedPackages -join ', ')" -ForegroundColor Yellow
}

$fullInstallFailed = $failedPackages.Count -gt 3

if ($fullInstallFailed) {
    Write-Host ""
    Write-Host "[!] Too many packages failed to install." -ForegroundColor Red
    Write-Host "    The tool may not work correctly." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Try running manually:" -ForegroundColor Cyan
    Write-Host "  $pythonExe -m pip install python-dotenv litellm pandas" -ForegroundColor White
}

Write-Host ""

# Create batch file launcher
Write-Host "Creating 'cses' command..." -ForegroundColor Cyan
$BinDir = "$env:USERPROFILE\.local\bin"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null

$BatchContent = @"
@echo off
set INSTALL_DIR=%USERPROFILE%\.cses-agent
set PYTHONPATH=%INSTALL_DIR%;%PYTHONPATH%
"%INSTALL_DIR%\.venv\Scripts\python.exe" "%INSTALL_DIR%\cses_cli.py" %*
"@

Set-Content -Path "$BinDir\cses.bat" -Value $BatchContent

# Add to user PATH
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$BinDir*") {
    Write-Host "Adding to user PATH..." -ForegroundColor Cyan
    [Environment]::SetEnvironmentVariable("Path", "$BinDir;$UserPath", "User")
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
if ($IsUpdate) {
    Write-Host "           Update Complete!                                           " -ForegroundColor Green
} else {
    Write-Host "           Installation Complete!                                     " -ForegroundColor Green
}
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""

if ($IsUpdate) {
    Write-Host "Your configuration has been preserved." -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To use: Open a new PowerShell and run 'cses'" -ForegroundColor White
} else {
    Write-Host "NEXT STEPS:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  1. CLOSE and REOPEN PowerShell" -ForegroundColor White
    Write-Host ""
    Write-Host "  2. Run 'cses' to start the setup wizard" -ForegroundColor White
    Write-Host "     (It will ask for your GESIS API key and Stata path)" -ForegroundColor Gray
    Write-Host ""
    Write-Host "  3. Navigate to a folder with collaborator files and run 'cses'" -ForegroundColor White
}
Write-Host ""
Write-Host "To reconfigure later: cses setup --force" -ForegroundColor Gray
Write-Host ""
