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

# Install dependencies
Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Cyan
Write-Host ""

$pipExe = "$InstallDir\.venv\Scripts\pip.exe"

# Upgrade pip and setuptools first
Write-Host "Upgrading pip and setuptools..." -ForegroundColor Cyan
& "$InstallDir\.venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Warning: pip upgrade had issues, continuing..." -ForegroundColor Yellow
}

# Install requirements with --upgrade to get latest Python 3.14-compatible versions
Write-Host "Installing packages..." -ForegroundColor Cyan
& $pipExe install --upgrade -r requirements.txt 2>&1 | Tee-Object -Variable pipOutput | ForEach-Object {
    if ($_ -match "error|Error|ERROR") {
        Write-Host $_ -ForegroundColor Red
    } elseif ($_ -match "Successfully installed") {
        Write-Host $_ -ForegroundColor Green
    }
}

$fullInstallFailed = $LASTEXITCODE -ne 0

if ($fullInstallFailed) {
    Write-Host ""
    Write-Host "[!] Full requirements failed. Installing core packages one by one..." -ForegroundColor Yellow
    Write-Host ""

    # Essential packages with Python 3.14-compatible minimum versions
    $corePackages = @(
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.0",
        "lxml>=6.0.1",
        "reportlab>=4.4.0",
        "pypdf>=4.0.0",
        "python-docx>=1.0.0",
        "openpyxl>=3.1.0",
        "pandas>=3.0.0",
        "pyreadstat>=1.3.2",
        "litellm>=1.60.0",
        "anthropic>=0.39.0",
        "openai>=1.0.0"
    )

    $installedCount = 0
    $failedPackages = @()

    foreach ($pkg in $corePackages) {
        $pkgName = $pkg -replace ">=.*", ""
        Write-Host "  Installing $pkgName..." -ForegroundColor Cyan -NoNewline
        $result = & $pipExe install --upgrade $pkg 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host " [OK]" -ForegroundColor Green
            $installedCount++
        } else {
            Write-Host " [FAILED]" -ForegroundColor Red
            $failedPackages += $pkgName
        }
    }

    Write-Host ""
    Write-Host "Installed $installedCount of $($corePackages.Count) packages" -ForegroundColor Cyan

    if ($failedPackages.Count -gt 0) {
        Write-Host "Failed packages: $($failedPackages -join ', ')" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "The tool may still work with limited functionality." -ForegroundColor Yellow
    }
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
