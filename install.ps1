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
$LockFile = "$env:TEMP\cses_agent.lock"

# CRITICAL: Check for running CSES instance
# This prevents parallel execution with the CLI
Write-Host "Checking for running instances..." -ForegroundColor Cyan
if (Test-Path $LockFile) {
    try {
        # Try to open the lock file exclusively
        $lockHandle = [System.IO.File]::Open($LockFile, 'Open', 'Read', 'None')
        $lockHandle.Close()
        # If we got here, no one else has the lock - delete stale lock file
        Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
    } catch {
        # Lock file is held by another process
        Write-Host ""
        Write-Host "[X] CSES Assistant is currently running!" -ForegroundColor Red
        Write-Host "    Please close all CSES windows before updating." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "    If you believe this is an error, delete:" -ForegroundColor Gray
        Write-Host "    $LockFile" -ForegroundColor Gray
        Write-Host ""
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Create our own lock during installation
try {
    $script:InstallLock = [System.IO.File]::Open($LockFile, 'Create', 'Write', 'None')
    "INSTALL-$PID" | Out-File $LockFile -NoNewline
} catch {
    Write-Host "[!] Could not create lock file (non-critical)" -ForegroundColor Yellow
}

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

# Check for existing installation
$ExistingEnv = $null
$ExistingVenv = $null
$IsUpdate = $false

if (Test-Path $InstallDir) {
    $IsUpdate = $true
    Write-Host "Existing installation found - updating..." -ForegroundColor Yellow

    # Preserve .env file
    if (Test-Path "$InstallDir\.env") {
        Write-Host "Preserving your configuration..." -ForegroundColor Cyan
        $ExistingEnv = Get-Content "$InstallDir\.env" -Raw
    }

    # Preserve venv (so we don't reinstall all packages)
    if (Test-Path "$InstallDir\.venv") {
        Write-Host "Preserving Python environment..." -ForegroundColor Cyan
        $ExistingVenv = "$env:TEMP\cses_venv_backup"
        if (Test-Path $ExistingVenv) {
            Remove-Item -Recurse -Force $ExistingVenv -ErrorAction SilentlyContinue
        }
        try {
            Move-Item "$InstallDir\.venv" $ExistingVenv -ErrorAction Stop
        } catch {
            Write-Host "  Could not preserve venv, will reinstall packages" -ForegroundColor Yellow
            $ExistingVenv = $null
        }
    }

    # Stop any Python processes that might lock files
    Write-Host "Stopping Python processes..." -ForegroundColor Cyan
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1

    # Remove old installation (venv already moved out)
    try {
        Remove-Item -Recurse -Force $InstallDir -ErrorAction Stop
    } catch {
        Write-Host "Cannot remove old installation (files locked)." -ForegroundColor Yellow
        # Restore venv if we moved it
        if ($ExistingVenv -and (Test-Path $ExistingVenv)) {
            New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
            Move-Item $ExistingVenv "$InstallDir\.venv" -ErrorAction SilentlyContinue
        }
        Write-Host "[X] Please close all terminals and try again." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
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

# Restore or create virtual environment
Set-Location $InstallDir
$pipExe = "$InstallDir\.venv\Scripts\pip.exe"
$pythonExe = "$InstallDir\.venv\Scripts\python.exe"

if ($ExistingVenv -and (Test-Path $ExistingVenv)) {
    # Restore preserved venv (fast update - no package reinstall needed)
    Write-Host "Restoring Python environment..." -ForegroundColor Cyan
    Move-Item $ExistingVenv "$InstallDir\.venv"
    Write-Host "[OK] Python environment restored (packages preserved)" -ForegroundColor Green
} else {
    # Create new virtual environment
    Write-Host "Creating Python environment..." -ForegroundColor Cyan

    if ($script:PythonArgs.Length -gt 0) {
        & $script:PythonCmd $script:PythonArgs -m venv .venv
    } else {
        & $script:PythonCmd -m venv .venv
    }

    if (-not (Test-Path $pythonExe)) {
        Write-Host "[X] Failed to create Python virtual environment" -ForegroundColor Red
        Write-Host "    Try running: $script:PythonCmd -m venv $InstallDir\.venv" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }

    Write-Host "[OK] Virtual environment created" -ForegroundColor Green
}

# Install dependencies
Write-Host ""
Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Cyan
Write-Host ""

# Upgrade pip first
Write-Host "Upgrading pip..." -ForegroundColor Cyan
& $pythonExe -m pip install --upgrade pip setuptools wheel 2>&1 | ForEach-Object {
    if ($_ -match "Successfully") { Write-Host $_ -ForegroundColor Green }
}

# Check which packages need to be installed (only install missing ones)
Write-Host ""
Write-Host "Checking installed packages..." -ForegroundColor Cyan

$packagesToInstall = @(
    @{pkg="python-dotenv"; mod="dotenv"},
    @{pkg="pyyaml"; mod="yaml"},
    @{pkg="lxml"; mod="lxml"},
    @{pkg="reportlab"; mod="reportlab"},
    @{pkg="pypdf"; mod="pypdf"},
    @{pkg="python-docx"; mod="docx"},
    @{pkg="openpyxl"; mod="openpyxl"},
    @{pkg="pandas"; mod="pandas"},
    @{pkg="pyreadstat"; mod="pyreadstat"},
    @{pkg="litellm"; mod="litellm"},
    @{pkg="anthropic"; mod="anthropic"},
    @{pkg="openai"; mod="openai"},
    @{pkg="rich"; mod="rich"},
    @{pkg="tqdm"; mod="tqdm"},
    @{pkg="textual"; mod="textual"}
)

$missingPackages = @()
$alreadyInstalled = 0

foreach ($item in $packagesToInstall) {
    & $pythonExe -c "import $($item.mod)" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $alreadyInstalled++
    } else {
        $missingPackages += $item.pkg
    }
}

Write-Host "  Already installed: $alreadyInstalled" -ForegroundColor Green
Write-Host "  Missing: $($missingPackages.Count)" -ForegroundColor $(if ($missingPackages.Count -gt 0) { "Yellow" } else { "Green" })

$installedCount = $alreadyInstalled
$failedPackages = @()

if ($missingPackages.Count -gt 0) {
    Write-Host ""
    Write-Host "Installing missing packages..." -ForegroundColor Cyan

    foreach ($pkg in $missingPackages) {
        Write-Host "  Installing $pkg..." -ForegroundColor Cyan -NoNewline
        $output = & $pipExe install $pkg 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host " [OK]" -ForegroundColor Green
            $installedCount++
        } else {
            Write-Host " [FAILED]" -ForegroundColor Red
            $failedPackages += $pkg
        }
    }
}

Write-Host ""
Write-Host "Installed $installedCount of $($packagesToInstall.Count) packages" -ForegroundColor Cyan

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

# CRITICAL: Verify ALL packages can actually be imported
Write-Host ""
Write-Host "Verifying installation (testing imports)..." -ForegroundColor Cyan

# Test each module individually - simpler and more reliable
$modulesToTest = @(
    @("dotenv", "python-dotenv"),
    @("yaml", "pyyaml"),
    @("lxml", "lxml"),
    @("reportlab", "reportlab"),
    @("pypdf", "pypdf"),
    @("docx", "python-docx"),
    @("openpyxl", "openpyxl"),
    @("pandas", "pandas"),
    @("pyreadstat", "pyreadstat"),
    @("litellm", "litellm"),
    @("anthropic", "anthropic"),
    @("openai", "openai"),
    @("rich", "rich"),
    @("tqdm", "tqdm"),
    @("textual", "textual")
)

$failedModules = @()
foreach ($item in $modulesToTest) {
    $mod = $item[0]
    $pkg = $item[1]
    & $pythonExe -c "import $mod" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] $pkg" -ForegroundColor Green
    } else {
        Write-Host "  [X] $pkg - FAILED" -ForegroundColor Red
        $failedModules += $pkg
    }
}

if ($failedModules.Count -gt 0) {
    Write-Host ""
    Write-Host "[X] Some packages failed to install!" -ForegroundColor Red
    Write-Host "    Missing: $($failedModules -join ', ')" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Attempting to reinstall failed packages..." -ForegroundColor Yellow

    foreach ($pkg in $failedModules) {
        Write-Host "  Reinstalling $pkg..." -ForegroundColor Yellow -NoNewline
        & $pipExe install --upgrade --force-reinstall $pkg 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host " [OK]" -ForegroundColor Green
        } else {
            Write-Host " [FAILED]" -ForegroundColor Red
        }
    }

    # Final check
    Write-Host ""
    Write-Host "Final verification..." -ForegroundColor Cyan
    $stillFailed = @()
    foreach ($item in $modulesToTest) {
        $mod = $item[0]
        $pkg = $item[1]
        & $pythonExe -c "import $mod" 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            $stillFailed += $pkg
        }
    }

    if ($stillFailed.Count -gt 0) {
        Write-Host "[X] Still missing: $($stillFailed -join ', ')" -ForegroundColor Red
        Write-Host ""
        Read-Host "Press Enter to continue anyway"
    } else {
        Write-Host "[OK] All packages now verified" -ForegroundColor Green
    }
} else {
    Write-Host ""
    Write-Host "[OK] All packages verified" -ForegroundColor Green
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

# Release install lock
if ($script:InstallLock) {
    try {
        $script:InstallLock.Close()
        Remove-Item $LockFile -Force -ErrorAction SilentlyContinue
    } catch {}
}
