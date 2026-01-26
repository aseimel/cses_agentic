# CSES Agent - One-line installer for Windows
# Usage: irm https://raw.githubusercontent.com/YOUR_ORG/cses_agentic/main/install.ps1 | iex
# Or: powershell -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║           CSES Data Harmonization Agent                      ║" -ForegroundColor Cyan
Write-Host "║           Windows Installation                               ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Installation paths (no admin needed - user directory)
$InstallDir = "$env:USERPROFILE\.cses-agent"
$RepoUrl = "https://github.com/aseimel/cses_agentic/archive/refs/heads/main.zip"

# Check Python (3.10, 3.11, 3.12, 3.13 all work)
function Test-Python {
    # Try different python commands
    $pythonCmds = @("python", "python3", "py -3")

    foreach ($cmd in $pythonCmds) {
        try {
            $version = & $cmd.Split()[0] $cmd.Split()[1..99] --version 2>&1
            if ($version -match "Python 3\.(\d+)") {
                $minor = [int]$Matches[1]
                if ($minor -ge 10) {
                    Write-Host "[OK] $version" -ForegroundColor Green
                    $script:PythonCmd = $cmd.Split()[0]
                    return $true
                }
            }
        } catch {}
    }

    Write-Host "[X] Python 3.10+ required" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please ask IT to install Python, or install from:" -ForegroundColor Yellow
    Write-Host "https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "IMPORTANT: Make sure 'Add Python to PATH' is enabled!" -ForegroundColor Yellow
    return $false
}

$script:PythonCmd = "python"

# Check Claude CLI (optional)
function Test-Claude {
    try {
        $null = Get-Command claude -ErrorAction Stop
        Write-Host "[OK] Claude CLI found" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "[!] Claude CLI not found (optional)" -ForegroundColor Yellow
        Write-Host "    For Claude Max subscription support, install Node.js and run:" -ForegroundColor Gray
        Write-Host "    npm install -g @anthropic-ai/claude-code" -ForegroundColor Gray
        Write-Host "    claude login" -ForegroundColor Gray
        return $false
    }
}

Write-Host "Checking requirements..." -ForegroundColor Cyan
if (-not (Test-Python)) {
    exit 1
}
Test-Claude
Write-Host ""

# Remove old installation
if (Test-Path $InstallDir) {
    Write-Host "Removing old installation..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $InstallDir
}

# Download and extract
Write-Host "Downloading CSES Agent..." -ForegroundColor Cyan
$ZipFile = "$env:TEMP\cses_agentic.zip"
Invoke-WebRequest -Uri $RepoUrl -OutFile $ZipFile

Write-Host "Extracting..." -ForegroundColor Cyan
Expand-Archive -Path $ZipFile -DestinationPath $env:TEMP -Force
Move-Item "$env:TEMP\cses_agentic-main" $InstallDir
Remove-Item $ZipFile

# Create virtual environment
Write-Host "Setting up Python environment..." -ForegroundColor Cyan
Set-Location $InstallDir
& $script:PythonCmd -m venv .venv

# Activate and install
Write-Host "Installing dependencies (this may take a few minutes)..." -ForegroundColor Cyan
& "$InstallDir\.venv\Scripts\python.exe" -m pip install --upgrade pip 2>&1 | Out-Null
& "$InstallDir\.venv\Scripts\python.exe" -m pip install -r requirements.txt

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

# Copy default .env if not exists
if (-not (Test-Path "$InstallDir\.env")) {
    if (Test-Path "$InstallDir\.env.example") {
        Copy-Item "$InstallDir\.env.example" "$InstallDir\.env"
    }
}

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║           Installation Complete!                             ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "To get started:" -ForegroundColor Cyan
Write-Host "  1. CLOSE and REOPEN your terminal/PowerShell" -ForegroundColor White
Write-Host "  2. Navigate to a folder with collaborator files" -ForegroundColor White
Write-Host "  3. Run: cses" -ForegroundColor White
Write-Host ""
Write-Host "Configuration file:" -ForegroundColor Cyan
Write-Host "  $InstallDir\.env" -ForegroundColor White
Write-Host ""
Write-Host "To configure API keys, edit the .env file or run:" -ForegroundColor Cyan
Write-Host "  notepad $InstallDir\.env" -ForegroundColor White
Write-Host ""
