# CSES Data Harmonization Agent - Setup Guide

## Quick Start

```bash
# 1. Activate the environment
source activate_cses.sh

# 2. Configure API keys (edit .env file)
nano .env

# 3. Test the CLI
cses --help

# 4. Go to collaborator files and start
cd /path/to/collaborator/files
cses
```

## Environment Setup (Already Done)

The environment is set up at `.venv/` with all dependencies installed.

### To Reinstall/Update

```bash
# Recreate virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Configuration

Edit `.env` file with your API keys:

```bash
# Required: At least one LLM provider
OPENAI_API_KEY=sk-...         # For GPT models (matching)

# Model configuration
LLM_MODEL_MATCH=openai/gpt-4o

# Validation model - choose ONE option:
# Option 1: Use Claude CLI with your Max subscription (recommended)
LLM_MODEL_VALIDATE=claude-cli

# Option 2: Use Anthropic API key
# ANTHROPIC_API_KEY=sk-ant-...
# LLM_MODEL_VALIDATE=anthropic/claude-sonnet-4-20250514
```

### Claude CLI Setup (for Max subscribers)

If you have a Claude Max subscription, you can use it for validation without an API key:

```bash
# Install Claude Code CLI
npm install -g @anthropic-ai/claude-code

# Login with your Max subscription
claude login

# Set in .env
LLM_MODEL_VALIDATE=claude-cli
```

## Usage

### CLI Mode (Recommended)

```bash
# Navigate to a folder with collaborator files
cd /path/to/collaborator/deposit/

# Start interactive mode
cses

# Or use specific commands
cses init                 # Initialize study
cses status               # Show progress
cses step 7               # Work on step 7
cses match                # Run variable matching
cses export               # Export mappings
```

## Folder Structure After Initialization

All files in ONE folder with clear, standardized names:

```
KOR_2024/
├── KOR_2024_original_data.dta        # Original data from collaborator
├── KOR_2024_questionnaire.pdf        # Questionnaire
├── KOR_2024_codebook.docx            # Codebook (if provided)
├── KOR_2024_design_report.pdf        # Design report (if provided)
│
│   --- Generated during processing ---
├── KOR_2024_processing.do            # Stata do file
├── KOR_2024_variable_mappings.xlsx   # Variable mappings
├── KOR_2024_tracking_sheet.xlsx      # CSES tracking sheet
├── KOR_2024_processing_log.txt       # Processing log with notes
├── KOR_2024_frequencies.txt          # Frequency tables
├── KOR_2024_M6.dta                   # Final harmonized dataset
│
└── .cses/                            # Agent state (hidden)
    └── state.json
```

## 16-Step Workflow

| Step | Name | Automated |
|------|------|-----------|
| 0 | Set Up Country Folder | Yes |
| 1 | Check Completeness of Deposit | Yes |
| 2 | Read Design Report | Partial |
| 3 | Fill Variable Tracking Sheet | Yes |
| 4 | Write Study Design Overview | Yes |
| 5 | Request Election Results | Manual |
| 6 | Run Frequencies | Yes |
| 7 | Process Variables (Matching) | Yes |
| 8 | Complete Documentation | Yes |
| 9 | Collect District Data | Manual |
| 10 | Update Label Files | Yes |
| 11 | Finish Processing | Yes |
| 12 | Run Check Files | Yes |
| 13 | Write Collaborator Questions | Yes |
| 14 | Follow Up Questions | Manual |
| 15 | Transfer ESNs | Yes |
| 16 | Final Deposit | Yes |

## Troubleshooting

### Import Errors
```bash
# Make sure you've activated the environment
source activate_cses.sh
```

### LLM API Errors
```bash
# Check your API keys in .env
cat .env | grep API_KEY
```

### Missing Files
```bash
# List detected files in current folder
cses files
```
