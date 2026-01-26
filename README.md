# CSES Data Harmonization Agent

A command-line tool that helps process CSES (Comparative Study of Electoral Systems) Module 6 election studies. It guides you through the 16-step workflow with LLM-assisted variable matching.

## Features

- **Automatic file detection**: Detects data files, questionnaires, and codebooks
- **Standardized naming**: All files renamed with `COUNTRY_YEAR_` prefix
- **Variable matching**: LLM-powered matching of source variables to CSES schema
- **Dual-model validation**: Cross-checks mappings using Claude for accuracy
- **Generates all outputs**: Stata .do files, tracking sheets, processing logs

## Installation (Windows)

### Prerequisites

1. **Python 3.10+** - Download from [python.org](https://www.python.org/downloads/)
   - **IMPORTANT**: Check "Add Python to PATH" during installation!

2. **Claude CLI** (optional, for Claude Max subscribers):
   - Install [Node.js](https://nodejs.org/)
   - Run: `npm install -g @anthropic-ai/claude-code`
   - Run: `claude login`

### One-Line Install

Open **PowerShell** and run:

```powershell
irm https://raw.githubusercontent.com/YOUR_ORG/cses_agentic/main/install.ps1 | iex
```

Or download `install.ps1` and run:
```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

### Manual Install

```powershell
# Clone repository
git clone https://github.com/YOUR_ORG/cses_agentic.git
cd cses_agentic

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run
python cses_cli.py
```

## Usage

1. **Navigate to a folder with collaborator files**:
   ```
   cd C:\Users\YourName\Downloads\Korea_2024_deposit
   ```

2. **Start the CLI**:
   ```
   cses
   ```

3. **Follow the prompts** to:
   - Detect and organize files
   - Run variable matching
   - Review and approve mappings
   - Generate outputs

## Generated Output Structure

```
KOR_2024/
├── KOR_2024_original_data.dta        # Original data from collaborator
├── KOR_2024_questionnaire.pdf        # Questionnaire
├── KOR_2024_codebook.docx            # Codebook
│
│   --- Generated during processing ---
├── KOR_2024_processing.do            # Stata do file
├── KOR_2024_variable_mappings.xlsx   # Variable mappings
├── KOR_2024_tracking_sheet.xlsx      # CSES tracking sheet
├── KOR_2024_processing_log.txt       # Processing log
├── KOR_2024_frequencies.txt          # Frequency tables
├── KOR_2024_M6.dta                   # Final harmonized dataset
```

## Configuration

Edit `~/.cses-agent/.env` to configure:

```bash
# LLM for variable matching (via GESIS API)
LLM_MODEL_MATCH=openai/gpt-oss:120b
OPENAI_API_KEY=your-key-here
OPENAI_API_BASE=https://your-api-endpoint

# Validation model
# Option 1: Use Claude CLI with Max subscription (no API key needed)
LLM_MODEL_VALIDATE=claude-cli

# Option 2: Use Anthropic API
# LLM_MODEL_VALIDATE=anthropic/claude-sonnet-4-20250514
# ANTHROPIC_API_KEY=sk-ant-...
```

## Commands

| Command | Description |
|---------|-------------|
| `cses` | Start interactive mode |
| `cses init` | Initialize study from files |
| `cses status` | Show workflow progress |
| `cses match` | Run variable matching |
| `cses export` | Export mappings |
| `cses --help` | Show all commands |

## Troubleshooting

### "cses is not recognized"

Close and reopen your terminal/PowerShell after installation.

### Python not found

Make sure Python is installed and added to PATH. Reinstall Python and check "Add Python to PATH".

### Permission errors

The installer doesn't require admin privileges. Everything is installed to your user folder (`~/.cses-agent`).

## Support

For issues, contact your system administrator or open an issue on GitHub.
