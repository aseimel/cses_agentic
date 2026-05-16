# CSES Assistant

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
irm https://raw.githubusercontent.com/aseimel/cses_agentic/main/install.ps1 | iex
```

Or download `install.ps1` and run:
```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

### Manual Install

```powershell
# Clone repository
git clone https://github.com/aseimel/cses_agentic.git
cd cses_agentic

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run
python cses_cli.py

# Open the Windows GUI
python cses_cli.py gui
```

## Usage

1. **Navigate to a folder with collaborator files**:
   ```
   cd C:\Users\YourName\Downloads\Korea_2024_deposit
   ```

2. **Start the CLI or GUI**:
   ```
   cses
   cses gui
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

Use the Windows GUI Settings tab, or edit `~/.cses-agent/.env`, to configure:

```bash
# LiteLLM provider keys
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
XAI_API_KEY=your-key-here
GEMINI_API_KEY=your-key-here

# Curated model profile selected in the GUI
CSES_MODEL_PROFILE=gesis_recommended

# Optional: main chat dropdown models selected in the GUI
CSES_CHAT_MODELS=xai/grok-4-1-fast,openai/gpt-4.1
CSES_CHAT_MODEL=xai/grok-4-1-fast

# Optional OpenWebUI/OpenAI-compatible service.
# Leave disabled for direct vendor APIs.
CSES_USE_OPENWEBUI=false
# OPENAI_API_BASE=https://your-openwebui.example.org/api/v1

# Stata executable path
STATA_PATH=C:\Program Files\Stata18\StataMP-64.exe
```

Stata execution runs through the app's bundled Stata bridge. Select the Stata
executable in the GUI Settings tab, or set `STATA_PATH` in `~/.cses-agent/.env`.

## Project Context

Each study can define plain-text context files that the chat assistant reads:

```text
agent.md                         # assistant role and project boundaries
workflow.md                      # project-specific workflow notes
.agents/skills/<skill>/SKILL.md  # reusable project-specific instructions
```

The GUI's Project Context tab can create starter files for non-technical users.

## Commands

| Command | Description |
|---------|-------------|
| `cses` | Start interactive mode |
| `cses init` | Initialize study from files |
| `cses status` | Show workflow progress |
| `cses match` | Run variable matching |
| `cses export` | Export mappings |
| `cses gui` | Open the Windows GUI |
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
