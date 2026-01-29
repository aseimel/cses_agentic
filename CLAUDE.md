# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## CRITICAL - READ FIRST

**EXAMPLES ARE JUST EXAMPLES - NEVER HARDCODE FOR SPECIFIC COUNTRIES OR FILES**

When the user mentions a specific country (Korea, Germany, etc.) or specific folder (SouthKorea_2024, etc.), this is ONLY an example to illustrate the requirement. The code must:
- Work for ANY country, ANY year, ANY folder structure
- NEVER contain hardcoded country names, paths, or country-specific logic
- NEVER treat one country differently from another
- Use generic patterns that work universally

If you find yourself writing code that mentions a specific country name, STOP and make it generic.

## Project Overview

CLI tool for harmonizing election study data to a standardized schema using LLM-powered variable matching.

## Architecture

```
src/
├── agent/           # Claude validation logic
│   ├── validator.py # Dual-model validation
│   └── cses_agent.py
├── ingest/          # Data and document loading
│   ├── data_loader.py    # .dta, .sav, .csv, .xlsx
│   ├── doc_parser.py     # .docx, .pdf, .txt
│   └── context_extractor.py
├── matching/        # LLM variable matching
│   └── llm_matcher.py
├── workflow/        # Workflow state and execution
│   ├── state.py     # WorkflowState persistence
│   ├── organizer.py # File detection/organization
│   └── steps.py     # Step execution
└── preprocessing/   # Document preprocessing
    └── document_aggregator.py

cses_cli.py          # Main CLI entry point
install.ps1          # Windows installer
```

## Critical Rules

1. **ALWAYS TEST CODE BEFORE RETURNING TO USER**
   - NEVER return untested code - this is strictly forbidden
   - Run the code locally and verify it works
   - Test the full flow, not just individual functions
   - If you cannot test, explicitly tell the user and explain why
   - Broken code wastes the user's time and destroys trust

2. **FIX THE ENTIRE PROCESS, NOT JUST SYMPTOMS**
   - "Immediate fixes" or workarounds are strictly forbidden
   - When something is broken, fix the root cause in the proper place
   - The install script, update command, and CLI must all work correctly
   - Never tell the user to run manual commands as a workaround
   - If the install process is broken, fix the install process

3. **NEVER write format-specific document parsing code**
   - NO regex patterns to parse collaborator documents
   - NO assumptions about document structure
   - Pass FULL document text to LLM and let it interpret semantically

4. **Keep solutions simple**
   - Don't over-engineer
   - Don't add features beyond what's requested

5. **ALWAYS provide user feedback during operations**
   - Users are non-technical and need to know the tool is working
   - Print progress messages before any operation that takes time
   - Examples: "Loading data file...", "Connecting to Claude...", "Processing documents..."
   - Never leave users waiting with no output - they will think the tool is frozen
   - NO emojis (Windows cp1252 encoding compatibility)

6. **NEVER allow skipping workflow steps**
   - Every step in the CSES workflow is required
   - Do NOT implement "skip" options for steps
   - Users can either proceed with a step or provide custom instructions
   - This is similar to Claude Code plan mode: approve or provide feedback

## Key Files

| File | Purpose |
|------|---------|
| `cses_cli.py` | Main CLI entry point |
| `src/workflow/state.py` | Workflow state management |
| `src/workflow/organizer.py` | File detection and organization |
| `src/matching/llm_matcher.py` | LLM variable matching engine |
| `src/agent/validator.py` | Claude validation logic |

## Technology Stack

| Component | Technology |
|-----------|------------|
| LLM Provider | LiteLLM (unified API) |
| Data | Polars, Pandas |
| CLI | argparse |
| File formats | pyreadstat, openpyxl, pypdf, python-docx |

## Workflow Behavior

When `cses` is run in a folder with deposited files:
1. Files are detected and copied to `micro/original_deposit/`
2. Source files are automatically removed from root after copying
3. Log file and collaborator questions file are created in `micro/`

This keeps the root folder clean - all originals are preserved in `original_deposit/`.

## Development

```bash
# Activate environment
source .venv/bin/activate

# Run CLI
python cses_cli.py --help

# Run in a folder with data files
cd /path/to/study/files
cses
```
