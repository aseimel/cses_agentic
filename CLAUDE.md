# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

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

1. **NEVER write format-specific document parsing code**
   - NO regex patterns to parse collaborator documents
   - NO assumptions about document structure
   - Pass FULL document text to LLM and let it interpret semantically

2. **Keep solutions simple**
   - Don't over-engineer
   - Don't add features beyond what's requested

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
