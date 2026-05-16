# CSES Assistant

CSES Assistant is a Windows-oriented, human-in-the-loop tool for processing
Comparative Study of Electoral Systems (CSES) Module 6 studies. The core
interface is a conversational assistant: the processor loads a study, chats with
the agent, reviews evidence and proposed coding decisions, and presses
`Proceed` to advance one workflow step at a time.

The goal is not unattended automation. The goal is to make the CSES processing
workflow reproducible, standards-backed, and reviewable while producing
professional Stata syntax, checks, documentation, and readiness reports.

## Current Capabilities

- Windows GUI with a central chat interface, `Load Study`, and one-step
  `Proceed` workflow control.
- CSES standards wiki in `cses_wiki/`, including questionnaire, syntax,
  documentation, validation, administrative, party, macro-context, and district
  patterns.
- OpenRouter-backed model routing for different workflow tasks, with
  cost-efficient defaults and role-specific settings in the GUI.
- Study material review that extracts eligibility, sample design, fieldwork,
  mode, weights, questionnaire coverage, and missing items before decisions are
  made.
- Registry-driven Module 6 matching:
  canonical CSES questionnaire item -> collaborator questionnaire item -> source
  data variable -> CSES target variable.
- Separate review gates for administrative information, demographics, party
  order, macro context, party metadata, district data, recoding plans, Stata
  execution, checks, documentation, and final readiness.
- Plan-driven Stata generation using CSES-style `**>>>` variable blocks,
  explicit missing-value handling, source-target checks, labels, final order,
  save, and `log close`.
- Stata execution through the package-owned MCP-Stata bridge rather than direct
  interactive Stata launches.
- Benchmark harnesses for Sweden and for signed-off example studies. These are
  development checks only; installed/runtime workflows must not depend on raw
  example-study folders.

## What This Tool Is Not

- It is not a menu replacement for the conversational processor workflow.
- It is not intended to process studies without human review.
- It must not contain country-specific runtime logic. Example studies are used
  to discover general workflow requirements and regression cases.
- It must not require `example_studies/`, `auto_macro/`, local Stata installs, or
  raw benchmark data to be committed to Git.

## Installation

### Requirements

- Windows
- Python 3.10 or newer
- Stata, if you want to run generated `.do` files
- An OpenRouter API key for AI-assisted workflow steps

### Manual Setup

```powershell
git clone https://github.com/aseimel/cses_agentic.git
cd cses_agentic

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run The GUI

```powershell
python -B cses_cli.py gui
```

You can also launch the GUI module directly during development:

```powershell
python -B src\gui_app.py
```

In the GUI Settings tab, add:

- OpenRouter API key
- Stata executable path, for example
  `C:\Program Files\Stata19\StataSE-64.exe`

Settings are stored under the user-level CSES settings folder. API keys and raw
study data should never be committed.

## Typical Workflow

1. Put a study deposit in a study folder.
2. Open the GUI.
3. Click `Load Study`.
   - If the folder has not been initialized, the app initializes it.
   - If it is already initialized, the app loads the existing study state.
4. Chat with the assistant about the current study.
5. Click `Proceed` to run exactly one workflow step.
6. Review and approve, edit, or reject proposed decisions.
7. Continue through matching, recoding, Stata execution, checks, documentation,
   and readiness.

The assistant should explain processor-facing decisions in CSES terms: sample
eligibility, sample type, fieldwork, mode, weights, CSES item coverage, missing
items, party order, district review, coding decisions, and final readiness.
Internal implementation terms should stay out of the normal GUI and chat.

## CLI Commands

The GUI is the preferred interface, but the CLI remains available for
development, testing, and scripted checks.

```powershell
python -B cses_cli.py --help
python -B cses_cli.py init
python -B cses_cli.py status
python -B cses_cli.py step 1
python -B cses_cli.py match
python -B cses_cli.py generate
python -B cses_cli.py benchmark
python -B cses_cli.py wiki audit
```

## Stata Execution

Generated Stata syntax is run through the bundled MCP-Stata integration:

- Python package dependency: `mcp-stata`
- Local wrapper: `src/stata_mcp.py`
- Isolated runner: `src/stata_mcp_runner.py`

Configure `STATA_PATH` in the GUI Settings tab or in the user settings file. The
workflow should use MCP-backed execution for Stata runs so the app can capture
results, parse failures, and avoid relying on manual interactive execution.

## CSES Standards Wiki

Runtime standards live in `cses_wiki/`. This is the installed knowledge source
for:

- workflow procedures
- CSES Module 6 schema
- canonical Module 6 questionnaire registry
- Stata syntax patterns
- documentation templates
- validation checks
- administrative, demographic, party, macro, and district rules

Development-only source material can be distilled into this wiki, but runtime
processing should depend on the checked-in wiki files rather than raw example
folders.

Useful maintenance command:

```powershell
python -B cses_cli.py wiki audit
```

## Benchmarks

Benchmarks are development tools. They help verify that the generic workflow can
handle real signed-off studies without hardcoding study-specific behavior.

List discovered final-reference studies:

```powershell
python -B scripts\benchmark_example_studies.py --list-only
```

Run one study:

```powershell
python -B scripts\benchmark_example_studies.py --study Sweden_2022 --stata-path "C:\Path\To\StataSE-64.exe"
```

Run the full final-reference corpus:

```powershell
python -B scripts\benchmark_example_studies.py --stata-path "C:\Path\To\StataSE-64.exe"
```

Benchmark outputs classify differences as workflow bugs, missing inputs,
reference-selection issues, legitimate processor judgments, or stricter
standard-backed divergences. A benchmark pass does not remove the need for
processor review in production.

## Repository Structure

```text
src/
  agent/          conversational assistant and workflow tools
  benchmark.py    replication benchmark services
  gui_app.py      Windows GUI
  workflow/       canonical workflow steps and state
  standards/      CSES standards, tracking, and validators
  matching/       registry-driven and ensemble matching
  preprocessing/  document, evidence, and study-material processing
  stata_mcp.py    MCP-Stata execution wrapper

cses_wiki/
  patterns/       schema, questionnaire registry, syntax, docs, checks
  procedures/     workflow standards
  topics/         CSES reference guidance
  retrieval/      generated retrieval index

scripts/
  benchmark_sweden_replication.py
  benchmark_example_studies.py
```

## Development Rules

- Keep the conversational chat workflow intact.
- Make changes generic across countries, years, and folder structures.
- Do not hardcode study names, country names, or example-study paths in runtime
  logic.
- Do not commit raw study data, benchmark working copies, Stata installs,
  `.cses` study state, or API keys.
- Stage explicit source, test, wiki, and documentation files only.
- Run relevant tests before committing.

## Quick Verification

Useful lightweight checks before committing:

```powershell
python -B -m py_compile cses_cli.py src\gui_app.py
python -B cses_cli.py --help
python -B scripts\benchmark_example_studies.py --list-only
git status --short
```
