# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## !!! ABSOLUTE PROHIBITION - READ THIS FIRST !!!

**NEVER WRITE FORMAT-SPECIFIC DOCUMENT PARSING CODE. EVER.**

This means:
- NO regex patterns to parse collaborator documents (codebooks, questionnaires, etc.)
- NO assumptions about delimiters like `>>>`, `---`, `###`, or ANY specific markers
- NO assumptions about variable naming conventions in collaborator files
- NO assumptions about document structure, sections, or formatting
- NO "extracting" structured data from collaborator documents with pattern matching

**WHY:** Collaborator files come from 50+ countries in dozens of formats. They are MESSY, INCONSISTENT, and UNPREDICTABLE. Any format-specific parsing WILL FAIL on other countries.

**WHAT TO DO INSTEAD:**
1. Pass the FULL raw document text to the LLM
2. Let the LLM understand and interpret the content semantically
3. The LLM handles diverse formats - that's the whole point of using an LLM

**IF YOU CATCH YOURSELF writing regex to parse collaborator document content - STOP IMMEDIATELY.**

This rule has been violated multiple times. It must NEVER happen again.

---

## Project Status

**Phase:** LLM Variable Mapping Validated - Building Human-in-the-Loop UI

This project aims to automate the CSES (Comparative Study of Electoral Systems) data harmonization workflow, reducing processing time from weeks to days per country study.

### LLM Mapping Test Results (Multi-Country Validation)

| Metric | Result |
|--------|--------|
| Overall Accuracy | **98%** (147/150) |
| Countries Tested | Australia, Brazil, Denmark, France, Portugal x2 |
| Error Rate | 3 errors out of 150 mappings |
| **Verdict** | Production-quality for human-in-the-loop |

Key success factors:
1. Correct data file selection (original vs processed)
2. Full CSES codebook as RAG context
3. Few-shot examples + chain-of-thought reasoning

See `docs/assessment/09_multicountry_validation_results.md` for full analysis.

## Critical Rules

1. **NEVER modify files in `/repo` subdirectory** - This is READ ONLY historical data
2. **All new code goes in `/src`, `/tests`, `/docs`, `/config`, `/outputs` only**
3. **The `workflow.md` file is the primary reference** for understanding the 16-step processing workflow
4. **RAW DATA IS ONLY IN E-MAIL FOLDERS** - For testing, use ONLY data files from e-mail deposit folders (e.g., `repo/.../Election Studies/COUNTRY/a) e-mail/`). Do NOT use processed data from other locations.
5. **NEVER ASSUME SPECIFIC DOCUMENT FORMATS** - Collaborator files are messy and highly diverse. Do NOT write code that assumes:
   - Specific codebook formats (like `>>>` delimiters)
   - Specific variable naming conventions
   - Specific document structures
   - Any particular pattern in collaborator-supplied files
   The LLM must handle raw, unstructured input. Pass full document text to the LLM and let IT figure out the structure. No regex patterns for parsing collaborator documents.

## Repository Structure

```
/home/armin/cses_agentic/
├── repo/                    # READ ONLY - Historical CSES data (25,913 files)
│   └── CSES Data Products/
│       └── CSES Standalone Modules/
│           └── Module 6/    # Current active module
│               ├── Election Studies/  # 27 country datasets
│               ├── CSES6 AR3/        # Current archive release
│               └── Macro/Micro/      # Processing resources
├── src/                     # New Python code
│   ├── ingest/              # Format-agnostic data/document loading
│   │   ├── data_loader.py   # .dta, .sav, .csv, .xlsx, etc.
│   │   ├── doc_parser.py    # .docx, .pdf, .txt, .rtf
│   │   └── context_extractor.py  # Adaptive context combination
│   ├── matching/            # LLM variable matching
│   │   └── llm_matcher.py   # Matching engine with confidence scores
│   ├── workflow/            # Workflow state and execution
│   │   ├── state.py         # WorkflowState persistence
│   │   ├── organizer.py     # File detection and organization
│   │   └── steps.py         # Step execution logic
│   └── agent/               # Claude validation agent
│       ├── validator.py     # Dual-model validation
│       └── cses_agent.py    # Main agent logic
├── tests/                   # Test files
├── docs/
│   └── assessment/          # Feasibility assessment documents
├── config/                  # Configuration files
├── outputs/                 # Generated files (gitignored)
├── cses_cli.py              # Main CLI entry point
├── workflow.md              # 16-STEP WORKFLOW REFERENCE
└── knowledge_base.md        # Guidelines and FAQs
```

## Key Reference Files

| File | Purpose | Priority |
|------|---------|----------|
| `workflow.md` | 16-step micro-processing workflow | **Critical** |
| `knowledge_base.md` | FAQs and guidelines | High |
| `repo/.../z) COUNTRY_YEAR_M6/` | Template folder | High |
| `docs/assessment/*.md` | Feasibility analysis | Reference |

## 16-Step Workflow Summary

The CSES micro-processing workflow transforms raw national election study data into harmonized datasets:

| Phase | Steps | Automation Potential |
|-------|-------|---------------------|
| Setup | 0-3: Folder setup, deposit check, design review, variable tracking | 70% |
| Processing | 4-11: Documentation, frequencies, variable processing, labels | 65% |
| QA | 12-14: Check files, collaborator questions, follow-up | 75% |
| Finalization | 15-16: Codebook transfer, final deposit | 90% |

## Variable Naming Schema

- **F1XXX:** ID, weight, and administration variables
- **F2XXX:** Demographic variables (age, gender, education, occupation)
- **F3XXX:** Survey variables (vote choice, party ID, attitudes)
- **F4XXX:** District-level variables
- **F5XXX:** Macro-level aggregate data

## Stata to Python/Polars Patterns

Common transformations to implement:

```python
# Pattern 1: Simple assignment
df = df.with_columns(pl.lit("CSES-MODULE-6").alias("F1001"))

# Pattern 2: Conditional recoding
df = df.with_columns(
    pl.when(pl.col("gender") == 1).then(0)
      .when(pl.col("gender") == 2).then(1)
      .otherwise(9)
      .alias("F2002")
)

# Pattern 3: Missing value standardization
# 7/97/997 = Refused, 8/98/998 = Don't know, 9/99/999 = Missing
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Orchestration | Prefect 3.0 | Workflow management |
| Agents | LangGraph | Multi-step LLM reasoning |
| LLM Provider | LiteLLM | Unified API (Claude/DeepSeek) |
| UI | CLI (cses) | Interactive terminal interface |
| RAG | Haystack | Knowledge retrieval |
| Data | Polars | Fast DataFrame operations |
| Vector DB | Qdrant | Semantic search |
| Checkpoints | PostgreSQL | State persistence |

## Automation Feasibility Summary

| Step | Automation Level | LLM Required |
|------|------------------|--------------|
| 6. Run Frequencies | HIGH (98%) | No |
| 12. Run Checks | HIGH (95%) | No |
| 0, 5, 10, 11, 15, 16 | HIGH (85-95%) | No |
| 1, 3, 4, 7, 8, 13 | MEDIUM (55-70%) | Yes |
| 2, 9, 14 | LOW (30-40%) | Yes |

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start CLI (the main workflow application)
source activate_cses.sh
cd /path/to/collaborator/files
cses

# Run Prefect flow (future)
python -m prefect deployment run 'study-processing/default'
```

## Key Insights from Assessment

1. **Template-driven processing:** All studies follow `z) COUNTRY_YEAR_M6/` template
2. **Main complexity in Step 7:** Variable processing requires semantic understanding
3. **Occupation mapping is hardest:** 170+ lines of many-to-many mappings
4. **Well-structured Stata code:** Section markers (\\\\\\, >>>) aid parsing
5. **Good documentation:** Log files and ESNs capture processing decisions

## Success Metrics

| Metric | Target |
|--------|--------|
| Automation rate | 70-80% of tasks |
| Time reduction | 75% (weeks → days) |
| Accuracy | 90%+ match with expert decisions |
| User satisfaction | Non-programmers can operate |

## Assessment Documents

Detailed analysis available in `docs/assessment/`:
1. `01_repo_structure.md` - Repository analysis
2. `02_workflow_understanding.md` - 16-step workflow details
3. `03_stata_patterns.md` - Stata→Python transformation patterns
4. `04_past_studies.md` - Historical processing analysis
5. `05_feasibility_matrix.md` - Step-by-step automation assessment
6. `06_data_quality.md` - Data quality evaluation
7. `07_technical_design.md` - Architecture proposal
8. `08_llm_mapping_test_results.md` - Initial LLM test (single country, 85% accuracy)
9. `09_multicountry_validation_results.md` - Multi-country validation (98% accuracy)

## Current Architecture

**Purpose:** Map collaborator-provided national election study data to the CSES Module 6 schema.

**Inputs (from collaborators):**
- **Data file** (required): The deposited dataset (.dta, .sav, .csv, .xlsx, etc.)
- **Questionnaire** (required): Native language survey document (per CSES policy)
- **Codebook** (optional): Collaborator's variable documentation

**Output:** Variable mappings from source → CSES schema (F2XXX demographics, F3XXX survey)

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   COLLABORATOR      │     │    CONTEXT       │     │      LLM        │     │  HUMAN REVIEW    │
│   UPLOADS           │────▶│  EXTRACTION      │────▶│    MATCHING     │────▶│    UI            │
│                     │     │                  │     │                 │     │                  │
│ - Data file (req)   │     │ - DataLoader     │     │ - Built-in CSES │     │ - Confidence     │
│ - Questionnaire(req)│     │ - DocParser      │     │   Module 6 spec │     │ - Approve/Reject │
│ - Codebook (opt)    │     │ - Adaptive merge │     │ - Few-shot      │     │ - Edit mapping   │
└─────────────────────┘     └──────────────────┘     └─────────────────┘     └──────────────────┘
                                                                                   │
                                                                                   ▼
                                                                         ┌──────────────────┐
                                                                         │     EXPORT       │
                                                                         │ source → F2XXX   │
                                                                         │ source → F3XXX   │
                                                                         └──────────────────┘
```

## Next Steps

1. **Done:** CLI-based workflow with dual-model validation (original matcher + Claude)
2. **Next:** Test CLI with real country data and iterate on usability
3. **Then:** Add recode/transformation handling (beyond simple mappings)
4. **Future:** Extend to all 16 workflow steps
