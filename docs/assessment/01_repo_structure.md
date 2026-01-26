# Repository Structure Analysis

## Overview

The CSES repository contains **25,913 files** across **6,495 directories**, representing a comprehensive research data management system for the Comparative Study of Electoral Systems project.

## File Counts by Extension

| Extension | Count | Purpose |
|-----------|-------|---------|
| `.do` | 3,698 | Stata processing scripts |
| `.dta` | 1,889 | Stata data files |
| `.docx` | 3,964 | Documentation, log files, reports |
| `.xlsx` | 855 | Variable tracking, templates, data tables |
| `.pdf` | 3,599 | Research papers, technical reports |
| `.smcl` | 949 | Stata log output files |

## Directory Hierarchy

```
/repo/
├── CSES Bibliography/                           # Research articles and references
├── CSES Communication/                          # Website, presentations, awards
├── CSES Data Products/                          # [PRIMARY] Core data processing
│   ├── CSES IMD/                                # Integrated Module Dataset
│   │   ├── CSES IMD Phase 1-5/
│   │   ├── CSES IMD Processing Templates/
│   │   └── CSES IMD 3 Cross-Checks/
│   └── CSES Standalone Modules/
│       ├── Module 5/                            # Previous module (completed)
│       ├── Module 6/                            # [ACTIVE] Current module
│       │   ├── CSES6 AR1/                       # Archive Release 1
│       │   ├── CSES6 AR2/                       # Archive Release 2
│       │   ├── CSES6 AR3/                       # Archive Release 3 (current)
│       │   ├── Election Studies/                # 27 country datasets
│       │   ├── Macro/                           # Macro-level data
│       │   └── Micro/                           # Micro processing resources
│       └── Module 7/                            # Future module
├── CSES Guidelines & Policies/                  # Processing rules and standards
├── CSES Trainings/                              # Training materials (2014-2026)
└── CSES Event and Staff Calendar/               # Project management tracking
```

## Key Folders for Automation

### 1. Template Folder (Critical)
**Path:** `Module 6/Election Studies/z) COUNTRY_YEAR_M6/`

Contains all processing templates:
```
z) COUNTRY_YEAR_M6/
├── micro/
│   ├── cses-m6_micro_CNT_YEAR_DATE.do          # Main processing template (134KB)
│   ├── cses-m6_log-file_CNT_YEAR_DATE.doc      # Log file template
│   ├── deposited variables-m6_CNT_YEAR_DATE.xlsx
│   ├── data_checks/
│   │   ├── cses-m6_inc-check_CNT_YEAR.do       # Inconsistency checks
│   │   ├── cses-m6_the-check_CNT_YEAR.do       # Theoretical checks
│   │   └── cses-m6_validation-check_CNT_YEAR.do
│   └── labels/
│       ├── cses-m6_micro-var-labels.do
│       ├── cses-m6_micro-val-labels.do
│       └── cses-m6_Vote_Choice_Labels.do
├── macro/
├── election results/
└── E-mails/
```

### 2. Active Election Studies (Module 6)
**Path:** `Module 6/Election Studies/`

27 country studies currently in processing:
- Argentina_2023, Australia_2022, Austria_2024
- Brazil_2022, Denmark_2022, France_2022
- Germany_2025, Great Britain_2024, Mexico_2021
- Montenegro_2023, New Zealand_2023, North Macedonia_2024
- Philippines_2022, Poland_2023, Portugal_2022, Portugal_2024
- Slovakia_2023, Slovenia_2022, SouthKorea_2024
- Spain_2023, Sweden_2022, Switzerland_2023
- Taiwan_2024, Thailand_2023, Turkiye_2023
- United States_2024

### 3. Label Files (Cross-National)
**Path:** `Module 6/CSES6 AR3/Labels/`

Shared label files that MUST NOT be modified during individual study processing.

### 4. Guidelines & Policies
**Path:** `CSES Guidelines & Policies/`

- Study Eligibility Checklist
- Demographic Data Classification Schemes
- Party/Coalition/Leader Classification Schemes
- District Data Guidelines
- Variable and Value Label Update Procedures

## Per-Country Folder Structure

Each `[COUNTRY]_[YEAR]/` folder contains:

| Subfolder | Contents |
|-----------|----------|
| `micro/` | Processing do-files, labels, data checks |
| `macro/` | Macro-level processing files |
| `election results/` | Election results table template |
| `E-mails/` | Collaborator communication, data deposits |
| `collaborator questions/` | Questions sent to country collaborators |
| `District Data/` | District-level data files |
| `References/` | Papers and sources used |

## Naming Conventions

### File Naming
- **Do-files:** `cses-m6_micro_[CNT]_[YEAR]_[YYYYMMDD].do`
- **Log files:** `cses-m6_log-file_[CNT]_[YEAR]_[YYYYMMDD].doc`
- **Data files:** `cses-m6_[COUNTRY]_[YEAR].dta`
- **Check files:** `cses-m6_[check-type]_[CNT]_[YEAR].do`

### Variable Naming
- **F1XXX:** ID, weight, and administration variables
- **F2XXX:** Demographic variables
- **F3XXX:** Survey variables (vote choice, party ID, attitudes)
- **F4XXX:** District-level variables
- **F5XXX:** Macro-level variables

### Party Code Format
6-digit codes: `[3-digit UN country code][1-digit study number][2-digit party code]`
- Example: `03602022` = Australia (036) + first study (0) + party 22

## Organizational Patterns

1. **Versioning:** Files are dated in filename (YYYYMMDD format)
2. **Backup:** `old/` subfolders contain previous versions
3. **Templates:** `z)` prefix indicates template folders (sorts last alphabetically)
4. **Exclusions:** `z) INELIGIBLE & EXCLUDED` for rejected studies
5. **Phases:** AR1, AR2, AR3 represent successive archive releases

## Critical Observations for Automation

1. **Template-driven:** All processing follows templates from `z) COUNTRY_YEAR_M6/`
2. **Standardized structure:** Country folders follow consistent organization
3. **Documentation alongside code:** Log files track all processing decisions
4. **Quality checks integrated:** Three types of automated checks
5. **Cross-national consistency:** Shared label files ensure harmonization
6. **Communication trails:** Email folders preserve collaborator interactions

## Files Requiring RAG/Indexing

Priority files for knowledge base:
1. `workflow.md` - 16-step processing workflow
2. `knowledge_base.md` - FAQs and guidelines
3. Main processing template (134KB do-file)
4. Check file templates (3 files)
5. Variable tracking sheet template
6. Study eligibility checklist
7. Classification scheme documents
