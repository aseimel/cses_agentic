# Workflow Documentation Analysis

## Overview

The CSES micro-processing workflow consists of **16 steps** that transform raw national election study data into harmonized, cross-nationally comparable datasets. This document analyzes each step for automation potential.

## Required Input Files

### From Country Collaborators (via Email folder)
- Data deposit (raw survey data, typically .dta or .sav)
- Design Report (survey methodology summary)
- Macro Report (election context, electoral system details)
- Questionnaires (original language + English translation)
- Optional: Dataset codebooks, collaborator notes

### From CSES Templates (z) COUNTRY_YEAR_M6/)
- Main processing do-file template
- Log file template
- Variable tracking sheet template
- Label files (micro-var, micro-val, Vote_Choice)
- Check files (inconsistency, theoretical, validation)

## Output Files

- Processed dataset (.dta format)
- Country-specific do-file (edited from template)
- Check results (.smcl log files)
- Completed log file (processing decisions documented)
- Collaborator questions document
- Updated codebook entries (Election Study Notes)

## 16-Step Workflow Analysis

### Phase 1: Setup (Steps 0-3)

| Step | Name | Inputs | Outputs | Human Judgment |
|------|------|--------|---------|----------------|
| 0 | Set Up Country Folder | z) COUNTRY_YEAR_M6 template | Local working folder | LOW - folder copy/rename |
| 1 | Check Completeness of Deposit | Email folder contents | Release tracking sheet update | MEDIUM - verify documentation |
| 2 | Read Design Report | Design report PDF/DOCX | Notes in log file | HIGH - understand study design |
| 3 | Fill Variable Tracking Sheet | Deposited data + questionnaire | Variable tracking Excel | MEDIUM - match variables |

**Step 0 Details:**
- Copy `z) COUNTRY_YEAR_M6` template to local drive
- Rename to `[COUNTRY]_[YEAR]` (e.g., Germany_2025)
- Copy deposited data files to micro subfolder

**Step 1 Details:**
- Verify all required documents received
- Update release tracking sheet on Dropbox
- Mark: data deposit, design report, macro report, questionnaires

**Step 2 Details:**
- Review survey methodology
- Verify CSES eligibility criteria met
- Note questions about design in log file
- Check against Study Eligibility Checklist

**Step 3 Details:**
- Check each CSES variable against deposited data
- Mark: deposited, missing, unusually coded
- Flag missing F3XXX variables (survey questions)
- Notify project manager of significant gaps

### Phase 2: Processing (Steps 4-11)

| Step | Name | Inputs | Outputs | Human Judgment |
|------|------|--------|---------|----------------|
| 4 | Write Study Design Overview | Design Report | Log file documentation | MEDIUM - summarize design |
| 5 | Request Election Results | - | Email to macro coder | LOW - standard request |
| 6 | Run Frequencies | Deposited data | Original frequency log | LOW - automated |
| 7 | Process Variables | Template + data | Variable gen/recode commands | HIGH - mapping decisions |
| 8 | Complete Documentation | Processing decisions | Log file, codebook entries | HIGH - expert documentation |
| 9 | Integrate District Data | External sources + data | Merged dataset | MEDIUM - data sourcing |
| 10 | Update Label Files | Party coding | Label do-files | LOW - follow pattern |
| 11 | Finish Processing | All processed variables | Final dataset | MEDIUM - final assembly |

**Step 6 Details (Highly Automatable):**
```stata
log using "cses-m6_org-freq_CNT_YEAR_DATE.smcl", replace
foreach var of varlist _all {
  display "Variable `var':"
  tab `var', mis
}
log close
```

**Step 7 Details (Core Processing - Variable by Variable):**
For each CSES variable (F1001, F1004, F2001, F3001, etc.):
1. Find matching variable in deposited data
2. Write `gen` command to create CSES variable
3. Write `recode` commands to map values
4. Run `tab` to verify results
5. Document decisions in log file

**Step 9 Details (District Data):**
- If study includes district IDs: collect official election results by district
- If nationwide district: process national results
- If no district IDs: ask collaborator question about availability

**Step 11 Details:**
- Drop all original (non-CSES) variables
- Apply label files via `do` commands
- Run frequencies on processed data
- Save as `cses-m6_[COUNTRY]_[YEAR].dta`

### Phase 3: Quality Assurance (Steps 12-14)

| Step | Name | Inputs | Outputs | Human Judgment |
|------|------|--------|---------|----------------|
| 12 | Run Check Files | Processed dataset | Check log files | LOW - automated checks |
| 13 | Write Collaborator Questions | Check results + issues | Questions document | HIGH - formulate questions |
| 14 | Follow Up on Responses | Collaborator answers | Updated syntax/documentation | HIGH - interpret answers |

**Step 12 Check Types:**
1. **Inconsistency Checks:** Logical violations (e.g., non-voters reporting vote choice)
2. **Theoretical Checks:** Unexpected patterns (e.g., all parties rated same on scale)
3. **Validation Checks:** Interviewer/interview validation

### Phase 4: Finalization (Steps 15-16)

| Step | Name | Inputs | Outputs | Human Judgment |
|------|------|--------|---------|----------------|
| 15 | Transfer ESNs to Codebook | Log file ESNs | Updated codebook | LOW - copy/paste |
| 16 | Final Deposit | Processed dataset + docs | Cross-national phase files | LOW - file transfer |

**Step 16 Details:**
- Copy final dataset to designated Dropbox location
- Provide design report and questionnaires
- Email project manager with:
  - Final N (sample size)
  - Vetting questions (if any)
  - Confirmation of completion

## Key Rules

1. **Never modify shared label files** - used across all countries
2. **Always date filenames** - version control
3. **Document everything** - record all decisions in log file
4. **Template-first approach** - always start from latest template

## Dependencies Between Steps

```
Step 0 → Step 1 → Step 2 → Step 3
                    ↓
Step 4 ←――――――――――――┘
   ↓
Step 5 (parallel) → Step 6 → Step 7 → Step 8 → Step 9 → Step 10 → Step 11
                                                                      ↓
Step 12 ←―――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――┘
   ↓
Step 13 → Step 14 → Step 15 → Step 16
```

## Quality Control Checkpoints

| Checkpoint | Location | Description |
|------------|----------|-------------|
| Eligibility | Step 2 | Verify study meets CSES standards |
| Deposit completeness | Step 1 | All required documents received |
| Variable coverage | Step 3 | Required variables present |
| Processing accuracy | Step 7 | Tab after each variable |
| Consistency | Step 12 | Automated check files |
| Collaborator validation | Step 14 | Expert review of questions |
| Final N | Step 16 | Sample size verification |

## Pain Points Identified

1. **Manual variable mapping** (Step 7) - most time-consuming
2. **Occupation coding** - complex many-to-many mapping (e.g., ANZSCO to ISCO)
3. **Party code assignment** - requires coordination with macro coder
4. **District data collection** - often requires external web scraping
5. **Documentation overhead** - extensive log file entries required
6. **Collaborator question turnaround** - wait time varies

## Current Tools Used

- **Stata** - All data processing and checks
- **Excel** - Variable tracking sheets
- **Word** - Log files, collaborator questions
- **Dropbox** - File sharing and version control
- **Email** - Collaborator communication
- **Web calculators** - Date calculations (timeanddate.com)

## Estimated Time Per Study (Current Manual Process)

| Phase | Time Estimate | Notes |
|-------|---------------|-------|
| Setup (0-3) | 1-2 days | Depends on documentation quality |
| Processing (4-11) | 1-2 weeks | Most variable by study complexity |
| QA (12-14) | 2-3 days | Includes collaborator wait time |
| Finalization (15-16) | 0.5-1 day | Straightforward |
| **Total** | **2-4 weeks** | Varies significantly by study |
