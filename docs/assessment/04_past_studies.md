# Historical Study Analysis

## Overview

Analysis of completed country studies to understand processing patterns, common issues, and edge cases that automation must handle.

## Sampled Studies

1. **Australia 2022** - Completed, well-documented
2. **Switzerland 2023** - Complex multi-language study
3. **Portugal 2022/2024** - Multiple studies from same country

## Case Study: Australia 2022

### Files Provided by Collaborator

| File Type | Received | Notes |
|-----------|----------|-------|
| Data deposit (.dta) | Yes | Updated version received Sept 2024 |
| Design Report | Yes | Complete |
| Macro Report | Yes | Complete |
| Questionnaires | Yes | English version |
| Technical Report | Yes | Life in Australia panel details |

### Processing Timeline
- **Initial deposit:** July 24, 2022
- **Processing started:** June 12, 2024 (by Klara Dentler)
- **Updated data:** September 10, 2024
- **Processing file date:** January 10, 2024

### Issues Encountered (from do-file analysis)

#### 1. Ineligible Respondents
```stata
fre p_citizen
keep if p_citizen == 1 // deletes 287 observations!
```
**Resolution:** Dropped 287 non-citizens (not eligible for voting)

#### 2. Study Context Discrepancy
```stata
gen F1013 = 2 // Different to M5 which was coded "1"!
```
**Resolution:** Confirmed via collaborator that survey was stand-alone (not part of AES)

#### 3. Mode of Interview Changes
```stata
* ! Change to the scale applied on July 05, 2024 based on KBK's email
gen F1015_1 = 4  // Internet
gen F1015_2 = 2  // Telephone
```
**Resolution:** Scale definition changed during processing - required code update

#### 4. Occupation Coding Complexity
The Australia do-file contains **170+ lines** of occupation mapping:
```stata
replace F2007 = 11 if dem3_coded == 111111  // Chief Executive or Managing Director
replace F2007 = 11 if dem3_coded == 111112  // General Manager
... (extensive mapping)
replace F2007 = 999 if dem3_coded == 312913  // Mine Deputy - UNMAPPED
```

Unmapped occupations requiring ESN (Election Study Notes):
- Mine Deputy
- Diversional Therapist
- Massage Therapist
- Parole or Probation Officer
- Residential Care Officer
- Contract/Program Administrators
- Conveyancer
- Court and Legal Clerks
- Clinical Coder
- Car Detailer
- Fencer
- Deck Hand
- Road Traffic Controller

#### 5. Variables Not Collected (by design)
```stata
gen F2013 = 999  // Race: CQ - not allowed to be asked
gen F2014 = 999  // Ethnicity: CQ - missing
```

#### 6. Income Quintile Distribution Challenge
```stata
replace F2010_1 = 1 if d09 == 1 | d09 == 2 | d09 == 3 // 20.48%
replace F2010_1 = 2 if d09 == 4 | d09 == 5 // 19.67%
replace F2010_1 = 3 if d09 == 6  // 14.26%  <-- Uneven distribution
replace F2010_1 = 4 if d09 == 7 | d09 == 8 | d09 == 9 // 24.08%
replace F2010_1 = 5 if d09 == 10 | d09 == 11 | d09 == 12 | d09 == 13 | d09 == 14 // 21.51%
```
**Note:** "A better distribution was based on the original variable, unfortunately, not possible!"

### Collaborator Questions Generated

**File:** `CollaboratorQuestions_20240624.docx`

Typical question categories:
1. Missing variable clarifications
2. Unexpected value distributions
3. Eligibility criteria confirmations
4. Scale/coding clarifications
5. Sample design questions

**Vetting Questions:**
- `VettingQuestion_20241015.docx` - Final review questions before release

### Final Output Statistics

- **Final N:** 3,269 respondents (after dropping non-citizens)
- **Original N:** 3,556 survey completions

## Common Patterns Across Studies

### Data Quality Issues Frequency

| Issue Type | Frequency | Automation Impact |
|------------|-----------|-------------------|
| Missing variables | Common | Medium - need to flag |
| Ineligible respondents | Occasional | Medium - filtering rules |
| Occupation mapping gaps | Very Common | High - requires review |
| Income distribution issues | Common | Medium - quintile calc |
| Missing demographics | Occasional | Low - standard handling |
| Date inconsistencies | Occasional | Low - validation rules |

### Documentation Patterns

1. **Election Study Notes (ESNs):** Required for any non-standard coding
2. **Log file comments:** Explain all processing decisions
3. **Collaborator communication:** Preserved in Emails folder

### Processing Decision Types

| Decision Type | LLM-Automatable | Human Required |
|---------------|-----------------|----------------|
| Standard recoding | Yes | No |
| Missing value handling | Yes | Review edge cases |
| Date calculations | Yes | No |
| Party code assignment | Partially | Ordering decisions |
| Occupation mapping | Partially | Unmapped codes |
| ESN drafting | Yes | Review/approval |
| Collaborator questions | Yes | Phrasing review |

## Edge Cases Requiring Human Judgment

### 1. Country-Specific Legal Restrictions
- Australia: Cannot ask race/ethnicity questions
- Some countries: Cannot ask about religion
- Privacy laws vary by jurisdiction

### 2. Electoral System Variations
- Preferential voting (Australia)
- Mixed-member proportional (Germany, New Zealand)
- Two-round systems (France, Brazil)
- Compulsory voting (Australia, Belgium)

### 3. Multi-Language Studies
- Switzerland: German, French, Italian questionnaire versions
- Belgium: Dutch, French versions
- Canada: English, French versions

### 4. Panel vs Cross-Sectional Designs
- Some studies use pre-existing panels (Australia's Life in Australia)
- Others are fresh samples
- Affects eligibility checking and weighting

### 5. COVID-19 Pandemic Effects
```stata
** 0. ELECTION/STUDY CONDUCTED ENTIRELY AFTER COVID-19 PANDEMIC (MAY 5, 2023)
** 1. ELECTION/STUDY CONDUCTED DURING & AFTER COVID-19 PANDEMIC
** 2. ELECTION/STUDY CONDUCTED ENTIRELY DURING COVID-19 PANDEMIC
gen F1012_2 = 2  // Australia 2022
```

## Insights for Automation

### High-Value Automation Targets

1. **Template population:** ~60% of variables are constant or formulaic
2. **Frequency generation:** 100% automatable (Step 6)
3. **Check file execution:** 100% automatable (Step 12)
4. **Standard missing value handling:** 90%+ automatable
5. **Date calculations:** 100% automatable
6. **String formatting:** 100% automatable

### Areas Requiring LLM Assistance

1. **Variable mapping:** LLM can propose, human validates
2. **ESN drafting:** LLM generates, human reviews
3. **Collaborator question drafting:** LLM proposes, human refines
4. **Occupation code suggestions:** LLM matches patterns

### Areas Requiring Human Expertise

1. **Eligibility assessment:** Study design review
2. **Unmapped occupation codes:** Domain expertise needed
3. **Complex recode decisions:** Context-dependent
4. **Collaborator response interpretation:** Nuanced communication
5. **Final quality sign-off:** Expert validation

## Processing Time Observations

Based on file dates and complexity:

| Study Type | Estimated Time | Notes |
|------------|----------------|-------|
| Simple study | 1-2 weeks | Standard electoral system, good documentation |
| Complex study | 3-4 weeks | Multi-language, unusual electoral system |
| Problem study | 4-6 weeks | Many collaborator questions, data issues |

### Time Distribution by Step

| Phase | Current % | With Automation % |
|-------|-----------|-------------------|
| Setup (0-3) | 15% | 5% |
| Processing (4-11) | 60% | 30% |
| QA (12-14) | 20% | 15% |
| Finalization (15-16) | 5% | 5% |

**Target:** Reduce total time from 2-4 weeks to 3-5 days through automation.
