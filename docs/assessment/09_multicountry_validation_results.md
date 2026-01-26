# Multi-Country LLM Variable Mapping Validation

## Executive Summary

**Final Accuracy: 98%+ across 6 diverse countries**

Testing validates that LLM-assisted variable mapping achieves production-quality accuracy when:
1. Correct source data files are loaded (not processed/combined files)
2. Full CSES codebook is provided as context (RAG)
3. Few-shot examples demonstrate matching patterns
4. Chain-of-thought reasoning guides the process

## Test Results

### Initial Test (6 Countries, 150 mappings)

| Country | Year | Accuracy | Correct/Total |
|---------|------|----------|---------------|
| Australia | 2022 | 100% | 25/25 |
| Brazil | 2022 | 96% | 24/25 |
| Denmark | 2022 | 100% | 25/25 |
| France | 2022 | 96% | 24/25 |
| Portugal | 2022 | 96% | 24/25 |
| Portugal | 2024 | 100% | 25/25 |
| **Overall** | | **98%** | **147/150** |

### Error Analysis (3 errors out of 150)

1. **Brazil F2018**: Proposed `D17_REGION` vs expected `D17`
   - Type: Partial match (same semantic content)
   - Reason: Multiple similar variables exist

2. **France F2021**: Proposed `unknown` vs expected `eayy_c1_rec`
   - Type: Complex compound variable name
   - Reason: Naming convention doesn't match any pattern

3. **Portugal 2022 F2010_1**: Proposed `unknown` vs expected `D09`
   - Type: Income variable
   - Reason: Less clear semantic match

## Key Success Factors

### 1. Correct Data File Selection

The biggest improvement came from loading the **original deposited data** instead of processed/combined files:

| File Selection | Brazil Accuracy | Denmark Accuracy |
|----------------|-----------------|------------------|
| Wrong file (Combined) | 0% | 0% |
| Correct file (Original) | 96% | 100% |

Implementation:
- Penalize files with "combined", "processed", "update" in name
- Prefer files with "CSES", "module" in name
- Sort by score, then prefer .dta over .sav

### 2. Full Codebook Context (RAG)

Providing the official CSES Module 6 codebook with question codes (Q01, D04, etc.) enables pattern matching:

```
F2004    >>> D04     MARITAL STATUS OR CIVIL UNION STATUS
F3001    >>> Q01     POLITICAL INTEREST
F3002_1  >>> Q02a    MEDIA USAGE: PUBLIC TV NEWS
```

### 3. Few-Shot Examples

Demonstrating both pattern AND semantic matching:

```
Example: dem1 → F2004 (MARITAL STATUS)
Reasoning: "dem" prefix indicates demographic,
description asks about marital status
```

### 4. Chain-of-Thought Reasoning

Guiding the LLM through systematic matching:
1. Look at target's question code (Q01, D04)
2. Find source variables with matching patterns
3. Verify by checking semantic meaning
4. Assign confidence level

## Challenges Handled

### Variable Naming Inconsistency

Each country uses different conventions:

| Country | Demographic | Survey | Example |
|---------|-------------|--------|---------|
| Australia | dem1, ses | q01, q02a | dem1 → marital |
| Brazil | D04, D05 | Q01, Q02a | D04 → marital |
| France | eayy_a4 | fes4_Q01 | Complex prefixes |
| Denmark | D01b, D02 | Q01 | Direct CSES codes |

### Data File Variations

- Multiple files per country (updates, combined, original)
- Different formats (.dta, .sav)
- Nested folder structures

### Missing Data

Some countries have incomplete data files or partial extracts.
System correctly identifies and handles these cases.

## Comparison to Initial Tests

| Version | Australia | Overall | Key Change |
|---------|-----------|---------|------------|
| v1 | 0% | 0% | Wrong ground truth extraction |
| v2 | 55% | 55% | Fixed extraction, basic prompt |
| v3 | 85% | 85% | Enhanced semantic descriptions |
| Multi-country | 100% | **98%** | Full codebook + correct files |

## Recommendations for Production

### 1. Data File Selection
- Implement robust file scoring algorithm
- Validate file contains expected variables before processing
- Log which file was selected for audit trail

### 2. Confidence Thresholding
- High confidence (pattern + semantic match): Auto-approve
- Medium confidence: Flag for quick review
- Low confidence: Require manual mapping

### 3. Error Handling
- Partial matches (D17 vs D17_REGION) should be flagged, not rejected
- Complex variable names may need manual attention
- Log all "unknown" mappings for review

### 4. Incremental Learning
- Store approved mappings as examples for similar countries
- Build country-specific pattern libraries
- Track mapping decisions for future reference

## Conclusion

**98% accuracy validates the human-in-the-loop approach for production use.**

With proper data file selection and full codebook context, the LLM can:
- Correctly match 98% of variable mappings across diverse countries
- Handle different naming conventions (dem1, D04, eayy_a4)
- Match by both pattern (q01→Q01) and semantic meaning

The remaining 2% of errors are edge cases that require human judgment, which aligns perfectly with the human-in-the-loop design.
