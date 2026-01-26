# Stata Code Pattern Extraction

## Overview

This document extracts the key Stata transformation patterns used in CSES data processing and provides Python/Polars equivalents for automation.

## Pattern 1: Variable Generation (gen)

### Stata Pattern
```stata
gen str13 F1001 = "CSES-MODULE-6"
gen str F1004 = "AUS_2022"
gen long F1005 = 03602022
gen F1009 = 2022
```

### Python/Polars Equivalent
```python
import polars as pl

df = df.with_columns([
    pl.lit("CSES-MODULE-6").alias("F1001"),
    pl.lit("AUS_2022").alias("F1004"),
    pl.lit(3602022).cast(pl.Int64).alias("F1005"),
    pl.lit(2022).alias("F1009"),
])
```

## Pattern 2: Conditional Assignment (replace with if)

### Stata Pattern
```stata
gen F1016_1 = 4 if mode == 1
replace F1016_1 = 2 if mode == 2
```

### Python/Polars Equivalent
```python
df = df.with_columns(
    pl.when(pl.col("mode") == 1).then(4)
      .when(pl.col("mode") == 2).then(2)
      .otherwise(None)
      .alias("F1016_1")
)
```

## Pattern 3: Value Recoding (recode)

### Stata Pattern
```stata
gen F2002 = 0 if gender == 1
replace F2002 = 1 if gender == 2
replace F2002 = 3 if gender == 3
replace F2002 = 7 if gender == -99  // Refused
replace F2002 = 8 if gender == -98  // Don't know
replace F2002 = 9 if gender == .    // Missing
```

### Python/Polars Equivalent
```python
gender_mapping = {
    1: 0,    # Male
    2: 1,    # Female
    3: 3,    # Other
    -99: 7,  # Refused
    -98: 8,  # Don't know
}

df = df.with_columns(
    pl.col("gender")
      .replace(gender_mapping)
      .fill_null(9)
      .alias("F2002")
)
```

## Pattern 4: String Manipulation (substr, tostring)

### Stata Pattern
```stata
tostring anu_id, gen(F1003_2) format(%010.0f)
gen F1006_UN = substr(F1006,1,3)
gen str F1003_1 = F1006 + "2022" + F1003_2
```

### Python/Polars Equivalent
```python
df = df.with_columns([
    pl.col("anu_id").cast(pl.Utf8).str.zfill(10).alias("F1003_2"),
    pl.col("F1006").str.slice(0, 3).alias("F1006_UN"),
])
df = df.with_columns(
    (pl.col("F1006") + "2022" + pl.col("F1003_2")).alias("F1003_1")
)
```

## Pattern 5: Date Calculations

### Stata Pattern
```stata
gen str2 month1 = string(F1010_M,"%02.0f")
gen str2 day1   = string(F1010_D,"%02.0f")
gen str4 year1  = string(F1010_Y,"%04.0f")
gen str F1010_1 = year1 + "-" + month1 + "-" + day1

generate interview_date = mdy(F1019_M,F1019_D,F1019_Y)
generate election_date_1 = mdy(F1010_M,F1010_D,F1010_Y)
gen F1020_1 = interview_date - election_date_1
```

### Python/Polars Equivalent
```python
import polars as pl

df = df.with_columns([
    pl.format("{:04d}-{:02d}-{:02d}",
              pl.col("F1010_Y"), pl.col("F1010_M"), pl.col("F1010_D"))
      .alias("F1010_1"),
    pl.date(pl.col("F1019_Y"), pl.col("F1019_M"), pl.col("F1019_D"))
      .alias("interview_date"),
    pl.date(pl.col("F1010_Y"), pl.col("F1010_M"), pl.col("F1010_D"))
      .alias("election_date"),
])

df = df.with_columns(
    (pl.col("interview_date") - pl.col("election_date"))
      .dt.total_days()
      .alias("F1020_1")
)
```

## Pattern 6: Generational Cohort Assignment

### Stata Pattern
```stata
gen F2001_GG = 1 if F2001_Y < 1928
replace F2001_GG = 0 if F2001_Y > 1927 & F2001_Y < 9997
replace F2001_GG = 9 if F2001_Y > 9996

gen F2001_GBB = 1 if F2001_Y > 1945 & F2001_Y < 1965
replace F2001_GBB = 0 if F2001_Y < 1946
replace F2001_GBB = 0 if F2001_Y > 1964 & F2001_Y < 9997
replace F2001_GBB = 9 if F2001_Y > 9996
```

### Python/Polars Equivalent
```python
GENERATIONS = {
    "F2001_GG": (None, 1928),      # Greatest Generation
    "F2001_GS": (1928, 1946),      # Silent Generation
    "F2001_GBB": (1946, 1965),     # Baby Boomers
    "F2001_GX": (1965, 1981),      # Generation X
    "F2001_GY": (1981, 1997),      # Generation Y
    "F2001_GZ": (1997, None),      # Generation Z
}

for gen_var, (start, end) in GENERATIONS.items():
    condition = pl.col("F2001_Y") < 9997
    if start:
        condition = condition & (pl.col("F2001_Y") >= start)
    if end:
        condition = condition & (pl.col("F2001_Y") < end)

    df = df.with_columns(
        pl.when(pl.col("F2001_Y") > 9996).then(9)
          .when(condition).then(1)
          .otherwise(0)
          .alias(gen_var)
    )
```

## Pattern 7: Missing Value Codes

### Stata Standard Missing Codes
| Code | Meaning |
|------|---------|
| 7/97/997/9997 | VOLUNTEERED: REFUSED |
| 8/98/998/9998 | VOLUNTEERED: DON'T KNOW |
| 9/99/999/9999 | MISSING |
| 95/96/995/996 | NOT APPLICABLE (various reasons) |

### Python Handling
```python
MISSING_CODES = {
    "refused": [7, 97, 997, 9997, 99999997],
    "dont_know": [8, 98, 998, 9998, 99999998],
    "missing": [9, 99, 999, 9999, 99999999],
    "not_applicable": [95, 96, 995, 996, 9995, 9996],
}

def standardize_missing(col_name: str, original_refused: int, original_dk: int):
    """Map original missing codes to CSES standard codes."""
    return (
        pl.when(pl.col(col_name) == original_refused).then(97)
          .when(pl.col(col_name) == original_dk).then(98)
          .when(pl.col(col_name).is_null()).then(99)
          .otherwise(pl.col(col_name))
    )
```

## Pattern 8: Occupation Code Mapping

### Stata Pattern (Extensive many-to-many mapping)
```stata
gen F2007 = .
replace F2007 = 11 if dem3_coded == 111111
replace F2007 = 11 if dem3_coded == 111112
replace F2007 = 11 if dem3_coded == 111211
replace F2007 = 11 if dem3_coded == 111212
replace F2007 = 11 if dem3_coded == 111311
... (hundreds of lines)
replace F2007 = 999 if dem3_coded == 312913  // Mine Deputy - UNMAPPED
```

### Python/Polars Equivalent
```python
# Load mapping table from external file/database
occupation_mapping = pl.read_csv("occupation_crosswalk.csv")
# Contains: source_code, target_code, notes

df = df.join(
    occupation_mapping.select(["source_code", "target_code"]),
    left_on="dem3_coded",
    right_on="source_code",
    how="left"
).with_columns(
    pl.col("target_code").fill_null(999).alias("F2007")  # 999 = unmapped
)
```

## Pattern 9: Party Code Variable Processing

### Stata Pattern
```stata
* Vote choice - party list
gen F3011_LH_PL = .
replace F3011_LH_PL = 036001 if vote_lower == 1  // Liberal Party
replace F3011_LH_PL = 036002 if vote_lower == 2  // Labor Party
replace F3011_LH_PL = 036003 if vote_lower == 3  // Nationals
replace F3011_LH_PL = 036004 if vote_lower == 4  // Greens
...
replace F3011_LH_PL = 999993 if vote_lower == 95 // Voted blank
replace F3011_LH_PL = 999995 if vote_lower == 96 // Did not vote
replace F3011_LH_PL = 999997 if vote_lower == -99 // Refused
replace F3011_LH_PL = 999998 if vote_lower == -98 // Don't know
replace F3011_LH_PL = 999999 if vote_lower == .  // Missing
```

### Python/Polars Equivalent
```python
def create_party_mapping(country_code: str, election_results: dict) -> dict:
    """Generate party code mapping based on election results order."""
    party_mapping = {}
    for rank, party_name in enumerate(election_results.keys(), 1):
        party_code = int(f"{country_code}{rank:02d}")
        party_mapping[party_name] = party_code

    # Add standard non-response codes
    party_mapping.update({
        "blank": 999993,
        "did_not_vote": 999995,
        "refused": 999997,
        "dont_know": 999998,
        "missing": 999999,
    })
    return party_mapping

# Apply mapping
df = df.with_columns(
    pl.col("vote_lower")
      .replace(party_code_mapping)
      .alias("F3011_LH_PL")
)
```

## Pattern 10: Consistency Check Logic

### Stata Pattern (Inconsistency Check)
```stata
** Non-Voters who report vote choice
tab F3011_PR_1 F3010_PR_1 if F3011_PR_1<999995 & F3010_PR_1==0

** Party Like/Dislike all same score
egen F3018n = rownonmiss (F3018_Am-F3018_Im)
gen F3018s = 1 if F3018_Am!=. & F3018_Am==F3018_Bm & F3018_Am==F3018_Cm ...
```

### Python/Polars Equivalent
```python
def check_nonvoter_vote_choice(df: pl.DataFrame) -> pl.DataFrame:
    """Flag non-voters who reported a vote choice."""
    return df.filter(
        (pl.col("F3011_PR_1") < 999995) & (pl.col("F3010_PR_1") == 0)
    ).select(["F1003_1", "F3010_PR_1", "F3011_PR_1"])

def check_same_party_ratings(df: pl.DataFrame, rating_cols: list) -> pl.DataFrame:
    """Flag respondents who gave identical ratings to all parties."""
    valid_ratings = [c for c in rating_cols]

    return df.with_columns([
        pl.concat_list(valid_ratings)
          .list.drop_nulls()
          .list.n_unique()
          .alias("unique_ratings")
    ]).filter(
        (pl.col("unique_ratings") == 1) &
        (pl.concat_list(valid_ratings).list.drop_nulls().list.len() > 2)
    )
```

## Top 10 Transformation Patterns Summary

| Rank | Pattern | Frequency | Automation Complexity |
|------|---------|-----------|----------------------|
| 1 | Simple value assignment | Very High | LOW |
| 2 | Conditional recoding | Very High | LOW |
| 3 | Missing value standardization | High | LOW |
| 4 | String concatenation | High | LOW |
| 5 | Date calculations | Medium | LOW |
| 6 | Generation/cohort assignment | Medium | LOW |
| 7 | Party code mapping | High | MEDIUM |
| 8 | Occupation code mapping | Low | HIGH |
| 9 | Consistency checks | Medium | LOW |
| 10 | Label application | High | LOW |

## Pattern Complexity Assessment

### LOW Complexity (Fully Automatable)
- Fixed constant assignments (country codes, dates, study identifiers)
- Standard missing value handling
- Date calculations
- Generation cohort assignment
- Frequency tabulations
- Duplicate checks

### MEDIUM Complexity (LLM-Assisted)
- Variable mapping from original to CSES (requires understanding variable names)
- Party code assignment (requires election results ordering)
- Income quintile calculation (requires distribution analysis)
- Region/district coding (requires external lookup)

### HIGH Complexity (Human Review Required)
- Occupation code mapping (many-to-many with exceptions)
- Education level assignment (country-specific ISCED mapping)
- Religion coding (country-specific denominations)
- Race/ethnicity coding (country-specific categories)
- Interpreting collaborator responses
- Resolving inconsistency check anomalies
