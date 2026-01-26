"""
Test Variable Mapping - Human-in-the-Loop Prototype

This script tests whether an LLM can propose reasonable variable mappings
that a domain expert would approve.

Test approach:
1. Load deposited data (raw from collaborator)
2. Load historical do-file (what expert actually coded)
3. Extract what mappings the expert made
4. Have LLM propose mappings
5. Compare: Would expert approve the AI proposal?
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass

# Will need: pip install polars pyreadstat
try:
    import polars as pl
except ImportError:
    print("Run: pip install polars pyreadstat")
    exit(1)

# Paths
REPO_ROOT = Path("/home/armin/cses_agentic/repo")
AUSTRALIA_PATH = REPO_ROOT / "CSES Data Products/CSES Standalone Modules/Module 6/Election Studies/Australia_2022"

# Deposited data (raw from collaborator)
DEPOSITED_DATA = AUSTRALIA_PATH / "E-mails/CSES 2022 AUstralia 10 Sep 24.dta"

# Historical do-file (what expert actually coded)
HISTORICAL_DOFILE = AUSTRALIA_PATH / "micro/cses-m6_micro_AUS_2022_20240110.do"

# CSES variable schema (from workflow.md and template)
CSES_VARIABLE_SCHEMA = {
    # F1XXX - ID, Weight, Administration
    "F1001": {"desc": "Dataset identifier", "type": "string", "example": "CSES-MODULE-6"},
    "F1002_VER": {"desc": "Dataset version", "type": "string", "example": "VER2024-MMM-DD"},
    "F1003_1": {"desc": "Respondent ID (full)", "type": "string", "width": 18},
    "F1003_2": {"desc": "Respondent ID within study", "type": "string", "width": 10},
    "F1004": {"desc": "Election study ID (alpha)", "type": "string", "example": "AUS_2022"},
    "F1005": {"desc": "Election study ID (numeric)", "type": "long", "example": 3602022},
    "F1006": {"desc": "Polity CSES code", "type": "string", "width": 4},
    "F1006_UN": {"desc": "UN country code", "type": "numeric", "example": 36},
    "F1009": {"desc": "Election year", "type": "numeric", "example": 2022},
    "F1010_M": {"desc": "Election date - month", "type": "numeric", "range": "1-12"},
    "F1010_D": {"desc": "Election date - day", "type": "numeric", "range": "1-31"},
    "F1010_Y": {"desc": "Election date - year", "type": "numeric"},
    "F1012_1": {"desc": "Study timing", "type": "numeric", "codes": {1: "post", 2: "pre+post", 3: "between"}},
    "F1014": {"desc": "Election type", "type": "numeric", "codes": {10: "parliamentary", 20: "presidential"}},
    "F1015_1": {"desc": "Mode of interview (study)", "type": "numeric"},
    "F1016_1": {"desc": "Mode of interview (respondent)", "type": "numeric"},
    "F1019_M": {"desc": "Interview date - month", "type": "numeric"},
    "F1019_D": {"desc": "Interview date - day", "type": "numeric"},
    "F1019_Y": {"desc": "Interview date - year", "type": "numeric"},
    "F1021": {"desc": "Interview duration (minutes)", "type": "numeric"},
    "F1101_1": {"desc": "Sample weight", "type": "numeric"},

    # F2XXX - Demographics
    "F2001_Y": {"desc": "Year of birth", "type": "numeric"},
    "F2001_A": {"desc": "Age in years", "type": "numeric"},
    "F2002": {"desc": "Gender", "type": "numeric", "codes": {0: "male", 1: "female", 3: "other"}},
    "F2003": {"desc": "Education (ISCED)", "type": "numeric", "range": "1-9"},
    "F2004": {"desc": "Marital status", "type": "numeric"},
    "F2005": {"desc": "Union membership", "type": "numeric", "codes": {0: "no", 1: "yes"}},
    "F2006": {"desc": "Employment status", "type": "numeric"},
    "F2007": {"desc": "Occupation (ISCO)", "type": "numeric"},
    "F2008": {"desc": "Socioeconomic status", "type": "numeric"},
    "F2009": {"desc": "Employment type", "type": "numeric"},
    "F2010_1": {"desc": "Household income quintile", "type": "numeric", "range": "1-5"},
    "F2011": {"desc": "Religious denomination", "type": "numeric"},
    "F2012": {"desc": "Religious attendance", "type": "numeric"},
    "F2015": {"desc": "Country of birth", "type": "numeric"},
    "F2020": {"desc": "Urban/rural residence", "type": "numeric"},

    # F3XXX - Survey variables (subset)
    "F3001": {"desc": "Political interest", "type": "numeric", "range": "1-4"},
    "F3010": {"desc": "Turnout (main election)", "type": "numeric", "codes": {0: "no", 1: "yes"}},
    "F3010_LH": {"desc": "Turnout lower house", "type": "numeric"},
    "F3011_LH_PL": {"desc": "Vote choice lower house (party list)", "type": "numeric"},
    "F3020_R": {"desc": "Left-right self-placement", "type": "numeric", "range": "0-10"},
    "F3023_1": {"desc": "Close to any party", "type": "numeric"},
    "F3023_3": {"desc": "Party identification", "type": "numeric"},
}


@dataclass
class VariableInfo:
    """Information about a source variable."""
    name: str
    dtype: str
    n_unique: int
    n_missing: int
    n_total: int
    sample_values: list
    value_labels: dict | None = None


@dataclass
class ExpertMapping:
    """A mapping decision made by the expert in the historical do-file."""
    target_var: str  # CSES variable (e.g., F1004)
    source_var: str | None  # Original variable used (e.g., anu_id), None if constant
    mapping_type: str  # "constant", "direct", "recode", "computed"
    stata_code: str  # The actual Stata code used


def load_deposited_data(path: Path) -> tuple[pl.DataFrame, dict]:
    """Load deposited data and extract variable information."""
    print(f"Loading deposited data from: {path}")

    # Read with pyreadstat to get metadata
    import pyreadstat
    df_pd, meta = pyreadstat.read_dta(str(path))

    # Convert to polars
    df = pl.from_pandas(df_pd)

    # Extract variable info
    var_info = {}
    for col in df.columns:
        sample = df[col].drop_nulls().head(5).to_list()
        var_info[col] = VariableInfo(
            name=col,
            dtype=str(df[col].dtype),
            n_unique=df[col].n_unique(),
            n_missing=df[col].null_count(),
            n_total=len(df),
            sample_values=sample,
            value_labels=meta.variable_value_labels.get(col),
        )

    print(f"Loaded {len(df)} rows, {len(df.columns)} variables")
    return df, var_info


def extract_expert_mappings(dofile_path: Path) -> dict[str, ExpertMapping]:
    """
    Parse the historical do-file to extract what mappings the expert made.

    Returns dict mapping CSES variable name -> ExpertMapping
    """
    print(f"Parsing historical do-file: {dofile_path}")

    with open(dofile_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    mappings = {}

    # Pattern for variable sections: **>>> F1004 - description
    section_pattern = r'\*\*>>> (F\d+[_\w]*)\s+.*?\n(.*?)(?=\*\*>>>|\*{10,}\\\\\\|$)'

    for match in re.finditer(section_pattern, content, re.DOTALL):
        target_var = match.group(1)
        section_code = match.group(2).strip()

        # Skip empty sections
        if not section_code:
            continue

        # Determine mapping type and source variable
        mapping_type = "unknown"
        source_var = None

        # Check for constant assignment: gen F1004 = "AUS_2022"
        const_match = re.search(r'gen\s+(?:str\d*\s+)?' + re.escape(target_var) + r'\s*=\s*["\']?([^"\'\n]+)["\']?', section_code)
        if const_match:
            value = const_match.group(1).strip()
            # Check if it's a literal constant or references another variable
            if value.startswith('"') or value.replace('.','').isdigit() or value in ['1', '2', '0']:
                mapping_type = "constant"
            else:
                # References another variable
                source_var = value.split()[0] if ' ' in value else value
                mapping_type = "direct" if 'recode' not in section_code.lower() else "recode"

        # Check for variable reference: gen F2002 = gender or tostring var, gen(F1003_2)
        var_ref_match = re.search(r'(?:gen|replace)\s+' + re.escape(target_var) + r'\s*=\s*(\w+)', section_code)
        if var_ref_match and not source_var:
            potential_source = var_ref_match.group(1)
            if potential_source not in ['1', '2', '0', '96', '97', '98', '99', '9996', '9999']:
                source_var = potential_source
                mapping_type = "recode" if 'replace' in section_code else "direct"

        # Check for tostring pattern
        tostring_match = re.search(r'tostring\s+(\w+),\s*gen\(' + re.escape(target_var) + r'\)', section_code)
        if tostring_match:
            source_var = tostring_match.group(1)
            mapping_type = "direct"

        # Check for computed variables (using other F variables)
        if re.search(r'F\d{4}', section_code) and not source_var:
            mapping_type = "computed"

        mappings[target_var] = ExpertMapping(
            target_var=target_var,
            source_var=source_var,
            mapping_type=mapping_type,
            stata_code=section_code[:500]  # Truncate for readability
        )

    print(f"Extracted {len(mappings)} variable mappings from do-file")
    return mappings


def analyze_mapping_patterns(mappings: dict[str, ExpertMapping]) -> dict:
    """Analyze patterns in expert mappings."""
    stats = {
        "total": len(mappings),
        "by_type": {},
        "with_source_var": 0,
        "constants": 0,
        "computed": 0,
    }

    for var, mapping in mappings.items():
        # Count by type
        stats["by_type"][mapping.mapping_type] = stats["by_type"].get(mapping.mapping_type, 0) + 1

        if mapping.source_var:
            stats["with_source_var"] += 1
        if mapping.mapping_type == "constant":
            stats["constants"] += 1
        if mapping.mapping_type == "computed":
            stats["computed"] += 1

    return stats


def print_variable_summary(var_info: dict[str, VariableInfo], limit: int = 30):
    """Print summary of source variables."""
    print("\n" + "="*80)
    print("SOURCE VARIABLES (from deposited data)")
    print("="*80)

    for i, (name, info) in enumerate(list(var_info.items())[:limit]):
        labels_str = ""
        if info.value_labels:
            labels_str = f" | Labels: {dict(list(info.value_labels.items())[:3])}..."

        print(f"{name:20} | {info.dtype:10} | {info.n_unique:5} unique | "
              f"Sample: {info.sample_values[:3]}{labels_str}")

    if len(var_info) > limit:
        print(f"... and {len(var_info) - limit} more variables")


def print_mapping_summary(mappings: dict[str, ExpertMapping], limit: int = 30):
    """Print summary of expert mappings."""
    print("\n" + "="*80)
    print("EXPERT MAPPINGS (from historical do-file)")
    print("="*80)

    for i, (target, mapping) in enumerate(list(mappings.items())[:limit]):
        source_str = mapping.source_var if mapping.source_var else "[constant/computed]"
        print(f"{target:15} <- {source_str:20} | Type: {mapping.mapping_type}")

    if len(mappings) > limit:
        print(f"... and {len(mappings) - limit} more mappings")


def main():
    """Main test function."""
    print("="*80)
    print("CSES VARIABLE MAPPING TEST")
    print("Testing human-in-the-loop approach with Australia 2022")
    print("="*80)

    # Check files exist
    if not DEPOSITED_DATA.exists():
        print(f"ERROR: Deposited data not found: {DEPOSITED_DATA}")
        return

    if not HISTORICAL_DOFILE.exists():
        print(f"ERROR: Historical do-file not found: {HISTORICAL_DOFILE}")
        return

    # Step 1: Load deposited data
    df, var_info = load_deposited_data(DEPOSITED_DATA)
    print_variable_summary(var_info)

    # Step 2: Extract expert mappings from historical do-file
    mappings = extract_expert_mappings(HISTORICAL_DOFILE)
    print_mapping_summary(mappings)

    # Step 3: Analyze patterns
    stats = analyze_mapping_patterns(mappings)
    print("\n" + "="*80)
    print("MAPPING STATISTICS")
    print("="*80)
    print(f"Total mappings: {stats['total']}")
    print(f"With source variable: {stats['with_source_var']}")
    print(f"Constants: {stats['constants']}")
    print(f"Computed: {stats['computed']}")
    print(f"By type: {stats['by_type']}")

    # Step 4: Identify testable mappings (where we can verify AI proposal)
    testable = []
    for target, mapping in mappings.items():
        if mapping.source_var and mapping.source_var in var_info:
            testable.append((target, mapping))

    print(f"\nTestable mappings (source var exists in data): {len(testable)}")
    print("\nThese are the mappings where we can test if AI would propose the same:")
    for target, mapping in testable[:20]:
        source_info = var_info[mapping.source_var]
        print(f"  {target:15} <- {mapping.source_var:20} ({source_info.dtype}, {source_info.n_unique} unique)")

    # Save for next step
    print("\n" + "="*80)
    print("NEXT STEP: Test LLM mapping proposals")
    print("="*80)
    print("Run: python src/test_llm_proposals.py")
    print("This will use the LLM to propose mappings and compare to expert decisions.")

    return df, var_info, mappings, testable


if __name__ == "__main__":
    main()
