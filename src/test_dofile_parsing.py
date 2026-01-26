#!/usr/bin/env python3
"""
Test Do-File Parsing - No External Dependencies

This script parses the historical do-file to extract expert mapping decisions.
Shows what the expert actually did - this is our ground truth for testing AI proposals.
"""

import re
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path("/home/armin/cses_agentic/repo")
AUSTRALIA_DOFILE = REPO_ROOT / "CSES Data Products/CSES Standalone Modules/Module 6/Election Studies/Australia_2022/micro/cses-m6_micro_AUS_2022_20240110.do"


def extract_gen_statements(content: str) -> list[dict]:
    """Extract all gen/replace statements from do-file."""
    patterns = [
        # gen F1004 = "AUS_2022"
        r'gen\s+(?:str\d*\s+)?(?:long\s+)?(?:double\s+)?(F\d+[_\w]*)\s*=\s*(.+?)(?:\n|//|$)',
        # replace F1004 = value if condition
        r'replace\s+(F\d+[_\w]*)\s*=\s*(.+?)(?:\s+if\s+|$)',
        # tostring var, gen(F1003_2)
        r'tostring\s+(\w+),\s*gen\((F\d+[_\w]*)\)',
    ]

    statements = []
    for pattern in patterns:
        for match in re.finditer(pattern, content, re.MULTILINE):
            groups = match.groups()
            if len(groups) == 2:
                if 'tostring' in pattern:
                    statements.append({
                        "target": groups[1],
                        "source": groups[0],
                        "type": "tostring",
                        "raw": match.group(0)[:100]
                    })
                else:
                    statements.append({
                        "target": groups[0],
                        "expression": groups[1].strip(),
                        "type": "gen" if 'gen' in match.group(0) else "replace",
                        "raw": match.group(0)[:100]
                    })

    return statements


def identify_source_variables(expression: str) -> list[str]:
    """Identify source variable names in an expression."""
    # Exclude CSES variables (F1xxx, F2xxx, etc.) and common Stata functions/keywords
    exclude = {'if', 'in', 'replace', 'gen', 'string', 'substr', 'real', 'missing',
               'inrange', 'format', 'mdy', 'round', 'minutes'}

    # Find word tokens
    words = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', expression)

    sources = []
    for word in words:
        # Skip F variables (our targets)
        if re.match(r'^F\d{4}', word):
            continue
        # Skip keywords and functions
        if word.lower() in exclude:
            continue
        # Skip pure numbers
        if word.isdigit():
            continue
        sources.append(word)

    return list(set(sources))


def analyze_variable_mappings(dofile_path: Path) -> dict:
    """Analyze the do-file to extract mapping patterns."""

    with open(dofile_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Extract all gen/replace statements
    statements = extract_gen_statements(content)

    # Group by target variable
    by_target = defaultdict(list)
    for stmt in statements:
        by_target[stmt["target"]].append(stmt)

    # Analyze each target variable
    mappings = {}
    for target, stmts in by_target.items():
        # Find source variables used
        sources = set()
        for stmt in stmts:
            if "source" in stmt:
                sources.add(stmt["source"])
            elif "expression" in stmt:
                sources.update(identify_source_variables(stmt["expression"]))

        # Determine mapping type
        if not sources:
            mapping_type = "constant"
        elif len(sources) == 1:
            mapping_type = "direct"
        else:
            mapping_type = "computed"

        # Check if it's a recode (multiple replace statements)
        replace_count = sum(1 for s in stmts if s["type"] == "replace")
        if replace_count > 3:
            mapping_type = "recode"

        mappings[target] = {
            "sources": list(sources),
            "type": mapping_type,
            "n_statements": len(stmts),
            "first_statement": stmts[0]["raw"] if stmts else None
        }

    return mappings


def categorize_by_variable_type(mappings: dict) -> dict:
    """Categorize mappings by CSES variable type."""
    categories = {
        "F1XXX_admin": [],      # ID, weights, administration
        "F2XXX_demo": [],       # Demographics
        "F3XXX_survey": [],     # Survey questions
        "F4XXX_district": [],   # District data
        "F5XXX_macro": [],      # Macro data
        "other": []
    }

    for var, info in mappings.items():
        if var.startswith("F1"):
            categories["F1XXX_admin"].append((var, info))
        elif var.startswith("F2"):
            categories["F2XXX_demo"].append((var, info))
        elif var.startswith("F3"):
            categories["F3XXX_survey"].append((var, info))
        elif var.startswith("F4"):
            categories["F4XXX_district"].append((var, info))
        elif var.startswith("F5"):
            categories["F5XXX_macro"].append((var, info))
        else:
            categories["other"].append((var, info))

    return categories


def print_analysis(mappings: dict, categories: dict):
    """Print detailed analysis."""

    print("=" * 80)
    print("CSES VARIABLE MAPPING ANALYSIS - Australia 2022")
    print("=" * 80)

    # Overall stats
    type_counts = defaultdict(int)
    for var, info in mappings.items():
        type_counts[info["type"]] += 1

    print(f"\nTotal CSES variables processed: {len(mappings)}")
    print(f"\nMapping types:")
    for mtype, count in sorted(type_counts.items()):
        print(f"  {mtype:15} {count:4} ({100*count/len(mappings):.1f}%)")

    # By category
    print("\n" + "-" * 80)
    print("BY VARIABLE CATEGORY")
    print("-" * 80)

    for cat_name, vars in categories.items():
        if not vars:
            continue
        print(f"\n{cat_name}: {len(vars)} variables")

        # Show sample
        for var, info in vars[:5]:
            sources_str = ", ".join(info["sources"][:3]) if info["sources"] else "[constant]"
            if len(info["sources"]) > 3:
                sources_str += "..."
            print(f"  {var:15} <- {sources_str:30} ({info['type']})")

        if len(vars) > 5:
            print(f"  ... and {len(vars) - 5} more")

    # Key findings for automation
    print("\n" + "=" * 80)
    print("KEY FINDINGS FOR AUTOMATION")
    print("=" * 80)

    # Find direct mappings (easiest to automate)
    direct = [(v, i) for v, i in mappings.items() if i["type"] == "direct" and i["sources"]]
    print(f"\n1. DIRECT MAPPINGS (easiest - 1 source variable): {len(direct)}")
    for var, info in direct[:10]:
        print(f"   {var:15} <- {info['sources'][0]}")

    # Find recodes (need value mapping)
    recodes = [(v, i) for v, i in mappings.items() if i["type"] == "recode"]
    print(f"\n2. RECODES (need value mapping): {len(recodes)}")
    for var, info in recodes[:10]:
        sources_str = ", ".join(info["sources"][:2]) if info["sources"] else "[unknown]"
        print(f"   {var:15} <- {sources_str} ({info['n_statements']} statements)")

    # Find constants (trivial)
    constants = [(v, i) for v, i in mappings.items() if i["type"] == "constant"]
    print(f"\n3. CONSTANTS (trivial): {len(constants)}")

    # Find computed (complex)
    computed = [(v, i) for v, i in mappings.items() if i["type"] == "computed"]
    print(f"\n4. COMPUTED (complex - multiple sources): {len(computed)}")
    for var, info in computed[:5]:
        sources_str = ", ".join(info["sources"][:3])
        print(f"   {var:15} <- {sources_str}")


def find_source_variable_usage(mappings: dict) -> dict:
    """Find which source variables are used most often."""
    usage = defaultdict(list)
    for var, info in mappings.items():
        for source in info["sources"]:
            usage[source].append(var)

    # Sort by frequency
    return dict(sorted(usage.items(), key=lambda x: len(x[1]), reverse=True))


def main():
    if not AUSTRALIA_DOFILE.exists():
        print(f"ERROR: Do-file not found: {AUSTRALIA_DOFILE}")
        return

    # Analyze
    mappings = analyze_variable_mappings(AUSTRALIA_DOFILE)
    categories = categorize_by_variable_type(mappings)

    # Print analysis
    print_analysis(mappings, categories)

    # Source variable usage
    usage = find_source_variable_usage(mappings)
    print("\n" + "-" * 80)
    print("MOST USED SOURCE VARIABLES (from deposited data)")
    print("-" * 80)
    for source, targets in list(usage.items())[:15]:
        print(f"  {source:25} -> {len(targets)} CSES vars: {', '.join(targets[:5])}")

    print("\n" + "=" * 80)
    print("NEXT STEP")
    print("=" * 80)
    print("""
Once you have the Python environment set up:

1. Run: python src/test_variable_mapping.py
   - This loads the actual deposited data
   - Shows variable types, sample values, labels

2. Then: python src/test_llm_proposals.py  (to be created)
   - Uses LLM to propose mappings
   - Compares to expert decisions
   - Measures accuracy

The goal: Can AI propose mappings that match what the expert did?
If accuracy is high, the human-in-loop approach is viable.
""")


if __name__ == "__main__":
    main()
