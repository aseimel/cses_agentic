#!/usr/bin/env python3
"""
Test LLM Variable Mapping Proposals - V2 (Fixed)

Tests whether LLM can correctly match source variables to CSES targets.
Only tests ACTUAL variable mappings, not constants.
"""

import os
import json
import re
from pathlib import Path

# Load .env
env_file = Path("/home/armin/cses_agentic/.env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: No API key found")
    exit(1)

import polars as pl
import pyreadstat
from litellm import completion

# Paths
REPO_ROOT = Path("/home/armin/cses_agentic/repo")
AUSTRALIA_PATH = REPO_ROOT / "CSES Data Products/CSES Standalone Modules/Module 6/Election Studies/Australia_2022"
DEPOSITED_DATA = AUSTRALIA_PATH / "E-mails/CSES 2022 AUstralia 10 Sep 24.dta"
HISTORICAL_DOFILE = AUSTRALIA_PATH / "micro/cses-m6_micro_AUS_2022_20240110.do"


def extract_real_mappings(dofile_path: Path) -> list[tuple[str, str]]:
    """Extract ACTUAL variable mappings (not constants)."""
    content = dofile_path.read_text(errors='ignore')

    # Pattern: gen F3001 = q01 (source is lowercase variable name)
    pattern = r'gen\s+(?:str\d*\s+)?(F\d+[_\w]*)\s*=\s*([a-z][a-z0-9_]*)\b'

    mappings = []
    skip = {'if', 'string', 'substr', 'length', 'real', 'round', 'mdy', 'minutes',
            'year1', 'year2', 'month1', 'month2', 'day1', 'day2', 'interview_date',
            'election_date', 'election_date_1', 'election_date_2'}

    for match in re.finditer(pattern, content):
        target = match.group(1)
        source = match.group(2)
        if source not in skip:
            mappings.append((target, source))

    return mappings


def load_source_variables(data_path: Path) -> dict:
    """Load source variable info."""
    df_pd, meta = pyreadstat.read_dta(str(data_path))

    var_info = {}
    for col in df_pd.columns:
        sample = df_pd[col].dropna().head(5).tolist()
        labels = meta.variable_value_labels.get(col, {})
        desc = meta.column_names_to_labels.get(col, "")

        var_info[col] = {
            "sample": sample[:5],
            "labels": dict(list(labels.items())[:5]) if labels else None,
            "desc": desc[:80] if desc else ""
        }

    return var_info


# CSES Variable descriptions
CSES_VARS = """
F2004: Marital status (1=married, 2=widowed, 3=divorced, 4=single)
F2006: Employment status (1=full-time, 2=part-time, 3=unemployed, etc.)
F2008: Socioeconomic status (1=white collar, 2=worker, 3=farmer, etc.)
F2012: Religious attendance (1=never to 6=weekly)
F2018: Region of residence (country-specific codes)
F2019: Electoral district
F2021: Number in household

F3001: Political interest (1=very to 4=not at all)
F3002_1: Media - public TV (days/week 0-7)
F3002_2: Media - private TV
F3002_3: Media - radio
F3002_4: Media - newspapers
F3002_5: Media - online
F3002_6_1: Media - social media (days/week)
F3002_6_2: Media - social media frequency per day
F3003: Internal efficacy (1=strongly agree to 5=strongly disagree)
F3004_1: Democracy preferable
F3004_2: Courts should stop government
F3004_3: Strong leader who bends rules
F3004_4: Women representation
F3005_1: Country better run by business leaders
F3005_2: Country better run by experts
F3005_3: Country better run by citizens in referendums
F3006: How democratic is country (0-10)
F3007_1: Trust in parliament
F3007_2: Trust in government
F3008_1: Government performance - general
F3008_2: Government performance - COVID
F3009: State of economy (1=better to 5=worse)
"""


def test_llm_mapping(source_vars: dict, ground_truth: list[tuple[str, str]], n_test: int = 20):
    """Test LLM ability to propose correct mappings."""

    # Take a sample of mappings to test
    test_mappings = ground_truth[:n_test]
    targets_to_test = [t for t, s in test_mappings]

    # Format source variables for prompt
    source_str = ""
    for name, info in list(source_vars.items())[:80]:
        desc = info.get('desc', '')
        labels = info.get('labels', {})
        sample = info.get('sample', [])[:3]

        line = f"- {name}"
        if desc:
            line += f": '{desc}'"
        if sample:
            line += f" sample={sample}"
        if labels:
            line += f" labels={dict(list(labels.items())[:3])}"
        source_str += line + "\n"

    prompt = f"""You are matching variables from an Australian election survey to standardized CSES variables.

SOURCE VARIABLES (from deposited Australian data):
{source_str}

TARGET CSES VARIABLES (what we need to map to):
{CSES_VARS}

TASK: For each target below, identify which SOURCE variable should map to it.
Look for semantic matches - variable names often follow patterns like q01, q02a, dem1, d11, etc.

Targets to map: {', '.join(targets_to_test)}

Respond in JSON:
{{
  "mappings": [
    {{"target": "F3001", "source": "q01", "confidence": "high", "reason": "q01 matches Q01 political interest"}},
    ...
  ]
}}

Rules:
- Match based on variable names, descriptions, and sample values
- "q01", "q02a" etc. are questionnaire items (Q01, Q02a in CSES)
- "dem1", "dem2" etc. are demographic questions
- "d09", "d10", "d11" etc. are survey items
- Use "unknown" if no clear match
"""

    print("\nCalling Claude...")
    response = completion(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )

    content = response.choices[0].message.content

    # Parse response
    json_match = re.search(r'\{[\s\S]*\}', content)
    if not json_match:
        print("ERROR: Could not parse LLM response")
        return None

    data = json.loads(json_match.group())
    proposals = {m["target"]: m["source"] for m in data.get("mappings", [])}

    # Evaluate
    correct = 0
    incorrect = 0
    results = []

    for target, expected_source in test_mappings:
        proposed = proposals.get(target, "unknown")
        is_correct = proposed.lower() == expected_source.lower()

        if is_correct:
            correct += 1
            status = "✓"
        else:
            incorrect += 1
            status = "✗"

        results.append({
            "target": target,
            "proposed": proposed,
            "expected": expected_source,
            "correct": is_correct
        })

    return {
        "total": len(test_mappings),
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": correct / len(test_mappings) if test_mappings else 0,
        "results": results
    }


def main():
    print("=" * 70)
    print("CSES VARIABLE MAPPING TEST v2")
    print("Testing: Can LLM match source variables to CSES targets?")
    print("=" * 70)

    # Load data
    print("\n1. Loading deposited data...")
    source_vars = load_source_variables(DEPOSITED_DATA)
    print(f"   Found {len(source_vars)} source variables")

    # Extract ground truth
    print("\n2. Extracting expert mappings...")
    ground_truth = extract_real_mappings(HISTORICAL_DOFILE)
    print(f"   Found {len(ground_truth)} actual variable mappings")
    print(f"   Examples: {ground_truth[:5]}")

    # Test LLM
    print("\n3. Testing LLM proposals...")
    results = test_llm_mapping(source_vars, ground_truth, n_test=20)

    if results:
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(f"\nAccuracy: {results['accuracy']:.0%} ({results['correct']}/{results['total']})")

        print("\nDetails:")
        for r in results["results"]:
            status = "✓" if r["correct"] else "✗"
            print(f"  {status} {r['target']:15} proposed: {r['proposed']:15} expected: {r['expected']}")

        print("\n" + "=" * 70)
        if results['accuracy'] >= 0.8:
            print("VERDICT: HIGH ACCURACY - Human-in-loop approach is VIABLE")
        elif results['accuracy'] >= 0.5:
            print("VERDICT: MODERATE - Needs refinement but promising")
        else:
            print("VERDICT: LOW - Needs significant improvement")
        print("=" * 70)


if __name__ == "__main__":
    main()
