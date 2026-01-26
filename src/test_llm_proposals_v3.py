#!/usr/bin/env python3
"""
Test LLM Variable Mapping Proposals - V3 (Improved Demographics)

Improvements over v2:
- Better CSES variable descriptions for demographics
- Clearer semantic matching hints in prompt
- Ensures demographic source variables are included
- More explicit matching guidance
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
            "desc": desc if desc else ""
        }

    return var_info


# Enhanced CSES Variable descriptions with semantic keywords
CSES_VARS = """
DEMOGRAPHIC VARIABLES (F2XXX):
F2004: Marital status - married, widowed, divorced, separated, single, never married
F2006: Employment status - full-time, part-time, unemployed, retired, student, homemaker
F2008: Socioeconomic status - professional, white collar, blue collar, worker, farmer
F2012: Religious attendance - how often attend religious services, church attendance frequency
F2018: Region of residence - state, province, geographic region (country-specific codes)
F2019: Electoral district - constituency, electorate, voting district
F2021: Number of people in household - household size

SURVEY/ATTITUDINAL VARIABLES (F3XXX):
F3001: Political interest - interested in politics (1=very to 4=not at all)
F3002_1: Media usage - public TV news (days per week 0-7)
F3002_2: Media usage - private TV news (days per week)
F3002_3: Media usage - radio news (days per week)
F3002_4: Media usage - newspapers print (days per week)
F3002_5: Media usage - online/internet news (days per week)
F3002_6_1: Media usage - social media (days per week)
F3002_6_2: Media usage - social media frequency per day
F3003: Internal efficacy - politics too complicated, understand political issues
F3004_1: Democracy preference - democracy is preferable to other forms
F3004_2: Courts should stop government - rule of law, judicial independence
F3004_3: Strong leader who bends rules - authoritarian tendency
F3004_4: Women representation - gender in politics
F3005_1: Business leaders - country better run by business people
F3005_2: Experts - country better run by experts/technocrats
F3005_3: Referendums - country better run by citizens in referendums
F3006: How democratic is country (0-10 scale)
F3007_1: Trust in parliament - institutional trust
F3007_2: Trust in government - institutional trust
F3008_1: Government performance - general satisfaction
F3008_2: Government performance - COVID handling
F3009: State of economy - economic perception (1=better to 5=worse)
"""


def test_llm_mapping(source_vars: dict, ground_truth: list[tuple[str, str]], n_test: int = 20):
    """Test LLM ability to propose correct mappings."""

    # Take a sample of mappings to test
    test_mappings = ground_truth[:n_test]
    targets_to_test = [t for t, s in test_mappings]

    # Format source variables - ensure demographic vars are included
    # Prioritize variables that are likely mappings (dem*, q*, d*, ses, etc.)
    priority_prefixes = ['dem', 'q0', 'q1', 'q2', 'd0', 'd1', 'ses', 'p_']

    priority_vars = {}
    other_vars = {}

    for name, info in source_vars.items():
        is_priority = any(name.lower().startswith(p) for p in priority_prefixes)
        if is_priority:
            priority_vars[name] = info
        else:
            other_vars[name] = info

    # Build source string with priority vars first
    source_str = ""
    all_vars = list(priority_vars.items()) + list(other_vars.items())[:40]

    for name, info in all_vars:
        desc = info.get('desc', '')
        labels = info.get('labels', {})
        sample = info.get('sample', [])[:3]

        line = f"- {name}"
        if desc:
            line += f": \"{desc}\""
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

MATCHING RULES:
1. Variable names often follow patterns:
   - "q01", "q02a", etc. = questionnaire items for attitudes/opinions → F3XXX
   - "dem1", "dem2", etc. = demographic questions → F2XXX
   - "d09", "d11", etc. = other survey items
   - "ses" = socioeconomic status → F2008
   - "p_state" = region/state → F2018 or F2019

2. MATCH BY CONTENT, not just name patterns:
   - Look at the variable DESCRIPTION (in quotes after the name)
   - "marital status" → F2004
   - "employment" → F2006
   - "socioeconomic" → F2008
   - "religious services" or "church" → F2012
   - "political interest" → F3001
   - "media" or "TV" or "news" → F3002_X

3. Match by sample VALUES and LABELS when description is unclear

Targets to map: {', '.join(targets_to_test)}

Respond in JSON:
{{
  "mappings": [
    {{"target": "F3001", "source": "q01", "confidence": "high", "reason": "q01 description matches political interest"}},
    {{"target": "F2004", "source": "dem1", "confidence": "high", "reason": "dem1 asks about marital status"}},
    ...
  ]
}}

IMPORTANT: Use "unknown" ONLY if there is truly no matching variable. Most targets WILL have a match!
"""

    print("\nCalling Claude...")
    response = completion(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )

    content = response.choices[0].message.content

    # Show raw response for debugging
    print("\n--- LLM Response Preview ---")
    print(content[:500] + "..." if len(content) > 500 else content)
    print("----------------------------\n")

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

    # Separate F2XXX and F3XXX for analysis
    f2_correct, f2_total = 0, 0
    f3_correct, f3_total = 0, 0

    for target, expected_source in test_mappings:
        proposed = proposals.get(target, "unknown")
        is_correct = proposed.lower() == expected_source.lower()

        if target.startswith("F2"):
            f2_total += 1
            if is_correct:
                f2_correct += 1
        elif target.startswith("F3"):
            f3_total += 1
            if is_correct:
                f3_correct += 1

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
        "f2_accuracy": f2_correct / f2_total if f2_total else 0,
        "f3_accuracy": f3_correct / f3_total if f3_total else 0,
        "f2_stats": f"{f2_correct}/{f2_total}",
        "f3_stats": f"{f3_correct}/{f3_total}",
        "results": results
    }


def main():
    print("=" * 70)
    print("CSES VARIABLE MAPPING TEST v3")
    print("Testing: Improved demographic variable matching")
    print("=" * 70)

    # Load data
    print("\n1. Loading deposited data...")
    source_vars = load_source_variables(DEPOSITED_DATA)
    print(f"   Found {len(source_vars)} source variables")

    # Show some demographic variables
    print("\n   Key demographic variables found:")
    for name in ['dem1', 'dem2', 'ses', 'd11', 'p_state']:
        if name in source_vars:
            desc = source_vars[name].get('desc', 'No desc')
            print(f"   - {name}: {desc[:60]}")

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
        print(f"\nOverall Accuracy: {results['accuracy']:.0%} ({results['correct']}/{results['total']})")
        print(f"F2XXX (Demographics): {results['f2_accuracy']:.0%} ({results['f2_stats']})")
        print(f"F3XXX (Survey items): {results['f3_accuracy']:.0%} ({results['f3_stats']})")

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

        # Specific feedback
        if results['f2_accuracy'] < results['f3_accuracy']:
            print("\nNOTE: Demographics still harder than survey items - may need richer context")
        print("=" * 70)


if __name__ == "__main__":
    main()
