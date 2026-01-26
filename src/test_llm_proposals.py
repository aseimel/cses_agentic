#!/usr/bin/env python3
"""
Test LLM Variable Mapping Proposals

This tests whether an LLM can correctly propose variable mappings
that match what a domain expert actually did.

Requires: pip install litellm anthropic polars pyreadstat
Set ANTHROPIC_API_KEY environment variable
"""

import os
import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Load .env file if it exists
env_file = Path("/home/armin/cses_agentic/.env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip()

# Check for API key
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: No API key found")
    print("Option 1: export ANTHROPIC_API_KEY='sk-ant-...'")
    print("Option 2: Create .env file with ANTHROPIC_API_KEY=sk-ant-...")
    exit(1)

try:
    import polars as pl
    import pyreadstat
    from litellm import completion
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install litellm anthropic polars pyreadstat")
    exit(1)


# Paths
REPO_ROOT = Path("/home/armin/cses_agentic/repo")
AUSTRALIA_PATH = REPO_ROOT / "CSES Data Products/CSES Standalone Modules/Module 6/Election Studies/Australia_2022"
DEPOSITED_DATA = AUSTRALIA_PATH / "E-mails/CSES 2022 AUstralia 10 Sep 24.dta"
HISTORICAL_DOFILE = AUSTRALIA_PATH / "micro/cses-m6_micro_AUS_2022_20240110.do"
WORKFLOW_FILE = Path("/home/armin/cses_agentic/workflow.md")


@dataclass
class GroundTruth:
    """What the expert actually did."""
    target_var: str
    source_var: str
    mapping_type: str


@dataclass
class LLMProposal:
    """What the LLM proposes."""
    target_var: str
    proposed_source: str
    confidence: str
    reasoning: str


def load_workflow_context() -> str:
    """Load workflow.md for context."""
    if WORKFLOW_FILE.exists():
        return WORKFLOW_FILE.read_text()[:4000]  # First 4000 chars
    return ""


def load_source_variables(data_path: Path, limit: int = 50) -> dict:
    """Load source variable information from deposited data."""
    df_pd, meta = pyreadstat.read_dta(str(data_path))
    df = pl.from_pandas(df_pd)

    var_info = {}
    for col in list(df.columns)[:limit]:
        sample = df[col].drop_nulls().head(5).to_list()
        labels = meta.variable_value_labels.get(col, {})

        var_info[col] = {
            "dtype": str(df[col].dtype),
            "n_unique": df[col].n_unique(),
            "sample_values": sample[:5],
            "labels": dict(list(labels.items())[:5]) if labels else None,
            "description": meta.column_names_to_labels.get(col, "")
        }

    return var_info


def extract_ground_truth(dofile_path: Path) -> list[GroundTruth]:
    """Extract what the expert actually mapped from the do-file."""
    content = dofile_path.read_text(errors='ignore')

    ground_truth = []

    # Pattern: gen F3001 = q01
    direct_pattern = r'gen\s+(?:str\d*\s+)?(F\d+[_\w]*)\s*=\s*(\w+)\s*(?:\n|//|$)'

    for match in re.finditer(direct_pattern, content):
        target = match.group(1)
        source = match.group(2)

        # Filter out constants and computed values
        if source in ['1', '2', '0', '96', '97', '98', '99', '999', '9996', '9999']:
            continue
        if source.startswith('F'):  # Skip references to other CSES variables
            continue

        ground_truth.append(GroundTruth(
            target_var=target,
            source_var=source,
            mapping_type="direct"
        ))

    # Pattern: tostring var, gen(F1003_2)
    tostring_pattern = r'tostring\s+(\w+),\s*gen\((F\d+[_\w]*)\)'
    for match in re.finditer(tostring_pattern, content):
        ground_truth.append(GroundTruth(
            target_var=match.group(2),
            source_var=match.group(1),
            mapping_type="tostring"
        ))

    return ground_truth


# CSES Variable descriptions for the prompt
CSES_VARIABLES = """
F1003_2: Respondent ID within election study (string, 10 chars, zero-padded)
F1004: Election study ID alphabetic (e.g., "AUS_2022")
F1009: Election year (numeric, e.g., 2022)
F1016_1: Mode of interview for respondent (1=face-to-face, 2=phone, 4=internet)
F1019_M: Interview date - month (1-12)
F1019_D: Interview date - day (1-31)
F1019_Y: Interview date - year
F1021: Interview duration in minutes

F2001_Y: Year of birth of respondent
F2001_A: Age of respondent in years
F2002: Gender (0=male, 1=female, 3=other)
F2003: Education level (ISCED 0-9)
F2004: Marital status
F2005: Union membership (0=no, 1=yes)
F2006: Employment status
F2007: Occupation (ISCO codes)
F2008: Socioeconomic status
F2009: Employment type (public/private)
F2010_1: Household income quintile (1-5)
F2011: Religious denomination
F2012: Religious attendance frequency

F3001: Political interest (1=very interested to 4=not at all)
F3002_1: Media usage - public TV news (days per week 0-7)
F3002_2: Media usage - private TV news
F3002_3: Media usage - radio news
F3002_4: Media usage - newspapers
F3002_5: Media usage - online news
F3003: Internal political efficacy
F3006: How democratic is your country (0-10 scale)
F3010: Turnout in main election (0=no, 1=yes)
F3010_LH: Turnout in lower house election
F3011_LH_PL: Vote choice lower house - party list
F3020_R: Left-right self-placement (0-10 scale)
F3023_1: Feel close to any party (0=no, 1=yes)
F3023_3: Party identification (party code)
"""


def get_llm_proposals(source_vars: dict, target_vars: list[str], batch_size: int = 10) -> list[LLMProposal]:
    """Ask LLM to propose mappings for target variables."""

    workflow_context = load_workflow_context()

    # Format source variables
    source_str = ""
    for name, info in source_vars.items():
        desc = info.get('description', '') or ''
        labels = info.get('labels', {})
        labels_str = str(labels)[:100] if labels else ""
        source_str += f"- {name}: {info['dtype']}, {info['n_unique']} unique, sample={info['sample_values'][:3]}"
        if desc:
            source_str += f", desc='{desc[:50]}'"
        if labels_str:
            source_str += f", labels={labels_str}"
        source_str += "\n"

    prompt = f"""You are helping process CSES (Comparative Study of Electoral Systems) election study data.

TASK: For each target CSES variable, identify which source variable from the deposited data should be mapped to it.

CSES TARGET VARIABLES (what we need to create):
{CSES_VARIABLES}

SOURCE VARIABLES (from collaborator's deposited data):
{source_str}

For each of these target variables, propose which source variable maps to it:
{', '.join(target_vars[:batch_size])}

Respond in JSON format:
{{
  "proposals": [
    {{
      "target": "F3001",
      "source": "q01",
      "confidence": "high",
      "reasoning": "Variable name q01 matches Q01 in questionnaire for political interest"
    }},
    ...
  ]
}}

Rules:
- Only propose a source if you're confident it maps to the target
- Use "none" for source if no clear match exists
- confidence: "high", "medium", or "low"
- Be specific about reasoning
"""

    try:
        response = completion(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )

        content = response.choices[0].message.content

        # Extract JSON
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            data = json.loads(json_match.group())
            proposals = []
            for p in data.get("proposals", []):
                proposals.append(LLMProposal(
                    target_var=p["target"],
                    proposed_source=p["source"],
                    confidence=p.get("confidence", "unknown"),
                    reasoning=p.get("reasoning", "")
                ))
            return proposals
    except Exception as e:
        print(f"LLM Error: {e}")

    return []


def evaluate_proposals(proposals: list[LLMProposal], ground_truth: list[GroundTruth]) -> dict:
    """Compare LLM proposals to expert decisions."""

    # Build ground truth lookup
    gt_lookup = {gt.target_var: gt.source_var for gt in ground_truth}

    results = {
        "total": len(proposals),
        "correct": 0,
        "incorrect": 0,
        "no_ground_truth": 0,
        "details": []
    }

    for proposal in proposals:
        expected = gt_lookup.get(proposal.target_var)

        if expected is None:
            status = "no_ground_truth"
            results["no_ground_truth"] += 1
        elif proposal.proposed_source.lower() == expected.lower():
            status = "correct"
            results["correct"] += 1
        elif proposal.proposed_source == "none":
            status = "skipped"
            results["no_ground_truth"] += 1
        else:
            status = "incorrect"
            results["incorrect"] += 1

        results["details"].append({
            "target": proposal.target_var,
            "proposed": proposal.proposed_source,
            "expected": expected,
            "confidence": proposal.confidence,
            "status": status,
            "reasoning": proposal.reasoning[:100]
        })

    # Calculate accuracy
    evaluated = results["correct"] + results["incorrect"]
    if evaluated > 0:
        results["accuracy"] = results["correct"] / evaluated
    else:
        results["accuracy"] = 0

    return results


def print_results(results: dict):
    """Print evaluation results."""
    print("\n" + "=" * 80)
    print("LLM PROPOSAL EVALUATION RESULTS")
    print("=" * 80)

    print(f"\nTotal proposals: {results['total']}")
    print(f"Correct: {results['correct']}")
    print(f"Incorrect: {results['incorrect']}")
    print(f"No ground truth: {results['no_ground_truth']}")
    print(f"\nACCURACY: {results['accuracy']:.1%}")

    print("\n" + "-" * 80)
    print("DETAILS")
    print("-" * 80)

    for d in results["details"]:
        status_icon = "✓" if d["status"] == "correct" else "✗" if d["status"] == "incorrect" else "?"
        print(f"{status_icon} {d['target']:15} proposed: {d['proposed']:15} expected: {str(d['expected']):15} ({d['confidence']})")
        if d["status"] == "incorrect" and d["reasoning"]:
            print(f"   Reasoning: {d['reasoning']}")


def main():
    print("=" * 80)
    print("CSES LLM VARIABLE MAPPING TEST")
    print("Testing human-in-the-loop approach with Australia 2022")
    print("=" * 80)

    # Load source variables
    print("\n1. Loading deposited data...")
    source_vars = load_source_variables(DEPOSITED_DATA, limit=60)
    print(f"   Loaded {len(source_vars)} source variables")

    # Extract ground truth
    print("\n2. Extracting expert mappings from historical do-file...")
    ground_truth = extract_ground_truth(HISTORICAL_DOFILE)
    print(f"   Found {len(ground_truth)} direct mappings")

    # Get target variables to test (ones with ground truth)
    target_vars = [gt.target_var for gt in ground_truth[:15]]  # Test first 15
    print(f"\n3. Testing LLM proposals for {len(target_vars)} variables...")
    print(f"   Targets: {', '.join(target_vars)}")

    # Get LLM proposals
    print("\n4. Calling LLM...")
    proposals = get_llm_proposals(source_vars, target_vars)
    print(f"   Got {len(proposals)} proposals")

    # Evaluate
    print("\n5. Evaluating against ground truth...")
    results = evaluate_proposals(proposals, ground_truth)

    # Print results
    print_results(results)

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if results["accuracy"] >= 0.8:
        print("✓ HIGH ACCURACY - Human-in-the-loop approach is VIABLE")
        print("  AI proposals can be used as starting point for expert review")
    elif results["accuracy"] >= 0.5:
        print("~ MODERATE ACCURACY - Approach needs refinement")
        print("  May need better prompting or more context")
    else:
        print("✗ LOW ACCURACY - Significant issues to address")
        print("  Review failure cases and improve approach")

    return results


if __name__ == "__main__":
    main()
