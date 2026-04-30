"""
LLM Validation Logic for CSES Variable Mapping.

This module implements Stage 2 of the dual-model validation pipeline:
1. Stage 1 (Original LLM matcher): Proposes mappings
2. Stage 2 (THIS MODULE - Validation LLM): Validates each proposal
3. Stage 3 (Human): Makes final decision

The validator reviews each proposal and returns:
- AGREE: Mapping appears correct
- DISAGREE: Mapping appears wrong, with explanation
- UNCERTAIN: Cannot verify, needs human review
"""

import logging
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional import - role-aware model runtime for API calls
try:
    from src.model_runtime import ModelRole, ModelTaskRunner
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    ModelRole = None
    ModelTaskRunner = None
    logger.warning("litellm not available - LLM validation will not work")

# Import existing modules - may fail if dependencies missing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from src.matching.llm_matcher import MatchProposal, CSES_TARGET_VARIABLES
except ImportError:
    MatchProposal = None
    CSES_TARGET_VARIABLES = []

try:
    from src.ingest.context_extractor import ExtractionResult, SourceVariableContext
except ImportError:
    ExtractionResult = None
    SourceVariableContext = None

from src.settings import apply_settings_to_environment


class ValidationVerdict(Enum):
    """Possible validation verdicts."""
    AGREE = "AGREE"
    DISAGREE = "DISAGREE"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class ValidationResult:
    """Result of LLM validation for a single proposal."""
    # Original proposal
    proposal: MatchProposal
    # Validation results
    verdict: ValidationVerdict
    reasoning: str
    suggested_alternative: Optional[str] = None
    # Recoding information (NEW)
    needs_recoding: bool = False
    recoding_notes: str = ""
    # Comparison
    models_agree: bool = False
    # Metadata
    validation_model: str = "openai/gpt-oss:120b"

    def to_dict(self) -> dict:
        return {
            "cses_target": self.proposal.target_variable,
            "source_variable": self.proposal.source_variable,
            "original_confidence": self.proposal.confidence,
            "original_reasoning": self.proposal.reasoning,
            "validation_verdict": self.verdict.value,
            "validation_reasoning": self.reasoning,
            "suggested_alternative": self.suggested_alternative,
            "needs_recoding": self.needs_recoding,
            "recoding_notes": self.recoding_notes,
            "models_agree": self.models_agree,
            "validation_model": self.validation_model
        }


# CSES validation knowledge embedded in prompts
CSES_VALIDATION_PROMPT = """You are validating a variable mapping proposal for CSES (Comparative Study of Electoral Systems) Module 6 data harmonization.

## Your Task
Review the proposed mapping and determine if it's correct OR can be made correct with recoding.

IMPORTANT: If the source variable measures the SAME CONCEPT but has different coding, DO NOT DISAGREE.
Instead, AGREE and note that recoding is needed. The goal is to maximize valid matches.

## CSES Target Variable
- Code: {target_variable}
- Description: {target_description}
- Expected coding: {expected_coding}

## Proposed Mapping
- Source variable: {source_variable}
- Original confidence: {confidence}%
- Original reasoning: {original_reasoning}

## Source Variable Information
{source_info}

## Validation Rules

1. AGREE if:
   - Source measures the SAME concept as target (even if coding differs)
   - Value labels can be recoded to match CSES scheme
   - Source has categorical values that can be converted (e.g., age groups -> age midpoints)

2. DISAGREE only if:
   - Source measures a COMPLETELY DIFFERENT concept
   - There is NO reasonable way to derive the target from source
   - The match is clearly an error (e.g., gender matched to political interest)

3. UNCERTAIN if:
   - Insufficient information to determine if concepts match
   - Multiple interpretations possible

## CSES Missing Value Codes
- 7/97/997 = Refused
- 8/98/998 = Don't know
- 9/99/999 = Missing/NA

## Respond with JSON:
{{
  "verdict": "AGREE" or "DISAGREE" or "UNCERTAIN",
  "reasoning": "Your detailed explanation",
  "needs_recoding": true/false,
  "recoding_notes": "Brief note on what recoding is needed (e.g., 'reverse scale', 'map age groups to midpoints')",
  "suggested_alternative": null or "alternative_source_variable_name"
}}

Only respond with valid JSON, no other text."""


# Expected coding for common CSES variables
EXPECTED_CODINGS = {
    "F2002": "1=Male, 2=Female, 9=Missing",
    "F2001_A": "Numeric age in years (18-120), 99=Missing",
    "F2001_Y": "4-digit birth year (1900-2010), 9999=Missing",
    "F2003": "Education level scale, typically 1-8, 9=Missing",
    "F3001": "1=Very interested, 2=Fairly, 3=Not very, 4=Not at all, 7=Refused, 8=DK, 9=Missing",
    "F3006": "0-10 scale (how democratic), 97=Refused, 98=DK, 99=Missing",
    "F3020": "0-10 scale (0=Left, 10=Right), 97=Refused, 98=DK, 99=Missing",
    "F3010": "1=Voted, 2=Did not vote, 7=Refused, 8=DK, 9=Missing",
    "F3021": "1=Has party ID, 2=Does not, 7=Refused, 8=DK, 9=Missing",
    "F3024": "1=Very satisfied, 2=Fairly, 3=Not very, 4=Not at all, 7=Refused, 8=DK, 9=Missing",
}


def get_expected_coding(target_var: str) -> str:
    """Get expected coding for a CSES target variable."""
    if target_var in EXPECTED_CODINGS:
        return EXPECTED_CODINGS[target_var]

    # General patterns based on variable type
    if target_var.startswith("F2"):
        return "See CSES Module 6 codebook for specific coding"
    elif target_var.startswith("F3"):
        if "F3018" in target_var or "F3019" in target_var:
            return "0-10 scale, 97=Refused, 98=DK, 99=Missing"
        elif "F3002" in target_var:
            return "Frequency scale, typically 1-5, 7=Refused, 8=DK, 9=Missing"
        elif "F3004" in target_var or "F3005" in target_var or "F3007" in target_var:
            return "Likert scale, check direction, 7=Refused, 8=DK, 9=Missing"
        else:
            return "See CSES Module 6 codebook for specific coding"
    else:
        return "See CSES Module 6 codebook"


def format_source_info(
    proposal: MatchProposal,
    extraction_result: Optional[ExtractionResult] = None
) -> str:
    """Format source variable information for validation prompt."""
    source_var = proposal.source_variable

    if source_var in ["NOT_FOUND", "ERROR"]:
        return f"Source variable: {source_var} (no actual source data)"

    parts = [f"Variable name: {source_var}"]

    # Try to get detailed info from extraction result
    if extraction_result:
        for var in extraction_result.variables:
            if var.name == source_var:
                if var.description:
                    parts.append(f"Description: {var.description}")
                if var.matched_question_text:
                    parts.append(f"Question text: {var.matched_question_text}")
                if var.value_labels:
                    labels_str = ", ".join(f"{k}={v}" for k, v in list(var.value_labels.items())[:8])
                    parts.append(f"Value labels: {labels_str}")
                if var.sample_values:
                    samples_str = ", ".join(str(v) for v in var.sample_values[:10])
                    parts.append(f"Sample values: {samples_str}")
                break

    if len(parts) == 1:
        parts.append("(No additional metadata available)")

    return "\n".join(parts)


def validate_proposal(
    proposal: MatchProposal,
    extraction_result: Optional[ExtractionResult] = None
) -> ValidationResult:
    """
    Validate a single variable mapping proposal using LLM.

    Args:
        proposal: The original LLM matcher's proposal
        extraction_result: Full context extraction (optional)

    Returns:
        ValidationResult with verdict and reasoning
    """
    from src.config import LLM_MODEL_VALIDATE
    apply_settings_to_environment()
    runner = ModelTaskRunner() if ModelTaskRunner else None
    validation_model = runner.model_for_role(ModelRole.VERIFIER) if runner and ModelRole else LLM_MODEL_VALIDATE

    target_var = proposal.target_variable
    target_desc = CSES_TARGET_VARIABLES.get(target_var, "Unknown CSES variable")
    expected_coding = get_expected_coding(target_var)

    # Format source variable info
    source_info = format_source_info(proposal, extraction_result)

    # Build validation prompt
    prompt = CSES_VALIDATION_PROMPT.format(
        target_variable=target_var,
        target_description=target_desc,
        expected_coding=expected_coding,
        source_variable=proposal.source_variable,
        confidence=int(proposal.confidence * 100),
        original_reasoning=proposal.reasoning,
        source_info=source_info
    )

    try:
        response = runner.response(
            ModelRole.VERIFIER,
            model_override=validation_model,
            max_tokens=1024,
            temperature=0,  # Deterministic validation
            timeout=60,
            drop_params=True,
            purpose=f"Validate mapping {proposal.source_variable} to {proposal.target_variable}",
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = response.choices[0].message.content.strip()

        # Parse JSON response
        result_data = _parse_validation_response(response_text)

        verdict_str = result_data.get("verdict", "UNCERTAIN")
        verdict = ValidationVerdict[verdict_str] if verdict_str in ValidationVerdict.__members__ else ValidationVerdict.UNCERTAIN

        # Extract recoding information
        needs_recoding = result_data.get("needs_recoding", False)
        recoding_notes = result_data.get("recoding_notes", "")

        return ValidationResult(
            proposal=proposal,
            verdict=verdict,
            reasoning=result_data.get("reasoning", "No reasoning provided"),
            suggested_alternative=result_data.get("suggested_alternative"),
            needs_recoding=needs_recoding,
            recoding_notes=recoding_notes,
            models_agree=(verdict == ValidationVerdict.AGREE),
            validation_model=validation_model
        )
    except Exception as e:
        logger.error(f"Validation failed for {proposal.target_variable}: {e}")
        return ValidationResult(
            proposal=proposal,
            verdict=ValidationVerdict.UNCERTAIN,
            reasoning=f"Validation error: {str(e)}",
            models_agree=False,
            validation_model=validation_model
        )


def _connection_kwargs() -> dict:
    kwargs = {}
    api_base = os.environ.get("OPENAI_API_BASE", "").strip()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def _parse_validation_response(response_text: str) -> dict:
    """Parse JSON from validation response."""
    import re

    # Try to extract JSON from code block
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try direct JSON parse
    if '{' in response_text:
        start = response_text.find('{')
        depth = 0
        end = start
        for i, c in enumerate(response_text[start:], start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end > start:
            try:
                return json.loads(response_text[start:end])
            except json.JSONDecodeError:
                pass

    # Fallback: try to extract verdict from text
    verdict = "UNCERTAIN"
    for v in ["AGREE", "DISAGREE", "UNCERTAIN"]:
        if v in response_text.upper():
            verdict = v
            break

    return {
        "verdict": verdict,
        "reasoning": response_text[:500],
        "suggested_alternative": None
    }


def validate_proposals(
    proposals: list[MatchProposal],
    extraction_result: Optional[ExtractionResult] = None,
    progress_callback: Optional[callable] = None
) -> list[ValidationResult]:
    """
    Validate a batch of proposals.

    Args:
        proposals: List of proposals from original LLM matcher
        extraction_result: Full context extraction
        progress_callback: Optional callback for progress updates

    Returns:
        List of ValidationResult objects
    """
    total = len(proposals)
    results = [None] * total
    max_workers = min(3, total) if total else 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(validate_proposal, proposal, extraction_result): i
            for i, proposal in enumerate(proposals)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            proposal = proposals[i]
            if progress_callback:
                progress_callback(f"Validated {i+1}/{total}: {proposal.target_variable}")
            result = future.result()
            results[i] = result
            logger.info(
                f"Validated {proposal.target_variable}: "
                f"{proposal.source_variable} -> {result.verdict.value}"
            )

    return [result for result in results if result is not None]


def format_validation_result(result: ValidationResult) -> str:
    """Format a validation result for display."""
    proposal = result.proposal

    status_markers = {
        ValidationVerdict.AGREE: "[OK]",
        ValidationVerdict.DISAGREE: "[X]",
        ValidationVerdict.UNCERTAIN: "[?]"
    }

    lines = [
        f"## {proposal.source_variable} -> {proposal.target_variable}",
        "",
        f"**CSES Target:** {CSES_TARGET_VARIABLES.get(proposal.target_variable, 'Unknown')}",
        "",
        "### Original Matcher",
        f"- Confidence: {proposal.confidence:.0%}",
        f"- Reasoning: {proposal.reasoning}",
        "",
        "### LLM Validation",
        f"- Verdict: {status_markers[result.verdict]} {result.verdict.value}",
        f"- Reasoning: {result.reasoning}",
    ]

    if result.suggested_alternative:
        lines.append(f"- Suggested alternative: {result.suggested_alternative}")

    # Recoding information (NEW)
    if result.needs_recoding:
        lines.extend([
            "",
            "### Recoding Required",
            f"- Recoding needed: YES",
            f"- Notes: {result.recoding_notes}",
        ])

    # Overall status
    if result.models_agree:
        if result.needs_recoding:
            lines.extend(["", "**Status:** APPROVED with recoding - include recode commands in .do file"])
        else:
            lines.extend(["", "**Status:** APPROVED - direct mapping"])
    else:
        lines.extend(["", f"**Status:** {status_markers[result.verdict]} Requires review"])

    return "\n".join(lines)


def compute_validation_summary(results: list[ValidationResult]) -> dict:
    """Compute summary statistics from validation results."""
    total = len(results)
    agrees = sum(1 for r in results if r.verdict == ValidationVerdict.AGREE)
    disagrees = sum(1 for r in results if r.verdict == ValidationVerdict.DISAGREE)
    uncertain = sum(1 for r in results if r.verdict == ValidationVerdict.UNCERTAIN)
    models_agree_count = sum(1 for r in results if r.models_agree)

    # Count by original confidence
    high_conf = sum(1 for r in results if r.proposal.confidence >= 0.85)
    medium_conf = sum(1 for r in results if 0.60 <= r.proposal.confidence < 0.85)
    low_conf = sum(1 for r in results if r.proposal.confidence < 0.60)

    return {
        "total": total,
        "agrees": agrees,
        "disagrees": disagrees,
        "uncertain": uncertain,
        "agreement_rate": agrees / total if total > 0 else 0,
        "models_agree_count": models_agree_count,
        "original_high_confidence": high_conf,
        "original_medium_confidence": medium_conf,
        "original_low_confidence": low_conf
    }
