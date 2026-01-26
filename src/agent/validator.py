"""
Claude Validation Logic for CSES Variable Mapping.

This module implements Stage 2 of the dual-model validation pipeline:
1. Stage 1 (Original LLM matcher): Proposes mappings
2. Stage 2 (THIS MODULE - Claude): Validates each proposal
3. Stage 3 (Human): Makes final decision

The validator reviews each proposal and returns:
- AGREE: Mapping appears correct
- DISAGREE: Mapping appears wrong, with explanation
- UNCERTAIN: Cannot verify, needs human review
"""

import logging
import json
import os
import subprocess
import shutil
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from pathlib import Path

from litellm import completion

# Import existing modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.matching.llm_matcher import MatchProposal, CSES_TARGET_VARIABLES
from src.ingest.context_extractor import ExtractionResult, SourceVariableContext

logger = logging.getLogger(__name__)


class ValidationVerdict(Enum):
    """Possible validation verdicts."""
    AGREE = "AGREE"
    DISAGREE = "DISAGREE"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class ValidationResult:
    """Result of Claude validation for a single proposal."""
    # Original proposal
    proposal: MatchProposal
    # Validation results
    verdict: ValidationVerdict
    reasoning: str
    suggested_alternative: Optional[str] = None
    # Comparison
    models_agree: bool = False
    # Metadata
    validation_model: str = "claude"

    def to_dict(self) -> dict:
        return {
            "cses_target": self.proposal.target_variable,
            "source_variable": self.proposal.source_variable,
            "original_confidence": self.proposal.confidence,
            "original_reasoning": self.proposal.reasoning,
            "validation_verdict": self.verdict.value,
            "validation_reasoning": self.reasoning,
            "suggested_alternative": self.suggested_alternative,
            "models_agree": self.models_agree,
            "validation_model": self.validation_model
        }


# CSES validation knowledge embedded in prompts
CSES_VALIDATION_PROMPT = """You are validating a variable mapping proposal for CSES (Comparative Study of Electoral Systems) Module 6 data harmonization.

## Your Task
Review the proposed mapping and determine if it's correct.

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

## Validation Checklist
1. Does the source variable measure the SAME concept as the CSES target?
2. Do the value labels align with CSES coding scheme?
3. Are there any red flags (ambiguous name, multiple candidates, etc.)?

## CSES Missing Value Codes
- 7/97/997 = Refused
- 8/98/998 = Don't know
- 9/99/999 = Missing/NA

## Respond with JSON:
{{
  "verdict": "AGREE" or "DISAGREE" or "UNCERTAIN",
  "reasoning": "Your detailed explanation",
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


def _call_claude_cli(prompt: str, timeout: int = 120) -> str:
    """
    Call Claude CLI to get a response using the user's Max subscription.

    This allows users with Claude Max subscription to use their subscription
    for validation without needing a separate API key.

    Args:
        prompt: The prompt to send to Claude
        timeout: Timeout in seconds (default: 120)

    Returns:
        Claude's response text

    Raises:
        RuntimeError: If Claude CLI is not installed or fails
    """
    # Check if claude CLI is available
    claude_path = shutil.which("claude")
    if not claude_path:
        raise RuntimeError(
            "Claude CLI not found. Please install Claude Code CLI and authenticate:\n"
            "  1. Install: npm install -g @anthropic-ai/claude-code\n"
            "  2. Authenticate: claude login\n"
            "Or set LLM_MODEL_VALIDATE to an API model like 'anthropic/claude-sonnet-4-20250514'"
        )

    try:
        # Call claude CLI with the prompt via stdin
        # Using --print flag to output response directly
        result = subprocess.run(
            [claude_path, "--print", "--dangerously-skip-permissions"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown error"
            if "not authenticated" in error_msg.lower() or "login" in error_msg.lower():
                raise RuntimeError(
                    "Claude CLI not authenticated. Please run 'claude login' to authenticate "
                    "with your Max subscription."
                )
            raise RuntimeError(f"Claude CLI error: {error_msg}")

        return result.stdout.strip()

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Claude CLI timed out after {timeout} seconds")
    except FileNotFoundError:
        raise RuntimeError("Claude CLI not found in PATH")


def check_claude_cli_available() -> tuple[bool, str]:
    """
    Check if Claude CLI is available and authenticated.

    Returns:
        Tuple of (is_available: bool, message: str)
    """
    claude_path = shutil.which("claude")
    if not claude_path:
        return False, "Claude CLI not installed. Run: npm install -g @anthropic-ai/claude-code"

    # Try a simple test command
    try:
        result = subprocess.run(
            [claude_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            return True, f"Claude CLI available: {version}"
        else:
            return False, f"Claude CLI error: {result.stderr.strip()}"
    except Exception as e:
        return False, f"Error checking Claude CLI: {e}"


def validate_proposal(
    proposal: MatchProposal,
    extraction_result: Optional[ExtractionResult] = None,
    model: Optional[str] = None
) -> ValidationResult:
    """
    Validate a single variable mapping proposal using Claude.

    Args:
        proposal: The original LLM matcher's proposal
        extraction_result: Full context extraction (optional)
        model: Model to use for validation (default: from env)
              Use "claude-cli" to use Claude Code CLI with Max subscription

    Returns:
        ValidationResult with verdict and reasoning
    """
    # Get model from environment or use default
    validation_model = model or os.getenv("LLM_MODEL_VALIDATE") or os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")

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
        # Check if using Claude CLI (Max subscription)
        if validation_model == "claude-cli":
            logger.info(f"Using Claude CLI for validation of {target_var}")
            response_text = _call_claude_cli(prompt)
        else:
            # Use LiteLLM API
            response = completion(
                model=validation_model,
                max_tokens=1024,
                temperature=0,  # Deterministic validation
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content.strip()

        # Parse JSON response
        result_data = _parse_validation_response(response_text)

        verdict_str = result_data.get("verdict", "UNCERTAIN")
        verdict = ValidationVerdict[verdict_str] if verdict_str in ValidationVerdict.__members__ else ValidationVerdict.UNCERTAIN

        return ValidationResult(
            proposal=proposal,
            verdict=verdict,
            reasoning=result_data.get("reasoning", "No reasoning provided"),
            suggested_alternative=result_data.get("suggested_alternative"),
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
    model: Optional[str] = None,
    progress_callback: Optional[callable] = None
) -> list[ValidationResult]:
    """
    Validate a batch of proposals.

    Args:
        proposals: List of proposals from original LLM matcher
        extraction_result: Full context extraction
        model: Model to use for validation
        progress_callback: Optional callback for progress updates

    Returns:
        List of ValidationResult objects
    """
    results = []
    total = len(proposals)

    for i, proposal in enumerate(proposals):
        if progress_callback:
            progress_callback(f"Validating {i+1}/{total}: {proposal.target_variable}")

        result = validate_proposal(proposal, extraction_result, model)
        results.append(result)

        logger.info(
            f"Validated {proposal.target_variable}: "
            f"{proposal.source_variable} -> {result.verdict.value}"
        )

    return results


def format_validation_result(result: ValidationResult) -> str:
    """Format a validation result for display."""
    proposal = result.proposal

    status_markers = {
        ValidationVerdict.AGREE: "[OK]",
        ValidationVerdict.DISAGREE: "[X]",
        ValidationVerdict.UNCERTAIN: "[?]"
    }

    lines = [
        f"## {proposal.source_variable} → {proposal.target_variable}",
        "",
        f"**CSES Target:** {CSES_TARGET_VARIABLES.get(proposal.target_variable, 'Unknown')}",
        "",
        "### Original Matcher",
        f"- Confidence: {proposal.confidence:.0%}",
        f"- Reasoning: {proposal.reasoning}",
        "",
        "### Claude Validation",
        f"- Verdict: {status_markers[result.verdict]} {result.verdict.value}",
        f"- Reasoning: {result.reasoning}",
    ]

    if result.suggested_alternative:
        lines.append(f"- Suggested alternative: {result.suggested_alternative}")

    # Overall status
    if result.models_agree:
        lines.extend(["", "**Status:** Both models agree - recommend approval"])
    else:
        lines.extend(["", f"**Status:** {status_markers[result.verdict]} Models disagree - requires review"])

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
