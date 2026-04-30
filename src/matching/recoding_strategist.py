"""
Recoding strategist for CSES variable transformations.

Generates specific recoding strategies for each variable mapping, going beyond
just "needs_recoding: True" to provide:
- Specific transformation rules (value-to-value mappings)
- Stata pattern recommendations
- Missing value handling
- Confidence scores for the recoding strategy

This enables automatic Stata code generation with proper transformations.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Callable

from src.config import LLM_MODEL_RECODE, LLM_TEMPERATURE
from src.matching.llm_matcher import CSES_TARGET_VARIABLES
from src.model_runtime import ModelRole, ModelTaskRunner

logger = logging.getLogger(__name__)


@dataclass
class RecodeRule:
    """A single value transformation rule."""
    from_value: str
    to_value: str
    comment: str


@dataclass
class RecodingStrategy:
    """Complete recoding strategy for a variable mapping."""
    target_var: str
    source_var: str
    transformation_type: str  # direct, recode, calculate, complex
    requires_recoding: bool
    recode_rules: list[RecodeRule] = field(default_factory=list)
    stata_pattern: str = "gen"  # gen, recode, replace, calculate
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "target_var": self.target_var,
            "source_var": self.source_var,
            "transformation_type": self.transformation_type,
            "requires_recoding": self.requires_recoding,
            "recode_rules": [{"from": r.from_value, "to": r.to_value, "comment": r.comment} for r in self.recode_rules],
            "stata_pattern": self.stata_pattern,
            "warnings": self.warnings,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


# CSES expected codings for target variables
CSES_EXPECTED_CODINGS = {
    "F2001_Y": {
        "desc": "Year of birth",
        "coding": "4-digit year (1900-2010)",
        "missing": "9999=Missing"
    },
    "F2001_A": {
        "desc": "Age in years",
        "coding": "Numeric age 18-120",
        "missing": "9999=Missing",
        "note": "If source is categorical age groups, convert to midpoints"
    },
    "F2002": {
        "desc": "Gender",
        "coding": "1=Male, 2=Female, 3=Other",
        "missing": "7=Refused, 8=DK, 9=Missing"
    },
    "F2003": {
        "desc": "Education level",
        "coding": "1-8 scale (1=None to 8=Post-grad)",
        "missing": "97=Refused, 98=DK, 99=Missing"
    },
    "F2004": {
        "desc": "Marital status",
        "coding": "1=Married, 2=Living with partner, 3=Separated, 4=Divorced, 5=Widowed, 6=Single",
        "missing": "7=Refused, 8=DK, 9=Missing"
    },
    "F2005": {
        "desc": "Union membership",
        "coding": "1=Member, 2=Household member, 3=Not member",
        "missing": "7=Refused, 8=DK, 9=Missing"
    },
    "F2006": {
        "desc": "Employment status",
        "coding": "1=Full-time, 2=Part-time, 3=Unemployed, 4=Student, 5=Retired, 6=Homemaker, 7=Other",
        "missing": "97=Refused, 98=DK, 99=Missing"
    },
    "F2011": {
        "desc": "Religious denomination",
        "coding": "Country-specific codes",
        "missing": "9997=Refused, 9998=DK, 9999=Missing"
    },
    "F2012": {
        "desc": "Religious attendance",
        "coding": "1=Every week, 2=Almost every week, 3=Once/twice a month, 4=Few times a year, 5=Once a year, 6=Less often, 7=Never",
        "missing": "97=Refused, 98=DK, 99=Missing"
    },
    "F2020": {
        "desc": "Rural/Urban",
        "coding": "1=Rural, 2=Small town, 3=Suburb, 4=Urban",
        "missing": "7=Refused, 8=DK, 9=Missing"
    },
    "F2021": {
        "desc": "Household size",
        "coding": "Numeric count 1-20",
        "missing": "97=Refused, 98=DK, 99=Missing"
    },
    "F3001": {
        "desc": "Political interest",
        "coding": "1=Very interested, 2=Somewhat, 3=Not very, 4=Not at all",
        "missing": "7=Refused, 8=DK, 9=Missing"
    },
    "F3002_1": {
        "desc": "Media: Public TV",
        "coding": "1=Every day, 2=3-4 days, 3=1-2 days, 4=Less often, 5=Never",
        "missing": "7=Refused, 8=DK, 9=Missing"
    },
    "F3003": {
        "desc": "Internal efficacy",
        "coding": "1=Strongly agree, 2=Agree, 3=Neither, 4=Disagree, 5=Strongly disagree",
        "missing": "7=Refused, 8=DK, 9=Missing"
    },
    "F3006": {
        "desc": "How democratic",
        "coding": "0-10 scale (0=Not democratic, 10=Completely democratic)",
        "missing": "97=Refused, 98=DK, 99=Missing"
    },
    "F3007_1": {
        "desc": "Trust: Parliament",
        "coding": "0-10 scale (0=No trust, 10=Complete trust)",
        "missing": "97=Refused, 98=DK, 99=Missing"
    },
    "F3009": {
        "desc": "State of economy",
        "coding": "1=Very good, 2=Good, 3=Neither, 4=Bad, 5=Very bad",
        "missing": "7=Refused, 8=DK, 9=Missing"
    },
    "F3010": {
        "desc": "Voted in election",
        "coding": "1=Yes voted, 2=No did not vote",
        "missing": "7=Refused, 8=DK, 9=Missing"
    },
    "F3018_A": {
        "desc": "Like/Dislike Party A",
        "coding": "0-10 scale (0=Strongly dislike, 10=Strongly like)",
        "missing": "96=Haven't heard, 97=Refused, 98=DK, 99=Missing"
    },
    "F3019_A": {
        "desc": "Left-Right Party A",
        "coding": "0-10 scale (0=Left, 10=Right)",
        "missing": "95=Don't know L-R, 96=Haven't heard, 97=Refused, 98=DK, 99=Missing"
    },
    "F3020": {
        "desc": "Left-Right self",
        "coding": "0-10 scale (0=Left, 10=Right)",
        "missing": "95=Don't know L-R, 97=Refused, 98=DK, 99=Missing"
    },
    "F3021": {
        "desc": "Party identification",
        "coding": "1=Yes has party ID, 2=No",
        "missing": "7=Refused, 8=DK, 9=Missing"
    },
    "F3024": {
        "desc": "Satisfaction with democracy",
        "coding": "1=Very satisfied, 2=Fairly, 3=Not very, 4=Not at all",
        "missing": "7=Refused, 8=DK, 9=Missing"
    },
}


class RecodingStrategist:
    """Generate specific recoding strategies for each variable mapping."""

    def __init__(self, working_dir=None):
        """Initialize the recoding strategist."""
        from pathlib import Path
        self.runner = ModelTaskRunner(Path(working_dir) if working_dir else Path.cwd())
        self.model = self.runner.model_for_role(ModelRole.RECODE_CODEGEN) or LLM_MODEL_RECODE
        self.temperature = LLM_TEMPERATURE
        logger.info(f"RecodingStrategist initialized with model: {self.model}")

    def generate_strategies(
        self,
        mappings: list[dict],
        source_contexts: list[dict],
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> dict[str, RecodingStrategy]:
        """
        Generate recoding strategies for all mappings.

        Args:
            mappings: List of variable mappings from matcher
            source_contexts: Source variable metadata
            progress_callback: Optional callback for progress updates

        Returns:
            Dict mapping target variable to RecodingStrategy
        """
        def update_progress(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        # Build source lookup
        source_lookup = {}
        for ctx in source_contexts:
            name = ctx.get('name', '')
            if name:
                source_lookup[name] = ctx

        strategies = {}
        total = len([m for m in mappings if m.get("source") not in ["NOT_FOUND", "ERROR"]])
        processed = 0

        update_progress(f"Generating recoding strategies for {total} mappings...")

        for mapping in mappings:
            target = mapping.get("target") or mapping.get("target_variable", "")
            source = mapping.get("source") or mapping.get("source_variable", "")

            if not target or source in ["NOT_FOUND", "ERROR"]:
                continue

            processed += 1
            update_progress(f"  [{processed}/{total}] Analyzing {target} <- {source}")

            source_info = source_lookup.get(source, {})
            target_info = CSES_EXPECTED_CODINGS.get(target, {})

            # For similar variables (F3002_2 is like F3002_1), use base pattern
            if not target_info:
                base_var = re.sub(r'_[A-I]$', '_A', target)  # F3018_B -> F3018_A
                base_var = re.sub(r'_\d+$', '_1', base_var)  # F3002_2 -> F3002_1
                target_info = CSES_EXPECTED_CODINGS.get(base_var, {})

            strategy = self.generate_strategy(
                target_var=target,
                source_var=source,
                source_info=source_info,
                target_info=target_info,
                mapping_reasoning=mapping.get("reasoning", "")
            )

            strategies[target] = strategy

        update_progress(f"Generated {len(strategies)} recoding strategies")
        return strategies

    def generate_strategy(
        self,
        target_var: str,
        source_var: str,
        source_info: dict,
        target_info: dict,
        mapping_reasoning: str = ""
    ) -> RecodingStrategy:
        """
        Generate detailed recoding strategy for a single mapping.

        Args:
            target_var: CSES target variable code
            source_var: Source variable name
            source_info: Dict with source variable metadata
            target_info: Dict with CSES target coding expectations
            mapping_reasoning: Original reasoning from matcher

        Returns:
            RecodingStrategy with transformation rules
        """
        # Build source info string
        source_desc = source_info.get('description', '') or source_info.get('desc', '')
        source_labels = source_info.get('value_labels', {})
        source_samples = source_info.get('sample_values', [])

        labels_str = ""
        if source_labels and isinstance(source_labels, dict):
            labels_str = ", ".join(f"{k}={v}" for k, v in list(source_labels.items())[:15])

        samples_str = ""
        if source_samples:
            samples_str = ", ".join(str(s) for s in source_samples[:10])

        # Build target info string
        target_desc = target_info.get("desc", CSES_TARGET_VARIABLES.get(target_var, ""))
        target_coding = target_info.get("coding", "See CSES codebook")
        target_missing = target_info.get("missing", "97=Refused, 98=DK, 99=Missing")
        target_note = target_info.get("note", "")

        prompt = f"""Analyze this variable mapping and generate a recoding strategy.

SOURCE VARIABLE:
- Name: {source_var}
- Description: {source_desc}
- Value labels: {labels_str if labels_str else 'Not available'}
- Sample values: {samples_str if samples_str else 'Not available'}

TARGET CSES VARIABLE:
- Name: {target_var}
- Description: {target_desc}
- Expected coding: {target_coding}
- Missing codes: {target_missing}
{f"- Note: {target_note}" if target_note else ""}

MATCHING REASONING: {mapping_reasoning}

TASK: Analyze the source and target coding and generate a recoding strategy.

Consider:
1. DIRECT COPY: If source coding exactly matches target (same scale, same values)
2. SIMPLE RECODE: If source has different codes but same scale (e.g., 1-4 vs 0-3)
3. SCALE CONVERSION: If source has different number of categories
4. MISSING VALUE MAPPING: How to map source missing to CSES missing codes (97/98/99 or 7/8/9)
5. SPECIAL CASES: Age groups to midpoints, string to numeric, etc.

Return JSON:
{{
  "transformation_type": "direct|recode|calculate|complex",
  "requires_recoding": true,
  "recode_rules": [
    {{"from": "1", "to": "1", "comment": "Male stays 1"}},
    {{"from": "2", "to": "2", "comment": "Female stays 2"}},
    {{"from": ".", "to": "9", "comment": "System missing to 9"}}
  ],
  "stata_pattern": "gen|recode|replace",
  "warnings": ["any issues to flag for review"],
  "confidence": 0.85,
  "reasoning": "Brief explanation of the transformation"
}}

RULES:
- Include ALL necessary recode rules (don't skip values)
- Always map missing values explicitly
- Use "direct" type only if no changes needed
- Confidence should reflect how certain you are about the mapping
- Return ONLY valid JSON"""

        try:
            response = self.runner.response(
                ModelRole.RECODE_CODEGEN,
                model_override=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                timeout=60,
                purpose=f"Generate recoding strategy for {target_var}",
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content.strip() if response.choices[0].message.content else ""

            # Parse JSON
            json_data = self._extract_json(response_text)

            if json_data:
                recode_rules = []
                for rule in json_data.get("recode_rules", []):
                    recode_rules.append(RecodeRule(
                        from_value=str(rule.get("from", "")),
                        to_value=str(rule.get("to", "")),
                        comment=rule.get("comment", "")
                    ))

                return RecodingStrategy(
                    target_var=target_var,
                    source_var=source_var,
                    transformation_type=json_data.get("transformation_type", "recode"),
                    requires_recoding=json_data.get("requires_recoding", True),
                    recode_rules=recode_rules,
                    stata_pattern=json_data.get("stata_pattern", "gen"),
                    warnings=json_data.get("warnings", []),
                    confidence=float(json_data.get("confidence", 0.5)),
                    reasoning=json_data.get("reasoning", "")
                )

        except Exception as e:
            logger.error(f"Recoding strategy generation failed for {target_var}: {e}")

        # Fallback: simple direct copy
        return RecodingStrategy(
            target_var=target_var,
            source_var=source_var,
            transformation_type="direct",
            requires_recoding=False,
            recode_rules=[],
            stata_pattern="gen",
            warnings=["LLM strategy generation failed - using direct copy"],
            confidence=0.3,
            reasoning="Fallback to direct copy"
        )

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from LLM response."""
        # Try code block first
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON object directly
        if '{' in text:
            start = text.find('{')
            depth = 0
            end = start
            for i, c in enumerate(text[start:], start):
                if c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass

        return None


def create_recoding_strategist() -> RecodingStrategist:
    """Factory function to create a RecodingStrategist instance."""
    return RecodingStrategist()
