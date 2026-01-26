"""
LLM-based variable matching engine for CSES harmonization.

This is the CORE of the agentic approach - using AI to semantically understand:
1. Questionnaire content (question text in any language)
2. Codebook documentation (variable descriptions, value labels)
3. Data file metadata (variable labels, sample values)

The LLM reads ALL available context and reasons about what each source variable
represents, then maps it to the appropriate CSES Module 6 target variable.

Uses TOON format (Token-Oriented Object Notation) for efficient context encoding.
See: https://github.com/toon-format/spec
"""

import logging
import json
import re
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from litellm import completion

from src.utils.toon_encoder import encode_table

logger = logging.getLogger(__name__)


# CSES Module 6 target variables - clean descriptions
CSES_TARGET_VARIABLES = {
    # Demographics
    "F2001_Y": "Year of birth",
    "F2001_A": "Age in years",
    "F2002": "Gender",
    "F2003": "Education level",
    "F2004": "Marital status",
    "F2005": "Union membership",
    "F2006": "Employment status",
    "F2007": "Occupation",
    "F2008": "Socio-economic status",
    "F2009": "Public or private sector",
    "F2010_1": "Household income",
    "F2011": "Religious denomination",
    "F2012": "Religious attendance",
    "F2013": "Race",
    "F2014": "Ethnicity",
    "F2015": "Country of birth",
    "F2016": "Parent born abroad",
    "F2017": "Language at home",
    "F2018": "Region of residence",
    "F2020": "Rural or urban",
    "F2021": "Household size",
    # Survey questions
    "F3001": "Interest in politics",
    "F3002_1": "Media: Public TV news",
    "F3002_2": "Media: Private TV news",
    "F3002_3": "Media: Radio news",
    "F3002_4": "Media: Newspapers",
    "F3002_5": "Media: Online news",
    "F3002_6_1": "Media: Social media",
    "F3003": "Internal efficacy",
    "F3004_1": "Democracy is preferable",
    "F3004_2": "Courts should stop government",
    "F3004_3": "Need for strong leader",
    "F3004_4": "Representation in democracy",
    "F3005_1": "Country run by business leaders",
    "F3005_2": "Country run by experts",
    "F3005_3": "Country run by referendums",
    "F3006": "How democratic is country",
    "F3007_1": "Trust in parliament",
    "F3007_2": "Trust in government",
    "F3007_3": "Trust in judiciary",
    "F3007_4": "Trust in scientists",
    "F3007_5": "Trust in political parties",
    "F3007_6": "Trust in traditional media",
    "F3007_7": "Trust in social media",
    "F3008_1": "Government performance",
    "F3008_2": "Government COVID-19 response",
    "F3009": "State of the economy",
    "F3010": "Voted in election",
    "F3011_LH_PL": "Vote choice (party)",
    "F3012_1": "Satisfaction with vote",
    "F3013": "Satisfaction with choices",
    "F3014": "Fairness of election",
    "F3017": "External efficacy",
    "F3018_A": "Like/dislike Party A",
    "F3018_B": "Like/dislike Party B",
    "F3018_C": "Like/dislike Party C",
    "F3019_A": "Left-right Party A",
    "F3019_B": "Left-right Party B",
    "F3019_C": "Left-right Party C",
    "F3020": "Left-right self-placement",
    "F3021": "Party identification",
    "F3022": "Party ID: which party",
    "F3023": "Party ID: strength",
    "F3024": "Satisfaction with democracy",
}


@dataclass
class MatchProposal:
    """Proposed variable mapping with confidence."""
    source_variable: str
    target_variable: str
    confidence: float
    confidence_level: str
    reasoning: str
    matched_by: str
    needs_review: bool = False

    def to_dict(self) -> dict:
        return {
            "source": self.source_variable,
            "target": self.target_variable,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level,
            "reasoning": self.reasoning,
            "matched_by": self.matched_by,
            "needs_review": self.needs_review
        }


@dataclass
class MatchingResult:
    """Result of batch matching operation."""
    proposals: list[MatchProposal] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    high_confidence_count: int = 0
    medium_confidence_count: int = 0
    low_confidence_count: int = 0

    def summary(self) -> str:
        return (
            f"Total proposals: {len(self.proposals)}\n"
            f"  High confidence: {self.high_confidence_count}\n"
            f"  Medium confidence: {self.medium_confidence_count}\n"
            f"  Low confidence: {self.low_confidence_count}\n"
            f"Unmatched: {len(self.unmatched)}"
        )


class LLMMatcher:
    """
    LLM-based variable matcher that uses FULL document context.

    This is the agentic approach - the AI reads questionnaire and codebook
    content to understand what each variable measures, then maps to CSES schema.

    Supports two modes:
    1. Pre-aggregated mode: Uses summary from DocumentAggregator (Stage 1)
    2. On-the-fly mode: Extracts definitions during matching (legacy)

    The pre-aggregated mode is preferred as it's more token-efficient:
    - Stage 1 (DocumentAggregator with gemma3): Reads all docs, creates summary
    - Stage 2 (this class with gpt-oss:120b): Reads summary, does matching
    """

    def __init__(self, model: Optional[str] = None):
        # Load model from environment - prefer LLM_MODEL_MATCH for two-stage pipeline
        # Falls back to LLM_MODEL for backwards compatibility
        self.model = model or os.getenv("LLM_MODEL_MATCH") or os.getenv("LLM_MODEL", "openai/gpt-oss:120b")
        # Use temperature=0 for consistent, deterministic outputs
        self.temperature = 0
        logger.info(f"LLMMatcher initialized with model: {self.model}")

    def _extract_codebook_definitions(self, codebook_text: str, source_varnames: list[str]) -> dict[str, str]:
        """
        Use LLM to extract variable definitions from raw codebook.

        This is FORMAT-AGNOSTIC - the LLM interprets whatever document structure
        the codebook uses. We do NOT assume any specific delimiters or patterns.

        Args:
            codebook_text: Raw codebook document text
            source_varnames: List of variable names to look for

        Returns:
            Dict mapping variable name -> extracted definition
        """
        if not codebook_text or not source_varnames:
            return {}

        logger.info(f"Extracting definitions for {len(source_varnames)} variables from {len(codebook_text)} char codebook")

        # Process in chunks to handle large codebooks
        # Each chunk gets analyzed for variable definitions
        chunk_size = 30000  # chars per chunk
        all_definitions = {}

        # Build compact variable list for the prompt
        varnames_str = ", ".join(source_varnames[:200])  # Limit to avoid huge prompts

        # Process codebook in overlapping chunks to catch variables at boundaries
        for i in range(0, len(codebook_text), chunk_size - 2000):  # 2000 char overlap
            chunk = codebook_text[i:i + chunk_size]
            chunk_num = i // (chunk_size - 2000) + 1

            prompt = f"""Extract variable definitions from this codebook section.

VARIABLES TO FIND (look for these exact names in the text):
{varnames_str}

CODEBOOK TEXT:
{chunk}

INSTRUCTIONS:
1. Find each variable name and its definition/description in the codebook
2. Include question text, value labels, and any other relevant information
3. Be comprehensive - capture the full meaning of each variable
4. Only include variables that are actually documented in this text section
5. Use the EXACT variable name as it appears (case-sensitive)

Return JSON (only include variables found in this section):
{{"definitions": {{"VARNAME1": "Full definition including question text and value labels", "VARNAME2": "Definition..."}}}}

Return ONLY valid JSON."""

            try:
                response = completion(
                    model=self.model,
                    max_tokens=4096,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )

                response_text = response.choices[0].message.content.strip()

                # Extract JSON
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
                if json_match:
                    response_text = json_match.group(1)
                if not response_text.startswith('{'):
                    json_start = response_text.find('{')
                    if json_start >= 0:
                        response_text = response_text[json_start:]

                result = json.loads(response_text)
                chunk_defs = result.get("definitions", {})

                # Merge into all_definitions (later chunks don't overwrite)
                for var, defn in chunk_defs.items():
                    if var not in all_definitions:
                        all_definitions[var] = defn

                logger.info(f"Chunk {chunk_num}: extracted {len(chunk_defs)} definitions")

            except Exception as e:
                logger.warning(f"Chunk {chunk_num} extraction failed: {e}")
                continue

        logger.info(f"Total extracted definitions: {len(all_definitions)}")
        return all_definitions

    def match_variables(
        self,
        source_contexts: list[dict],
        questionnaire_text: str = "",
        codebook_text: str = "",
        pre_aggregated_summary: Optional[str] = None,
        batch_size: int = 12  # Ignored - kept for backwards compatibility
    ) -> MatchingResult:
        """
        Match source variables to CSES targets in ONE LLM call.

        Args:
            source_contexts: List of dicts with variable metadata from data file
            questionnaire_text: Raw questionnaire text (used in on-the-fly mode)
            codebook_text: Raw codebook text (used in on-the-fly mode)
            pre_aggregated_summary: TOON summary from DocumentAggregator (Stage 1)
            batch_size: IGNORED - matching is now done in one call

        Returns:
            MatchingResult with proposals, confidence scores, and errors
        """
        result = MatchingResult()

        # Determine which mode we're using
        if pre_aggregated_summary:
            logger.info("Using pre-aggregated summary (one-shot matching)")
            logger.info(f"Summary size: {len(pre_aggregated_summary)} chars")
            source_index = pre_aggregated_summary
        else:
            logger.info("Using on-the-fly extraction (legacy mode)")
            logger.info(f"Source variables: {len(source_contexts)}")
            logger.info(f"Questionnaire: {len(questionnaire_text)} chars, Codebook: {len(codebook_text)} chars")

            # PASS 1: Extract codebook definitions using LLM (format-agnostic)
            source_varnames = [ctx.get('name', '') for ctx in source_contexts if ctx.get('name')]
            extracted_definitions = {}
            if codebook_text:
                logger.info("Pass 1: Extracting variable definitions from codebook...")
                extracted_definitions = self._extract_codebook_definitions(codebook_text, source_varnames)
                logger.info(f"Pass 1 complete: {len(extracted_definitions)} definitions extracted")

            # Build source index with extracted definitions
            source_index = self._build_source_index(source_contexts, extracted_definitions)

        logger.info(f"Source index size: {len(source_index)} chars")

        cses_targets = list(CSES_TARGET_VARIABLES.keys())
        logger.info(f"CSES targets to match: {len(cses_targets)} (one-shot)")

        # ONE-SHOT MATCHING - all targets in a single call
        try:
            proposals = self._match_all_targets(cses_targets, source_index)

            for proposal in proposals:
                result.proposals.append(proposal)

                if proposal.confidence_level == "high":
                    result.high_confidence_count += 1
                elif proposal.confidence_level == "medium":
                    result.medium_confidence_count += 1
                else:
                    result.low_confidence_count += 1

                if proposal.source_variable == "NOT_FOUND":
                    result.unmatched.append(proposal.target_variable)

        except Exception as e:
            logger.error(f"Matching failed: {e}")
            result.errors.append(f"Matching error: {str(e)}")
            for target in cses_targets:
                result.proposals.append(MatchProposal(
                    source_variable="ERROR", target_variable=target,
                    confidence=0.0, confidence_level="low",
                    reasoning=f"Error: {str(e)[:100]}", matched_by="error", needs_review=True
                ))
                result.unmatched.append(target)
                result.low_confidence_count += 1

        matched_count = len([p for p in result.proposals if p.source_variable not in ["NOT_FOUND", "ERROR"]])
        logger.info(f"Complete. {matched_count}/{len(cses_targets)} targets matched.")
        return result

    def _match_all_targets(
        self,
        cses_targets: list[str],
        source_index: str
    ) -> list[MatchProposal]:
        """
        Match ALL CSES targets to source variables in ONE LLM call.

        This is for CSES data where source variables should closely match
        the standardized CSES questions.
        """
        # Build TOON table for all targets
        target_rows = [{'code': c, 'desc': CSES_TARGET_VARIABLES[c]} for c in cses_targets]
        targets_toon = 'targets' + encode_table(target_rows, ['code', 'desc'], '|')

        prompt = f"""Match CSES target variables to source variables from the dataset.

This is CSES (Comparative Study of Electoral Systems) data - source variables should closely match
the standardized CSES Module 6 schema based on question text and meaning.

CSES TARGET VARIABLES TO MATCH:
{targets_toon}

SOURCE VARIABLES FROM DATASET:
{source_index}

TASK: For each CSES target, find the source variable that measures the SAME concept.
Match based on:
1. Question text similarity (most important)
2. Variable description
3. Value labels (response options)

Return JSON with ALL {len(cses_targets)} mappings:
{{"mappings":[
  {{"target":"F2001_Y","source":"SOURCE_VAR_NAME","confidence":0.95,"reasoning":"Question asks year of birth"}},
  {{"target":"F2002","source":"GENDER_VAR","confidence":0.98,"reasoning":"Measures gender with 1=Male,2=Female"}},
  ...
]}}

RULES:
- Return a mapping for EVERY target (use "NOT_FOUND" if no match)
- Confidence: 0.95+ for exact question match, 0.80-0.94 for similar, <0.80 for uncertain
- Each source variable can only map to ONE target
- Return ONLY valid JSON"""

        # Call LLM with enough tokens for all mappings
        # ~50 tokens per mapping × 64 targets = ~3200 tokens
        response = completion(
            model=self.model,
            max_tokens=8192,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        logger.info(f"Response length: {len(response_text)} chars")

        # Parse JSON response
        json_data = self._extract_json(response_text)

        if json_data is None:
            logger.error(f"JSON parse failed. Response preview: {response_text[:300]}")
            return [MatchProposal(
                source_variable="ERROR", target_variable=t, confidence=0.0,
                confidence_level="low", reasoning="JSON parse error",
                matched_by="error", needs_review=True
            ) for t in cses_targets]

        # Build proposals from mappings
        proposals = []
        seen_targets = set()
        assigned_sources = set()

        for mapping in json_data.get("mappings", []):
            target = mapping.get("target", "unknown")
            source = mapping.get("source", "NOT_FOUND")
            confidence = float(mapping.get("confidence", 0.0))
            reasoning = mapping.get("reasoning", "")

            # Validate: don't allow duplicate source assignments
            if source in assigned_sources and source not in ["NOT_FOUND", "ERROR"]:
                source = "NOT_FOUND"
                confidence = 0.0
                reasoning = f"Source already assigned to another target"

            if source not in ["NOT_FOUND", "ERROR"]:
                assigned_sources.add(source)

            level = "high" if confidence >= 0.85 else "medium" if confidence >= 0.60 else "low"

            proposals.append(MatchProposal(
                source_variable=source,
                target_variable=target,
                confidence=confidence,
                confidence_level=level,
                reasoning=reasoning,
                matched_by="llm_semantic",
                needs_review=(level != "high")
            ))
            seen_targets.add(target)

        # Add NOT_FOUND for any missing targets
        for target in cses_targets:
            if target not in seen_targets:
                proposals.append(MatchProposal(
                    source_variable="NOT_FOUND",
                    target_variable=target,
                    confidence=0.0,
                    confidence_level="low",
                    reasoning="Not returned by LLM",
                    matched_by="missing",
                    needs_review=True
                ))

        return proposals

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from LLM response."""
        # Try 1: Look for JSON in code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try 2: Find JSON object directly
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

        # Try 3: Parse whole response
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        return None

    def _build_source_index(self, source_contexts: list[dict], extracted_defs: dict[str, str] = None) -> str:
        """
        Build a TOON-formatted index of source variables.

        Combines:
        1. Data file metadata (variable labels, value labels)
        2. Extracted codebook definitions (from Pass 1 LLM extraction)

        This is token-efficient because the codebook extraction already
        distilled the key information from the raw document.
        """
        extracted_defs = extracted_defs or {}
        cses_direct_matches = []

        # Build rows for TOON table
        rows = []
        for ctx in source_contexts:
            name = ctx.get('name', '?')

            # Check if this variable already uses CSES naming
            if re.match(r'^F[12345]\d{3}', name):
                cses_direct_matches.append(name)

            # Priority: extracted codebook definition > data file description
            desc = ''
            if name in extracted_defs:
                # Use LLM-extracted definition (most informative)
                desc = extracted_defs[name]
            elif ctx.get('description'):
                desc = ctx['description']
            elif ctx.get('question_text'):
                desc = ctx['question_text']

            # Compact labels from data file
            labels_str = ''
            if ctx.get('value_labels') and isinstance(ctx['value_labels'], dict):
                labels = list(ctx['value_labels'].items())[:5]
                labels_str = ';'.join(f"{k}={v}" for k, v in labels)

            rows.append({
                'name': name,
                'desc': desc[:250] if desc else '',  # Allow longer since it's already distilled
                'labels': labels_str
            })

        # Build TOON table with pipe delimiter
        toon_table = 'vars' + encode_table(rows, ['name', 'desc', 'labels'], delimiter='|')

        # Add note about CSES-named variables if present
        if cses_direct_matches:
            header = f"note: Source contains CSES-named vars: {','.join(cses_direct_matches[:10])}\n"
            toon_table = header + toon_table

        return toon_table

    def _match_batch_with_index(
        self,
        cses_targets: list[str],
        source_index: str,
        assigned_sources: set
    ) -> list[MatchProposal]:
        """
        Match a batch of CSES targets using TOON-formatted source index.

        The source_index contains pre-extracted codebook definitions (from Pass 1),
        so no need to pass raw codebook again - this is token-efficient.
        """
        # Build TOON table for targets
        target_rows = [{'code': c, 'desc': CSES_TARGET_VARIABLES[c]} for c in cses_targets]
        targets_toon = 'targets' + encode_table(target_rows, ['code', 'desc'], '|')

        # Compact exclusion list
        excluded = ','.join(sorted(assigned_sources)[:30]) if assigned_sources else 'none'

        prompt = f"""Match CSES targets to source variables. Return JSON ONLY.

TARGETS:
{targets_toon}

SOURCE VARIABLES:
{source_index}

ALREADY ASSIGNED (cannot reuse): {excluded}

For each target, find the source variable measuring the SAME concept.
Use NOT_FOUND if no match exists.

RESPOND WITH VALID JSON ONLY. NO OTHER TEXT.
Example format:
{{"mappings":[{{"target":"F2002","source":"GENDER_VAR","confidence":0.95,"reasoning":"measures gender"}}]}}"""

        # Try up to 2 times for valid JSON response
        for attempt in range(2):
            response = completion(
                model=self.model,
                max_tokens=4096,
                temperature=self.temperature,  # 0 for consistency
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
            if response_text and '{' in response_text:
                break
            logger.warning(f"Attempt {attempt + 1}: Empty or invalid response, retrying...")

        # Robust JSON extraction
        json_data = None

        # Try 1: Look for JSON in code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            try:
                json_data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try 2: Find JSON object directly
        if json_data is None and '{' in response_text:
            # Find the outermost JSON object
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
                    json_data = json.loads(response_text[start:end])
                except json.JSONDecodeError:
                    pass

        # Try 3: Parse the whole response as JSON
        if json_data is None:
            try:
                json_data = json.loads(response_text)
            except json.JSONDecodeError:
                pass

        if json_data is None:
            logger.error(f"JSON parse failed. Response preview: {response_text[:200]}")
            return [MatchProposal(
                source_variable="ERROR", target_variable=t, confidence=0.0,
                confidence_level="low", reasoning="JSON parse error",
                matched_by="error", needs_review=True
            ) for t in cses_targets]

        result = json_data

        proposals = []
        for mapping in result.get("mappings", []):
            source = mapping.get("source", "NOT_FOUND")
            confidence = float(mapping.get("confidence", 0.0))

            # Validate: don't allow already-assigned sources
            if source in assigned_sources:
                source = "NOT_FOUND"
                confidence = 0.0
                reasoning = f"Originally matched to {mapping.get('source')} but already assigned"
            else:
                reasoning = mapping.get("reasoning", "")

            level = "high" if confidence >= 0.85 else "medium" if confidence >= 0.60 else "low"

            proposals.append(MatchProposal(
                source_variable=source,
                target_variable=mapping.get("target", "unknown"),
                confidence=confidence,
                confidence_level=level,
                reasoning=reasoning,
                matched_by="llm_semantic",
                needs_review=(level != "high")
            ))

        return proposals

    def _match_cses_targets(
        self,
        cses_targets: list[str],
        source_contexts: list[dict],
        questionnaire_text: str,
        codebook_text: str,
        already_matched: set
    ) -> list[MatchProposal]:
        """For each CSES target, find the best matching source variable."""

        # Build detailed source variable list with all available context
        source_lines = []
        for ctx in source_contexts:
            name = ctx.get('name', '?')
            parts = [f"[{name}]"]

            if ctx.get('description'):
                parts.append(f" Desc: {ctx['description'][:80]}")
            if ctx.get('question_text'):
                parts.append(f" Q: {ctx['question_text'][:80]}")
            if ctx.get('value_labels') and isinstance(ctx['value_labels'], dict):
                labels = ", ".join(f"{k}={v}" for k, v in list(ctx['value_labels'].items())[:5])
                parts.append(f" Labels: {labels}")
            if ctx.get('sample_values'):
                parts.append(f" Samples: {ctx['sample_values'][:3]}")

            source_lines.append("".join(parts))

        sources_text = "\n".join(source_lines)

        # Build CSES targets to find
        targets_text = "\n".join(f"- {code}: {CSES_TARGET_VARIABLES[code]}" for code in cses_targets)

        # Use more document context - smarter truncation
        q_text = questionnaire_text[:20000] if questionnaire_text else "(no questionnaire provided)"
        c_text = codebook_text[:15000] if codebook_text else "(no codebook provided)"

        # Note already matched vars
        excluded_note = ""
        if already_matched:
            excluded_note = f"\nALREADY MATCHED (do not use): {', '.join(list(already_matched)[:20])}"

        prompt = f"""Find the best source variable for each CSES target.

CSES TARGETS TO FIND:
{targets_text}
{excluded_note}

SOURCE VARIABLES (with all available metadata):
{sources_text}

=== QUESTIONNAIRE DOCUMENT ===
{q_text}

=== CODEBOOK DOCUMENT ===
{c_text}

INSTRUCTIONS:
1. For each CSES target, find the source variable that measures the same concept
2. Use ALL evidence: variable name, description, question text, value labels, codebook entries
3. In reasoning, cite the SPECIFIC evidence (e.g., "Codebook says Q1 measures political interest" or "Value labels 1=Male,2=Female indicate gender")
4. Use "NOT_FOUND" if no source variable measures this concept

Return JSON:
{{"mappings": [
  {{"target": "F3001", "source": "Q1", "confidence": 0.95, "reasoning": "Codebook: 'Q1. How interested are you in politics?' with labels 1=Very interested to 4=Not at all"}}
]}}

Return ONLY valid JSON."""

        response = completion(
            model=self.model,
            max_tokens=4096,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.choices[0].message.content.strip()

        # Extract JSON
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response_text)
        if json_match:
            response_text = json_match.group(1)

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response: {response_text[:500]}")
            return [MatchProposal(
                source_variable="ERROR",
                target_variable=t,
                confidence=0.0,
                confidence_level="low",
                reasoning="Failed to parse LLM response",
                matched_by="error",
                needs_review=True
            ) for t in cses_targets]

        proposals = []
        for mapping in result.get("mappings", []):
            confidence = float(mapping.get("confidence", 0.0))
            level = "high" if confidence >= 0.85 else "medium" if confidence >= 0.60 else "low"

            proposals.append(MatchProposal(
                source_variable=mapping.get("source", "NOT_FOUND"),
                target_variable=mapping.get("target", "unknown"),
                confidence=confidence,
                confidence_level=level,
                reasoning=mapping.get("reasoning", ""),
                matched_by="llm_semantic",
                needs_review=(level != "high")
            ))

        return proposals


# Keep these for backwards compatibility with __init__.py exports
class PatternMatcher:
    """Deprecated - kept for backwards compatibility."""
    def __init__(self):
        pass
    def match(self, var_name: str, description: str = None, value_labels: dict = None):
        return None


def create_matcher(model: Optional[str] = None) -> LLMMatcher:
    """Create a matcher instance using environment configuration."""
    return LLMMatcher(model=model)
