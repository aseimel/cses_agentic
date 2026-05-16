"""
Tool Wrappers for CSES Agent Integration.

Wraps existing modules as callable tools for agent orchestration.
These wrappers provide a consistent interface for the agent to
interact with the underlying functionality.
"""

import logging
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass
import time

# Import existing modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingest import extract_context, ExtractionResult
from src.ingest.data_loader import DataLoader, DatasetInfo
from src.ingest.doc_parser import DocumentParser, DocumentInfo
from src.matching import create_matcher, MatchingResult
from src.matching.party_codes import (
    PartyOrderResult,
    get_un_country_code,
    generate_party_codes,
    extract_party_results_from_macro,
    generate_party_code_stata_section,
)
from src.preprocessing import DocumentAggregator
from src.matching.recoding_strategist import RecodingStrategist, RecodingStrategy
from src.reference.sweden_patterns import load_sweden_patterns, get_examples_for_code_generation

logger = logging.getLogger(__name__)


def generate_stata_code_for_variable(
    target_var: str,
    source_var: str,
    source_info: dict,
    reasoning: str = "",
    recoding_strategy: Optional[RecodingStrategy] = None,
    use_sweden_patterns: bool = True
) -> str:
    """
    Use LLM to generate appropriate Stata code for a single variable mapping.

    This is the key innovation - instead of using templates, the LLM generates
    contextually appropriate Stata code based on the specific source and target,
    using Sweden reference patterns as examples and recoding strategies for guidance.

    Args:
        target_var: CSES target variable code (e.g., "F2001_A")
        source_var: Source variable name from deposited data
        source_info: Dict with source variable metadata (description, value_labels, sample_values)
        reasoning: Original reasoning from the matcher
        recoding_strategy: Optional RecodingStrategy with transformation rules
        use_sweden_patterns: Whether to include Sweden reference examples

    Returns:
        String of Stata code for this variable
    """
    from src.config import LLM_MODEL_CODE, LLM_TEMPERATURE
    from src.model_runtime import ModelRole, ModelTaskRunner
    import json
    import re

    # Get CSES coding expectations
    cses_info = CSES_EXPECTED_CODINGS.get(target_var, {})
    target_desc = cses_info.get("desc", "See CSES codebook")
    target_coding = cses_info.get("coding", "See CSES codebook")

    # For similar variables (F3002_2 is like F3002_1), use base pattern
    base_var = re.sub(r'_[A-I]$', '_A', target_var)  # F3018_B -> F3018_A
    base_var = re.sub(r'_\d+$', '_1', base_var)  # F3002_2 -> F3002_1
    if base_var != target_var and base_var in CSES_EXPECTED_CODINGS:
        target_coding = CSES_EXPECTED_CODINGS[base_var].get("coding", target_coding)

    # Build source info string
    source_desc = source_info.get('description', '') or source_info.get('desc', '')
    source_labels = source_info.get('value_labels', {})
    source_samples = source_info.get('sample_values', [])

    labels_str = ""
    if source_labels and isinstance(source_labels, dict):
        labels_str = ", ".join(f"{k}={v}" for k, v in list(source_labels.items())[:10])

    samples_str = ""
    if source_samples:
        samples_str = ", ".join(str(s) for s in source_samples[:10])

    # Build recoding strategy section
    recode_section = ""
    if recoding_strategy:
        recode_section = f"""
RECODING STRATEGY (pre-analyzed transformation):
- Transformation type: {recoding_strategy.transformation_type}
- Stata pattern: {recoding_strategy.stata_pattern}
- Confidence: {recoding_strategy.confidence:.2f}
- Reasoning: {recoding_strategy.reasoning}
"""
        if recoding_strategy.recode_rules:
            recode_section += "- Recode rules:\n"
            for rule in recoding_strategy.recode_rules[:10]:
                recode_section += f"    {rule.from_value} -> {rule.to_value} ({rule.comment})\n"
        if recoding_strategy.warnings:
            recode_section += f"- Warnings: {', '.join(recoding_strategy.warnings)}\n"

    # Get Sweden reference patterns
    sweden_examples = ""
    if use_sweden_patterns:
        try:
            pattern_type = recoding_strategy.stata_pattern if recoding_strategy else "gen"
            sweden_examples = get_examples_for_code_generation(target_var, pattern_type)
            if sweden_examples and sweden_examples != "# No reference patterns available":
                sweden_examples = f"""
SWEDEN REFERENCE EXAMPLES (follow this style exactly):
{sweden_examples}
"""
        except Exception as e:
            logger.debug(f"Could not load Sweden patterns: {e}")

    prompt = f"""Generate Stata code to create CSES variable {target_var} from source variable {source_var}.

TARGET CSES VARIABLE:
- Name: {target_var}
- Description: {target_desc}
- Expected coding: {target_coding}

SOURCE VARIABLE:
- Name: {source_var}
- Description: {source_desc}
- Value labels: {labels_str if labels_str else 'Not available'}
- Sample values: {samples_str if samples_str else 'Not available'}

MATCHING REASONING: {reasoning}
{recode_section}
{sweden_examples}

REQUIREMENTS:
1. Start with section header: **>>> {target_var} - {target_desc}
2. Add coding notes comment block if complex transformation
3. Use appropriate command: gen, recode, or replace
4. Always end with: tab {target_var}, mis
5. Add cross-tab if recoding: tab SOURCE {target_var}, mis
6. Handle missing values explicitly (., 97, 98, 99 codes)

INSTRUCTIONS:
Generate the Stata code to create {target_var} from {source_var}. Consider:

1. If source and target codings match directly, use simple gen:
   gen {target_var} = {source_var}

2. If source needs recoding (different scale, different labels), use recode or replace:
   gen {target_var} = {source_var}
   recode {target_var} (old1=new1) (old2=new2) ...

3. If source is categorical (e.g., age groups) but target needs numeric (e.g., actual age):
   gen {target_var} = .
   replace {target_var} = 24 if {source_var} == 1  // 18-29 midpoint
   replace {target_var} = 35 if {source_var} == 2  // 30-39 midpoint
   ...

4. Handle missing values - map source missing to CSES missing codes:
   - 7/97/997 = Refused
   - 8/98/998 = Don't know
   - 9/99/999 = Missing/NA

Return ONLY the Stata code, no explanations. Include brief comments with * prefix where helpful.
Do NOT include any markdown formatting or code blocks."""

    try:
        runner = ModelTaskRunner()
        response = runner.response(
            ModelRole.RECODE_CODEGEN,
            model_override=runner.model_for_role(ModelRole.RECODE_CODEGEN) or LLM_MODEL_CODE,
            max_tokens=1024,
            temperature=LLM_TEMPERATURE,
            timeout=60,
            purpose=f"Generate Stata code for {target_var}",
            messages=[{"role": "user", "content": prompt}]
        )

        code = response.choices[0].message.content.strip()

        # Clean up any markdown code blocks if present
        code = re.sub(r'^```stata?\s*', '', code)
        code = re.sub(r'\s*```$', '', code)
        code = code.strip()

        # Ensure it ends with tab command
        if f"tab {target_var}" not in code:
            code += f"\ntab {target_var}, mis"

        return code

    except Exception as e:
        logger.warning(f"LLM code generation failed for {target_var}: {e}")
        # Fallback to simple gen
        return f"* LLM code generation failed: {str(e)[:50]}\ngen {target_var} = {source_var}\ntab {target_var}, mis"


def generate_do_file_with_llm(
    mappings: list[dict],
    source_contexts: list[dict],
    country_code: str,
    country_name: str,
    year: str,
    author: str = "CSES Tool",
    data_file_path: str = "",
    party_result: Optional[PartyOrderResult] = None,
    study_number: int = 0,
    progress_callback: Optional[callable] = None,
    recoding_strategies: Optional[dict[str, RecodingStrategy]] = None
) -> str:
    """
    Generate complete .do file content using LLM for variable code generation.

    This function orchestrates the LLM-powered code generation for each variable,
    building a complete, executable .do file. Optionally uses pre-generated
    recoding strategies and Sweden reference patterns for better code quality.

    Args:
        mappings: List of variable mappings from matcher
        source_contexts: Source variable metadata
        country_code: ISO alpha-3 country code
        country_name: Full country name
        year: Election year
        author: Author for header
        data_file_path: Path to source data file
        party_result: Optional party code information
        study_number: Study number within country
        progress_callback: Optional callback for progress updates
        recoding_strategies: Optional dict of target_var -> RecodingStrategy

    Returns:
        Complete .do file content as string
    """
    from datetime import datetime

    def update_progress(msg: str):
        logger.info(msg)
        if progress_callback:
            progress_callback(msg)

    today = datetime.now().strftime("%B %d, %Y")
    un_code = get_un_country_code(country_code) or get_un_country_code(country_name) or "XXX"

    # Build lookups
    mapping_lookup = {}
    for m in mappings:
        target = m.get("target") or m.get("cses_variable") or m.get("target_variable")
        if target:
            mapping_lookup[target] = m

    source_lookup = {}
    for ctx in source_contexts:
        name = ctx.get('name', '')
        if name:
            source_lookup[name] = ctx

    lines = []

    # Header
    lines.append("/***************************************************************************")
    lines.append("**                     Process CSES-M6 Micro-Data                         **")
    lines.append("**                     **************************                         **")
    lines.append("**                                                                        **")
    lines.append(f"** File Author:      {author:<52} **")
    lines.append(f"** Date:             {today:<52} **")
    lines.append(f"** CSES MODULE 6:    {country_name.upper()} {year:<40} **")
    lines.append("**                                                                        **")
    lines.append("** Generated with LLM-powered code generation                            **")
    lines.append("***************************************************************************/")
    lines.append("")

    # Syntax instructions
    lines.append("*-------------------------------------------------------------------------**")
    lines.append("****************************************************************************")
    lines.append("**\\\\\\              SYNTAX INSTRUCTIONS                                    **")
    lines.append("****************************************************************************")
    lines.append("*-------------------------------------------------------------------------**")
    lines.append('* \\\\\\  Section Headings')
    lines.append('* >>> Variable Headings')
    lines.append("****************************************************************************")
    lines.append("")

    # Open data section
    lines.append("*-------------------------------------------------------------------------*")
    lines.append("***************************************************************************")
    lines.append("**\\\\\\              OPEN DATA")
    lines.append("***************************************************************************")
    lines.append("*-------------------------------------------------------------------------*")
    lines.append("")
    lines.append("clear")
    lines.append("set more off")
    lines.append("capture log close")
    lines.append("")
    if data_file_path:
        lines.append(f'use "{data_file_path}", clear')
    else:
        lines.append('* TODO: Specify path to data file')
        lines.append('use "YOUR_DATA_FILE.dta", clear')
    lines.append("")

    # Fixed ID variables
    lines.append("*-------------------------------------------------------------------------*")
    lines.append("***************************************************************************")
    lines.append("**\\\\\\       ID, WEIGHT, AND ADMINISTRATION VARIABLES")
    lines.append("***************************************************************************")
    lines.append("*-------------------------------------------------------------------------*")
    lines.append("")

    # F1001 - Dataset
    lines.append("***************************************************************************")
    lines.append("**>>> F1001 - DATASET")
    lines.append("***************************************************************************")
    lines.append('gen str13 F1001 = "CSES-MODULE-6"')
    lines.append("tab F1001, mis")
    lines.append("")

    # F1004 - Study ID
    lines.append("***************************************************************************")
    lines.append("**>>> F1004 - ID VARIABLE - ELECTION STUDY (ALPHABETIC)")
    lines.append("***************************************************************************")
    lines.append(f'gen str F1004 = "{country_code}_{year}"')
    lines.append("tab F1004, mis")
    lines.append("")

    # F1005 - Numeric study ID
    lines.append("***************************************************************************")
    lines.append("**>>> F1005 - ID VARIABLE - ELECTION STUDY (NUMERIC)")
    lines.append("***************************************************************************")
    lines.append(f"gen long F1005 = {un_code}{study_number}{year}")
    lines.append("format F1005 %8.0f")
    lines.append("tab F1005, mis")
    lines.append("")

    # F1006 - Polity code
    lines.append("***************************************************************************")
    lines.append("**>>> F1006 - ID COMPONENT - POLITY CSES CODE")
    lines.append("***************************************************************************")
    lines.append(f'gen str F1006 = "{un_code}{study_number}"')
    lines.append("tab F1006, mis")
    lines.append("")

    # F1009 - Election year
    lines.append("***************************************************************************")
    lines.append("**>>> F1009 - ID COMPONENT - ELECTION YEAR")
    lines.append("***************************************************************************")
    lines.append(f"gen F1009 = {year}")
    lines.append("tab F1009, mis")
    lines.append("")

    # Demographics section
    lines.append("")
    lines.append("*-------------------------------------------------------------------------*")
    lines.append("***************************************************************************")
    lines.append("**\\\\\\       DEMOGRAPHICS")
    lines.append("***************************************************************************")
    lines.append("*-------------------------------------------------------------------------*")

    # Variable lists by category
    demo_vars = ["F2001_Y", "F2001_A", "F2002", "F2003", "F2004", "F2005", "F2006",
                 "F2007", "F2008", "F2009", "F2010_2", "F2011", "F2012", "F2013",
                 "F2014", "F2015", "F2016", "F2017", "F2018", "F2019", "F2020", "F2021"]

    survey_vars = ["F3001", "F3002_1", "F3002_2", "F3002_3", "F3002_4", "F3002_5",
                   "F3002_6_1", "F3003", "F3004_1", "F3004_2", "F3004_3", "F3004_4",
                   "F3005_1", "F3005_2", "F3005_3", "F3006", "F3007_1", "F3007_2",
                   "F3007_3", "F3007_4", "F3007_5", "F3007_6", "F3007_7", "F3008_1",
                   "F3008_2", "F3009", "F3010", "F3011_LH_PL", "F3012_1", "F3013",
                   "F3014", "F3017", "F3018_A", "F3018_B", "F3018_C", "F3019_A",
                   "F3019_B", "F3019_C", "F3020", "F3021", "F3022", "F3023", "F3024"]

    # Process demographics
    matched_count = 0
    total_count = 0

    for var_code in demo_vars:
        total_count += 1
        m = mapping_lookup.get(var_code, {})
        source = m.get("source") or m.get("source_variable") or ""
        reasoning = m.get("reasoning") or m.get("notes") or ""

        lines.append("")
        lines.append("***************************************************************************")
        cses_desc = CSES_EXPECTED_CODINGS.get(var_code, {}).get("desc", var_code)
        lines.append(f"**>>> {var_code} - {cses_desc}")
        lines.append("***************************************************************************")
        lines.append("")

        if source and source not in ["NOT_FOUND", "ERROR", "NO_CONSENSUS"]:
            matched_count += 1
            update_progress(f"Generating Stata code for {var_code} <- {source}")

            # Get source info
            source_info = source_lookup.get(source, {})

            # Get recoding strategy if available
            recode_strategy = recoding_strategies.get(var_code) if recoding_strategies else None

            # Generate code using LLM
            stata_code = generate_stata_code_for_variable(
                target_var=var_code,
                source_var=source,
                source_info=source_info,
                reasoning=reasoning,
                recoding_strategy=recode_strategy,
                use_sweden_patterns=True
            )

            lines.append(f"* Source: {source}")
            if reasoning:
                lines.append(f"* Reasoning: {reasoning[:100]}")
            lines.append(stata_code)
        else:
            lines.append(f"* Source: NOT FOUND - requires manual mapping or collaborator input")
            lines.append(f"* gen {var_code} = .")
            lines.append(f"* tab {var_code}, mis")

        lines.append("")

    # Survey questions section
    lines.append("")
    lines.append("*-------------------------------------------------------------------------*")
    lines.append("***************************************************************************")
    lines.append("**\\\\\\       SURVEY QUESTIONS")
    lines.append("***************************************************************************")
    lines.append("*-------------------------------------------------------------------------*")

    for var_code in survey_vars:
        total_count += 1
        m = mapping_lookup.get(var_code, {})
        source = m.get("source") or m.get("source_variable") or ""
        reasoning = m.get("reasoning") or m.get("notes") or ""

        lines.append("")
        lines.append("***************************************************************************")
        cses_desc = CSES_EXPECTED_CODINGS.get(var_code, {}).get("desc", var_code)
        lines.append(f"**>>> {var_code} - {cses_desc}")
        lines.append("***************************************************************************")
        lines.append("")

        if source and source not in ["NOT_FOUND", "ERROR", "NO_CONSENSUS"]:
            matched_count += 1
            update_progress(f"Generating Stata code for {var_code} <- {source}")

            source_info = source_lookup.get(source, {})

            # Get recoding strategy if available
            recode_strategy = recoding_strategies.get(var_code) if recoding_strategies else None

            stata_code = generate_stata_code_for_variable(
                target_var=var_code,
                source_var=source,
                source_info=source_info,
                reasoning=reasoning,
                recoding_strategy=recode_strategy,
                use_sweden_patterns=True
            )

            lines.append(f"* Source: {source}")
            if reasoning:
                lines.append(f"* Reasoning: {reasoning[:100]}")
            lines.append(stata_code)
        else:
            lines.append(f"* Source: NOT FOUND - requires manual mapping or collaborator input")
            lines.append(f"* gen {var_code} = .")
            lines.append(f"* tab {var_code}, mis")

        lines.append("")

    # Save section
    lines.append("")
    lines.append("*-------------------------------------------------------------------------*")
    lines.append("***************************************************************************")
    lines.append("**\\\\\\       SAVE PROCESSED DATA")
    lines.append("***************************************************************************")
    lines.append("*-------------------------------------------------------------------------*")
    lines.append("")
    lines.append(f'save "cses-m6_micro_{country_code}_{year}.dta", replace')
    lines.append("")
    lines.append(f"* Matched {matched_count}/{total_count} variables")
    lines.append("")

    update_progress(f"Generated .do file with {matched_count}/{total_count} variables matched")

    return "\n".join(lines)


# CSES target variable expected codings for LLM code generation
CSES_EXPECTED_CODINGS = {
    "F2001_Y": {"desc": "Year of birth", "coding": "4-digit year (1900-2010), 9999=Missing"},
    "F2001_A": {"desc": "Age in years", "coding": "Numeric age 18-120, 9999=Missing. If source is categorical age groups, convert to midpoints."},
    "F2002": {"desc": "Gender", "coding": "1=Male, 2=Female, 3=Other, 7=Refused, 8=DK, 9=Missing"},
    "F2003": {"desc": "Education level", "coding": "1-8 scale (1=None to 8=Post-grad), 7=Refused, 8=DK, 9=Missing"},
    "F2004": {"desc": "Marital status", "coding": "1=Married, 2=Living with partner, 3=Separated, 4=Divorced, 5=Widowed, 6=Single, 7=Refused, 8=DK, 9=Missing"},
    "F2005": {"desc": "Union membership", "coding": "1=Member, 2=Household member, 3=Not member, 7=Refused, 8=DK, 9=Missing"},
    "F2006": {"desc": "Employment status", "coding": "1=Full-time, 2=Part-time, 3=Unemployed, 4=Student, 5=Retired, 6=Homemaker, 7=Other, 97=Refused, 98=DK, 99=Missing"},
    "F2011": {"desc": "Religious denomination", "coding": "Numeric codes vary by country, 9997=Refused, 9998=DK, 9999=Missing"},
    "F2012": {"desc": "Religious attendance", "coding": "1=Every week, 2=Almost every week, 3=Once/twice a month, 4=Few times a year, 5=Once a year, 6=Less often, 7=Never, 97=Refused, 98=DK, 99=Missing"},
    "F2020": {"desc": "Rural/Urban", "coding": "1=Rural, 2=Small town, 3=Suburb, 4=Urban, 7=Refused, 8=DK, 9=Missing"},
    "F2021": {"desc": "Household size", "coding": "Numeric count 1-20, 97=Refused, 98=DK, 99=Missing"},
    "F3001": {"desc": "Political interest", "coding": "1=Very interested, 2=Somewhat, 3=Not very, 4=Not at all, 7=Refused, 8=DK, 9=Missing"},
    "F3002_1": {"desc": "Media: Public TV", "coding": "1=Every day, 2=3-4 days, 3=1-2 days, 4=Less often, 5=Never, 7=Refused, 8=DK, 9=Missing"},
    "F3003": {"desc": "Internal efficacy", "coding": "1=Strongly agree, 2=Agree, 3=Neither, 4=Disagree, 5=Strongly disagree, 7=Refused, 8=DK, 9=Missing"},
    "F3006": {"desc": "How democratic", "coding": "0-10 scale (0=Not democratic, 10=Completely democratic), 97=Refused, 98=DK, 99=Missing"},
    "F3007_1": {"desc": "Trust: Parliament", "coding": "0-10 scale (0=No trust, 10=Complete trust), 97=Refused, 98=DK, 99=Missing"},
    "F3009": {"desc": "State of economy", "coding": "1=Very good, 2=Good, 3=Neither, 4=Bad, 5=Very bad, 7=Refused, 8=DK, 9=Missing"},
    "F3010": {"desc": "Voted in election", "coding": "1=Yes voted, 2=No did not vote, 7=Refused, 8=DK, 9=Missing"},
    "F3018_A": {"desc": "Like/Dislike Party A", "coding": "0-10 scale (0=Strongly dislike, 10=Strongly like), 96=Haven't heard, 97=Refused, 98=DK, 99=Missing"},
    "F3019_A": {"desc": "Left-Right Party A", "coding": "0-10 scale (0=Left, 10=Right), 95=Don't know L-R, 96=Haven't heard, 97=Refused, 98=DK, 99=Missing"},
    "F3020": {"desc": "Left-Right self", "coding": "0-10 scale (0=Left, 10=Right), 95=Don't know L-R, 97=Refused, 98=DK, 99=Missing"},
    "F3021": {"desc": "Party identification", "coding": "1=Yes has party ID, 2=No, 7=Refused, 8=DK, 9=Missing"},
    "F3024": {"desc": "Satisfaction with democracy", "coding": "1=Very satisfied, 2=Fairly, 3=Not very, 4=Not at all, 7=Refused, 8=DK, 9=Missing"},
}


@dataclass
class ToolResult:
    """Standard result wrapper for tool calls."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============================================================================
# DATA LOADING TOOLS
# ============================================================================

def load_data_file(file_path: Path | str) -> ToolResult:
    """
    Load a data file and extract variable metadata.

    Supports: .dta, .sav, .csv, .xlsx, .json, .parquet

    Returns:
        ToolResult with DatasetInfo on success
    """
    try:
        file_path = Path(file_path)
        loader = DataLoader()
        dataset_info = loader.load(file_path)

        if dataset_info:
            return ToolResult(
                success=True,
                data=dataset_info,
                metadata={
                    "file": str(file_path),
                    "n_variables": len(dataset_info.variables),
                    "n_rows": dataset_info.n_rows,
                    "format": file_path.suffix.lower()
                }
            )
        else:
            return ToolResult(
                success=False,
                error=f"Failed to load data from {file_path}"
            )
    except Exception as e:
        logger.error(f"Error loading data file: {e}")
        return ToolResult(success=False, error=str(e))


def load_document(file_path: Path | str) -> ToolResult:
    """
    Parse a documentation file.

    Supports: .docx, .pdf, .txt, .rtf

    Returns:
        ToolResult with DocumentInfo on success
    """
    try:
        file_path = Path(file_path)
        parser = DocumentParser()
        doc_info = parser.parse(file_path)

        if doc_info:
            return ToolResult(
                success=True,
                data=doc_info,
                metadata={
                    "file": str(file_path),
                    "text_length": len(doc_info.full_text) if doc_info.full_text else 0,
                    "n_questions": len(doc_info.questions) if doc_info.questions else 0,
                    "is_questionnaire": doc_info.is_questionnaire
                }
            )
        else:
            return ToolResult(
                success=False,
                error=f"Failed to parse document {file_path}"
            )
    except Exception as e:
        logger.error(f"Error parsing document: {e}")
        return ToolResult(success=False, error=str(e))


# ============================================================================
# CONTEXT EXTRACTION TOOLS
# ============================================================================

def extract_full_context(
    data_file: Path | str,
    doc_files: list[Path | str]
) -> ToolResult:
    """
    Extract full context from data file and documentation.

    Combines variable metadata, questionnaire text, and codebook
    into a unified context for LLM matching.

    Returns:
        ToolResult with ExtractionResult on success
    """
    try:
        result = extract_context(
            data_file=Path(data_file),
            doc_files=[Path(f) for f in doc_files] if doc_files else None
        )

        return ToolResult(
            success=True,
            data=result,
            metadata={
                "n_variables": len(result.variables),
                "n_questions": len(result.questionnaire_questions),
                "data_quality": result.data_quality,
                "doc_quality": result.doc_quality,
                "overall_quality": result.overall_quality,
                "errors": result.errors
            }
        )
    except Exception as e:
        logger.error(f"Error extracting context: {e}")
        return ToolResult(success=False, error=str(e))


# ============================================================================
# MATCHING TOOLS
# ============================================================================

def run_llm_matching(
    source_contexts: list[dict],
    questionnaire_text: str = "",
    codebook_text: str = "",
    pre_aggregated_summary: Optional[str] = None,
    model: Optional[str] = None
) -> ToolResult:
    """
    Run LLM-based variable matching.

    Uses the existing LLMMatcher to propose source -> CSES target mappings.

    Returns:
        ToolResult with MatchingResult on success
    """
    try:
        matcher = create_matcher(model)
        result = matcher.match_variables(
            source_contexts=source_contexts,
            questionnaire_text=questionnaire_text,
            codebook_text=codebook_text,
            pre_aggregated_summary=pre_aggregated_summary
        )

        return ToolResult(
            success=True,
            data=result,
            metadata={
                "n_proposals": len(result.proposals),
                "high_confidence": result.high_confidence_count,
                "medium_confidence": result.medium_confidence_count,
                "low_confidence": result.low_confidence_count,
                "unmatched": len(result.unmatched),
                "errors": result.errors
            }
        )
    except Exception as e:
        logger.error(f"Error in LLM matching: {e}")
        return ToolResult(success=False, error=str(e))


def aggregate_documents(
    source_variables: list[dict],
    codebook_text: str = "",
    questionnaire_text: str = "",
    design_report_text: str = ""
) -> ToolResult:
    """
    Aggregate document information for matching.

    Uses DocumentAggregator to create a token-efficient summary
    of all document context.

    Returns:
        ToolResult with aggregated summary string on success
    """
    try:
        aggregator = DocumentAggregator()
        summary = aggregator.aggregate_variable_info(
            source_variables=source_variables,
            codebook_text=codebook_text,
            questionnaire_text=questionnaire_text,
            design_report_text=design_report_text
        )

        return ToolResult(
            success=True,
            data=summary,
            metadata={
                "summary_length": len(summary),
                "input_codebook_length": len(codebook_text),
                "input_questionnaire_length": len(questionnaire_text),
                "input_design_report_length": len(design_report_text),
                "compression_ratio": len(summary) / max(1, len(codebook_text) + len(questionnaire_text))
            }
        )
    except Exception as e:
        logger.error(f"Error in document aggregation: {e}")
        return ToolResult(success=False, error=str(e))


# ============================================================================
# VALIDATION TOOLS
# ============================================================================

def validate_single_mapping(
    source_variable: str,
    target_variable: str,
    source_info: dict,
    original_confidence: float,
    original_reasoning: str
) -> ToolResult:
    """
    Validate a single variable mapping using the validation LLM.

    Returns:
        ToolResult with validation verdict and reasoning
    """
    try:
        from src.agent.validator import validate_proposal, ValidationResult
        from src.matching.llm_matcher import MatchProposal

        # Create proposal object
        proposal = MatchProposal(
            source_variable=source_variable,
            target_variable=target_variable,
            confidence=original_confidence,
            confidence_level="high" if original_confidence >= 0.85 else "medium" if original_confidence >= 0.60 else "low",
            reasoning=original_reasoning,
            matched_by="llm_semantic"
        )

        # Run validation
        result = validate_proposal(proposal)

        return ToolResult(
            success=True,
            data={
                "verdict": result.verdict.value,
                "reasoning": result.reasoning,
                "suggested_alternative": result.suggested_alternative,
                "models_agree": result.models_agree
            },
            metadata={
                "validation_model": result.validation_model
            }
        )
    except Exception as e:
        logger.error(f"Error in validation: {e}")
        return ToolResult(success=False, error=str(e))


# ============================================================================
# FREQUENCY & QUALITY CHECK TOOLS
# ============================================================================

def generate_frequency_table(
    data_file: Path | str,
    variable_name: str
) -> ToolResult:
    """
    Generate frequency table for a variable.

    Returns:
        ToolResult with frequency counts and percentages
    """
    try:
        import polars as pl

        file_path = Path(data_file)
        loader = DataLoader()
        dataset_info = loader.load(file_path)

        if not dataset_info:
            return ToolResult(success=False, error=f"Could not load {file_path}")

        # Get raw data
        df = dataset_info.df

        if variable_name not in df.columns:
            return ToolResult(success=False, error=f"Variable {variable_name} not found")

        # Compute frequencies
        counts = (
            df.group_by(variable_name)
            .agg(pl.count().alias("frequency"))
            .sort(variable_name)
        )

        total = counts["frequency"].sum()
        counts = counts.with_columns([
            (pl.col("frequency") / total * 100).round(1).alias("percent")
        ])

        # Get value labels if available
        var_info = dataset_info.variables.get(variable_name)
        value_labels = var_info.value_labels if var_info else {}

        return ToolResult(
            success=True,
            data={
                "variable": variable_name,
                "total": int(total),
                "frequencies": counts.to_dicts(),
                "value_labels": value_labels
            }
        )
    except Exception as e:
        logger.error(f"Error generating frequencies: {e}")
        return ToolResult(success=False, error=str(e))


def check_value_range(
    data_file: Path | str,
    variable_name: str,
    min_value: float,
    max_value: float,
    missing_codes: list = None
) -> ToolResult:
    """
    Check if variable values fall within expected range.

    Returns:
        ToolResult with pass/fail status and out-of-range cases
    """
    try:
        import polars as pl

        if missing_codes is None:
            missing_codes = [7, 8, 9, 97, 98, 99, 997, 998, 999]

        file_path = Path(data_file)
        loader = DataLoader()
        dataset_info = loader.load(file_path)

        if not dataset_info:
            return ToolResult(success=False, error=f"Could not load {file_path}")

        df = dataset_info.df

        if variable_name not in df.columns:
            return ToolResult(success=False, error=f"Variable {variable_name} not found")

        # Filter out missing codes and check range
        valid = df.filter(~pl.col(variable_name).is_in(missing_codes))
        out_of_range = valid.filter(
            (pl.col(variable_name) < min_value) | (pl.col(variable_name) > max_value)
        )

        n_out_of_range = out_of_range.height
        passed = n_out_of_range == 0

        return ToolResult(
            success=True,
            data={
                "variable": variable_name,
                "expected_range": [min_value, max_value],
                "passed": passed,
                "out_of_range_count": n_out_of_range,
                "out_of_range_values": out_of_range[variable_name].unique().to_list() if n_out_of_range > 0 else []
            }
        )
    except Exception as e:
        logger.error(f"Error checking range: {e}")
        return ToolResult(success=False, error=str(e))


# ============================================================================
# EXPORT TOOLS
# ============================================================================

def export_mappings_json(
    mappings: list[dict],
    output_path: Path | str,
    metadata: dict = None
) -> ToolResult:
    """
    Export mappings to JSON file.

    Returns:
        ToolResult with output path on success
    """
    try:
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "metadata": metadata or {},
            "mappings": mappings
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        return ToolResult(
            success=True,
            data=str(output_path),
            metadata={"n_mappings": len(mappings)}
        )
    except Exception as e:
        logger.error(f"Error exporting JSON: {e}")
        return ToolResult(success=False, error=str(e))


def export_tracking_sheet(
    mappings: list[dict],
    output_path: Path | str,
    country_code: str = "CNT",
    year: str = "YEAR"
) -> ToolResult:
    """
    Export mappings to CSES tracking sheet format (Excel).

    Matches CSES "deposited variables" format:
    - Column 0 (VARIABLES): Questionnaire code + description (e.g., "D02     GENDER")
    - Column 1 (CSES code): Target variable code (e.g., "F2002")
    - Column 2 (SOURCE_VAR): Original variable name or "not asked"/"missing"
    - Column 3 (REMARKS): Notes about the mapping

    Returns:
        ToolResult with output path on success
    """
    try:
        from datetime import datetime
        import pandas as pd

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        today = datetime.now().strftime("%B %d, %Y")

        # CSES variable descriptions from M6 codebook
        # Format: questionnaire_code, description
        cses_variable_labels = {
            # Administration variables
            "F1003_2": ("A01", "ID VARIABLE - RESPONDENT"),
            "F1016_1": ("", "MODE OF INTERVIEW - RESPONDENT - FIRST"),
            "F1016_2": ("", "MODE OF INTERVIEW - RESPONDENT - SECOND"),
            "F1016_3": ("", "MODE OF INTERVIEW - RESPONDENT - THIRD"),
            "F1021": ("", "DURATION OF INTERVIEW"),
            "F1022_1": ("A02", "INTERVIEWER ID WITHIN ELECTION STUDY"),
            "F1022_2": ("A03", "INTERVIEWER GENDER"),
            "F1019_M": ("A04a", "DATE QUESTIONNAIRE ADMINISTERED - MONTH"),
            "F1019_D": ("A04b", "DATE QUESTIONNAIRE ADMINISTERED - DAY"),
            "F1019_Y": ("A04c", "DATE QUESTIONNAIRE ADMINISTERED - YEAR"),
            "F1100": ("", "SAMPLE COMPONENT"),
            "F1101_": ("A05", "ORIGINAL WEIGHT: SAMPLE / DEMOGRAPHIC / POLITICAL"),
            "F1023": ("A06", "LANGUAGE OF QUESTIONNAIRE ADMINISTRATION"),
            # Demographics
            "F2001_M": ("D01a", "DATE OF BIRTH OF RESPONDENT - MONTH"),
            "F2001_Y": ("D01b", "DATE OF BIRTH OF RESPONDENT - YEAR"),
            "F2002": ("D02", "GENDER"),
            "F2003": ("D03", "EDUCATION"),
            "F2004": ("D04", "MARITAL OR CIVIL UNION STATUS"),
            "F2005": ("D05", "UNION MEMBERSHIP OF RESPONDENT"),
            "F2006": ("D06", "CURRENT EMPLOYMENT STATUS"),
            "F2007": ("D07", "MAIN OCCUPATION"),
            "F2008": ("D07a", "SOCIO ECONOMIC STATUS"),
            "F2009": ("D08", "EMPLOYMENT TYPE - PUBLIC OR PRIVATE"),
            "F2010_2": ("D09", "HOUSEHOLD INCOME - ORIGINAL VARIABLE"),
            "F2011": ("D10", "RELIGIOUS DENOMINATION"),
            "F2012": ("D11", "RELIGIOUS SERVICES ATTENDANCE"),
            "F2013": ("D12", "RACE"),
            "F2014": ("D13", "ETHNICITY"),
            "F2015": ("D14", "COUNTRY OF BIRTH"),
            "F2016": ("D15", "WAS EITHER BIOLOGICAL PARENT BORN OUTSIDE OF THE COUNTRY"),
            "F2017": ("D16", "LANGUAGE USUALLY SPOKEN AT HOME"),
            "F2018": ("D17", "REGION OF RESIDENCE"),
            "F2019": ("D18", "PRIMARY ELECTORAL DISTRICT"),
            "F2020": ("D19", "RURAL OR URBAN RESIDENCE"),
            "F2021": ("D20", "NUMBER IN HOUSEHOLD IN TOTAL"),
            # Survey questions
            "F3001": ("Q01", "POLITICAL INTEREST"),
            "F3002_1": ("Q02a", "MEDIA USAGE: WATCH NEWS ON A PUBLIC TELEVISION BROADCASTER"),
            "F3002_2": ("Q02b", "MEDIA USAGE: WATCH NEWS ON A PRIVATE TELEVISION BROADCASTER"),
            "F3002_3": ("Q02c", "MEDIA USAGE: LISTEN TO THE NEWS ON RADIO"),
            "F3002_4": ("Q02d", "MEDIA USAGE: READ NEWSPAPERS"),
            "F3002_5": ("Q02e", "MEDIA USAGE: ONLINE NEWS SITES"),
            "F3002_6_1": ("Q02f", "MEDIA USAGE: SOCIAL MEDIA"),
            "F3002_6_2": ("Q02g", "MEDIA USAGE: SOCIAL MEDIA - PER DAY"),
            "F3003": ("Q03", "INTERNAL EFFICACY"),
            "F3004_1": ("Q04a", "TRUST AND SUPPORT FOR DEMOCRACY: PREFERABLE"),
            "F3004_2": ("Q04b", "TRUST AND SUPPORT FOR DEMOCRACY: COURTS"),
            "F3004_3": ("Q04c", "TRUST AND SUPPORT FOR DEMOCRACY: STRONG LEADER BENDS THE RULES"),
            "F3004_4": ("Q04d", "TRUST AND SUPPORT FOR DEMOCRACY: REPRESENTATION"),
            "F3005_1": ("Q05a", "ATTITUDES TOWARD FORMS OF GOVERNMENT: BUSINESS LEADERS"),
            "F3005_2": ("Q05b", "ATTITUDES TOWARD FORMS OF GOVERNMENT: EXPERTS"),
            "F3005_3": ("Q05c", "ATTITUDES TOWARD FORMS OF GOVERNMENT: REFERENDUMS"),
            "F3006": ("Q06", "HOW DEMOCRATIC IS THE COUNTRY"),
            "F3007_1": ("Q07a", "TRUST IN: PARLIAMENT"),
            "F3007_2": ("Q07b", "TRUST IN: GOVERNMENT"),
            "F3007_3": ("Q07c", "TRUST IN: JUDICIARY"),
            "F3007_4": ("Q07d", "TRUST IN: SCIENTISTS"),
            "F3007_5": ("Q07e", "TRUST IN: POLITICAL PARTIES"),
            "F3007_6": ("Q07f", "TRUST IN: TRADITIONAL MEDIA"),
            "F3007_7": ("Q07g", "TRUST IN: SOCIAL MEDIA"),
            "F3008_1": ("Q08a", "GOVERNMENT PERFORMANCE"),
            "F3008_2": ("Q08b", "GOVERNMENT COVID-19 RESPONSE"),
            "F3009": ("Q09", "STATE OF THE ECONOMY"),
            "F3010": ("Q10", "VOTED IN ELECTION"),
            "F3011_LH_PL": ("Q11", "VOTE CHOICE - PARTY"),
            "F3012_1": ("Q12", "SATISFACTION WITH VOTE"),
            "F3013": ("Q13", "SATISFACTION WITH CHOICES"),
            "F3014": ("Q14", "FAIRNESS OF ELECTION"),
            "F3017": ("Q17", "EXTERNAL EFFICACY"),
            "F3018_A": ("Q18a", "LIKE/DISLIKE: PARTY A"),
            "F3018_B": ("Q18b", "LIKE/DISLIKE: PARTY B"),
            "F3018_C": ("Q18c", "LIKE/DISLIKE: PARTY C"),
            "F3019_A": ("Q19a", "LEFT-RIGHT: PARTY A"),
            "F3019_B": ("Q19b", "LEFT-RIGHT: PARTY B"),
            "F3019_C": ("Q19c", "LEFT-RIGHT: PARTY C"),
            "F3020": ("Q20", "LEFT-RIGHT SELF-PLACEMENT"),
            "F3021": ("Q21", "PARTY IDENTIFICATION"),
            "F3022": ("Q22", "PARTY ID: WHICH PARTY"),
            "F3023": ("Q23", "PARTY ID: STRENGTH"),
            "F3024": ("Q24", "SATISFACTION WITH DEMOCRACY"),
        }

        # Build tracking sheet rows matching CSES format
        rows = []

        # Header row
        rows.append({0: "VARIABLES", 1: None, 2: f"{country_code}_{year}", 3: "REMARKS"})
        # Questionnaire row
        rows.append({0: "QUESTIONNAIRE", 1: "CSES M6", 2: "(X = missing)", 3: None})
        # Empty row
        rows.append({0: None, 1: None, 2: None, 3: None})

        # Build mapping lookup by target variable
        mapping_lookup = {}
        for m in mappings:
            target = m.get("target") or m.get("cses_variable") or m.get("target_variable")
            if target:
                mapping_lookup[target] = m

        # Add rows for each CSES variable in order
        for cses_var, (q_code, description) in cses_variable_labels.items():
            # Format variable description like CSES template
            if q_code:
                var_desc = f"{q_code}     {description}"
            else:
                var_desc = description

            # Get source variable from mappings
            m = mapping_lookup.get(cses_var, {})
            source = m.get("source") or m.get("source_variable") or ""

            # Handle special cases
            if not source or source == "NOT_FOUND":
                source = "not asked"

            # Get remarks
            remarks = m.get("notes") or m.get("reasoning") or ""
            if len(remarks) > 50:
                remarks = remarks[:47] + "..."

            rows.append({
                0: var_desc,
                1: cses_var,
                2: source,
                3: remarks if remarks else None
            })

        df = pd.DataFrame(rows)
        df.to_excel(output_path, index=False, header=False, sheet_name="deposited variables")

        return ToolResult(
            success=True,
            data=str(output_path),
            metadata={"n_mappings": len(mappings)}
        )
    except Exception as e:
        logger.error(f"Error exporting Excel: {e}")
        return ToolResult(success=False, error=str(e))


def export_do_file(
    mappings: list[dict],
    output_path: Path | str,
    country_code: str = "CNT",
    country_name: str = "COUNTRY",
    year: str = "YEAR",
    author: str = "CSES Tool",
    data_file_path: str = "",
    party_result: Optional[PartyOrderResult] = None,
    study_number: int = 0,
    source_contexts: Optional[list[dict]] = None,
    use_llm_generation: bool = False,
    progress_callback: Optional[callable] = None
) -> ToolResult:
    """
    Export mappings to CSES Stata .do file format.

    Generates a Stata script following CSES conventions:
    - Header with author, date, country
    - Section markers (\\\\\\) for major sections
    - Variable markers (>>>) for individual variables
    - gen command + tab command for each variable
    - Party code mappings (if party_result provided)

    Args:
        mappings: List of variable mappings
        output_path: Path to write .do file
        country_code: ISO alpha-3 country code (e.g., "CHE")
        country_name: Full country name (e.g., "SWITZERLAND")
        year: Election year (e.g., "2023")
        author: Author name for header
        data_file_path: Path to source data file
        party_result: Optional PartyOrderResult with party codes
        study_number: Study number within country (usually 0)
        source_contexts: Source variable metadata (required for LLM generation)
        use_llm_generation: If True, use LLM to generate Stata code for each variable
        progress_callback: Optional callback for progress updates

    Returns:
        ToolResult with output path on success
    """
    try:
        from datetime import datetime

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        today = datetime.now().strftime("%B %d, %Y")

        # Get UN country code for F1005 and F1006
        un_code = get_un_country_code(country_code) or get_un_country_code(country_name) or "XXX"

        # Build mapping lookup by target variable
        mapping_lookup = {}
        for m in mappings:
            target = m.get("target") or m.get("cses_variable") or m.get("target_variable")
            if target:
                mapping_lookup[target] = m

        # CSES variable definitions with their gen commands
        cses_variables = {
            # Fixed variables (generated, not from source data)
            "F1001": {
                "desc": "DATASET",
                "type": "string",
                "gen": 'gen str13 F1001 = "CSES-MODULE-6"'
            },
            "F1002_VER": {
                "desc": "DATASET VERSION",
                "type": "string",
                "gen": 'gen str14 F1002_VER = "VER2024-MMM-DD"'
            },
            "F1002_DOI": {
                "desc": "DATASET DIGITAL OBJECT IDENTIFIER (DOI)",
                "type": "string",
                "gen": 'gen str35 F1002_DOI = "doi:10.7804/cses.module6.2024-MM-DD"'
            },
            "F1004": {
                "desc": "ID VARIABLE - ELECTION STUDY (ALPHABETIC POLITY)",
                "type": "string",
                "gen": f'gen str F1004 = "{country_code}_{year}"'
            },
            "F1005": {
                "desc": "ID VARIABLE - ELECTION STUDY (NUMERIC POLITY)",
                "type": "numeric",
                "gen": f'gen long F1005 = {un_code}{study_number}{year}\nformat F1005 %8.0f'
            },
            "F1006": {
                "desc": "ID COMPONENT - POLITY CSES CODE",
                "type": "string",
                "gen": f'gen str F1006 = "{un_code}{study_number}"'
            },
            "F1009": {
                "desc": "ID COMPONENT - ELECTION YEAR",
                "type": "numeric",
                "gen": f'gen F1009 = {year}'
            },
        }

        # Variable mappings from source data
        source_mapped_vars = {
            "F1003_2": ("A01", "ID VARIABLE - RESPONDENT WITHIN ELECTION STUDY"),
            "F1019_M": ("A04a", "DATE QUESTIONNAIRE ADMINISTERED - MONTH"),
            "F1019_D": ("A04b", "DATE QUESTIONNAIRE ADMINISTERED - DAY"),
            "F1019_Y": ("A04c", "DATE QUESTIONNAIRE ADMINISTERED - YEAR"),
            "F1021": ("", "DURATION OF INTERVIEW"),
            "F1022_1": ("A02", "INTERVIEWER ID WITHIN ELECTION STUDY"),
            "F1022_2": ("A03", "INTERVIEWER GENDER"),
            "F1023": ("A06", "LANGUAGE OF QUESTIONNAIRE ADMINISTRATION"),
            "F1100": ("", "SAMPLE COMPONENT"),
            "F1101_": ("A05", "ORIGINAL WEIGHT"),
            "F2001_M": ("D01a", "DATE OF BIRTH OF RESPONDENT - MONTH"),
            "F2001_Y": ("D01b", "DATE OF BIRTH OF RESPONDENT - YEAR"),
            "F2002": ("D02", "GENDER"),
            "F2003": ("D03", "EDUCATION"),
            "F2004": ("D04", "MARITAL OR CIVIL UNION STATUS"),
            "F2005": ("D05", "UNION MEMBERSHIP OF RESPONDENT"),
            "F2006": ("D06", "CURRENT EMPLOYMENT STATUS"),
            "F2007": ("D07", "MAIN OCCUPATION"),
            "F2008": ("D07a", "SOCIO ECONOMIC STATUS"),
            "F2009": ("D08", "EMPLOYMENT TYPE - PUBLIC OR PRIVATE"),
            "F2010_2": ("D09", "HOUSEHOLD INCOME - ORIGINAL VARIABLE"),
            "F2011": ("D10", "RELIGIOUS DENOMINATION"),
            "F2012": ("D11", "RELIGIOUS SERVICES ATTENDANCE"),
            "F2013": ("D12", "RACE"),
            "F2014": ("D13", "ETHNICITY"),
            "F2015": ("D14", "COUNTRY OF BIRTH"),
            "F2016": ("D15", "WAS EITHER BIOLOGICAL PARENT BORN OUTSIDE OF THE COUNTRY"),
            "F2017": ("D16", "LANGUAGE USUALLY SPOKEN AT HOME"),
            "F2018": ("D17", "REGION OF RESIDENCE"),
            "F2019": ("D18", "PRIMARY ELECTORAL DISTRICT"),
            "F2020": ("D19", "RURAL OR URBAN RESIDENCE"),
            "F2021": ("D20", "NUMBER IN HOUSEHOLD IN TOTAL"),
            "F3001": ("Q01", "POLITICAL INTEREST"),
            "F3002_1": ("Q02a", "MEDIA USAGE: PUBLIC TV"),
            "F3002_2": ("Q02b", "MEDIA USAGE: PRIVATE TV"),
            "F3002_3": ("Q02c", "MEDIA USAGE: RADIO"),
            "F3002_4": ("Q02d", "MEDIA USAGE: NEWSPAPERS"),
            "F3002_5": ("Q02e", "MEDIA USAGE: ONLINE NEWS"),
            "F3002_6_1": ("Q02f", "MEDIA USAGE: SOCIAL MEDIA"),
            "F3003": ("Q03", "INTERNAL EFFICACY"),
            "F3004_1": ("Q04a", "DEMOCRACY IS PREFERABLE"),
            "F3004_2": ("Q04b", "COURTS SHOULD STOP GOVERNMENT"),
            "F3004_3": ("Q04c", "STRONG LEADER"),
            "F3004_4": ("Q04d", "REPRESENTATION"),
            "F3005_1": ("Q05a", "BUSINESS LEADERS"),
            "F3005_2": ("Q05b", "EXPERTS"),
            "F3005_3": ("Q05c", "REFERENDUMS"),
            "F3006": ("Q06", "HOW DEMOCRATIC IS COUNTRY"),
            "F3007_1": ("Q07a", "TRUST: PARLIAMENT"),
            "F3007_2": ("Q07b", "TRUST: GOVERNMENT"),
            "F3007_3": ("Q07c", "TRUST: JUDICIARY"),
            "F3007_4": ("Q07d", "TRUST: SCIENTISTS"),
            "F3007_5": ("Q07e", "TRUST: POLITICAL PARTIES"),
            "F3007_6": ("Q07f", "TRUST: TRADITIONAL MEDIA"),
            "F3007_7": ("Q07g", "TRUST: SOCIAL MEDIA"),
            "F3008_1": ("Q08a", "GOVERNMENT PERFORMANCE"),
            "F3008_2": ("Q08b", "COVID-19 RESPONSE"),
            "F3009": ("Q09", "STATE OF ECONOMY"),
            "F3010": ("Q10", "VOTED IN ELECTION"),
            "F3011_LH_PL": ("Q11", "VOTE CHOICE - PARTY"),
            "F3012_1": ("Q12", "SATISFACTION WITH VOTE"),
            "F3013": ("Q13", "SATISFACTION WITH CHOICES"),
            "F3014": ("Q14", "FAIRNESS OF ELECTION"),
            "F3017": ("Q17", "EXTERNAL EFFICACY"),
            "F3018_A": ("Q18a", "LIKE/DISLIKE: PARTY A"),
            "F3018_B": ("Q18b", "LIKE/DISLIKE: PARTY B"),
            "F3018_C": ("Q18c", "LIKE/DISLIKE: PARTY C"),
            "F3019_A": ("Q19a", "LEFT-RIGHT: PARTY A"),
            "F3019_B": ("Q19b", "LEFT-RIGHT: PARTY B"),
            "F3019_C": ("Q19c", "LEFT-RIGHT: PARTY C"),
            "F3020": ("Q20", "LEFT-RIGHT SELF-PLACEMENT"),
            "F3021": ("Q21", "PARTY IDENTIFICATION"),
            "F3022": ("Q22", "PARTY ID: WHICH PARTY"),
            "F3023": ("Q23", "PARTY ID: STRENGTH"),
            "F3024": ("Q24", "SATISFACTION WITH DEMOCRACY"),
        }

        lines = []

        # Header
        lines.append("/***************************************************************************")
        lines.append("**                     Process CSES-M6 Micro-Data                         **")
        lines.append("**                     **************************                         **")
        lines.append("**                                                                        **")
        lines.append(f"** File Author:      {author:<52} **")
        lines.append(f"** Date:             {today:<52} **")
        lines.append(f"** CSES MODULE 6:    {country_name.upper()} {year:<40} **")
        lines.append("**                                                                        **")
        lines.append("***************************************************************************/")
        lines.append("")

        # Syntax instructions
        lines.append("*-------------------------------------------------------------------------**")
        lines.append("****************************************************************************")
        lines.append("**\\\\\\              SYNTAX INSTRUCTIONS                                    **")
        lines.append("****************************************************************************")
        lines.append("*-------------------------------------------------------------------------**")
        lines.append("*                                                                         **")
        lines.append('* 1) FILE NAVIGATION: The Syntax File can be navigated using different    **')
        lines.append('*         combinations of symbols. "\\\\\\" will direct the user to          **')
        lines.append('*         the different sections of the file and ">>>" allows to          **')
        lines.append('*         jump from variable to variable.                                 **')
        lines.append('*         \\\\\\  Section Headings                                            **')
        lines.append('*         >>> Variable Headings                                           **')
        lines.append("**                                                                        **")
        lines.append("****************************************************************************")
        lines.append("")

        # Open data section
        lines.append("*-------------------------------------------------------------------------*")
        lines.append("***************************************************************************")
        lines.append("**\\\\\\              OPEN DATA")
        lines.append("***************************************************************************")
        lines.append("*-------------------------------------------------------------------------*")
        lines.append("")
        lines.append("clear")
        lines.append("set more off")
        lines.append("capture log close")
        lines.append("")
        lines.append("* Set Working Directory")
        lines.append(f'cd "{output_path.parent}"')
        lines.append("")
        lines.append("* Open File")
        if data_file_path:
            lines.append(f'use "{data_file_path}", clear')
        else:
            lines.append('* TODO: Specify path to data file')
            lines.append('use "YOUR_DATA_FILE.dta", clear')
        lines.append("")
        lines.append("")

        # ID and Admin section
        lines.append("*-------------------------------------------------------------------------*")
        lines.append("***************************************************************************")
        lines.append("**\\\\\\       ID, WEIGHT, AND ADMINISTRATION VARIABLES")
        lines.append("***************************************************************************")
        lines.append("*-------------------------------------------------------------------------*")
        lines.append("")

        # Add fixed variables
        for var_code, var_info in cses_variables.items():
            lines.append("")
            lines.append("***************************************************************************")
            lines.append(f"**>>> {var_code} - {var_info['desc']} -> {var_info['type']} variable")
            lines.append("***************************************************************************")
            lines.append("")
            lines.append(var_info["gen"])
            lines.append(f"tab {var_code}, mis")
            lines.append("")

        # Demographics section
        lines.append("")
        lines.append("*-------------------------------------------------------------------------*")
        lines.append("***************************************************************************")
        lines.append("**\\\\\\       DEMOGRAPHICS")
        lines.append("***************************************************************************")
        lines.append("*-------------------------------------------------------------------------*")

        # Add source-mapped variables
        for var_code, (q_code, description) in source_mapped_vars.items():
            lines.append("")
            lines.append("***************************************************************************")
            if q_code:
                lines.append(f"**>>> {var_code} - {q_code} {description}")
            else:
                lines.append(f"**>>> {var_code} - {description}")
            lines.append("***************************************************************************")
            lines.append("")

            # Get source variable from mappings
            m = mapping_lookup.get(var_code, {})
            source = m.get("source") or m.get("source_variable") or ""
            reasoning = m.get("reasoning") or m.get("notes") or ""

            if source and source != "NOT_FOUND":
                lines.append(f"* Source variable: {source}")
                if reasoning:
                    # Wrap reasoning in comments
                    for line in reasoning.split("\n")[:3]:
                        lines.append(f"* {line[:70]}")
                lines.append(f"gen {var_code} = {source}")
                lines.append("* TODO: Add recode rules if needed")
            else:
                lines.append("* Source variable: NOT FOUND - requires manual review")
                lines.append(f"gen {var_code} = .")

            lines.append(f"tab {var_code}, mis")
            lines.append("")

        # Party codes section (if party_result provided)
        if party_result and party_result.parties:
            lines.append("")
            lines.append("*-------------------------------------------------------------------------*")
            lines.append("***************************************************************************")
            lines.append("**\\\\\\       PARTY CODES")
            lines.append("***************************************************************************")
            lines.append("*-------------------------------------------------------------------------*")
            lines.append("")
            lines.append("/*")
            lines.append("Party Code Conventions (CSES Module 6):")
            lines.append("- Parties A-F are ordered by DESCENDING vote share")
            lines.append("- Party A = highest vote share, Party B = second highest, etc.")
            lines.append("- Supplemental parties G-I have no fixed ordering")
            lines.append("")
            lines.append("6-digit party code structure: [XXX][Y][ZZ]")
            lines.append("  XXX = UN country code (3 digits)")
            lines.append("  Y   = Study number (0 if only one study)")
            lines.append("  ZZ  = Party sequence (01-99)")
            lines.append("")
            lines.append(f"Country code: {party_result.country_code}")
            lines.append(f"Study number: {party_result.study_number}")
            lines.append(f"Ordering basis: {party_result.ordering_basis}")
            lines.append("")
            lines.append("Party Mapping:")
            for party in party_result.parties:
                if party.code_letter:
                    lines.append(f"  PARTY {party.code_letter}: {party.name} ({party.vote_share*100:.1f}%)")
                    lines.append(f"         Code: {party.numerical_code}")
            lines.append("*/")
            lines.append("")

            # Generate F3018_A-I (Like/Dislike Party) variables
            lines.append("***************************************************************************")
            lines.append("**>>> F3018_A-I - LIKE/DISLIKE PARTIES")
            lines.append("***************************************************************************")
            lines.append("")
            lines.append("** 00. STRONGLY DISLIKE * 10. STRONGLY LIKE * 96. HAVEN'T HEARD OF PARTY *")
            lines.append("** 97. VOLUNTEERED: REFUSED * 98. DON'T KNOW * 99. MISSING")
            lines.append("")

            for party in party_result.parties:
                if party.code_letter:
                    var_name = f"F3018_{party.code_letter}"
                    m = mapping_lookup.get(var_name, {})
                    source = m.get("source") or m.get("source_variable") or ""

                    lines.append(f"* {var_name}: {party.name}")
                    if source and source != "NOT_FOUND":
                        lines.append(f"gen {var_name} = {source}")
                        lines.append("* TODO: Add recode rules if needed")
                    else:
                        lines.append(f"gen {var_name} = 99  // Source not found - requires manual mapping")
                    lines.append(f"tab {var_name}, mis")
                    lines.append("")

            # Generate F3019_A-I (Like/Dislike Leader) variables
            lines.append("***************************************************************************")
            lines.append("**>>> F3019_A-I - LIKE/DISLIKE LEADERS")
            lines.append("***************************************************************************")
            lines.append("")
            lines.append("** 00. STRONGLY DISLIKE * 10. STRONGLY LIKE * 96. HAVEN'T HEARD OF LEADER *")
            lines.append("** 97. VOLUNTEERED: REFUSED * 98. DON'T KNOW * 99. MISSING")
            lines.append("")

            for party in party_result.parties:
                if party.code_letter:
                    var_name = f"F3019_{party.code_letter}"
                    m = mapping_lookup.get(var_name, {})
                    source = m.get("source") or m.get("source_variable") or ""

                    lines.append(f"* {var_name}: Leader of {party.name}")
                    if source and source != "NOT_FOUND":
                        lines.append(f"gen {var_name} = {source}")
                        lines.append("* TODO: Add recode rules if needed")
                    else:
                        lines.append(f"gen {var_name} = 99  // Source not found - requires manual mapping")
                    lines.append(f"tab {var_name}, mis")
                    lines.append("")

            # Generate F3020_A-I (Left-Right Party Placement) variables
            lines.append("***************************************************************************")
            lines.append("**>>> F3020_A-I - LEFT-RIGHT PARTY PLACEMENT")
            lines.append("***************************************************************************")
            lines.append("")
            lines.append("** 00. LEFT * 10. RIGHT * 95. HAVEN'T HEARD OF LEFT-RIGHT *")
            lines.append("** 96. HAVEN'T HEARD OF PARTY * 97. REFUSED * 98. DON'T KNOW * 99. MISSING")
            lines.append("")

            for party in party_result.parties:
                if party.code_letter:
                    var_name = f"F3020_{party.code_letter}"
                    m = mapping_lookup.get(var_name, {})
                    source = m.get("source") or m.get("source_variable") or ""

                    lines.append(f"* {var_name}: {party.name}")
                    if source and source != "NOT_FOUND":
                        lines.append(f"gen {var_name} = {source}")
                        lines.append("* TODO: Add recode rules if needed")
                    else:
                        lines.append(f"gen {var_name} = 99  // Source not found - requires manual mapping")
                    lines.append(f"tab {var_name}, mis")
                    lines.append("")

            # Generate vote choice recoding section
            lines.append("***************************************************************************")
            lines.append("**>>> F3011 - VOTE CHOICE (6-digit party codes)")
            lines.append("***************************************************************************")
            lines.append("")
            lines.append("* Convert source party variable to 6-digit CSES party codes")
            lines.append("* TODO: Map source party values to numerical codes below")
            lines.append("")
            lines.append("/*")
            lines.append("Example recode pattern:")
            for party in party_result.parties:
                if party.numerical_code:
                    lines.append(f"replace F3011_LH_PL = {party.numerical_code} if source_party == X  // {party.name}")
            lines.append("*/")
            lines.append("")

        # Save section
        lines.append("")
        lines.append("*-------------------------------------------------------------------------*")
        lines.append("***************************************************************************")
        lines.append("**\\\\\\       SAVE PROCESSED DATA")
        lines.append("***************************************************************************")
        lines.append("*-------------------------------------------------------------------------*")
        lines.append("")
        lines.append("* Drop original variables and helper variables")
        lines.append("* drop <list of original variables>")
        lines.append("")
        lines.append("* Save processed data")
        lines.append(f'save "cses-m6_micro_{country_code}_{year}.dta", replace')
        lines.append("")
        lines.append("* Run label files")
        lines.append('capture confirm file "labels/cses-m6_micro-var-labels.do"')
        lines.append('if _rc == 0 do "labels/cses-m6_micro-var-labels.do"')
        lines.append('capture confirm file "labels/cses-m6_micro-val-labels.do"')
        lines.append('if _rc == 0 do "labels/cses-m6_micro-val-labels.do"')
        lines.append("")
        lines.append(f'save "cses-m6_micro_{country_code}_{year}.dta", replace')
        lines.append("")
        lines.append("log close")
        lines.append("")

        # Write to file
        content = "\n".join(lines)
        output_path.write_text(content, encoding="utf-8")

        return ToolResult(
            success=True,
            data=str(output_path),
            metadata={"n_mappings": len(mappings), "n_lines": len(lines)}
        )
    except Exception as e:
        logger.error(f"Error exporting .do file: {e}")
        return ToolResult(success=False, error=str(e))


# ============================================================================
# STATA DEBUG TOOLS
# ============================================================================

def run_stata_debug(
    do_file_path: Path | str,
    stata_path: str = None
) -> ToolResult:
    """
    Run a Stata .do file and return errors for debugging.

    This tool allows the agent to:
    1. Execute a .do file through the package-owned Stata bridge
    2. Parse the log file for errors
    3. Return structured error information for fixing

    Args:
        do_file_path: Path to the .do file to run
        stata_path: Optional path to Stata executable (uses STATA_PATH env var if not provided)

    Returns:
        ToolResult with:
        - success: True if .do file ran without errors
        - data: Dict with log content, errors, and context
        - metadata: Execution details
    """
    import re
    import os
    from src.settings import apply_settings_to_environment
    from src.stata_mcp import MCPStataRunner

    try:
        apply_settings_to_environment()
        do_file_path = Path(do_file_path)

        if not do_file_path.exists():
            return ToolResult(
                success=False,
                error=f".do file not found: {do_file_path}"
            )

        # Get Stata path
        if stata_path is None:
            stata_path = os.environ.get("STATA_PATH", "")

        if not stata_path:
            return ToolResult(
                success=False,
                error="Stata path not configured. Set STATA_PATH environment variable or run 'cses setup'."
            )

        stata_candidate = Path(stata_path)
        if stata_candidate.is_dir():
            for exe_name in ["StataSE-64.exe", "StataMP-64.exe", "StataBE-64.exe", "Stata-64.exe"]:
                candidate = stata_candidate / exe_name
                if candidate.exists():
                    stata_path = str(candidate)
                    break

        if not Path(stata_path).exists():
            return ToolResult(
                success=False,
                error=f"Stata executable not found: {stata_path}"
            )

        print(f"Running Stata on: {do_file_path.name}")
        result = MCPStataRunner(stata_path=stata_path).run_do_file(do_file_path)

        # Read log file
        log_path = Path(result.log_path) if result.log_path else do_file_path.with_suffix(".log")
        if not log_path.exists():
            # Try smcl file
            smcl_path = do_file_path.with_suffix(".smcl")
            if smcl_path.exists():
                log_path = smcl_path

        if not log_path.exists():
            return ToolResult(
                success=False,
                error=result.error or "Stata ran but no log file was generated",
                data={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.rc,
                }
            )

        log_content = log_path.read_text(encoding="utf-8", errors="replace")

        # Parse errors from log
        errors = []
        error_context = []
        lines = log_content.split("\n")

        for i, line in enumerate(lines):
            # Stata error codes look like r(123) or "no; data in memory would be lost"
            error_match = re.search(r'r\((\d+)\)', line)
            if error_match:
                error_code = error_match.group(1)
                # Get surrounding context (5 lines before, 2 after)
                start = max(0, i - 5)
                end = min(len(lines), i + 3)
                context = "\n".join(lines[start:end])

                errors.append({
                    "line_number": i + 1,
                    "error_code": error_code,
                    "error_line": line.strip(),
                    "context": context
                })
            elif "error" in line.lower() and "no error" not in line.lower():
                # General error mention
                start = max(0, i - 3)
                end = min(len(lines), i + 2)
                context = "\n".join(lines[start:end])

                # Avoid duplicate entries
                if not any(e["line_number"] == i + 1 for e in errors):
                    errors.append({
                        "line_number": i + 1,
                        "error_code": "unknown",
                        "error_line": line.strip(),
                        "context": context
                    })

        if result.rc not in (None, 0) and not errors:
            errors.append({
                "line_number": 0,
                "error_code": str(result.rc),
                "error_line": f"Stata returned r({result.rc}).",
                "context": result.stderr or result.stdout[-1000:],
            })

        if errors or not result.success:
            # Build error summary for the agent
            error_summary = []
            for e in errors[:10]:  # Limit to first 10 errors
                error_summary.append(
                    f"Error at line {e['line_number']} (r{e['error_code']}): {e['error_line']}\n"
                    f"Context:\n{e['context']}"
                )

            return ToolResult(
                success=False,
                data={
                    "error_count": len(errors),
                    "errors": errors[:10],
                    "error_summary": "\n\n---\n\n".join(error_summary),
                    "log_path": str(log_path),
                    "do_file_path": str(do_file_path)
                },
                metadata={
                    "total_errors": len(errors),
                    "log_lines": len(lines),
                    "stata_bridge": "mcp-stata",
                },
                error=result.error or f"Found {len(errors)} error(s) in Stata execution"
            )
        else:
            return ToolResult(
                success=True,
                data={
                    "message": "Stata executed successfully with no errors",
                    "log_path": str(log_path),
                    "log_preview": log_content[-2000:] if len(log_content) > 2000 else log_content
                },
                metadata={
                    "log_lines": len(lines),
                    "log_size": len(log_content),
                    "stata_bridge": "mcp-stata",
                }
            )

    except Exception as e:
        logger.error(f"Error running Stata debug: {e}")
        return ToolResult(success=False, error=str(e))


def llm_fix_stata_error(
    do_file_path: Path | str,
    error_info: dict,
    max_attempts: int = 3
) -> ToolResult:
    """
    Use LLM to fix a Stata error in a .do file.

    This function reads the .do file, identifies the problematic code,
    uses the LLM to suggest a fix, applies the fix, and re-runs Stata
    to verify the fix worked.

    Args:
        do_file_path: Path to the .do file
        error_info: Dict with error details (line_number, error_code, context)
        max_attempts: Maximum number of fix attempts

    Returns:
        ToolResult with fixed code or error if unfixable
    """
    from src.config import LLM_TEMPERATURE
    from src.model_runtime import ModelRole, ModelTaskRunner
    import re

    do_file_path = Path(do_file_path)

    if not do_file_path.exists():
        return ToolResult(success=False, error=f".do file not found: {do_file_path}")

    try:
        content = do_file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        line_num = error_info.get("line_number", 0)
        error_code = error_info.get("error_code", "unknown")
        error_context = error_info.get("context", "")
        error_line = error_info.get("error_line", "")

        # Get surrounding code for context
        start_line = max(0, line_num - 10)
        end_line = min(len(lines), line_num + 5)
        surrounding_code = "\n".join(f"{i+1}: {lines[i]}" for i in range(start_line, end_line))

        prompt = f"""Fix the following Stata error in a CSES data processing .do file.

ERROR:
- Error code: r({error_code})
- Line {line_num}: {error_line}

ERROR CONTEXT:
{error_context}

SURROUNDING CODE:
{surrounding_code}

COMMON STATA ERRORS:
- r(111): Observation out of range
- r(198): Invalid syntax
- r(100): Variable not found
- r(110): Variable already defined
- r(109): Type mismatch

Provide the CORRECTED Stata code to fix this error. Return ONLY the fixed code lines.
If multiple lines need to change, include all of them.
Include brief comments explaining the fix.

Do NOT include any markdown formatting or code blocks."""

        runner = ModelTaskRunner()
        response = runner.response(
            ModelRole.STATA_REPAIR,
            model_override=runner.model_for_role(ModelRole.STATA_REPAIR),
            max_tokens=1024,
            temperature=LLM_TEMPERATURE,
            timeout=60,
            purpose=f"Repair Stata error r({error_code})",
            messages=[{"role": "user", "content": prompt}]
        )

        fixed_code = response.choices[0].message.content.strip()
        # Clean up markdown if present
        fixed_code = re.sub(r'^```stata?\s*', '', fixed_code)
        fixed_code = re.sub(r'\s*```$', '', fixed_code)

        return ToolResult(
            success=True,
            data={
                "original_error": error_info,
                "suggested_fix": fixed_code,
                "line_number": line_num,
                "file_path": str(do_file_path)
            },
            metadata={"fix_generated": True}
        )

    except Exception as e:
        logger.error(f"Error in LLM fix generation: {e}")
        return ToolResult(success=False, error=str(e))


def read_do_file(do_file_path: Path | str) -> ToolResult:
    """
    Read a Stata .do file content for editing.

    Args:
        do_file_path: Path to the .do file

    Returns:
        ToolResult with file content
    """
    try:
        do_file_path = Path(do_file_path)

        if not do_file_path.exists():
            return ToolResult(
                success=False,
                error=f".do file not found: {do_file_path}"
            )

        content = do_file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        return ToolResult(
            success=True,
            data={
                "content": content,
                "lines": lines,
                "path": str(do_file_path)
            },
            metadata={
                "line_count": len(lines),
                "size": len(content)
            }
        )
    except Exception as e:
        logger.error(f"Error reading .do file: {e}")
        return ToolResult(success=False, error=str(e))


def write_do_file(
    do_file_path: Path | str,
    content: str
) -> ToolResult:
    """
    Write updated content to a Stata .do file.

    Args:
        do_file_path: Path to the .do file
        content: New file content

    Returns:
        ToolResult with success status
    """
    try:
        do_file_path = Path(do_file_path)
        do_file_path.parent.mkdir(parents=True, exist_ok=True)

        do_file_path.write_text(content, encoding="utf-8")

        return ToolResult(
            success=True,
            data=str(do_file_path),
            metadata={
                "line_count": len(content.split("\n")),
                "size": len(content)
            }
        )
    except Exception as e:
        logger.error(f"Error writing .do file: {e}")
        return ToolResult(success=False, error=str(e))


def fix_do_file_line(
    do_file_path: Path | str,
    line_number: int,
    old_line: str,
    new_line: str
) -> ToolResult:
    """
    Fix a specific line in a .do file.

    Args:
        do_file_path: Path to the .do file
        line_number: Line number to fix (1-indexed)
        old_line: Expected current content (for verification)
        new_line: New content for the line

    Returns:
        ToolResult with success status
    """
    try:
        do_file_path = Path(do_file_path)

        if not do_file_path.exists():
            return ToolResult(
                success=False,
                error=f".do file not found: {do_file_path}"
            )

        content = do_file_path.read_text(encoding="utf-8")
        lines = content.split("\n")

        if line_number < 1 or line_number > len(lines):
            return ToolResult(
                success=False,
                error=f"Line number {line_number} out of range (1-{len(lines)})"
            )

        current_line = lines[line_number - 1]

        # Verify the current content matches (optional safety check)
        if old_line and old_line.strip() != current_line.strip():
            return ToolResult(
                success=False,
                error=f"Line content mismatch. Expected: '{old_line.strip()}', Found: '{current_line.strip()}'"
            )

        # Replace the line
        lines[line_number - 1] = new_line
        new_content = "\n".join(lines)

        do_file_path.write_text(new_content, encoding="utf-8")

        return ToolResult(
            success=True,
            data={
                "path": str(do_file_path),
                "line_number": line_number,
                "old_line": current_line,
                "new_line": new_line
            },
            metadata={"lines_modified": 1}
        )
    except Exception as e:
        logger.error(f"Error fixing .do file line: {e}")
        return ToolResult(success=False, error=str(e))


# ============================================================================
# PARTY CODE TOOLS
# ============================================================================

def generate_party_codes_from_results(
    country: str,
    parties_by_vote_share: list[tuple[str, float]],
    study_number: int = 0,
    electoral_system: str = "single-tier"
) -> ToolResult:
    """
    Generate CSES party codes from election results.

    Assigns alphabetical codes (A-F) by descending vote share and
    generates 6-digit numerical codes.

    Args:
        country: Country name or ISO alpha-3 code (e.g., "Switzerland" or "CHE")
        parties_by_vote_share: List of (party_name, vote_share) tuples
            Example: [("SVP", 0.279), ("SP", 0.183), ("FDP", 0.147)]
        study_number: Study number within country (usually 0)
        electoral_system: "single-tier", "multi-tier-one-vote", or "two-votes"

    Returns:
        ToolResult with PartyOrderResult data
    """
    try:
        result = generate_party_codes(
            country=country,
            study_number=study_number,
            parties_by_vote_share=parties_by_vote_share,
            electoral_system=electoral_system
        )

        if result.errors:
            return ToolResult(
                success=False,
                error="; ".join(result.errors)
            )

        # Convert to serializable format
        parties_data = []
        for party in result.parties:
            parties_data.append({
                "name": party.name,
                "vote_share": party.vote_share,
                "code_letter": party.code_letter,
                "numerical_code": party.numerical_code,
                "is_supplemental": party.is_supplemental
            })

        return ToolResult(
            success=True,
            data={
                "country_code": result.country_code,
                "study_number": result.study_number,
                "electoral_system": result.electoral_system,
                "ordering_basis": result.ordering_basis,
                "parties": parties_data
            },
            metadata={
                "n_parties": len(result.parties),
                "n_main_parties": len([p for p in result.parties if not p.is_supplemental])
            }
        )
    except Exception as e:
        logger.error(f"Error generating party codes: {e}")
        return ToolResult(success=False, error=str(e))


def extract_party_codes_from_macro(
    working_dir: Path | str,
    model: Optional[str] = None
) -> ToolResult:
    """
    Extract party information from the dedicated Election Results folder using LLM.

    Searches Election Results/ for election results files and uses
    LLM to semantically extract party names and vote shares.

    Args:
        working_dir: Working directory containing macro/ folder
        model: Optional LLM model to use

    Returns:
        ToolResult with extracted party data
    """
    try:
        working_dir = Path(working_dir)

        parties = extract_party_results_from_macro(working_dir, model)

        if parties is None:
            return ToolResult(
                success=False,
                error="No election results found in Election Results/. Please provide the standardized election-results workbook or source table there."
            )

        if not parties:
            return ToolResult(
                success=False,
                error="Could not extract party data from files in Election Results/."
            )

        return ToolResult(
            success=True,
            data={
                "parties": parties,
                "n_parties": len(parties)
            },
            metadata={
                "source": "election_results_folder",
                "top_party": parties[0].get("name") if parties else None
            }
        )
    except Exception as e:
        logger.error(f"Error extracting party codes from macro: {e}")
        return ToolResult(success=False, error=str(e))


def get_party_order_prompt() -> ToolResult:
    """
    Get the prompt for requesting party data from user.

    Returns standard prompt text explaining what party information
    is needed for CSES coding.

    Returns:
        ToolResult with prompt text
    """
    from src.matching.party_codes import prompt_for_party_data

    return ToolResult(
        success=True,
        data={
            "prompt": prompt_for_party_data(),
            "required_info": [
                "Party names",
                "Vote share percentages",
                "Electoral tier (if multi-tier system)"
            ]
        }
    )


def lookup_country_code(country: str) -> ToolResult:
    """
    Look up UN M49 country code from country name or ISO alpha-3.

    Args:
        country: Country name (e.g., "Switzerland") or ISO alpha-3 (e.g., "CHE")

    Returns:
        ToolResult with UN country code
    """
    code = get_un_country_code(country)

    if code:
        return ToolResult(
            success=True,
            data={
                "country": country,
                "un_code": code,
                "formatted": f"{code}0"  # With study number 0
            }
        )
    else:
        return ToolResult(
            success=False,
            error=f"Unknown country: {country}. Please provide the UN M49 code manually."
        )
