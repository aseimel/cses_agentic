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

# Import existing modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingest import extract_context, ExtractionResult
from src.ingest.data_loader import DataLoader, DatasetInfo
from src.ingest.doc_parser import DocumentParser, DocumentInfo
from src.matching import create_matcher, MatchingResult
from src.preprocessing import DocumentAggregator

logger = logging.getLogger(__name__)


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
    design_report_text: str = "",
    model: Optional[str] = None
) -> ToolResult:
    """
    Aggregate document information for matching.

    Uses DocumentAggregator to create a token-efficient summary
    of all document context.

    Returns:
        ToolResult with aggregated summary string on success
    """
    try:
        aggregator = DocumentAggregator(model=model)
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
    original_reasoning: str,
    model: Optional[str] = None
) -> ToolResult:
    """
    Validate a single variable mapping using Claude.

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
        result = validate_proposal(proposal, model=model)

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

    Returns:
        ToolResult with output path on success
    """
    try:
        import pandas as pd

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build tracking sheet rows
        rows = [
            {"VARIABLES": "VARIABLES", "CSES_CODE": "", "SOURCE_VAR": f"{country_code}_{year}", "REMARKS": "REMARKS"},
            {"VARIABLES": "QUESTIONNAIRE", "CSES_CODE": "CSES M6", "SOURCE_VAR": "(X = missing)", "REMARKS": ""},
            {"VARIABLES": "", "CSES_CODE": "", "SOURCE_VAR": "", "REMARKS": ""}
        ]

        for m in mappings:
            rows.append({
                "VARIABLES": f"{m['cses_variable']}     {m.get('cses_description', '')}",
                "CSES_CODE": m["cses_variable"],
                "SOURCE_VAR": m.get("source_variable", "missing"),
                "REMARKS": m.get("notes", "")[:50]
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
