"""
Adaptive context extractor for CSES variable mapping.

Combines information from:
- Data files (variable names, descriptions, labels, sample values)
- Documentation (questionnaire text, response options)

Builds unified context for LLM matching regardless of what metadata is available.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from .data_loader import DataLoader, DatasetInfo, VariableInfo
from .doc_parser import DocumentParser, DocumentInfo, QuestionInfo

logger = logging.getLogger(__name__)


@dataclass
class SourceVariableContext:
    """Context for a single source variable, combining all available info."""
    name: str
    # From data file (may be None)
    description: Optional[str] = None
    value_labels: Optional[dict] = None
    sample_values: list = field(default_factory=list)
    dtype: Optional[str] = None
    n_unique: Optional[int] = None
    # From questionnaire (may be None)
    matched_question_code: Optional[str] = None
    matched_question_text: Optional[str] = None
    matched_response_options: list = field(default_factory=list)
    # Computed
    context_richness: str = "minimal"  # "rich", "partial", "minimal"

    def to_llm_context(self) -> str:
        """Format for LLM consumption."""
        parts = [f"Variable: {self.name}"]

        if self.description:
            parts.append(f"  Description: {self.description}")

        if self.matched_question_code:
            parts.append(f"  Questionnaire code: {self.matched_question_code}")

        if self.matched_question_text:
            parts.append(f"  Question text: {self.matched_question_text}")

        if self.value_labels:
            labels_str = ", ".join(f"{k}={v}" for k, v in list(self.value_labels.items())[:5])
            parts.append(f"  Value labels: {labels_str}")

        if self.matched_response_options:
            options_str = ", ".join(self.matched_response_options[:5])
            parts.append(f"  Response options: {options_str}")

        if self.sample_values:
            samples_str = ", ".join(str(v) for v in self.sample_values[:5])
            parts.append(f"  Sample values: {samples_str}")

        if self.dtype:
            parts.append(f"  Type: {self.dtype}")

        return "\n".join(parts)


@dataclass
class ExtractionResult:
    """Result of adaptive context extraction."""
    # Source info
    data_file: Optional[Path] = None
    doc_files: list[Path] = field(default_factory=list)
    # Extracted context
    variables: list[SourceVariableContext] = field(default_factory=list)
    questionnaire_questions: list[QuestionInfo] = field(default_factory=list)
    # FULL document texts for LLM semantic matching
    questionnaire_full_text: str = ""  # Full questionnaire content
    codebook_full_text: str = ""  # Full codebook content (if provided)
    design_report_full_text: str = ""  # Full design report content (if provided)
    # Quality metrics
    data_quality: str = "none"  # "rich", "partial", "minimal", "none"
    doc_quality: str = "none"  # "rich", "partial", "none"
    overall_quality: str = "minimal"
    # Errors
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Get extraction summary."""
        return (
            f"Data file: {self.data_file.name if self.data_file else 'None'}\n"
            f"Doc files: {len(self.doc_files)}\n"
            f"Variables extracted: {len(self.variables)}\n"
            f"Questions extracted: {len(self.questionnaire_questions)}\n"
            f"Data quality: {self.data_quality}\n"
            f"Doc quality: {self.doc_quality}\n"
            f"Overall quality: {self.overall_quality}\n"
            f"Errors: {len(self.errors)}"
        )

    def get_variables_by_richness(self, richness: str) -> list[SourceVariableContext]:
        """Get variables filtered by context richness."""
        return [v for v in self.variables if v.context_richness == richness]


class AdaptiveContextExtractor:
    """
    Extracts and combines context from whatever sources are available.

    Handles scenarios:
    1. Rich metadata: Full variable descriptions + questionnaire
    2. Partial metadata: Some descriptions OR questionnaire
    3. Minimal: Only variable names (still usable via pattern matching)
    """

    def __init__(self):
        self.data_loader = DataLoader()
        self.doc_parser = DocumentParser()

    def extract(
        self,
        data_file: Optional[Path | str] = None,
        doc_files: Optional[list[Path | str]] = None
    ) -> ExtractionResult:
        """
        Extract context from available sources.

        Args:
            data_file: Path to data file (.dta, .sav, .csv, etc.)
            doc_files: List of documentation files (.docx, .pdf, .txt, etc.)

        Returns:
            ExtractionResult with combined context
        """
        result = ExtractionResult()

        # Load data file if provided
        dataset_info = None
        if data_file:
            data_file = Path(data_file)
            result.data_file = data_file
            dataset_info = self.data_loader.load(data_file)
            if dataset_info:
                result.data_quality = dataset_info.metadata_quality
            else:
                result.errors.append(f"Failed to load data file: {data_file}")

        # Parse documentation files if provided
        doc_infos = []
        all_doc_texts = []  # Collect all document texts for LLM
        if doc_files:
            for doc_file in doc_files:
                doc_file = Path(doc_file)
                result.doc_files.append(doc_file)
                doc_info = self.doc_parser.parse(doc_file)
                if doc_info:
                    doc_infos.append(doc_info)
                    if doc_info.questions:
                        result.questionnaire_questions.extend(doc_info.questions)
                    # Store full text for LLM semantic matching
                    if doc_info.full_text:
                        all_doc_texts.append(f"=== Document: {doc_file.name} ===\n{doc_info.full_text}")
                        # Heuristic: determine document type from filename
                        fname_lower = doc_file.name.lower()
                        is_codebook = any(x in fname_lower for x in ['codebook', 'code book', 'variable', 'dictionary'])
                        is_questionnaire_file = any(x in fname_lower for x in ['questionnaire', 'survey', 'translation', 'pregunta', 'fragebogen'])
                        is_design_report = any(x in fname_lower for x in ['design', 'report', 'methodology', 'technical', 'micro'])

                        if is_codebook:
                            result.codebook_full_text += doc_info.full_text + "\n\n"
                        elif is_questionnaire_file or doc_info.is_questionnaire:
                            result.questionnaire_full_text += doc_info.full_text + "\n\n"
                        elif is_design_report:
                            result.design_report_full_text += doc_info.full_text + "\n\n"
                        else:
                            # Default: treat as additional documentation (goes to codebook)
                            result.codebook_full_text += doc_info.full_text + "\n\n"
                else:
                    result.errors.append(f"Failed to parse document: {doc_file}")

        # Assess doc quality
        if result.questionnaire_questions:
            if len(result.questionnaire_questions) >= 20:
                result.doc_quality = "rich"
            elif len(result.questionnaire_questions) >= 5:
                result.doc_quality = "partial"
            else:
                result.doc_quality = "minimal"

        # Build variable contexts
        if dataset_info:
            result.variables = self._build_variable_contexts(
                dataset_info, result.questionnaire_questions
            )

        # Compute overall quality
        result.overall_quality = self._compute_overall_quality(
            result.data_quality, result.doc_quality
        )

        return result

    def _build_variable_contexts(
        self,
        dataset_info: DatasetInfo,
        questions: list[QuestionInfo]
    ) -> list[SourceVariableContext]:
        """Build context for each variable, matching with questionnaire if possible."""
        contexts = []

        # Create lookup for questions by code
        question_lookup = {q.code: q for q in questions}
        # Also create normalized lookup (Q1 -> Q01, etc.)
        for q in questions:
            normalized = self._normalize_code(q.code)
            if normalized not in question_lookup:
                question_lookup[normalized] = q

        for var_name, var_info in dataset_info.variables.items():
            ctx = SourceVariableContext(
                name=var_name,
                description=var_info.description,
                value_labels=var_info.value_labels,
                sample_values=var_info.sample_values,
                dtype=var_info.dtype,
                n_unique=var_info.n_unique
            )

            # Try to match variable to questionnaire question
            matched_question = self._match_variable_to_question(var_name, question_lookup)
            if matched_question:
                ctx.matched_question_code = matched_question.code
                ctx.matched_question_text = matched_question.text
                ctx.matched_response_options = matched_question.response_options

            # Compute richness
            ctx.context_richness = self._compute_richness(ctx)

            contexts.append(ctx)

        return contexts

    def _match_variable_to_question(
        self,
        var_name: str,
        question_lookup: dict[str, QuestionInfo]
    ) -> Optional[QuestionInfo]:
        """Try to match a variable name to a questionnaire question."""
        import re

        # Direct lookup (exact match)
        if var_name in question_lookup:
            return question_lookup[var_name]

        # Uppercase lookup
        if var_name.upper() in question_lookup:
            return question_lookup[var_name.upper()]

        # Try to extract question code from variable name
        # Patterns: q01, Q01, dem_q01, fes4_Q01, q01a, etc.
        patterns = [
            r'[_]?([QDA]\d{1,2}[a-z]?)(?:_|$)',  # Q01, D04, A01 with optional suffix
            r'([Qq]\d{1,2}[a-z]?)',  # q01, Q01, q01a
            r'([Dd]\d{1,2}[a-z]?)',  # d04, D04
        ]

        for pattern in patterns:
            match = re.search(pattern, var_name, re.IGNORECASE)
            if match:
                code = match.group(1).upper()
                normalized = self._normalize_code(code)
                if normalized in question_lookup:
                    return question_lookup[normalized]
                if code in question_lookup:
                    return question_lookup[code]

        return None

    def _normalize_code(self, code: str) -> str:
        """Normalize question code (Q1 -> Q01)."""
        import re
        match = re.match(r'([A-Z])(\d+)([a-z]?)', code, re.IGNORECASE)
        if match:
            prefix = match.group(1).upper()
            num = int(match.group(2))
            suffix = match.group(3).lower() if match.group(3) else ''
            return f"{prefix}{num:02d}{suffix}"
        return code.upper()

    def _compute_richness(self, ctx: SourceVariableContext) -> str:
        """Compute context richness for a variable."""
        score = 0

        if ctx.description:
            score += 2
        if ctx.value_labels:
            score += 2
        if ctx.matched_question_text:
            score += 2
        if ctx.matched_response_options:
            score += 1
        if ctx.sample_values:
            score += 1

        if score >= 4:
            return "rich"
        elif score >= 2:
            return "partial"
        else:
            return "minimal"

    def _compute_overall_quality(self, data_quality: str, doc_quality: str) -> str:
        """Compute overall extraction quality."""
        quality_scores = {"rich": 3, "partial": 2, "minimal": 1, "none": 0}

        data_score = quality_scores.get(data_quality, 0)
        doc_score = quality_scores.get(doc_quality, 0)

        combined = data_score + doc_score

        if combined >= 5:
            return "rich"
        elif combined >= 3:
            return "partial"
        elif combined >= 1:
            return "minimal"
        else:
            return "none"


def extract_context(
    data_file: Optional[Path | str] = None,
    doc_files: Optional[list[Path | str]] = None
) -> ExtractionResult:
    """Convenience function to extract context."""
    extractor = AdaptiveContextExtractor()
    return extractor.extract(data_file, doc_files)
