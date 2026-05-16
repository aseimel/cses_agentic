"""
Document quality assessor for CSES data processing.

Assesses document quality BEFORE expensive LLM processing to:
- Alert users early if documents are insufficient
- Score documents to weight extraction confidence
- Avoid wasting LLM calls on poor-quality documents

Each document type (codebook, questionnaire, design report) has specific
quality criteria that are evaluated using a fast, cheap model.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable

from src.config import LLM_MODEL_QUALITY, LLM_TEMPERATURE
from src.model_runtime import ModelRole, ModelTaskRunner

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality score for a single criterion."""
    criterion: str
    score: float  # 0-10
    explanation: str


@dataclass
class DocumentQuality:
    """Complete quality assessment for a document."""
    doc_type: str
    overall_score: float  # 0-10
    criterion_scores: dict[str, float]
    issues: list[str]
    recommendations: list[str]
    is_usable: bool  # True if score >= 4

    def summary(self) -> str:
        """Return a human-readable summary."""
        status = "USABLE" if self.is_usable else "INSUFFICIENT"
        lines = [
            f"{self.doc_type.upper()}: {self.overall_score:.1f}/10 [{status}]",
        ]
        if self.issues:
            lines.append("  Issues:")
            for issue in self.issues[:3]:
                lines.append(f"    - {issue}")
        if self.recommendations and not self.is_usable:
            lines.append("  Recommendations:")
            for rec in self.recommendations[:3]:
                lines.append(f"    - {rec}")
        return "\n".join(lines)


@dataclass
class QualityReport:
    """Complete quality assessment for all documents."""
    documents: dict[str, DocumentQuality] = field(default_factory=dict)
    overall_score: float = 0.0
    can_proceed: bool = False
    critical_issues: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary of all documents."""
        lines = [
            "=" * 50,
            "DOCUMENT QUALITY ASSESSMENT",
            "=" * 50,
        ]
        for doc_type, quality in self.documents.items():
            lines.append(quality.summary())
            lines.append("")

        lines.append(f"Overall Score: {self.overall_score:.1f}/10")
        lines.append(f"Can Proceed: {'YES' if self.can_proceed else 'NO - Documents insufficient'}")

        if self.critical_issues:
            lines.append("\nCritical Issues:")
            for issue in self.critical_issues:
                lines.append(f"  - {issue}")

        return "\n".join(lines)


class DocumentQualityAssessor:
    """Assess document quality before processing."""

    # Quality criteria by document type
    QUALITY_CRITERIA = {
        "codebook": {
            "variable_definitions": "Does the codebook define all or most variables with clear descriptions?",
            "value_labels": "Are value labels (response codes) provided for categorical variables?",
            "missing_codes": "Are missing value codes documented (refused, don't know, missing)?",
            "completeness": "Does the codebook appear to cover all variables in the dataset?",
        },
        "questionnaire": {
            "question_text": "Are the full survey questions included (not just variable names)?",
            "response_options": "Are response options/answer categories clearly listed?",
            "question_numbers": "Are questions numbered or labeled consistently (Q1, Q2, etc.)?",
            "clarity": "Is the questionnaire clear and readable?",
        },
        "design_report": {
            "sample_design": "Is the sample design described (population, sampling method)?",
            "field_dates": "Are field dates (when data was collected) included?",
            "response_rates": "Are response rates reported?",
            "weighting": "Is weighting information provided?",
        },
    }

    def __init__(self):
        """Initialize the quality assessor."""
        self.runner = ModelTaskRunner()
        self.model = self.runner.model_for_role(ModelRole.VERIFIER) or LLM_MODEL_QUALITY
        self.temperature = LLM_TEMPERATURE
        logger.info(f"DocumentQualityAssessor initialized with model: {self.model}")

    def assess_all_documents(
        self,
        codebook_text: str = "",
        questionnaire_text: str = "",
        design_report_text: str = "",
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> QualityReport:
        """
        Assess quality of all provided documents.

        Args:
            codebook_text: Full text of codebook document
            questionnaire_text: Full text of questionnaire document
            design_report_text: Full text of design report document
            progress_callback: Optional callback for progress updates

        Returns:
            QualityReport with scores and recommendations
        """
        def update_progress(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        update_progress("Starting document quality assessment...")

        report = QualityReport()

        # Assess each document that was provided
        documents_to_assess = []
        if codebook_text:
            documents_to_assess.append(("codebook", codebook_text))
        if questionnaire_text:
            documents_to_assess.append(("questionnaire", questionnaire_text))
        if design_report_text:
            documents_to_assess.append(("design_report", design_report_text))

        if not documents_to_assess:
            report.critical_issues.append("No documents provided")
            report.can_proceed = False
            return report

        # Assess each document
        for doc_type, doc_text in documents_to_assess:
            update_progress(f"  Assessing {doc_type}...")
            quality = self.assess_document(doc_type, doc_text)
            report.documents[doc_type] = quality

            if not quality.is_usable:
                report.critical_issues.append(f"{doc_type}: Score {quality.overall_score:.1f}/10 (below 4.0 threshold)")

        # Calculate overall score (weighted average)
        if report.documents:
            weights = {"codebook": 0.5, "questionnaire": 0.35, "design_report": 0.15}
            total_weight = sum(weights.get(dt, 0.2) for dt in report.documents.keys())
            weighted_sum = sum(
                report.documents[dt].overall_score * weights.get(dt, 0.2)
                for dt in report.documents.keys()
            )
            report.overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Determine if we can proceed
        # Need at least codebook OR questionnaire to be usable
        has_usable_codebook = report.documents.get("codebook", DocumentQuality("", 0, {}, [], [], False)).is_usable
        has_usable_questionnaire = report.documents.get("questionnaire", DocumentQuality("", 0, {}, [], [], False)).is_usable

        report.can_proceed = has_usable_codebook or has_usable_questionnaire

        if not report.can_proceed:
            report.critical_issues.append("Neither codebook nor questionnaire is usable - cannot proceed with matching")

        update_progress(f"Quality assessment complete: {report.overall_score:.1f}/10")
        return report

    def assess_document(self, doc_type: str, doc_text: str) -> DocumentQuality:
        """
        Assess quality of a single document.

        Args:
            doc_type: Type of document (codebook, questionnaire, design_report)
            doc_text: Full document text

        Returns:
            DocumentQuality with scores and recommendations
        """
        if not doc_text or len(doc_text) < 100:
            return DocumentQuality(
                doc_type=doc_type,
                overall_score=0.0,
                criterion_scores={},
                issues=["Document is empty or too short"],
                recommendations=["Provide a complete document with all required information"],
                is_usable=False
            )

        criteria = self.QUALITY_CRITERIA.get(doc_type, {})
        criteria_text = "\n".join(f"- {name}: {desc}" for name, desc in criteria.items())

        # Use first 15k chars for assessment (enough to judge quality)
        doc_preview = doc_text[:15000]

        prompt = f"""Assess the quality of this {doc_type} document for CSES election study data processing.

DOCUMENT TEXT:
{doc_preview}

CRITERIA TO EVALUATE:
{criteria_text}

For each criterion, score 0-10 and provide a brief explanation:
- 0-3: Missing or unusable (major problems)
- 4-6: Partial, may need clarification (some issues)
- 7-10: Complete and clear (good quality)

Return JSON:
{{
  "overall_score": 7.5,
  "criterion_scores": {{
    "criterion_name": 8,
    "another_criterion": 6
  }},
  "issues": ["List of specific problems found"],
  "recommendations": ["What the collaborator should provide or fix"]
}}

Be specific about what is missing or unclear.
Return ONLY valid JSON."""

        try:
            response = self.runner.response(
                ModelRole.VERIFIER,
                model_override=self.model,
                max_tokens=1024,
                temperature=self.temperature,
                timeout=60,
                purpose=f"Assess {doc_type} document quality",
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.choices[0].message.content.strip() if response.choices[0].message.content else ""

            # Parse JSON
            json_data = self._extract_json(response_text)

            if json_data:
                overall_score = float(json_data.get("overall_score", 5.0))
                return DocumentQuality(
                    doc_type=doc_type,
                    overall_score=overall_score,
                    criterion_scores=json_data.get("criterion_scores", {}),
                    issues=json_data.get("issues", []),
                    recommendations=json_data.get("recommendations", []),
                    is_usable=overall_score >= 4.0
                )

        except Exception as e:
            logger.error(f"Quality assessment failed for {doc_type}: {e}")

        # Fallback: basic heuristic assessment
        return self._heuristic_assessment(doc_type, doc_text)

    def _heuristic_assessment(self, doc_type: str, doc_text: str) -> DocumentQuality:
        """Fallback heuristic assessment when LLM fails."""
        issues = []
        score = 5.0  # Start at medium

        # Basic length check
        if len(doc_text) < 1000:
            issues.append("Document is very short")
            score -= 2.0
        elif len(doc_text) > 10000:
            score += 1.0  # Longer documents tend to have more info

        # Check for common keywords based on doc type
        text_lower = doc_text.lower()

        if doc_type == "codebook":
            if "variable" in text_lower:
                score += 0.5
            if "label" in text_lower or "value" in text_lower:
                score += 0.5
            if "missing" in text_lower or "refused" in text_lower:
                score += 0.5

        elif doc_type == "questionnaire":
            # Count question-like patterns
            import re
            question_count = len(re.findall(r'\?|Q\d+|question', text_lower, re.IGNORECASE))
            if question_count >= 10:
                score += 1.0
            elif question_count < 3:
                issues.append("Few questions detected")
                score -= 1.0

        elif doc_type == "design_report":
            if "sample" in text_lower:
                score += 0.5
            if "weight" in text_lower:
                score += 0.5
            if "response rate" in text_lower:
                score += 0.5

        # Clamp score
        score = max(0.0, min(10.0, score))

        return DocumentQuality(
            doc_type=doc_type,
            overall_score=score,
            criterion_scores={},
            issues=issues,
            recommendations=["LLM assessment failed - using heuristic estimate"],
            is_usable=score >= 4.0
        )

    def _extract_json(self, text: str) -> Optional[dict]:
        """Extract JSON from LLM response."""
        import re

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


def create_quality_assessor() -> DocumentQualityAssessor:
    """Factory function to create a DocumentQualityAssessor instance."""
    return DocumentQualityAssessor()
