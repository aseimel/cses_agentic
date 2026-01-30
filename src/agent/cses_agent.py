"""
CSES Data Harmonization Agent - Main Entry Point.

Orchestrates the dual-model validation pipeline:
1. Extract context from uploaded files
2. Run original LLM matcher for proposals
3. Run LLM validation on each proposal
4. Present results for human review

This module provides both programmatic API and CLI interface.
"""

import logging
import json
from dataclasses import dataclass, field
from typing import Optional, Generator
from pathlib import Path
from datetime import datetime, timezone

# Import existing modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingest import extract_context, ExtractionResult
from src.matching import create_matcher, MatchingResult
from src.preprocessing import DocumentAggregator
from src.agent.validator import (
    validate_proposals,
    ValidationResult,
    ValidationVerdict,
    compute_validation_summary
)

# Import audit logger
sys.path.insert(0, str(Path(__file__).parent.parent.parent / ".claude" / "hooks"))
try:
    from audit_logger import AuditLogger, create_session_id
except ImportError:
    # Fallback if hooks not in path
    AuditLogger = None
    create_session_id = lambda: f"ses_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

logger = logging.getLogger(__name__)


@dataclass
class DualModelResult:
    """
    Result of dual-model validation pipeline.

    Contains both original proposals and validation results,
    with computed agreement statistics.
    """
    # Input info
    country: str
    year: str
    session_id: str

    # Results
    extraction: ExtractionResult
    matching: MatchingResult
    validations: list[ValidationResult] = field(default_factory=list)

    # Summary
    total_targets: int = 0
    matched_count: int = 0
    agree_count: int = 0
    disagree_count: int = 0
    uncertain_count: int = 0

    def to_dict(self) -> dict:
        return {
            "metadata": {
                "country": self.country,
                "year": self.year,
                "session_id": self.session_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "summary": {
                "total_targets": self.total_targets,
                "matched_count": self.matched_count,
                "agree_count": self.agree_count,
                "disagree_count": self.disagree_count,
                "uncertain_count": self.uncertain_count,
                "agreement_rate": self.agree_count / self.total_targets if self.total_targets > 0 else 0
            },
            "validations": [v.to_dict() for v in self.validations]
        }

    def get_agreements(self) -> list[ValidationResult]:
        """Get validations where both models agree."""
        return [v for v in self.validations if v.models_agree]

    def get_disagreements(self) -> list[ValidationResult]:
        """Get validations where models disagree."""
        return [v for v in self.validations if not v.models_agree]

    def get_by_verdict(self, verdict: ValidationVerdict) -> list[ValidationResult]:
        """Get validations by the validation LLM's verdict."""
        return [v for v in self.validations if v.verdict == verdict]


class CSESAgent:
    """
    Main CSES Data Harmonization Agent.

    Coordinates the dual-model validation pipeline and human review.
    """

    def __init__(
        self,
        country: str = "Unknown",
        year: str = "Unknown",
        output_dir: Optional[Path] = None,
        enable_audit: bool = True
    ):
        """
        Initialize the CSES agent.

        Args:
            country: Country name or code
            year: Election year
            output_dir: Directory for outputs (default: outputs/{country}_{year})
            enable_audit: Enable audit logging
        """
        self.country = country
        self.year = year
        self.session_id = create_session_id()

        # Setup output directory
        if output_dir is None:
            output_dir = Path("outputs") / f"{country}_{year}"
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.aggregator = DocumentAggregator()
        self.matcher = create_matcher()

        # Initialize audit logger
        self.audit_logger = None
        if enable_audit and AuditLogger:
            audit_dir = self.output_dir / "audit"
            self.audit_logger = AuditLogger(self.session_id, audit_dir)

        logger.info(f"CSESAgent initialized: {country} {year} (session: {self.session_id})")

    def run(
        self,
        data_file: Path,
        doc_files: list[Path],
        validate: bool = True,
        progress_callback: Optional[callable] = None
    ) -> DualModelResult:
        """
        Run the full dual-model harmonization pipeline.

        Args:
            data_file: Path to data file
            doc_files: List of documentation files
            validate: Run LLM validation (default: True)
            progress_callback: Optional callback for progress updates

        Returns:
            DualModelResult with all proposals and validations
        """
        def report(msg):
            if progress_callback:
                progress_callback(msg)
            logger.info(msg)

        # Stage 0: Extract context
        report("Stage 1/4: Extracting context from files...")
        extraction = extract_context(data_file, doc_files)

        if not extraction.variables:
            raise ValueError(f"No variables extracted from {data_file}")

        report(f"  Extracted {len(extraction.variables)} variables")

        # Build source contexts for matcher
        source_contexts = []
        for var in extraction.variables:
            ctx = {
                "name": var.name,
                "description": var.description,
                "value_labels": var.value_labels,
                "sample_values": var.sample_values,
            }
            if var.matched_question_text:
                ctx["question_text"] = var.matched_question_text
            source_contexts.append(ctx)

        # Stage 1: Document aggregation
        report("Stage 2/4: Aggregating document information...")
        summary = self.aggregator.aggregate_variable_info(
            source_variables=source_contexts,
            codebook_text=extraction.codebook_full_text or "",
            questionnaire_text=extraction.questionnaire_full_text or "",
            design_report_text=extraction.design_report_full_text or "",
            progress_callback=report
        )
        report(f"  Aggregated summary: {len(summary)} chars")

        # Stage 2: Original LLM matching
        report("Stage 3/4: Running LLM matching (one-shot)...")
        matching_result = self.matcher.match_variables(
            source_contexts=source_contexts,
            pre_aggregated_summary=summary
        )
        report(f"  Generated {len(matching_result.proposals)} proposals")

        # Log proposals to audit
        if self.audit_logger:
            for proposal in matching_result.proposals:
                self.audit_logger.log_proposal(
                    cses_target=proposal.target_variable,
                    source_variable=proposal.source_variable,
                    confidence=proposal.confidence,
                    reasoning=proposal.reasoning,
                    model=self.matcher.model
                )

        # Stage 3: LLM validation (optional)
        validations = []
        if validate:
            report("Stage 4/4: Running LLM validation...")
            validations = validate_proposals(
                proposals=matching_result.proposals,
                extraction_result=extraction,
                progress_callback=report
            )

            # Log validations to audit
            if self.audit_logger:
                for validation in validations:
                    self.audit_logger.log_validation(
                        cses_target=validation.proposal.target_variable,
                        source_variable=validation.proposal.source_variable,
                        verdict=validation.verdict.value,
                        reasoning=validation.reasoning,
                        suggested_alternative=validation.suggested_alternative,
                        model=validation.validation_model
                    )
        else:
            report("Stage 4/4: Skipped LLM validation")
            # Create placeholder validations
            for proposal in matching_result.proposals:
                validations.append(ValidationResult(
                    proposal=proposal,
                    verdict=ValidationVerdict.UNCERTAIN,
                    reasoning="Validation skipped",
                    models_agree=False
                ))

        # Build result
        result = DualModelResult(
            country=self.country,
            year=self.year,
            session_id=self.session_id,
            extraction=extraction,
            matching=matching_result,
            validations=validations,
            total_targets=len(validations),
            matched_count=len([v for v in validations if v.proposal.source_variable not in ["NOT_FOUND", "ERROR"]]),
            agree_count=len([v for v in validations if v.verdict == ValidationVerdict.AGREE]),
            disagree_count=len([v for v in validations if v.verdict == ValidationVerdict.DISAGREE]),
            uncertain_count=len([v for v in validations if v.verdict == ValidationVerdict.UNCERTAIN])
        )

        report(f"Complete: {result.agree_count} agree, {result.disagree_count} disagree, {result.uncertain_count} uncertain")

        return result

    def approve(
        self,
        cses_target: str,
        source_variable: str,
        user: str = "expert"
    ) -> None:
        """Log a human approval decision."""
        if self.audit_logger:
            self.audit_logger.log_human_decision(
                cses_target=cses_target,
                source_variable=source_variable,
                decision="approve",
                user=user
            )

    def reject(
        self,
        cses_target: str,
        source_variable: str,
        user: str = "expert",
        notes: Optional[str] = None
    ) -> None:
        """Log a human rejection decision."""
        if self.audit_logger:
            self.audit_logger.log_human_decision(
                cses_target=cses_target,
                source_variable=source_variable,
                decision="reject",
                user=user,
                notes=notes
            )

    def edit(
        self,
        cses_target: str,
        old_source: str,
        new_source: str,
        user: str = "expert"
    ) -> None:
        """Log a human edit decision."""
        if self.audit_logger:
            self.audit_logger.log_human_decision(
                cses_target=cses_target,
                source_variable=old_source,
                decision="edit",
                user=user,
                edit_to=new_source
            )

    def export(
        self,
        result: DualModelResult,
        format: str = "json",
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Export results to file.

        Args:
            result: DualModelResult to export
            format: "json" or "xlsx"
            output_path: Output path (default: output_dir/mappings/)

        Returns:
            Path to exported file
        """
        if output_path is None:
            mappings_dir = self.output_dir / "mappings"
            mappings_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if format == "xlsx":
                output_path = mappings_dir / f"mappings_{self.country}_{self.year}_{timestamp}.xlsx"
            else:
                output_path = mappings_dir / f"mappings_{self.country}_{self.year}_{timestamp}.json"

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
        elif format == "xlsx":
            self._export_xlsx(result, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Log export
        if self.audit_logger:
            self.audit_logger.log_export(
                format=format,
                output_path=str(output_path),
                mappings_count=len(result.validations),
                summary={
                    "agree": result.agree_count,
                    "disagree": result.disagree_count,
                    "uncertain": result.uncertain_count
                }
            )

        logger.info(f"Exported to {output_path}")
        return output_path

    def _export_xlsx(self, result: DualModelResult, output_path: Path) -> None:
        """Export to Excel tracking sheet format."""
        import pandas as pd

        rows = []
        for v in result.validations:
            rows.append({
                "CSES Variable": v.proposal.target_variable,
                "CSES Description": v.proposal.target_variable,  # Could look up
                "Source Variable": v.proposal.source_variable,
                "Original Confidence": f"{v.proposal.confidence:.0%}",
                "Original Reasoning": v.proposal.reasoning[:100],
                "Validation Verdict": v.verdict.value,
                "Validation Reasoning": v.reasoning[:100],
                "Models Agree": "Yes" if v.models_agree else "No",
                "Status": "Pending"
            })

        df = pd.DataFrame(rows)
        df.to_excel(output_path, index=False, sheet_name="Mappings")


def run_harmonization(
    data_file: Path,
    doc_files: list[Path],
    country: str = "Unknown",
    year: str = "Unknown",
    validate: bool = True,
    output_dir: Optional[Path] = None
) -> DualModelResult:
    """
    Convenience function to run full harmonization pipeline.

    Args:
        data_file: Path to data file
        doc_files: List of documentation files
        country: Country name/code
        year: Election year
        validate: Run LLM validation
        output_dir: Output directory

    Returns:
        DualModelResult with all proposals and validations
    """
    agent = CSESAgent(country, year, output_dir)
    return agent.run(data_file, doc_files, validate)


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CSES Data Harmonization Agent")
    parser.add_argument("data_file", type=Path, help="Path to data file")
    parser.add_argument("--docs", "-d", type=Path, nargs="+", help="Documentation files")
    parser.add_argument("--country", "-c", default="Unknown", help="Country name/code")
    parser.add_argument("--year", "-y", default="Unknown", help="Election year")
    parser.add_argument("--no-validate", action="store_true", help="Skip LLM validation")
    parser.add_argument("--output", "-o", type=Path, help="Output directory")
    parser.add_argument("--export", "-e", choices=["json", "xlsx", "both"], default="json", help="Export format")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Run pipeline
    result = run_harmonization(
        data_file=args.data_file,
        doc_files=args.docs or [],
        country=args.country,
        year=args.year,
        validate=not args.no_validate,
        output_dir=args.output
    )

    # Print summary
    print("\n" + "=" * 60)
    print(f"CSES Harmonization Complete: {args.country} {args.year}")
    print("=" * 60)
    print(f"Session ID: {result.session_id}")
    print(f"Total targets: {result.total_targets}")
    print(f"Matched: {result.matched_count}")
    print(f"Agreements: {result.agree_count} ({result.agree_count/result.total_targets:.1%})")
    print(f"Disagreements: {result.disagree_count}")
    print(f"Uncertain: {result.uncertain_count}")

    # Show disagreements
    if result.disagree_count > 0:
        print("\nDisagreements (require review):")
        for v in result.get_disagreements():
            print(f"  - {v.proposal.target_variable}: {v.proposal.source_variable}")
            print(f"    Original: {v.proposal.confidence:.0%} - {v.proposal.reasoning[:50]}")
            print(f"    Validator: {v.verdict.value} - {v.reasoning[:50]}")

    # Export
    agent = CSESAgent(args.country, args.year, args.output)
    if args.export in ["json", "both"]:
        agent.export(result, "json")
    if args.export in ["xlsx", "both"]:
        agent.export(result, "xlsx")
