"""
Step Executor for CSES Workflow.

Provides execution logic for each of the 16 workflow steps.
Each step can be executed interactively with user guidance.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Any

from .state import WorkflowState, StepStatus, WORKFLOW_STEPS
from src.workflow.evidence_packet import EvidencePacketBuilder
from src.workflow.phases import current_phase_id, phase_status_payload
from src.preprocessing.evidence_extractor import ParallelEvidenceExtractionService, summarize_evidence_index
from src.study_kb import StudyKnowledgeBase, StudyKnowledgeBaseBuilder
from src.ui_text import format_study_review_status, sanitize_processor_text
from src.matching.party_order import (
    ElectionResultsWorkbookParser,
    PartyOrderingRulesEngine,
    election_results_intake_summary,
    infer_election_context_from_macro_material,
    party_agreement_summary,
    party_order_message,
    write_election_results_intake,
)
from src.matching.party_metadata import (
    PartyMetadataReviewBuilder,
    party_metadata_message,
)
from src.matching.macro_context import (
    MacroContextBuilder,
    macro_context_message,
)

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of executing a workflow step."""
    success: bool
    message: str
    artifacts: list[str] = None
    issues: list[str] = None
    next_action: str = None  # Suggested next action
    standards_status: str = ""
    standards_checks: dict = None
    wiki_sources: list[str] = None
    human_decision_required: bool = False
    ready_for_next_step: bool = True

    def __post_init__(self):
        self.artifacts = self.artifacts or []
        self.issues = self.issues or []
        self.standards_checks = self.standards_checks or {}
        self.wiki_sources = self.wiki_sources or []


class StepExecutor:
    """
    Executes workflow steps with LLM assistance.

    Each step method:
    1. Checks prerequisites
    2. Executes the step (possibly with LLM)
    3. Updates workflow state
    4. Returns result with artifacts and issues
    """

    def __init__(self, state: WorkflowState, llm_callback: Callable = None):
        """
        Initialize executor.

        Args:
            state: Current workflow state
            llm_callback: Function to call LLM for assistance
        """
        self.state = state
        self.llm_callback = llm_callback
        self.working_dir = Path(state.working_dir) if state.working_dir else Path.cwd()

        # Initialize active logger for real-time logging
        from .active_logging import ActiveLogger
        from src.standards.engine import WorkflowStandardsEngine
        self.active_logger = ActiveLogger(state)
        self.standards = WorkflowStandardsEngine(self.working_dir)

    def _refresh_phase_state(self) -> None:
        self.state.current_phase = current_phase_id(self.state)
        self.state.phase_status = phase_status_payload(self.state)

    def _ensure_evidence_packet(self, force: bool = False) -> dict:
        builder = EvidencePacketBuilder(self.working_dir)
        packet = builder.build(self.state, force=force)
        return packet

    def _evidence_packet_status_line(self) -> str:
        status = EvidencePacketBuilder(self.working_dir).status()
        return format_study_review_status(status.status)

    def _ensure_study_kb(self, force: bool = False) -> dict:
        """Build or load the study-specific KB before evidence decisions."""
        kb = StudyKnowledgeBase(self.working_dir)
        if kb.exists() and not force and not kb.is_stale():
            payload = kb.payload
            StudyKnowledgeBaseBuilder(self.working_dir).update_state(self.state, payload)
            return payload
        builder = StudyKnowledgeBaseBuilder(
            self.working_dir,
            progress_callback=lambda message: print(f"  {sanitize_processor_text(message)}") if sanitize_processor_text(message) else None,
        )
        payload = builder.build(self.state, force=force)
        self.active_logger.log_message("Study materials reviewed and summarized for processing.")
        return payload

    def _record_missing_items_for_processor_review(
        self,
        step_num: int,
        missing_items: list[str],
        context: str,
        suggested_question: str,
    ) -> list[str]:
        """Record missing items without prematurely creating collaborator questions."""
        clean_items = [str(item) for item in missing_items if str(item).strip()]
        if not clean_items:
            return []
        self.active_logger.log_message(
            f"Step {step_num} missing items detected for processor review ({context}):\n"
            + "\n".join(f"  - {item}" for item in clean_items)
            + "\nThe agent must first use available data, documentation, and CSES wiki evidence. "
              "The processor decides whether a collaborator question is needed.",
            level="WARNING",
        )
        self.active_logger.add_candidate_collaborator_question(
            suggested_question,
            context,
            step_num=step_num,
            missing_items=clean_items,
        )
        return [f"Missing item for processor review: {item}" for item in clean_items]

    def _format_value(self, value: Any, default: str = "Not found in deposited files") -> str:
        text = str(value or "").strip()
        if not text or text.lower() in {"tbd", "not specified", "not specified in report", "none"}:
            return default
        return text

    def _review_party_ordering_inputs(self) -> dict[str, Any]:
        """Find early party-ordering inputs needed for vote-choice coding."""
        summary = election_results_intake_summary(self.working_dir)
        files = summary.get("files", [])
        tables = summary.get("tables", [])
        status = "found" if files else "needs_review"
        return {
            "status": status,
            "files": files,
            "tables": tables,
            "message": (
                f"Election results material found ({len(files)} file(s)); standardized table(s) detected: {len(tables)}."
                if files
                else "Election results material not found in the deposited files."
            ),
        }

    def _format_eligibility_brief(self, eligibility: Any, detected: Any, party_ordering: dict[str, Any] | None = None) -> str:
        probability_labels = {
            "confirmed_probability_sample": "Yes - evidence indicates a probability sample",
            "confirmed_non_probability_sample": "No - documentation indicates a non-probability sample",
            "unclear": "Needs processor review - not enough explicit sampling evidence yet",
        }
        probability = probability_labels.get(
            getattr(eligibility, "probability_sample_status", "unclear"),
            "Needs processor review",
        )
        sample_size = (
            getattr(eligibility, "documented_sample_size", "")
            or getattr(eligibility, "sample_size_rows", None)
        )
        missing_items = getattr(eligibility, "missing_cses_items", []) or []
        if missing_items:
            coverage_lines = [f"- Missing items: {', '.join(missing_items[:80])}"]
        else:
            coverage = self._clean_coverage_text(
                self._format_value(
                    getattr(eligibility, "cses_items_evidence", ""),
                    "No missing CSES questionnaire items were identified in the current evidence search.",
                )
            )
            coverage_lines = [f"- Coverage assessment: {coverage}"]

        evidence = getattr(eligibility, "sampling_evidence", []) or []
        evidence_text = "; ".join(str(item) for item in evidence[:3]) if evidence else "No direct sampling quote recorded."

        lines = [
            "**Step 1 - Deposit and Eligibility Brief**",
            "",
            f"Probability sample: {probability}",
            f"Sample design: {self._format_value(getattr(eligibility, 'sample_design', ''))}",
            f"Sample size: {self._format_value(sample_size)}",
            f"Mode: {self._format_value(getattr(eligibility, 'mode', ''))}",
            f"Fieldwork: {self._format_value(getattr(eligibility, 'fieldwork_dates', ''))}",
            f"Weights: {self._format_value(getattr(eligibility, 'weights', ''))}",
            f"Response rate: {self._format_value(getattr(eligibility, 'response_rate', ''))}",
            "",
            "CSES item coverage:",
            *coverage_lines,
            "",
            "Key sampling evidence:",
            f"- {evidence_text}",
            "",
            "Party ordering inputs:",
            f"- {self._format_value((party_ordering or {}).get('message'), 'Not reviewed yet')}",
            "- Processor should confirm the election-results source before vote-choice and party-code recoding.",
            "",
            "Processor decision needed:",
            "- Confirm one of: eligible, not eligible, or needs Secretariat review.",
        ]
        if detected.data_files or detected.questionnaire_files or detected.design_report_files:
            lines.extend([
                "",
                "Deposit inputs found:",
                f"- Data files: {len(detected.data_files)}",
                f"- Questionnaires: {len(detected.questionnaire_files)}",
                f"- Design reports: {len(detected.design_report_files)}",
            ])
        return "\n".join(lines)

    def _clean_coverage_text(self, text: str) -> str:
        text = " ".join(str(text or "").split())
        if not text:
            return "Not found in deposited files"
        if "full cses module" in text.casefold():
            first_sentence = text.split(".", 1)[0].strip()
            return first_sentence + "." if first_sentence else text[:220]
        return text[:420] + ("..." if len(text) > 420 else "")

    def _format_design_brief(self, design_report: str, extracted_info: dict[str, Any], missing_fields: list[str]) -> str:
        labels = {
            "sample_design": "Sample design",
            "sample_size": "Sample size",
            "response_rate": "Response rate",
            "weighting": "Weights",
            "collection_period": "Fieldwork",
            "mode": "Mode",
            "target_population": "Target population",
            "field_lag": "Field lag",
        }
        lines = [
            "**Step 2 - Study Design Documentation Brief**",
            "",
            f"Design source reviewed: {Path(design_report).name}",
            "",
            "Study design fields recorded:",
        ]
        for key, label in labels.items():
            lines.append(f"- {label}: {self._format_value(extracted_info.get(key))}")
        lines.append("")
        if missing_fields:
            lines.append("Needs processor review before collaborator contact:")
            for field in missing_fields[:20]:
                lines.append(f"- {field}")
        else:
            lines.append("Needs processor review before collaborator contact:")
            lines.append("- No missing study-design fields were identified.")
        return "\n".join(lines)

    def execute_step(self, step_num: int, **kwargs) -> StepResult:
        """
        Execute a specific workflow step.

        Args:
            step_num: Step number (0-16)
            **kwargs: Step-specific arguments

        Returns:
            StepResult with success status and details
        """
        if step_num not in WORKFLOW_STEPS:
            return StepResult(
                success=False,
                message=f"Unknown step number: {step_num}"
            )

        step_info = WORKFLOW_STEPS[step_num]
        step_name = step_info["name"]

        # Get step handler
        handler_name = f"_step_{step_num}"
        handler = getattr(self, handler_name, None)

        if handler is None:
            return StepResult(
                success=False,
                message=f"Step {step_num} handler not implemented yet",
                next_action=f"This step requires manual execution. See workflow.md for details."
            )

        # Log step start
        self.active_logger.log_step_start(step_num, step_name)

        # Mark step as in progress
        self.state.set_step_status(step_num, StepStatus.IN_PROGRESS)
        self.state.save()

        try:
            result = handler(**kwargs)
            self._apply_standards(step_num, result)
            result.message = sanitize_processor_text(result.message)
            result.next_action = sanitize_processor_text(result.next_action) if result.next_action else result.next_action
            result.issues = [sanitize_processor_text(issue) for issue in result.issues if sanitize_processor_text(issue)]

            # Update state based on result
            if result.success:
                self.state.set_step_status(
                    step_num,
                    StepStatus.COMPLETED,
                    note=result.message
                )
                for artifact in result.artifacts:
                    self.state.add_step_artifact(step_num, artifact)

                # Log step completion
                self.active_logger.log_step_complete(step_num, result.message, result.artifacts)
            else:
                # Log issues
                for issue in result.issues:
                    self.state.add_step_issue(step_num, issue)
                    self.active_logger.log_step_issue(step_num, issue)

            self._refresh_phase_state()
            self.state.save()
            return result

        except Exception as e:
            logger.error(f"Step {step_num} failed: {e}")
            error_msg = str(e)
            self.state.add_step_issue(step_num, error_msg)
            self.active_logger.log_step_issue(step_num, f"Error: {error_msg}")
            self._refresh_phase_state()
            self.state.save()
            return StepResult(
                success=False,
                message=f"Step failed with error: {e}",
                issues=[error_msg]
            )

    def _apply_standards(self, step_num: int, result: StepResult) -> None:
        """Run and persist CSES wiki-backed standards checks for a step."""
        try:
            check = self.standards.evaluate_step(self.state, step_num, result.artifacts)
            result.standards_status = check.status
            result.standards_checks = check.checks
            result.wiki_sources = check.wiki_sources
            result.human_decision_required = check.human_decision_required
            result.ready_for_next_step = check.ready_for_next_step
            for issue in check.issues:
                if issue not in result.issues:
                    result.issues.append(issue)
            self.state.record_standards_check(step_num, check.to_dict())
            self.active_logger.record_standards_check(step_num, check.to_dict())
            if check.issues:
                self.active_logger.log_message(
                    f"Standards check for Step {step_num}: {check.status}. "
                    f"Unresolved issues: {'; '.join(check.issues[:5])}",
                    level="WARNING"
                )
        except Exception as exc:
            logger.error(f"Standards check failed for step {step_num}: {exc}")
            result.issues.append(f"Standards check failed: {exc}")

    def _step_0(self, **kwargs) -> StepResult:
        """Step 0: Set Up Country Folder"""
        # This is typically done during initialization
        # Check that folder structure exists following CSES standard

        issues = []

        # Check micro/original_deposit/
        original_deposit = self.working_dir / "micro" / "original_deposit"
        if not original_deposit.exists():
            issues.append("Create folder: micro/original_deposit/")

        # Check micro/ folder exists
        if not (self.working_dir / "micro").exists():
            issues.append("Create folder: micro/")

        # Check macro/ folder exists
        if not (self.working_dir / "macro").exists():
            issues.append("Create folder: macro/")

        # Check Election Results/ folder
        if not (self.working_dir / "Election Results").exists():
            issues.append("Create folder: Election Results/")

        if issues:
            return StepResult(
                success=False,
                message=f"Missing required folders ({len(issues)} issues)",
                issues=issues,
                next_action="Run 'cses migrate' to create missing folders"
            )

        return StepResult(
            success=True,
            message="Folder structure verified (CSES standard)",
            artifacts=[str(self.working_dir)],
            next_action="Proceed to Step 1: Check deposit completeness"
        )

    def _step_1(self, **kwargs) -> StepResult:
        """Step 1: Check Completeness of Deposit"""
        from src.workflow.organizer import FileOrganizer

        print("Checking deposit completeness...")
        self.active_logger.log_message("Checking deposit completeness...")

        # Check micro/ directory first (new flow puts standardized files here)
        micro_dir = self.working_dir / "micro"

        # Also check original_deposit for legacy support
        original_deposit = micro_dir / "original_deposit"

        if micro_dir.exists():
            organizer = FileOrganizer(micro_dir)
            detected = organizer.detect_files()
            print(f"  Scanning micro/ directory...")
        elif original_deposit.exists():
            organizer = FileOrganizer(original_deposit)
            detected = organizer.detect_files()
            print(f"  Scanning micro/original_deposit/ directory...")
        else:
            organizer = FileOrganizer(self.working_dir)
            detected = organizer.detect_files()
            print(f"  Scanning working directory...")

        issues = []
        if not detected.data_files:
            issues.append("No data file found")
        if not detected.questionnaire_files:
            issues.append("No questionnaire found (required per CSES policy)")
        if not detected.design_report_files:
            issues.append("No design report found")
        self._record_missing_items_for_processor_review(
            1,
            issues,
            "Deposit completeness review",
            "Some deposit materials appear to be missing. Processor should confirm whether the information can be retrieved from available files before asking the collaborator.",
        )

        # Build a complete input manifest before choosing the primary data file.
        from src.workflow.input_manifest import InputManifestBuilder

        manifest = InputManifestBuilder(self.working_dir).build()
        manifest_path = InputManifestBuilder(self.working_dir).write(manifest)
        self.state.input_manifest_path = str(manifest_path)
        self.state.primary_input_selection = {
            "primary_data_file": manifest.primary_data_file,
            "reason": manifest.primary_data_reason,
            "warnings": manifest.warnings,
        }
        if manifest.primary_data_file:
            self.state.data_file = manifest.primary_data_file
            self.active_logger.log_message(
                f"Selected primary data file: {Path(manifest.primary_data_file).name}. "
                f"Reason: {manifest.primary_data_reason}"
            )
        for warning in manifest.warnings:
            self.active_logger.log_message(f"Input manifest warning: {warning}", level="WARNING")

        # Update state with detected files
        if detected.data_files and not self.state.data_file:
            self.state.data_file = str(detected.data_files[0])
            self.active_logger.log_message(f"Found data file: {detected.data_files[0].name}")
        self.state.questionnaire_files = [str(f) for f in detected.questionnaire_files]
        if detected.codebook_files:
            self.state.codebook_file = str(detected.codebook_files[0])
        if detected.design_report_files:
            self.state.design_report_file = str(detected.design_report_files[0])

        # Show what was found
        print(f"  Data files: {len(detected.data_files)}")
        print(f"  Questionnaires: {len(detected.questionnaire_files)}")
        print(f"  Design reports: {len(detected.design_report_files)}")
        print(f"  Codebooks: {len(detected.codebook_files)}")

        packet_status_before = EvidencePacketBuilder(self.working_dir).status()
        force_evidence_refresh = packet_status_before.status == "stale"
        if packet_status_before.status == "current":
            print("  Study materials were already reviewed.")
        elif packet_status_before.status == "stale":
            print("  Study files changed; reviewing study materials again.")

        print("  Reviewing study materials from deposited files...")
        study_kb = self._ensure_study_kb(force=force_evidence_refresh)
        kb_artifacts = [
            str(self.working_dir / ".cses" / "study_kb.json"),
            str(self.working_dir / ".cses" / "study_kb.toon"),
            str(self.working_dir / ".cses" / "study_kb_sources.json"),
            str(self.working_dir / ".cses" / "model_diagnostics.json"),
        ]
        evidence_service = ParallelEvidenceExtractionService(
            self.working_dir,
            progress_callback=lambda message: print(f"  {sanitize_processor_text(message)}") if sanitize_processor_text(message) else None,
        )
        evidence_index = evidence_service.build(self.state, force=force_evidence_refresh)
        self.state.evidence_index = evidence_index
        if evidence_index.get("failed_chunks"):
            self.active_logger.log_message(
                "Some study material sections need technical review before relying on extracted facts.",
                level="WARNING",
            )

        print("  Reviewing initial CSES eligibility...")
        from src.workflow.eligibility import review_initial_eligibility

        eligibility = review_initial_eligibility(
            working_dir=self.working_dir,
            data_files=detected.data_files,
            questionnaire_files=detected.questionnaire_files,
            codebook_files=detected.codebook_files,
            design_report_files=detected.design_report_files,
        )
        print(f"  Probability sample: {eligibility.probability_sample_status}")
        if eligibility.sample_design:
            print(f"  Sample design: {eligibility.sample_design[:180]}")
        if eligibility.documented_sample_size or eligibility.sample_size_rows is not None:
            print(f"  Sample size: {eligibility.documented_sample_size or eligibility.sample_size_rows}")
        if eligibility.mode:
            print(f"  Mode: {eligibility.mode}")
        if eligibility.missing_cses_items:
            print(f"  Missing CSES items: {', '.join(eligibility.missing_cses_items[:20])}")

        if eligibility.sample_size_rows is not None:
            self.active_logger.update_study_design_section({
                "sample_size": str(eligibility.sample_size_rows),
            })
        self.active_logger.update_study_design_section({
            "probability_sample_status": eligibility.probability_sample_status,
            "probability_sample_assessment": eligibility.probability_sample_assessment,
            "sampling_evidence": "\n".join(eligibility.sampling_evidence[:8]),
            "cses_item_coverage": eligibility.cses_items_evidence,
            "eligibility_assessment": eligibility.eligibility_assessment,
            "processor_eligibility_decision": eligibility.processor_eligibility_decision,
        })
        self.active_logger.log_message(eligibility.to_log_message())
        eligibility_issues = [
            issue for issue in eligibility.issues
            if "mapping" not in issue.casefold() and "f-code" not in issue.casefold()
        ]
        issues.extend(eligibility_issues)
        if eligibility.probability_sample_status == "unclear":
            if "Probability-sample eligibility needs processor review." not in issues:
                issues.append("Probability-sample eligibility needs processor review.")
        if eligibility.cses_items_included_count is None and eligibility.direct_cses_variable_count == 0:
            if not eligibility.missing_cses_items:
                issues.append("CSES questionnaire item coverage needs processor review.")

        party_ordering = self._review_party_ordering_inputs()
        self.active_logger.log_message(f"Party ordering input review: {party_ordering.get('message')}")
        if party_ordering.get("status") != "found":
            self._record_missing_items_for_processor_review(
                1,
                ["Election results material for party ordering"],
                "Early party-ordering review",
                "Processor should confirm whether election results can be retrieved from available files before asking for external material.",
            )
            issues.append("Election results material for party ordering needs processor review.")

        # UPDATE LOG FILE with deposit inventory
        self.active_logger.update_deposit_inventory(
            data_files=[str(f) for f in detected.data_files],
            questionnaires=[str(f) for f in detected.questionnaire_files],
            codebooks=[str(f) for f in detected.codebook_files],
            design_reports=[str(f) for f in detected.design_report_files],
            macro_reports=[str(f) for f in detected.macro_report_files] if hasattr(detected, 'macro_report_files') else []
        )
        print("  Log file updated with deposit inventory")
        self.active_logger.log_message(summarize_evidence_index(evidence_index))
        self.active_logger.log_message("Study materials review completed.")
        evidence_packet = self._ensure_evidence_packet(force=False)
        self.active_logger.log_message(f"Study materials review status: {evidence_packet.get('status')}.")

        if detected.has_minimum_requirements():
            message = self._format_eligibility_brief(eligibility, detected, party_ordering)
            review_status = self._evidence_packet_status_line()
            if review_status and "changed" in review_status.lower():
                message = f"{message}\n\n{review_status}"
            return StepResult(
                success=True,
                message=message,
                artifacts=[str(f) for f in detected.data_files + detected.questionnaire_files] + kb_artifacts + [str(manifest_path), str(self.working_dir / ".cses" / "evidence_packet.json")],
                issues=issues if issues else None,
                next_action="Record the processor eligibility decision, then proceed to Step 2 if appropriate.",
                human_decision_required=True,
            )
        else:
            print(f"  Issues: {issues}")
            return StepResult(
                success=False,
                message="Deposit incomplete - missing required files",
                issues=issues,
                next_action="Request missing files from collaborator"
            )

    def _step_2(self, **kwargs) -> StepResult:
        """Step 2: Confirm design facts and update log file with study design info."""
        design_report = self.state.design_report_file

        if not design_report or not Path(design_report).exists():
            return StepResult(
                success=False,
                message="Design report not found",
                issues=["Request design report from collaborator"],
                next_action="Cannot proceed without design report"
            )

        print("Confirming study design facts from reviewed study materials...")
        self.active_logger.log_message("Confirming study design facts from reviewed study materials...")

        packet_builder = EvidencePacketBuilder(self.working_dir)
        packet_status = packet_builder.status()
        if packet_status.status == "missing":
            packet = self._ensure_evidence_packet(force=False)
        else:
            packet = packet_builder.load()
            packet_builder.update_state(self.state, packet)
        evidence_index = self.state.evidence_index or {}
        study_kb = StudyKnowledgeBase(self.working_dir)
        design_facts = packet.get("design_facts", {}) or {}
        extracted_info = {
            'sample_design': design_facts.get("sample_design") or 'TBD',
            'sample_size': design_facts.get("sample_size") or 'TBD',
            'response_rate': design_facts.get("response_rate") or 'TBD',
            'weighting': design_facts.get("weights") or 'TBD',
            'collection_period': design_facts.get("fieldwork_dates") or 'TBD',
            'mode': design_facts.get("mode") or 'TBD',
            'target_population': design_facts.get("target_population") or 'TBD',
            'field_lag': design_facts.get("field_lag") or 'TBD',
            'consent_data_deposit': design_facts.get("consent_data_deposit") or 'TBD',
        }

        existing_design = getattr(self.active_logger.log_data, "study_design", {}) if self.active_logger.log_data else {}
        for key, existing_value in existing_design.items():
            current_value = extracted_info.get(key)
            current_missing = not current_value or str(current_value).strip().lower() in {
                "tbd",
                "not specified in report",
                "not specified",
            }
            existing_present = existing_value and str(existing_value).strip().lower() not in {
                "tbd",
                "not specified in report",
                "not specified",
            }
            if key in extracted_info and current_missing and existing_present:
                extracted_info[key] = existing_value

        # UPDATE LOG FILE with study design section
        print("Updating log file with study design information...")
        supported_log_fields = {
            key: value
            for key, value in extracted_info.items()
            if key in {
                "sample_design",
                "sample_size",
                "response_rate",
                "weighting",
                "collection_period",
                "mode",
                "field_lag",
            }
        }
        self.active_logger.update_study_design_section(supported_log_fields)

        # Log what was extracted
        missing_fields = [
            key for key, value in extracted_info.items()
            if key != "consent_data_deposit"
            if not value or str(value).strip().lower() in {"tbd", "not specified in report", "not specified"}
        ]
        resolved_aliases = {
            "fieldwork": "collection_period",
            "fieldwork_dates": "collection_period",
            "weights": "weighting",
            "weight": "weighting",
            "response": "response_rate",
        }
        for missing in study_kb.missing_fields():
            missing_key = str(missing).lower()
            resolved_key = next((field for token, field in resolved_aliases.items() if token in missing_key), missing)
            resolved_value = extracted_info.get(resolved_key)
            if resolved_value and str(resolved_value).strip().lower() not in {"tbd", "not specified in report", "not specified"}:
                continue
            if missing not in missing_fields and any(token in missing_key for token in ["sample", "response", "fieldwork", "mode", "weight"]):
                missing_fields.append(missing)
        missing_issues = self._record_missing_items_for_processor_review(
            2,
            missing_fields,
            f"Full evidence extraction from deposited files for {Path(design_report).name}",
            "Some study design fields could not be retrieved after full-file extraction across deposited materials. Processor should decide whether existing evidence is sufficient or collaborator clarification is needed.",
        )

        log_msg = (
            f"Study design info confirmed from reviewed study materials for {Path(design_report).name}:\n"
            f"Study materials review: {packet_status.status}. {packet_status.reason}\n"
        )
        for key, value in extracted_info.items():
            log_msg += f"  {key}: {value}\n"
        self.active_logger.log_message(log_msg)

        message = self._format_design_brief(design_report, extracted_info, missing_fields)
        review_status = self._evidence_packet_status_line()
        if review_status and "changed" in review_status.lower():
            message = f"{message}\n\n{review_status}"
        return StepResult(
            success=True,
            message=message,
            artifacts=[design_report, str(self.working_dir / ".cses" / "evidence_packet.json"), str(self.working_dir / ".cses" / "evidence_packet.toon")],
            issues=missing_issues,
            next_action="Review study design section in log file, then proceed to Step 3"
        )

    def _step_3(self, **kwargs) -> StepResult:
        """Step 3: Create initial Variable Tracking Sheet with all CSES target variables."""
        from datetime import datetime

        data_file = self.state.data_file

        if not data_file or not Path(data_file).exists():
            return StepResult(
                success=False,
                message="Data file not found",
                issues=["Data file required for variable tracking"]
            )

        # Load data and check variables
        print("Loading data file to get variable list...")
        from src.ingest.data_loader import DataLoader
        loader = DataLoader()
        dataset_info = loader.load(Path(data_file))

        if not dataset_info:
            return StepResult(
                success=False,
                message="Could not load data file",
                issues=["Check data file format"]
            )

        n_vars = len(dataset_info.variables)
        print(f"  Found {n_vars} variables in deposited data")

        # Create the initial schema-backed tracking sheet
        print("Creating schema-backed variable tracking sheet...")
        from src.standards.tracking import WorkflowTrackingModel

        # Create tracking sheet directory
        var_list_dir = self.working_dir / "micro" / "deposited variable list"
        var_list_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        country_code = self.state.country_code or self.state.country[:3].upper()
        year = self.state.year
        date_str = datetime.now().strftime("%Y%m%d")
        tracking_filename = f"deposited variables-m6_{country_code}_{year}_{date_str}.xlsx"
        tracking_path = var_list_dir / tracking_filename

        # Create Excel file with CSES format
        try:
            tracking_model = WorkflowTrackingModel(self.state)
            tracking_model.write_excel(tracking_path)
            self.state.workflow_tracking = tracking_model.to_state_payload()
            self.state.variable_tracking_file = str(tracking_path)
            print(f"  Created: {tracking_path.name}")

            # Log the creation
            self.active_logger.log_message(
                f"Created variable tracking sheet: {tracking_filename}\n"
                f"  CSES schema target variables: {self.state.workflow_tracking.get('target_count')}\n"
                f"  Source variables available: {n_vars}"
            )

            return StepResult(
                success=True,
                message=f"Variable tracking sheet created with {self.state.workflow_tracking.get('target_count')} CSES schema variables",
                artifacts=[str(tracking_path)],
                next_action="Proceed to Step 7 to match variables and fill the tracking sheet"
            )

        except Exception as e:
            return StepResult(
                success=False,
                message=f"Failed to create tracking sheet: {e}",
                issues=[str(e)]
            )

    def _step_4(self, **kwargs) -> StepResult:
        """Step 4: Write Study Design & Weights Overview."""
        self.active_logger.log_message("Writing Study Design and Weights Overview using CSES wiki standards...")
        packet_builder = EvidencePacketBuilder(self.working_dir)
        packet_status = packet_builder.status()
        packet = packet_builder.load() if packet_status.status in {"current", "stale"} else self._ensure_evidence_packet(force=False)
        design_facts = packet.get("design_facts", {}) or {}
        sd = self.active_logger.log_data.study_design if self.active_logger.log_data else {}
        packet_to_log = {
            "sample_design": design_facts.get("sample_design"),
            "sample_size": design_facts.get("sample_size"),
            "probability_sample_status": design_facts.get("probability_sample_status"),
            "probability_sample_assessment": design_facts.get("probability_sample_evidence"),
            "sampling_evidence": design_facts.get("probability_sample_evidence"),
            "response_rate": design_facts.get("response_rate"),
            "weighting": design_facts.get("weights"),
            "collection_period": design_facts.get("fieldwork_dates"),
            "mode": design_facts.get("mode"),
            "field_lag": design_facts.get("field_lag"),
        }
        update_from_packet = {
            key: value for key, value in packet_to_log.items()
            if value and (not sd.get(key) or str(sd.get(key)).strip().upper() == "TBD")
        }
        if update_from_packet:
            self.active_logger.update_study_design_section(update_from_packet)
            sd = self.active_logger.log_data.study_design if self.active_logger.log_data else sd
        required = ["sample_design", "sample_size", "probability_sample_status", "response_rate", "weighting", "collection_period", "mode"]
        missing = [field for field in required if not sd.get(field) or str(sd.get(field)).strip().upper() == "TBD"]

        missing_issues = self._record_missing_items_for_processor_review(
            4,
            missing,
            "Study Design and Weights Overview",
            "Study Design and Weights Overview fields remain missing after checking available records. Processor should decide whether collaborator clarification is needed.",
        )

        country_code = self.state.country_code or "CNT"
        year = self.state.year or "YEAR"
        overview_dir = self.working_dir / "micro" / "Documentation"
        overview_dir.mkdir(parents=True, exist_ok=True)
        overview_path = overview_dir / f"{country_code}_{year}_study_design_weights_overview.md"
        lines = [
            f"# Overview of Study Design and Weights: {country_code}_{year}_M6",
            "",
            "Source: cses_wiki/topics/documentation-standards.md",
            "Source: cses_wiki/procedures/study-eligibility-check.md",
            "",
        ]
        labels = {
            "sample_design": "Sample Design",
            "sample_size": "Sample Size",
            "probability_sample_status": "Probability Sample Status",
            "probability_sample_assessment": "Probability Sample Assessment",
            "sampling_evidence": "Sampling Evidence",
            "response_rate": "Response Rate",
            "weighting": "Weighting Methodology",
            "collection_period": "Data Collection Period",
            "mode": "Mode of Interview",
            "field_lag": "Field Lag",
        }
        for key, label in labels.items():
            lines.append(f"## {label}")
            lines.append(str(sd.get(key) or "TBD"))
            lines.append("")
        lines.append("## Study Materials Review")
        lines.append(self._evidence_packet_status_line() or "Study materials reviewed.")
        lines.append("")
        overview_path.write_text("\n".join(lines), encoding="utf-8")
        self.active_logger.log_message(f"Study Design and Weights Overview written: {overview_path.name}")

        return StepResult(
            success=True,
            message=(
                "Study Design and Weights Overview drafted from reviewed study materials."
            ),
            artifacts=[str(overview_path), str(self.working_dir / ".cses" / "evidence_packet.json")],
            issues=missing_issues,
            next_action="Resolve missing study design fields or proceed with recorded soft-gate issues"
        )

    def _step_5(self, **kwargs) -> StepResult:
        """Step 5: Register election-results material without locking party order."""
        self.active_logger.log_message("Checking election results material for later party-order agreement...")
        summary = election_results_intake_summary(self.working_dir)
        intake_path = write_election_results_intake(self.working_dir, summary)
        self.state.election_results_intake_path = str(intake_path)

        files = summary.get("files", []) or []
        tables = summary.get("tables", []) or []
        request_dir = self.working_dir / "Election Results"
        request_dir.mkdir(parents=True, exist_ok=True)
        review_path = request_dir / f"{self.state.country_code or 'CNT'}_{self.state.year or 'YEAR'}_election_results_review.md"

        if files:
            lines = [
                "# Election Results Material Review",
                "",
                "Election-results material was found. The final party order will be agreed later, immediately before party and vote-choice matching.",
                "",
                "Files found:",
                *[f"- {Path(path).name}" for path in files],
                "",
                "Standardized election-result tables detected:",
            ]
            if tables:
                for table in tables:
                    lines.append(
                        f"- {table.get('title') or table.get('sheet_name') or 'Untitled table'} "
                        f"({table.get('election_context', 'unknown').replace('_', ' ')}, "
                        f"{len(table.get('parties', []) or [])} rows)"
                    )
                issues = []
                message = (
                    "Election-results material registered.\n\n"
                    f"Files found: {len(files)}\n"
                    f"Standardized tables detected: {len(tables)}\n\n"
                    "Party order will be proposed later, after non-party matching has clarified the affected source variables."
                )
            else:
                lines.append("- None detected in the expected table format.")
                issues = ["Election-results file found, but no standardized party table was detected."]
                message = (
                    "Election-results material registered, but the standardized table format still needs processor review."
                )
        else:
            lines = [
                "# Election Results Material Review",
                "",
                "No election-results workbook was found in the study materials.",
                "",
                "Processor review needed:",
                "- Provide the standardized election-results workbook, or confirm whether an approved public-source lookup should be used.",
            ]
            self.active_logger.add_candidate_collaborator_question(
                "Please provide the standardized election-results workbook needed for CSES party order agreement.",
                "Election results are needed for party order, vote-choice coding, macro-party consistency, and labels. Processor should first confirm whether the file exists elsewhere.",
                step_num=5,
                missing_items=["Standardized election-results workbook"],
            )
            issues = ["Standardized election-results workbook not found."]
            message = (
                "Election-results material not found.\n\n"
                "Processor review needed:\n"
                "- Provide the standardized election-results workbook, or confirm an approved public-source lookup."
            )

        review_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.active_logger.log_message("Election-results material review prepared.")
        return StepResult(
            success=True,
            message=message,
            artifacts=[str(review_path), str(intake_path)] + files,
            issues=issues,
            next_action="Continue non-party source review. Party order agreement happens before party and vote-choice matching."
        )

    def _step_6(self, **kwargs) -> StepResult:
        """Step 6: Run Frequencies on Original Data"""
        data_file = self.state.data_file

        if not data_file or not Path(data_file).exists():
            return StepResult(
                success=False,
                message="Data file not found"
            )

        try:
            from src.ingest.data_loader import DataLoader

            loader = DataLoader()
            dataset_info = loader.load(Path(data_file))

            if not dataset_info:
                return StepResult(
                    success=False,
                    message="Could not load data file"
                )

            # Generate complete frequency/source summary for every original variable.
            freq_lines = [f"# Original Data Frequencies: {self.state.country} {self.state.year}\n"]
            for var_name, var_info in dataset_info.variables.items():
                freq_lines.append(f"\n## {var_name}")
                if var_info.description:
                    freq_lines.append(f"Description: {var_info.description}")
                if var_info.value_labels:
                    labels = ", ".join(f"{k}={v}" for k, v in var_info.value_labels.items())
                    freq_lines.append(f"Labels: {labels}")
                if var_info.sample_values:
                    freq_lines.append(f"Sample values: {var_info.sample_values}")

            # Save frequencies
            freq_dir = self.working_dir / "micro" / "frequencies"
            freq_dir.mkdir(parents=True, exist_ok=True)
            country_code = self.state.country_code or "CNT"
            year = self.state.year or "YEAR"
            freq_file = freq_dir / f"cses-m6_org-freq_{country_code}_{year}.txt"
            freq_file.write_text("\n".join(freq_lines), encoding="utf-8")
            self.active_logger.log_message(
                f"Original data frequencies generated for {len(dataset_info.variables)} variables: {freq_file.name}"
            )

            return StepResult(
                success=True,
                message=f"Generated frequencies for {len(dataset_info.variables)} variables",
                artifacts=[str(freq_file)],
                next_action="Review frequencies and proceed to variable processing"
            )

        except Exception as e:
            return StepResult(
                success=False,
                message=f"Failed to run frequencies: {e}",
                issues=[str(e)]
            )

    def _step_7(self, **kwargs) -> StepResult:
        """Step 7: Process Variables - Variable Matching"""
        result = self._step_7a(**kwargs)
        if result.success and not PartyOrderingRulesEngine().is_approved(self.working_dir):
            result.success = False
            if "Party Order Agreement requires micro processor and macro coder approval." not in result.issues:
                result.issues.append("Party Order Agreement requires micro processor and macro coder approval.")
            result.next_action = (
                "Review the proposed party order with the macro coder, record approval, "
                "then proceed to rerun party-aware matching."
            )
        return result

    def _step_7_legacy(self, **kwargs) -> StepResult:
        """Legacy variable matching path retained for reference."""
        # This is the main variable matching step
        data_file = self.state.data_file

        if not data_file:
            return StepResult(
                success=False,
                message="Data file not found"
            )

        doc_files = []
        if self.state.questionnaire_files:
            doc_files.extend(self.state.questionnaire_files)
        if self.state.codebook_file:
            doc_files.append(self.state.codebook_file)

        if not doc_files:
            return StepResult(
                success=False,
                message="No documentation files found for matching"
            )

        self.active_logger.log_message("Starting variable matching...")

        # Run the dual-model matching
        try:
            from src.agent import run_harmonization
            from datetime import datetime

            result = run_harmonization(
                data_file=Path(data_file),
                doc_files=[Path(f) for f in doc_files],
                country=self.state.country,
                year=self.state.year,
                validate=True,
                output_dir=self.working_dir / ".cses"
            )

            # Store mappings in state
            self.state.mappings = [v.to_dict() for v in result.validations]
            self.state.save()

            artifacts = [str(self.working_dir / ".cses" / "mappings")]

            # Log matching results
            self.active_logger.log_message(
                f"Variable matching completed: {result.matched_count}/{result.total_targets} matched, "
                f"{result.agree_count} agreements, {result.disagree_count} disagreements"
            )

            # Log disagreements as potential questions
            for v in result.validations:
                if v.verdict.name == "DISAGREE":
                    self.active_logger.add_candidate_collaborator_question(
                        f"Variable {v.proposal.target_variable} mapping needs clarification: "
                        f"matched to '{v.proposal.source_variable}' but validation disagreed",
                        f"Reasoning: {v.reasoning}. Processor should inspect available evidence before deciding whether collaborator contact is needed.",
                        step_num=7,
                        missing_items=[v.proposal.target_variable],
                    )

            # Auto-generate output files
            print("Generating CSES output files...")
            self.active_logger.log_message("Generating CSES output files...")
            country_code = self.state.country_code or "CNT"
            year = self.state.year or "YEAR"
            date_str = datetime.now().strftime("%Y%m%d")

            # Ensure micro folder structure exists
            micro_dir = self.working_dir / "micro"
            micro_dir.mkdir(exist_ok=True)

            # Create deposited variable list subfolder
            var_list_dir = micro_dir / "deposited variable list"
            var_list_dir.mkdir(exist_ok=True)

            # Generate tracking sheet in micro/deposited variable list/
            tracking_path = var_list_dir / f"deposited variables-m6_{country_code}_{year}_{date_str}.xlsx"
            from src.standards.tracking import WorkflowTrackingModel
            from src.codegen import TrackingSheetReader, StataCodeGenerator

            tracking_model = WorkflowTrackingModel(self.state)
            tracking_model.write_excel(tracking_path)
            self.state.workflow_tracking = tracking_model.to_state_payload()
            self.state.variable_tracking_file = str(tracking_path)
            artifacts.append(str(tracking_path))
            print(f"  Created: deposited variable list/{tracking_path.name}")

            # Generate .do file in micro/
            do_path = micro_dir / f"cses-m6_micro_{country_code}_{year}_{date_str}.do"
            tracking_data = TrackingSheetReader().read(tracking_path)
            do_result = StataCodeGenerator().generate(
                tracking_sheet=tracking_data,
                output_path=do_path,
                country_name=self.state.country or "COUNTRY",
                author="CSES Agent",
                data_file_path=str(data_file)
            )
            if do_result.success:
                artifacts.append(str(do_path))
                print(f"  Created: {do_path.name}")

            # VERIFY output files exist and have content
            issues = []

            # Check tracking sheet
            if not tracking_path.exists():
                issues.append("Tracking sheet was not created")
            else:
                try:
                    import openpyxl
                    wb = openpyxl.load_workbook(tracking_path)
                    ws = wb.active
                    if ws.max_row < 5:
                        issues.append("Tracking sheet is empty - no variable mappings recorded")
                except Exception as e:
                    issues.append(f"Could not verify tracking sheet: {e}")

            # Check .do file
            if not do_path.exists():
                issues.append("Stata .do file was not created")
            else:
                do_content = do_path.read_text()
                if len(do_content) < 100:
                    issues.append("Stata .do file is empty or has no recode syntax")
                elif "recode" not in do_content.lower() and "gen" not in do_content.lower():
                    issues.append("Stata .do file has no recode or generate commands")

            if issues:
                self.active_logger.log_message(f"Step 7 issues: {', '.join(issues)}")
                return StepResult(
                    success=False,
                    message="Variable matching incomplete - output files missing or empty",
                    issues=issues,
                    artifacts=artifacts,
                    next_action="Re-run variable matching or check for errors"
                )

            # Log successful completion
            self.active_logger.log_message(
                f"Step 7 completed successfully.\n"
                f"  Tracking sheet: {tracking_path.name}\n"
                f"  Stata .do file: {do_path.name}\n"
                f"  Variables matched: {result.matched_count}/{result.total_targets}"
            )

            return StepResult(
                success=True,
                message=f"Matched {result.matched_count}/{result.total_targets} variables. "
                        f"Agreements: {result.agree_count}, Disagreements: {result.disagree_count}. "
                        f"Output files generated in micro/ folder.",
                artifacts=artifacts,
                next_action="Review mappings and run Stata debugging if needed"
            )

        except Exception as e:
            logger.error(f"Variable matching failed: {e}")
            return StepResult(
                success=False,
                message=f"Variable matching failed: {e}",
                issues=[str(e)]
            )

    def _step_7a(self, **kwargs) -> StepResult:
        """Step 7a: AI Fill Tracking Sheet

        Uses ensemble matcher to propose variable mappings and fills the
        extended tracking sheet. Human reviews in Excel before code generation.
        """
        from datetime import datetime

        data_file = self.state.data_file
        if not data_file:
            return StepResult(
                success=False,
                message="Data file not found"
            )

        doc_files = []
        if self.state.questionnaire_files:
            doc_files.extend(self.state.questionnaire_files)
        if self.state.codebook_file:
            doc_files.append(self.state.codebook_file)

        self.active_logger.log_message("Starting AI variable matching (Step 7a)...")

        try:
            from src.ingest.data_loader import DataLoader
            from src.ingest.doc_parser import DocumentParser
            from src.matching.llm_matcher import CSES_TARGET_VARIABLES
            from src.matching.evidence import MatchingEvidenceBuilder, RemoteItemSimilarityService, build_similarity_pairs
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment

            # Load source data
            print("Loading data file...")
            loader = DataLoader()
            dataset_info = loader.load(Path(data_file))
            if not dataset_info:
                return StepResult(success=False, message="Could not load data file")

            source_contexts = [
                {
                    'name': v.name,
                    'description': v.description or '',
                    'value_labels': v.value_labels or {},
                    'sample_values': v.sample_values or []
                }
                for v in dataset_info.variables.values()
            ]
            print(f"  {len(source_contexts)} source variables found")

            print("Reviewing election results for later party-order agreement...")
            party_tables = ElectionResultsWorkbookParser().parse_directory(self.working_dir)
            party_order_engine = PartyOrderingRulesEngine()
            party_order_proposal = party_order_engine.propose(
                party_tables,
                source_variables=[item["name"] for item in source_contexts],
                context_hint=infer_election_context_from_macro_material(self.working_dir),
            )
            party_review_path, party_decision_path = party_order_engine.write_review(
                self.working_dir,
                party_order_proposal,
            )
            party_order_approved = party_order_engine.is_approved(self.working_dir)
            party_summary = party_agreement_summary(party_order_proposal)
            self.state.party_order_review_path = str(party_review_path)
            self.state.party_order_decision_path = str(party_decision_path)
            self.state.party_order_status = {
                "status": "approved" if party_order_approved else party_order_proposal.status,
                "party_count": party_summary.get("party_count", 0),
                "warnings": party_summary.get("warnings", 0),
                "micro_variables_affected": party_summary.get("micro_variables_affected", 0),
                "macro_variables_affected": party_summary.get("macro_variables_affected", 0),
            }
            print("Reviewing macro party metadata...")
            party_metadata_builder = PartyMetadataReviewBuilder(self.working_dir)
            party_metadata_review = party_metadata_builder.build()
            party_metadata_review_path, party_metadata_decision_path = party_metadata_builder.write_review(
                party_metadata_review
            )
            party_metadata_approved = party_metadata_builder.is_approved()
            print("Preparing macro context review...")
            macro_context_builder = MacroContextBuilder(self.working_dir, self.state)
            macro_context_review = macro_context_builder.build(
                party_order_proposal=party_order_proposal,
                party_order_approved=party_order_approved,
                party_metadata_review=party_metadata_review,
                party_metadata_approved=party_metadata_approved,
            )
            macro_context_review_path, macro_context_decision_path = macro_context_builder.write_review(
                macro_context_review
            )
            macro_context_approved = macro_context_builder.is_approved()
            self.state.macro_context_review_path = str(macro_context_review_path)
            self.state.macro_context_decision_path = str(macro_context_decision_path)
            self.state.macro_context_status = {
                "status": "approved" if macro_context_approved else macro_context_review.status,
                "missing": len(macro_context_review.missing_items),
                "conflicts": len(macro_context_review.conflicts),
                "external_provider": macro_context_review.external_provider.get("status", ""),
            }

            # Load documents
            print("Loading documentation...")
            doc_parser = DocumentParser()
            doc_infos = []
            questionnaire_text = ""
            codebook_text = ""
            for doc_path in doc_files:
                doc_info = doc_parser.parse(Path(doc_path))
                if doc_info and doc_info.full_text:
                    doc_infos.append(doc_info)
                    if "questionnaire" in doc_path.lower():
                        questionnaire_text += doc_info.full_text + "\n"
                    else:
                        codebook_text += doc_info.full_text + "\n"
            print(f"  Questionnaire: {len(questionnaire_text):,} chars")
            print(f"  Codebook: {len(codebook_text):,} chars")

            evidence_builder = MatchingEvidenceBuilder(self.working_dir)
            matching_evidence = evidence_builder.load()
            from src.standards.questionnaire_registry import Module6QuestionnaireRegistry
            registry_generated_at = Module6QuestionnaireRegistry().payload.get("generated_at", "")
            if (
                not matching_evidence
                or matching_evidence.get("schema_version") != 2
                or matching_evidence.get("registry_generated_at") != registry_generated_at
                or not any(
                    item.get("canonical_item_ids")
                    for item in matching_evidence.get("target_variable_profiles", []) or []
                    if isinstance(item, dict)
                )
            ):
                matching_evidence = evidence_builder.build_from_loaded(dataset_info, doc_infos)

            similarity_pairs = build_similarity_pairs(matching_evidence)
            remote_scores = RemoteItemSimilarityService(self.working_dir).score_pairs(similarity_pairs)
            from src.standards.administrative import (
                AdministrativeFactBuilder,
                AdministrativeVariablePlanner,
                administrative_plan_summary,
            )

            try:
                evidence_packet = EvidencePacketBuilder(self.working_dir).load()
            except Exception:
                evidence_packet = {}
            admin_facts = AdministrativeFactBuilder(self.working_dir).build(
                self.state,
                dataset_info=dataset_info,
                evidence_packet=evidence_packet,
            )
            admin_planner = AdministrativeVariablePlanner()
            administrative_plans = admin_planner.build_plans(
                admin_facts,
                source_variables={item["name"] for item in source_contexts if item.get("name")},
            )
            administrative_plan_lookup = {plan.target_variable: plan for plan in administrative_plans}
            administrative_plans_path = admin_planner.write_artifacts(
                self.working_dir,
                admin_facts,
                administrative_plans,
            )
            administrative_summary = administrative_plan_summary(administrative_plans)
            from src.matching.demographics import (
                DemographicRecodingDecisionStore,
                DemographicRecodingDossierBuilder,
                DemographicRecodingAssessmentEngine,
                demographic_assessment_summary,
            )

            demographic_engine = DemographicRecodingAssessmentEngine()
            demographic_assessments = demographic_engine.assess(
                matching_evidence=matching_evidence,
                source_contexts=source_contexts,
            )
            demographic_assessment_lookup = {
                item.target_variable: item for item in demographic_assessments
            }
            demographic_assessments_path = demographic_engine.write_artifacts(
                self.working_dir,
                demographic_assessments,
            )
            demographic_dossiers = DemographicRecodingDossierBuilder().build(
                demographic_assessments,
                matching_evidence=matching_evidence,
            )
            demographic_dossiers_path = DemographicRecodingDossierBuilder().write(
                self.working_dir,
                demographic_dossiers,
            )
            demographic_decisions_path = DemographicRecodingDecisionStore(self.working_dir).write_from_assessments(
                demographic_assessments,
                approved=False,
            )
            demographic_summary = demographic_assessment_summary(demographic_assessments)

            # Build reviewable matching decisions from deterministic evidence first.
            # A monolithic model call can block the workflow; processor review needs
            # a complete, inspectable sheet even when model scoring is unavailable.
            print()
            print(f"Preparing source-match proposals for {len(CSES_TARGET_VARIABLES)} core CSES targets...")
            match_result = type("MatchResult", (), {"proposals": []})()
            from src.matching.decision_engine import MatchingDecisionEngine, decision_summary, matching_category_summary

            study_kb_payload = {}
            try:
                study_kb_payload = StudyKnowledgeBase(self.working_dir).payload
            except Exception:
                study_kb_payload = {}
            decision_engine = MatchingDecisionEngine()
            decisions = decision_engine.decide(
                source_contexts=source_contexts,
                llm_proposals=match_result.proposals,
                study_kb=study_kb_payload,
                matching_evidence=matching_evidence,
                remote_similarity_scores=remote_scores,
                administrative_plans=administrative_plans,
                demographic_assessments=demographic_assessments,
                party_order_approved=party_order_approved,
                party_order_summary=party_summary,
            )
            candidates_path, decisions_path = decision_engine.write_artifacts(self.working_dir, decisions)
            status_counts = decision_summary(decisions)
            category_counts = matching_category_summary(decisions, matching_evidence)
            self.state.mappings = [decision.to_dict() for decision in decisions]
            self.state.matching_decisions_path = str(decisions_path)
            self.state.matching_coverage = {
                "target_count": len(decisions),
                "status_counts": status_counts,
                "category_counts": category_counts,
                "proposed_match_count": status_counts.get("proposed_match", 0),
                "blocked_count": status_counts.get("blocked_for_processor_review", 0),
                "external_input_required_count": status_counts.get("external_input_required", 0),
                "derived_metadata_count": status_counts.get("derived_metadata", 0),
                "administrative_generated_count": status_counts.get("generated_from_administrative_information", 0),
                "party_order_status": self.state.party_order_status,
                "administrative_summary": administrative_summary,
                "demographic_summary": demographic_summary,
                "party_metadata_status": {
                    "status": "approved" if party_metadata_approved else party_metadata_review.status,
                    "values_found": len(party_metadata_review.values),
                    "missing": len(party_metadata_review.missing_variables),
                    "warnings": len(party_metadata_review.warnings),
                },
                "macro_context_status": self.state.macro_context_status,
            }
            self.state.save()

            print()

            # Recoding strategy generation is deferred until mappings have been
            # reviewed. Running it for hundreds of unverified candidates makes
            # Step 7a slow and expensive without improving human review.
            print("Preparing tracking sheet; recoding strategy generation is deferred until mappings are reviewed.")
            strategies = {}

            # Create extended tracking sheet
            print()
            print("Creating tracking sheet...")
            country_code = self.state.country_code or "CNT"
            year = self.state.year or "YEAR"
            date_str = datetime.now().strftime("%Y%m%d")

            var_list_dir = self.working_dir / "micro" / "deposited variable list"
            var_list_dir.mkdir(parents=True, exist_ok=True)
            tracking_path = var_list_dir / f"deposited variables-m6_{country_code}_{year}_{date_str}.xlsx"

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "deposited variables"

            # Extended headers
            headers = [
                "CSES_DESC", "CSES_VAR", "SOURCE_VAR", "SOURCE_DESC",
                "SOURCE_VALUES", "TRANSFORM", "RECODE_MAP", "MISSING_MAP",
                "CONFIDENCE", "VERIFIED", "NOTES", "CANONICAL_ITEM",
                "ITEM_TYPE", "TOP_CANDIDATES", "EVIDENCE", "WARNINGS"
            ]
            header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF")

            for col, header in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col, value=header)
                cell.fill = header_fill
                cell.font = header_font

            # Fill data rows
            target_profile_lookup = {
                item.get("name"): item
                for item in matching_evidence.get("target_variable_profiles", []) or []
                if isinstance(item, dict)
            }
            row = 2
            for decision in decisions:
                target = decision.target_variable
                source = decision.source_variable
                desc = decision.description

                # Get source info
                source_info = next((s for s in source_contexts if s['name'] == source), {})
                source_desc = source_info.get('description', '')
                source_labels = source_info.get('value_labels', {})
                labels_str = "; ".join(f"{k}={v}" for k, v in list(source_labels.items())[:10])

                # Get strategy info
                strategy = strategies.get(target)
                transform_type = strategy.transformation_type if strategy else "direct"
                recode_map = ""
                missing_map = ""
                if strategy and strategy.recode_rules:
                    recode_map = "; ".join(f"{r.from_value}={r.to_value}" for r in strategy.recode_rules)
                missing_rules = getattr(strategy, "missing_rules", []) if strategy else []
                if missing_rules:
                    missing_map = "; ".join(f"{r.from_value}={r.to_value}" for r in missing_rules)

                # Confidence level
                confidence = decision.confidence

                # Highlight low confidence rows
                if source == "NOT_FOUND":
                    transform_type = "not_found"
                    confidence = "low"
                elif decision.status == "generated_from_administrative_information":
                    transform_type = "administrative_information"
                elif decision.status == "demographic_recoding_assessment":
                    assessment = demographic_assessment_lookup.get(target)
                    if assessment:
                        transform_type = assessment.recoding_plan_type
                        if assessment.value_map:
                            recode_map = "; ".join(f"{k}={v}" for k, v in assessment.value_map.items())
                        if assessment.missing_map:
                            missing_map = "; ".join(f"{k}={v}" for k, v in assessment.missing_map.items())
                elif decision.status == "awaiting_party_order_agreement":
                    transform_type = "party_order_agreement_required"
                elif decision.status == "generated_from_party_order":
                    transform_type = "party_order_information"
                elif decision.status == "generated_from_party_context":
                    transform_type = "party_context_derivative"
                elif decision.status in {"derived_metadata", "external_input_required", "blocked_for_processor_review"}:
                    transform_type = "not_found" if not source or source in {"DERIVED_METADATA", "EXTERNAL_INPUT_REQUIRED"} else transform_type

                ws.cell(row=row, column=1, value=desc)
                ws.cell(row=row, column=2, value=target)
                ws.cell(row=row, column=3, value=source)
                ws.cell(row=row, column=4, value=source_desc)
                ws.cell(row=row, column=5, value=labels_str)
                ws.cell(row=row, column=6, value=transform_type)
                ws.cell(row=row, column=7, value=recode_map)
                ws.cell(row=row, column=8, value=missing_map)
                ws.cell(row=row, column=9, value=confidence)
                ws.cell(row=row, column=10, value="FALSE")
                notes = " | ".join(
                    [decision.status, decision.notes]
                    + decision.conflict_flags[:3]
                    + decision.evidence[:2]
                )
                admin_plan = administrative_plan_lookup.get(target)
                if admin_plan:
                    extra = [
                        f"Administrative category: {admin_plan.generation_category}",
                        f"Administrative value: {admin_plan.value}" if admin_plan.value not in {"", None} else "",
                        f"Stata rule: {admin_plan.stata_rule}",
                    ]
                    notes = " | ".join([notes] + [item for item in extra if item])
                demographic_assessment = demographic_assessment_lookup.get(target)
                if demographic_assessment:
                    extra = [
                        f"Demographic concept: {demographic_assessment.concept}",
                        f"Source format: {demographic_assessment.source_format}",
                        f"Recoding action: {demographic_assessment.recoding_action}",
                    ]
                    notes = " | ".join([notes] + [item for item in extra if item])
                ws.cell(row=row, column=11, value=notes[:500])
                target_profile = target_profile_lookup.get(target, {})
                canonical_item = ", ".join(target_profile.get("canonical_item_ids", []) or target_profile.get("expected_item_ids", [])[:3])
                item_type = target_profile.get("canonical_item_type", "")
                candidate_text = "; ".join(
                    f"{candidate.source_variable} ({candidate.score:.2f})"
                    for candidate in decision.candidates[:3]
                )
                evidence_text = " | ".join(decision.evidence[:4])
                warning_text = "; ".join(decision.conflict_flags[:5])
                ws.cell(row=row, column=12, value=canonical_item[:250])
                ws.cell(row=row, column=13, value=item_type[:100])
                ws.cell(row=row, column=14, value=candidate_text[:500])
                ws.cell(row=row, column=15, value=evidence_text[:800])
                ws.cell(row=row, column=16, value=warning_text[:500])

                # Color-code by confidence
                if confidence == "low":
                    for col in range(1, 17):
                        ws.cell(row=row, column=col).fill = PatternFill(
                            start_color="FFC7CE", end_color="FFC7CE", fill_type="solid"
                        )
                elif confidence == "medium":
                    for col in range(1, 17):
                        ws.cell(row=row, column=col).fill = PatternFill(
                            start_color="FFEB9C", end_color="FFEB9C", fill_type="solid"
                        )

                row += 1

            # Set column widths
            ws.column_dimensions['A'].width = 40
            ws.column_dimensions['B'].width = 12
            ws.column_dimensions['C'].width = 15
            ws.column_dimensions['D'].width = 30
            ws.column_dimensions['E'].width = 40
            ws.column_dimensions['F'].width = 12
            ws.column_dimensions['G'].width = 30
            ws.column_dimensions['H'].width = 20
            ws.column_dimensions['I'].width = 12
            ws.column_dimensions['J'].width = 10
            ws.column_dimensions['K'].width = 40
            ws.column_dimensions['L'].width = 18
            ws.column_dimensions['M'].width = 24
            ws.column_dimensions['N'].width = 32
            ws.column_dimensions['O'].width = 60
            ws.column_dimensions['P'].width = 36

            wb.save(tracking_path)

            # Print summary
            print()
            print("Results:")
            total = len(decisions)
            high = len([d for d in decisions if d.confidence == "high"])
            medium = len([d for d in decisions if d.confidence == "medium"])
            low = len([d for d in decisions if d.confidence == "low"])
            not_found = len([d for d in decisions if d.status in {"blocked_for_processor_review", "external_input_required"}])
            valid_matched = len([
                d for d in decisions
                if d.status == "proposed_match" and d.source_variable not in {"", "NOT_FOUND", "ERROR", "NO_CONSENSUS"}
            ])
            error_count = len([d for d in decisions if d.source_variable == "ERROR"])
            print(f"  High confidence:   {high:3d} ({100*high//total if total else 0}%)")
            print(f"  Medium confidence: {medium:3d} ({100*medium//total if total else 0}%)")
            print(f"  Low confidence:    {low:3d} ({100*low//total if total else 0}%)")
            if not_found:
                print(f"  Not found:         {not_found:3d}")
            if error_count:
                print(f"  Errors:            {error_count:3d}")
            print("  By workflow category:")
            core = category_counts.get("core_questionnaire_items", {})
            demo = category_counts.get("demographic_items", {})
            admin = category_counts.get("administrative_metadata", {})
            party = category_counts.get("party_election_items", {})
            district = category_counts.get("district_items", {})
            print(f"    Core questionnaire items matched: {core.get('matched', 0)}/{core.get('total', 0)}")
            print(
                f"    Demographic recoding assessments prepared: {demo.get('matched', 0)}/{demo.get('total', 0)}"
                f" ({demographic_summary.get('needs_review', 0)} need processor review)"
            )
            print(
                f"    Administrative variables prepared: {admin.get('prepared', 0)}/{admin.get('total', 0)}"
                f" ({admin.get('needs_review', 0)} need processor review)"
            )
            print(f"    Party/election variables awaiting party ordering: {party.get('awaiting_party_ordering', 0)}/{party.get('total', 0)}")
            print(f"    District variables awaiting district input: {district.get('awaiting_district_input', 0)}/{district.get('total', 0)}")
            print(
                "    Party Order Agreement: "
                + ("approved" if party_order_approved else f"{party_order_proposal.status.replace('_', ' ')}")
            )
            print()
            print(f"Output: {tracking_path.relative_to(self.working_dir)}")

            # Log results
            self.active_logger.log_message(
                f"Step 7a completed:\n"
                f"  Tracking sheet: {tracking_path.name}\n"
                f"  High confidence: {high}\n"
                f"  Medium confidence: {medium}\n"
                f"  Low confidence: {low}\n"
                f"  Not found: {not_found}\n"
                f"  Errors: {error_count}\n"
                f"  Matching decisions: {decisions_path.name}"
            )

            if valid_matched == 0 or error_count > total // 2:
                return StepResult(
                    success=False,
                    message=f"Variable matching produced only {valid_matched}/{total} usable source matches",
                    artifacts=[
                        str(tracking_path),
                        str(candidates_path),
                        str(decisions_path),
                        str(administrative_plans_path),
                        str(demographic_assessments_path),
                        str(demographic_dossiers_path),
                        str(demographic_decisions_path),
                        str(party_review_path),
                        str(party_decision_path),
                        str(party_metadata_review_path),
                        str(party_metadata_decision_path),
                        str(macro_context_review_path),
                        str(macro_context_decision_path),
                    ],
                    issues=[
                        f"Matching errors: {error_count}",
                        f"Not found: {not_found}",
                        "Review model output and rerun Step 7a after fixing matching."
                    ],
                    next_action="Do not generate Stata syntax until matching produces usable reviewed mappings."
                )

            unresolved_count = len([
                d for d in decisions
                if d.status != "proposed_match"
            ])

            return StepResult(
                success=True,
                message=(
                    f"Proposed source matches for {valid_matched}/{total} CSES variables; "
                    f"{unresolved_count} remain unresolved.\n\n"
                    + party_order_message(party_order_proposal)
                    + "\n\n"
                    + party_metadata_message(party_metadata_review)
                    + "\n\n"
                    + macro_context_message(macro_context_review)
                ),
                artifacts=[
                    str(tracking_path),
                    str(candidates_path),
                    str(decisions_path),
                    str(administrative_plans_path),
                    str(demographic_assessments_path),
                    str(demographic_dossiers_path),
                    str(demographic_decisions_path),
                    str(party_review_path),
                    str(party_decision_path),
                    str(party_metadata_review_path),
                    str(party_metadata_decision_path),
                    str(macro_context_review_path),
                    str(macro_context_decision_path),
                ],
                next_action=(
                    "Review non-party matches, the Party Order Agreement, and the Macro Context Review. "
                    "Party-context coding stays paused until the micro processor and macro coder approve the shared context."
                )
            )

        except Exception as e:
            logger.error(f"Step 7a failed: {e}")
            return StepResult(
                success=False,
                message=f"AI matching failed: {e}",
                issues=[str(e)]
            )

    def _step_7c(self, tracking_sheet: str = None, **kwargs) -> StepResult:
        """Step 7c: Generate Stata Code from Tracking Sheet

        Reads the verified tracking sheet and generates deterministic Stata code.
        No AI is used in this step - pure template-based generation.
        """
        from src.codegen import TrackingSheetReader, StataCodeGenerator
        from datetime import datetime

        print("Step 7c: Generating Stata code from tracking sheet...")
        self.active_logger.log_message("Starting deterministic code generation (Step 7c)...")
        exclude_district = bool(kwargs.get("exclude_district") or kwargs.get("non_district_benchmark"))
        if exclude_district:
            self.state.readiness_mode = "release_ready_except_district"
            self.state.district_excluded_by_processor = True
            self.state.record_processor_decision(
                7,
                "District data not included in this run",
                "Non-district benchmark/finalization path selected by processor.",
            )

        # Find tracking sheet
        if tracking_sheet:
            sheet_path = Path(tracking_sheet)
        else:
            # Look for most recent tracking sheet
            var_list_dir = self.working_dir / "micro" / "deposited variable list"
            if not var_list_dir.exists():
                return StepResult(
                    success=False,
                    message="No tracking sheet directory found",
                    issues=["Run Step 7a first to create tracking sheet"]
                )

            sheets = list(var_list_dir.glob("deposited variables-m6_*.xlsx"))
            if not sheets:
                return StepResult(
                    success=False,
                    message="No tracking sheet found",
                    issues=["Run Step 7a first to create tracking sheet"]
                )
            sheet_path = max(sheets, key=lambda p: p.stat().st_mtime)

        print(f"  Reading: {sheet_path.name}")
        if not PartyOrderingRulesEngine().is_approved(self.working_dir):
            return StepResult(
                success=False,
                message=(
                    "Party Order Agreement is not approved yet.\n\n"
                    "Stata syntax for party, vote-choice, leader, and macro-party variables needs one locked party order "
                    "approved by both the micro processor and macro coder."
                ),
                issues=[
                    "Micro processor and macro coder approval are required before final party-related Stata generation.",
                ],
                next_action="Review the Party Order Agreement from Step 7 and record both approvals before rerunning Step 7c."
            )

        try:
            # Read tracking sheet
            reader = TrackingSheetReader()
            tracking_data = reader.read(sheet_path)
            from src.codegen.recoding_plan import RecodingPlanBuilder, StataSyntaxPlanner, PlanDrivenStataSyntaxGenerator
            from src.district_data import load_district_merge_plan

            plan_builder = RecodingPlanBuilder(self.state)
            recoding_plans = plan_builder.build(tracking_data)
            recoding_path = plan_builder.write(self.working_dir, recoding_plans)
            plan_counts = {}
            for plan in recoding_plans:
                plan_counts[plan.readiness_status] = plan_counts.get(plan.readiness_status, 0) + 1
            approved_count = sum(1 for plan in recoding_plans if plan.approved)
            non_district_plans = [plan for plan in recoding_plans if plan.dependency_class != "district_input"]
            non_district_approved = sum(1 for plan in non_district_plans if plan.approved)
            self.state.recoding_plans_path = str(recoding_path)
            self.state.recoding_coverage = {
                "target_count": len(recoding_plans),
                "approved_count": approved_count,
                "non_district_target_count": len(non_district_plans),
                "non_district_approved_count": non_district_approved,
                "readiness_counts": plan_counts,
            }
            self.state.approval_status = {
                "requires_processor_approval": len(recoding_plans) - approved_count,
                "approved": approved_count,
            }
            self.state.save()

            # Check for unverified mappings
            from src.standards.schema import SchemaRegistry
            schema_registry = SchemaRegistry()
            needs_review = [
                item for item in tracking_data.get_needs_review()
                if not (
                    exclude_district
                    and schema_registry.by_name(item.cses_var)
                    and schema_registry.by_name(item.cses_var).dependency_class == "district_input"
                )
            ]
            if needs_review:
                print(f"  Warning: {len(needs_review)} mappings need review")
                for m in needs_review[:5]:
                    print(f"    - {m.cses_var}: {m.source_var} ({m.confidence})")
                self._record_missing_items_for_processor_review(
                    7,
                    [f"{m.cses_var}: {m.source_var} ({m.confidence})" for m in needs_review[:25]],
                    "Tracking sheet human verification",
                    "Some mappings are unverified. Processor should review data/documentation evidence before deciding whether any collaborator clarification is needed.",
                )
                return StepResult(
                    success=False,
                    message="Tracking sheet contains unverified mappings",
                    artifacts=[str(sheet_path), str(recoding_path)],
                    issues=[
                        f"{len(needs_review)} mapping(s) require human verification before code generation",
                        f"Recoding plan approval coverage: {approved_count}/{len(recoding_plans)}",
                    ],
                    next_action="Open the tracking sheet, review mappings, set VERIFIED=TRUE for approved rows, then rerun Step 7c"
                )

            # Generate code
            print("  Generating Stata code...")
            planner = StataSyntaxPlanner()
            unresolved = planner.unresolved_required(recoding_plans, exclude_district=exclude_district)
            if unresolved:
                return StepResult(
                    success=False,
                    message="Recoding plans are not ready for final Stata generation",
                    artifacts=[str(sheet_path), str(recoding_path)],
                    issues=[
                        f"{len(unresolved)} recoding plan(s) require processor decision or external input",
                        *[f"{plan.target_variable}: {', '.join(plan.issues[:2]) or plan.readiness_status}" for plan in unresolved[:20]],
                    ],
                    next_action="Resolve recoding plan issues, external inputs, or recorded processor decisions before rerunning Step 7c"
                )

            country_code = tracking_data.country_code or self.state.country_code or "CNT"
            year = tracking_data.year or self.state.year or "YEAR"
            output_path = self.working_dir / "micro" / f"cses-m6_micro_{country_code}_{year}.do"

            generated_path = PlanDrivenStataSyntaxGenerator().generate(
                plans=recoding_plans,
                output_path=output_path,
                country_code=country_code,
                year=year,
                country_name=self.state.country or country_code,
                data_file_path=self.state.data_file or "",
                draft=False,
                exclude_district=exclude_district,
                district_merge_plan=load_district_merge_plan(self.working_dir),
            )
            result = type("GenerationResult", (), {
                "success": generated_path.exists(),
                "variables_generated": len(recoding_plans),
                "variables_skipped": 0,
                "warnings": [],
                "errors": [],
            })()

            if result.success:
                from src.standards.validators import validate_stata_syntax_text

                syntax_check = validate_stata_syntax_text(output_path.read_text(encoding="utf-8", errors="replace"))
                print(f"  Created: {output_path.name}")
                print(f"  Variables generated: {result.variables_generated}")
                print(f"  Variables skipped: {result.variables_skipped}")

                self.active_logger.log_message(
                    f"Step 7c completed:\n"
                    f"  .do file: {output_path.name}\n"
                    f"  Variables generated: {result.variables_generated}\n"
                    f"  Variables skipped: {result.variables_skipped}"
                )

                return StepResult(
                    success=True,
                    message=f"Generated Stata code for {result.variables_generated} variables",
                    artifacts=[str(output_path), str(recoding_path)],
                    issues=(result.warnings[:5] if result.warnings else []) + [f"Syntax standard missing: {issue}" for issue in syntax_check.issues],
                    next_action="Run Step 8 to test the .do file in Stata"
                )
            else:
                return StepResult(
                    success=False,
                    message="Code generation failed",
                    issues=result.errors
                )

        except Exception as e:
            logger.error(f"Step 7c failed: {e}")
            return StepResult(
                success=False,
                message=f"Code generation failed: {e}",
                issues=[str(e)]
            )

    def _step_8(self, do_file: str = None, **kwargs) -> StepResult:
        """Step 8: Debug Stata .do File

        This step runs the generated .do file through Stata and checks for errors.
        If errors are found, the agent can use the debugging tools to fix them.
        """
        from src.stata_execution import StataExecutionVerifier

        self.active_logger.log_message("Starting Stata debugging...")

        # Find .do file
        if do_file:
            do_path = Path(do_file)
        else:
            # Look for most recent .do file in micro folder
            micro_dir = self.working_dir / "micro"
            do_files = list(micro_dir.glob("cses-m6_micro_*.do"))
            if not do_files:
                generation = self._step_7c(**kwargs)
                if not generation.success:
                    return StepResult(
                        success=False,
                        message="Stata syntax is not ready for execution",
                        artifacts=generation.artifacts,
                        issues=generation.issues,
                        next_action=generation.next_action or "Resolve recoding plan issues, then rerun Stata execution"
                    )
                do_files = list(micro_dir.glob("cses-m6_micro_*.do"))
            if not do_files:
                return StepResult(
                    success=False,
                    message="No .do file found in micro/ folder",
                    issues=["Generate final CSES Stata syntax before running Stata"]
                )
            do_path = max(do_files, key=lambda p: p.stat().st_mtime)

        if not do_path.exists():
            return StepResult(
                success=False,
                message=f".do file not found: {do_path}"
            )

        self.active_logger.log_message(f"Running Stata on: {do_path.name}")

        execution_path = self.working_dir / ".cses" / "stata_execution.json"
        result = StataExecutionVerifier(self.working_dir).run(do_path, stata_path=kwargs.get("stata_path"))
        self.state.stata_execution_status = {
            "success": bool(result.success),
            "do_file": str(do_path),
            "execution_path": str(execution_path),
            "error": result.error or "",
            "output_dataset": result.output_dataset,
            "log_path": result.log_path,
        }
        self.state.save()

        if result.success:
            self.active_logger.log_message("Stata executed successfully - no errors found")
            return StepResult(
                success=True,
                message="Stata executed .do file successfully",
                artifacts=[result.log_path or str(do_path), result.output_dataset, str(execution_path)],
                next_action="Review log file and proceed to quality checks"
            )
        else:
            # Get error details for the agent to fix
            if result.errors:
                error_count = len(result.errors)
                errors = result.errors

                self.active_logger.log_message(
                    f"Stata found {error_count} error(s) - debugging required",
                    level="WARNING"
                )

                # Store error info in state for agent access
                self.state.pending_questions = [{
                    "type": "stata_debug",
                    "do_file": str(do_path),
                    "errors": errors,
                    "error_summary": result.error,
                }]
                self.state.save()

                issues = [f"Line {e.get('line_number')}: {e.get('error_line')}" for e in errors[:5]]

                return StepResult(
                    success=False,
                    message=f"Stata found {error_count} error(s) in .do file.",
                    artifacts=[result.log_path, str(execution_path)],
                    issues=issues,
                    next_action="Fix reproducible syntax errors, rerun Stata, and ask the processor before changing any coding decision"
                )
            else:
                return StepResult(
                    success=False,
                    message=result.error or "Stata execution failed",
                    issues=[result.error] if result.error else []
                )

    def _step_9(self, **kwargs) -> StepResult:
        """Step 9: Collect and Integrate District Data."""
        self.active_logger.log_message("Reviewing district data...")
        from src.district_data import (
            DistrictDataTemplateParser,
            DistrictDataValidator,
            DistrictMergePlanner,
        )

        parser = DistrictDataTemplateParser()
        district_file = kwargs.get("district_file")
        approve = bool(kwargs.get("approve") or kwargs.get("processor_approved"))
        source_variable = kwargs.get("source_variable", "")
        if district_file:
            table = parser.parse(Path(district_file))
        else:
            table = parser.parse_best(self.working_dir)

        review_dir = self.working_dir / "micro" / "district data"
        review_dir.mkdir(parents=True, exist_ok=True)
        review_path = review_dir / f"{self.state.country_code or 'CNT'}_{self.state.year or 'YEAR'}_district_data_review.md"

        if not table:
            lines = [
                f"# District Data Review: {self.state.country} {self.state.year}",
                "",
                "No standardized district data file was found.",
                "",
                "Processor review needed:",
                "- Provide the standardized district data file, or confirm that district data is not included in this run.",
                "- Confirm the respondent district variable in the survey data.",
            ]
            review_path.write_text("\n".join(lines), encoding="utf-8")
            self.state.district_review_path = str(review_path)
            self.state.district_data_status = {
                "status": "needs_processor_review",
                "approved": False,
                "district_file_found": False,
            }
            return StepResult(
                success=False,
                message="District data review needs a standardized district data file",
                artifacts=[str(review_path)],
                issues=["Standardized district data file not found."],
                next_action="Add the standardized district data file and rerun District Data Review"
            )

        validation = DistrictDataValidator().validate(
            table,
            self.state,
            approve=approve,
            source_variable=source_variable,
        )
        planner = DistrictMergePlanner()
        plan = planner.build(self.working_dir, table, validation)
        plan.source_data_file = self.state.data_file or ""
        plan_path = planner.write(self.working_dir, plan)
        self.state.district_review_path = str(review_path)
        self.state.district_merge_plan_path = str(plan_path)
        self.state.district_data_status = {
            "status": validation.status,
            "approved": validation.approved,
            "district_file_found": True,
            "district_count": table.district_count,
            "source_district_variable": validation.source_district_variable,
            "missing_observed_district_codes": validation.missing_observed_district_codes,
            "extra_district_codes": validation.extra_district_codes,
            "party_order_approved": validation.party_order_approved,
        }

        lines = [
            f"# District Data Review: {self.state.country} {self.state.year}",
            "",
            f"District file: {Path(table.source_file).name}",
            f"District rows: {table.district_count}",
            f"Respondent district variable: {validation.source_district_variable or 'Needs processor review'}",
            f"Party order approved: {'Yes' if validation.party_order_approved else 'No'}",
            f"Processor approval recorded: {'Yes' if validation.approved else 'No'}",
            "",
            "District variables found:",
        ]
        lines.extend(f"- {col}" for col in table.columns if col.startswith("F400"))
        if validation.generated_missing_party_slots:
            lines.extend(["", "Party slots generated as not applicable/missing:"])
            lines.extend(f"- {col}" for col in validation.generated_missing_party_slots)
        if validation.missing_observed_district_codes:
            lines.extend(["", "District codes needing review:"])
            lines.extend(f"- {code}" for code in validation.missing_observed_district_codes[:100])
        if validation.extra_district_codes:
            lines.extend(["", "District rows not observed in the survey data:"])
            lines.extend(f"- {code}" for code in validation.extra_district_codes[:100])
        if validation.issues:
            lines.extend(["", "Processor review needed:"])
            lines.extend(f"- {issue}" for issue in validation.issues)
        if validation.warnings:
            lines.extend(["", "Warnings:"])
            lines.extend(f"- {warning}" for warning in validation.warnings)
        review_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.active_logger.log_message("District data review completed.")

        if validation.approved:
            return StepResult(
                success=True,
                message=(
                    "District data review approved\n\n"
                    f"District file: {Path(table.source_file).name}\n"
                    f"District rows: {table.district_count}\n"
                    f"Respondent district variable: {validation.source_district_variable}\n"
                    "District merge syntax can now be generated."
                ),
                artifacts=[str(review_path), str(plan_path), plan.normalized_dta_path],
                issues=validation.warnings,
                next_action="Continue to final Stata syntax generation"
            )

        return StepResult(
            success=False,
            message=(
                "District data review needs processor approval\n\n"
                f"District file: {Path(table.source_file).name}\n"
                f"District rows: {table.district_count}\n"
                f"Respondent district variable: {validation.source_district_variable or 'Needs processor review'}"
            ),
            artifacts=[str(review_path), str(plan_path), plan.normalized_dta_path],
            issues=validation.issues + validation.warnings,
            next_action="Review the district file, source district variable, party order alignment, and district-code coverage"
        )

    def _step_10(self, **kwargs) -> StepResult:
        """Step 10: Update Stata Label Files."""
        self.active_logger.log_message("Preparing release-specific Stata label updates...")
        from src.standards.artifacts import LabelFileGenerator

        label_dir = self.working_dir / "micro" / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(label_dir.glob("*.do"))
        generated = LabelFileGenerator(self.state).generate_micro_labels(label_dir)
        issues = []
        if not existing:
            issues.append("No existing micro label files found; schema-backed release-specific update file created.")
        self.active_logger.log_message(f"Release-specific label update file prepared: {generated.path.name}")
        return StepResult(
            success=True,
            message="Release-specific micro label update file prepared",
            artifacts=[str(generated.path)] + [str(path) for path in existing],
            issues=issues,
            next_action="Add reviewed label changes here instead of editing master label files"
        )

    def _step_11(self, **kwargs) -> StepResult:
        """Step 11: Finish Data Processing."""
        self.active_logger.log_message("Reviewing final data-processing reproducibility...")
        micro_dir = self.working_dir / "micro"
        do_files = sorted(micro_dir.glob("cses-m6_micro_*.do"))
        final_candidates = list(micro_dir.glob("*final*.dta")) + list(micro_dir.glob("*processed*.dta")) + list(micro_dir.glob("cses-m6_micro_*.dta"))
        report_path = micro_dir / f"{self.state.country_code or 'CNT'}_{self.state.year or 'YEAR'}_finish_processing_review.md"

        issues = []
        syntax_result = None
        if do_files:
            from src.standards.validators import validate_stata_syntax_text

            do_path = max(do_files, key=lambda path: path.stat().st_mtime)
            syntax_result = validate_stata_syntax_text(do_path.read_text(encoding="utf-8", errors="replace"))
            issues.extend(f"Syntax check failed: {issue}" for issue in syntax_result.issues)
        else:
            issues.append("No generated CSES micro .do file found.")

        if not final_candidates:
            issues.append("No final/processed Stata dataset found in micro/.")

        lines = [
            f"# Finish Data Processing Review: {self.state.country} {self.state.year}",
            "",
            "Source: cses_wiki/patterns/validation_checks.json",
            "",
            f"Generated do-files: {len(do_files)}",
            f"Final/processed dataset candidates: {len(final_candidates)}",
            "",
            "## Syntax Checks",
        ]
        if syntax_result:
            lines.extend(f"- {key}: {'PASS' if value else 'NEEDS REVIEW'}" for key, value in syntax_result.checks.items())
        else:
            lines.append("- No syntax file available.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        self.active_logger.log_message(f"Finish processing review written: {report_path.name}")
        return StepResult(
            success=True,
            message="Finish data processing review completed",
            artifacts=[str(report_path)] + [str(path) for path in final_candidates],
            issues=issues,
            next_action="Resolve reproducibility issues before release checks"
        )

    def _step_12(self, **kwargs) -> StepResult:
        """Step 12: Run Check Files"""
        self.active_logger.log_message("Running CSES standards review for validation/check files...")
        from src.standards.artifacts import CheckFileGenerator

        micro_dir = self.working_dir / "micro"
        data_checks_dir = micro_dir / "data_checks"
        generated_checks = CheckFileGenerator(self.state).generate_checks(data_checks_dir)
        check_run_artifacts = []
        check_run_issues = []
        stata_status = getattr(self.state, "stata_execution_status", {}) or {}
        if stata_status.get("success") and stata_status.get("output_dataset"):
            from src.stata_mcp import MCPStataRunner

            runner = MCPStataRunner(stata_path=kwargs.get("stata_path") or "")
            for generated in generated_checks:
                result = runner.run_do_file(generated.path)
                if result.log_path:
                    check_run_artifacts.append(result.log_path)
                if not result.success:
                    check_run_issues.append(f"{generated.path.name} did not pass cleanly: {result.error or 'review log'}")
        check_files = list(micro_dir.glob("data_checks/*.do")) + list(micro_dir.glob("*check*.do"))
        check_outputs = list(micro_dir.glob("data_checks/*.log")) + list(micro_dir.glob("data_checks/*.smcl")) + list(micro_dir.glob("*check*.log")) + list(micro_dir.glob("*check*.smcl"))
        issues = []
        warnings = []

        if not check_files:
            issues.append("No CSES validation/inconsistency/theoretical check .do files found")
        if not check_outputs:
            issues.append("No check output logs found")

        # Check for mappings with disagreements
        disagreements = [
            m for m in self.state.mappings
            if m.get("validation_verdict") == "DISAGREE"
        ]
        if disagreements:
            warnings.append(f"{len(disagreements)} mappings had model disagreements")

        # Check for NOT_FOUND
        not_found = [
            m for m in self.state.mappings
            if m.get("source_variable") in ["NOT_FOUND", "ERROR"]
        ]
        if not_found:
            warnings.append(f"{len(not_found)} CSES variables not matched")

        report_path = micro_dir / f"{self.state.country_code or 'CNT'}_{self.state.year or 'YEAR'}_check_file_review.md"
        lines = [
            f"# CSES Check File Review: {self.state.country} {self.state.year}",
            "",
            "Source: cses_wiki/procedures/module-release-checklist.md",
            "Source: cses_wiki/patterns/validation_checks.json",
            "",
            "## Required Check Types",
            "- validation checks",
            "- inconsistency checks",
            "- theoretical checks",
            "- missing-value checks",
            "- label checks",
            "- party-code checks",
            "",
            f"Detected check files: {len(check_files)}",
            f"Detected check outputs: {len(check_outputs)}",
            "",
        ]
        lines.extend(f"- File: {path.relative_to(self.working_dir)}" for path in check_files)
        lines.extend(f"- Output: {path.relative_to(self.working_dir)}" for path in check_outputs)
        if check_run_issues:
            lines.extend(["", "## Check Results Needing Review"])
            lines.extend(f"- {issue}" for issue in check_run_issues)
        report_path.write_text("\n".join(lines), encoding="utf-8")
        for issue in issues:
            self.active_logger.add_todo_item(issue)

        return StepResult(
            success=True,
            message="CSES check-file review completed",
            artifacts=[str(report_path)] + [str(item.path) for item in generated_checks] + [str(path) for path in check_files + check_outputs] + check_run_artifacts,
            issues=issues + warnings + check_run_issues,
            next_action="Review any warnings and draft collaborator questions if needed"
        )

    def _step_13(self, **kwargs) -> StepResult:
        """Step 13: Write Up Collaborator Questions"""
        self.active_logger.log_message("Compiling collaborator questions...")

        # Use existing tracked questions from collaborator_questions
        existing_questions = self.state.collaborator_questions.copy()

        # Also check for step issues that weren't already added
        for step_num in WORKFLOW_STEPS:
            step = self.state.get_step(step_num)
            if step.issues:
                for issue in step.issues:
                    keywords = ["request", "ask", "clarify", "confirm", "provide", "missing"]
                    if any(kw in issue.lower() for kw in keywords):
                        # Check if already tracked
                        already_tracked = any(
                            q.get("question") == issue or issue in q.get("question", "")
                            for q in existing_questions
                        )
                        if not already_tracked:
                            self.state.add_candidate_collaborator_question(
                                issue,
                                f"From Step {step_num} issues; processor review required before contacting collaborator",
                                step_num,
                                missing_items=[issue],
                            )

        # Check for mapping disagreements not already tracked
        for m in self.state.mappings:
            if m.get("validation_verdict") == "DISAGREE":
                target = m.get("cses_target", "")
                issue_text = f"Variable {target} mapping needs clarification"
                already_tracked = any(
                    target in q.get("question", "") for q in self.state.collaborator_questions
                )
                if not already_tracked:
                    self.state.add_candidate_collaborator_question(
                        issue_text,
                        m.get("validation_reasoning", "Models disagreed on mapping") + " Processor should inspect available evidence before creating a collaborator question.",
                        step_num=7,
                        missing_items=[target],
                    )

        # Get all pending questions
        pending_questions = self.state.get_pending_questions()
        candidate_questions = self.state.candidate_collaborator_questions

        # Also update legacy pending_questions for backwards compatibility
        self.state.pending_questions = [
            {"step": q.get("step"), "issue": q.get("question"), "status": q.get("status")}
            for q in pending_questions
        ]
        self.state.save()

        if pending_questions or candidate_questions:
            self.active_logger.log_message(
                f"Found {len(pending_questions)} confirmed pending collaborator questions and "
                f"{len(candidate_questions)} potential questions for processor review"
            )

            # The questions are already in the Collaborator Questions Word doc
            # but also generate a summary text file for quick reference
            questions_file = self.working_dir / "micro" / "collaborator_questions_summary.txt"
            grouped = self._group_collaborator_questions(pending_questions)
            lines = [
                f"# Collaborator Questions Summary: {self.state.country} {self.state.year}",
                "Source: cses_wiki/procedures/collaborator-question-review.md",
                "",
                f"Confirmed pending questions: {len(pending_questions)}",
                f"Potential questions for processor review: {len(candidate_questions)}",
                f"Full questions document: {self.state.collaborator_questions_file}",
                ""
            ]
            for group_name, questions in grouped.items():
                lines.append(f"## {group_name}")
                lines.append("")
                for q in questions:
                    lines.append(f"### {q.get('id', 'Question')}")
                    lines.append(f"Step: {q.get('step', 'N/A')}")
                    lines.append(f"Question: {q.get('question', 'N/A')}")
                    lines.append(f"Context: {q.get('context', 'N/A')}")
                    lines.append("Suggested resolution: Confirm with collaborator/project manager and update the log, tracking sheet, and syntax as needed.")
                    lines.append(f"Status: {q.get('status', 'pending')}")
                    lines.append("")
            if candidate_questions:
                lines.append("## Potential Questions For Processor Review")
                lines.append("")
                lines.append("These are not outgoing collaborator questions yet. First inspect available data, documentation, and wiki evidence; then decide whether collaborator contact is needed.")
                lines.append("")
                for q in candidate_questions:
                    lines.append(f"### {q.get('id', 'Potential Question')}")
                    missing = q.get("missing_items", [])
                    if missing:
                        lines.append("Missing items:")
                        lines.extend(f"- {item}" for item in missing)
                    lines.append(f"Potential question: {q.get('question', 'N/A')}")
                    lines.append(f"Context: {q.get('context', 'N/A')}")
                    lines.append(f"Status: {q.get('status', 'processor_review')}")
                    lines.append("")

            questions_file.write_text("\n".join(lines), encoding="utf-8")

            artifacts = [str(questions_file)]
            if self.state.collaborator_questions_file:
                artifacts.append(self.state.collaborator_questions_file)

            return StepResult(
                success=True,
                message=f"Compiled {len(pending_questions)} confirmed questions and {len(candidate_questions)} potential questions for processor review",
                artifacts=artifacts,
                next_action="Review questions document and send to project manager"
            )
        else:
            self.active_logger.log_message("No collaborator questions found")
            return StepResult(
                success=True,
                message="No collaborator questions needed",
                next_action="Proceed to Step 14 or 15"
            )

    def _step_14(self, **kwargs) -> StepResult:
        """Step 14: Follow Up on Collaborator Questions."""
        from src.standards.artifacts import _extract_text

        pending = self.state.get_pending_questions()
        candidates = self.state.candidate_collaborator_questions
        followup_dir = self.working_dir / "micro" / "collaborator questions"
        followup_dir.mkdir(parents=True, exist_ok=True)
        followup_path = followup_dir / f"{self.state.country_code or 'CNT'}_{self.state.year or 'YEAR'}_question_followup.md"
        existing_question_docs = [
            path for path in sorted((self.working_dir / "micro" / "Collaborator Questions").glob("**/*"))
            if path.is_file()
            and path.resolve() != followup_path.resolve()
            and path.suffix.casefold() in {".docx", ".txt", ".md"}
        ]
        lines = [
            f"# Collaborator Question Follow-Up: {self.state.country} {self.state.year}",
            "",
            "Source: cses_wiki/procedures/collaborator-question-review.md",
            "",
            f"Pending questions: {len(pending)}",
            f"Potential questions for processor review: {len(candidates)}",
            "",
        ]
        for q in self.state.collaborator_questions:
            lines.append(f"## {q.get('id')}: {q.get('status')}")
            lines.append(q.get("question", ""))
            lines.append("")
            if q.get("answer"):
                lines.append(f"Answer: {q.get('answer')}")
                lines.append("")
        if candidates:
            lines.append("## Potential Questions For Processor Review")
            lines.append("")
            for q in candidates:
                lines.append(f"### {q.get('id')}: {q.get('status')}")
                missing = q.get("missing_items", [])
                if missing:
                    lines.append("Missing items:")
                    lines.extend(f"- {item}" for item in missing)
                lines.append(q.get("question", ""))
                lines.append("")
        if existing_question_docs:
            lines.extend([
                "## Existing Collaborator Correspondence Reviewed",
                "",
                "The following supplied correspondence was retained in the processing record so the processor can verify that resolved collaborator decisions are reflected in syntax and documentation.",
                "",
            ])
            for path in existing_question_docs:
                text = _extract_text(path).strip()
                if not text:
                    continue
                lines.extend([
                    f"### {path.name}",
                    "",
                    text,
                    "",
                ])
        followup_path.write_text("\n".join(lines), encoding="utf-8")
        issues = []
        if pending:
            issues.append(f"{len(pending)} collaborator question(s) remain pending")
        if candidates:
            issues.append(f"{len(candidates)} potential collaborator question(s) need processor review")
        for issue in issues:
            self.active_logger.add_todo_item(issue)
        return StepResult(
            success=True,
            message="Collaborator question follow-up status recorded",
            artifacts=[str(followup_path)],
            issues=issues,
            next_action="Integrate resolved responses into log, tracking sheet, syntax, and documentation"
        )

    def _step_15(self, **kwargs) -> StepResult:
        """Step 15: Transfer ESNs to Codebook."""
        from src.standards.artifacts import DocumentationRenderer, _extract_text

        doc_dir = self.working_dir / "micro" / "Documentation"
        doc_dir.mkdir(parents=True, exist_ok=True)
        esn_path = doc_dir / f"ESN - {self.state.country} {self.state.year}.txt"
        rendered_log = DocumentationRenderer(self.state).render_processing_log(
            self.working_dir / "micro" / f"cses-m6_log-file_{self.state.country_code or 'CNT'}_{self.state.year or 'YEAR'}.txt"
        )
        log_text = ""
        if rendered_log.path.exists():
            log_text = rendered_log.path.read_text(encoding="utf-8", errors="replace")
        elif self.state.log_file and Path(self.state.log_file).exists():
            log_text = Path(self.state.log_file).read_text(encoding="utf-8", errors="replace")
        sd = self.active_logger.log_data.study_design if self.active_logger.log_data else {}
        lines = [
            f"Election Study Notes: {self.state.country} {self.state.year}",
            "",
            "Source: cses_wiki/topics/documentation-standards.md",
            "Source: cses_wiki/procedures/party-leader-appendix.md",
            "",
            "Election Summary",
            self.active_logger.log_data.election_summary if self.active_logger.log_data else "TBD",
            "",
            "Overview of Study Design and Weights",
        ]
        for key, value in sd.items():
            lines.append(f"{key}: {value or 'TBD'}")
        lines.extend(["", "Parties and Leaders", self.active_logger.log_data.parties_leaders if self.active_logger.log_data else "TBD"])
        source_esns = [
            path for path in sorted([*self.working_dir.glob("macro/*ESN*"), *doc_dir.glob("*ESN*")])
            if path.is_file() and path.resolve() != esn_path.resolve()
        ]
        if source_esns:
            lines.extend([
                "",
                "Existing Election Study Notes Reviewed",
                "",
                "The following supplied ESN material was retained so the processor can verify that all source documentation details are transferred to the codebook.",
                "",
            ])
            for path in source_esns:
                text = _extract_text(path).strip()
                if text:
                    lines.extend([f"Source ESN: {path.name}", "", text, ""])
        esn_path.write_text("\n".join(lines), encoding="utf-8")

        from src.standards.validators import validate_documentation_text

        doc_check = validate_documentation_text(log_text) if log_text else None
        issues = []
        if doc_check and not doc_check.ok:
            issues.extend(f"Documentation section missing from log: {issue}" for issue in doc_check.issues)
        if not log_text:
            issues.append("Processing log not available for ESN transfer validation.")
        return StepResult(
            success=True,
            message="Election Study Notes draft created from processing log",
            artifacts=[str(esn_path), str(rendered_log.path)],
            issues=issues,
            next_action="Review ESN text and transfer approved notes to codebook"
        )

    def _step_16(self, **kwargs) -> StepResult:
        """Step 16: Final Deposit."""
        self.active_logger.log_message("Running final CSES release readiness review...")
        from src.standards.artifacts import DocumentationRenderer, FinalReadinessValidator

        standards_readiness = self.standards.final_readiness(self.state)
        artifact_readiness = FinalReadinessValidator(self.state).evaluate(self.working_dir)
        issues = list(dict.fromkeys((standards_readiness.get("issues", []) or []) + (artifact_readiness.get("issues", []) or [])))
        readiness = {
            "status": "ready" if not issues else "needs_review",
            "issues": issues,
            "step_results": standards_readiness.get("step_results", {}),
            "schema_target_count": artifact_readiness.get("schema_target_count"),
            "syntax_schema_coverage": artifact_readiness.get("syntax_schema_coverage"),
            "wiki_sources": list(dict.fromkeys((standards_readiness.get("wiki_sources", []) or []) + (artifact_readiness.get("wiki_sources", []) or []))),
        }
        self.state.final_readiness = readiness
        self.active_logger.update_final_readiness(readiness)
        try:
            from src.benchmark import ReplicationBenchmarkRunner

            scorecard = ReplicationBenchmarkRunner(self.working_dir).scorecard(profile="workflow")
            self.state.benchmark_scorecard_path = str(self.working_dir / ".cses" / "replication_scorecard.json")
            for issue in scorecard.get("issues", []):
                if issue not in issues:
                    issues.append(issue)
            readiness["benchmark_status"] = scorecard.get("status")
        except Exception as exc:
            issues.append(f"Benchmark scorecard could not be generated: {exc}")
        report_dir = self.working_dir / "micro"
        report_path = report_dir / f"{self.state.country_code or 'CNT'}_{self.state.year or 'YEAR'}_final_readiness_report.md"
        missing_report = DocumentationRenderer(self.state).render_missing_input_report(
            report_dir / f"{self.state.country_code or 'CNT'}_{self.state.year or 'YEAR'}_missing_input_report.md"
        )
        processing_log = DocumentationRenderer(self.state).render_processing_log(
            report_dir / f"cses-m6_log-file_{self.state.country_code or 'CNT'}_{self.state.year or 'YEAR'}.txt"
        )
        lines = [
            f"# Final Readiness Report: {self.state.country} {self.state.year}",
            "",
            "Source: cses_wiki/procedures/module-release-checklist.md",
            "Source: cses_wiki/patterns/module6_schema.json",
            "",
            f"Status: {readiness.get('status')}",
            f"Readiness mode: {readiness.get('mode', 'full_release')}",
            f"Schema target count: {readiness.get('schema_target_count')}",
            f"Syntax schema coverage: {readiness.get('syntax_schema_coverage')}",
            "",
            "## Unresolved Issues",
        ]
        lines.extend(f"- {issue}" for issue in issues) if issues else lines.append("- None recorded")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return StepResult(
            success=True,
            message=f"Final readiness review completed: {readiness.get('status')}",
            artifacts=[str(report_path), str(missing_report.path), str(processing_log.path)],
            issues=issues,
            next_action="Proceed with final deposit only after unresolved risks are accepted or resolved"
        )

    def _group_collaborator_questions(self, questions: list[dict]) -> dict[str, list[dict]]:
        """Group collaborator questions by CSES workflow theme."""
        groups: dict[str, list[dict]] = {
            "Eligibility and Study Design": [],
            "Variable Mapping and Recoding": [],
            "Election Results and Party Coding": [],
            "District Data": [],
            "Documentation and Release": [],
            "Other": [],
        }
        for q in questions:
            text = f"{q.get('question', '')} {q.get('context', '')}".lower()
            if any(term in text for term in ["sample", "design", "eligib", "weight", "response rate", "mode"]):
                groups["Eligibility and Study Design"].append(q)
            elif any(term in text for term in ["variable", "mapping", "recode", "missing"]):
                groups["Variable Mapping and Recoding"].append(q)
            elif any(term in text for term in ["election result", "party", "leader", "vote"]):
                groups["Election Results and Party Coding"].append(q)
            elif any(term in text for term in ["district", "constituency"]):
                groups["District Data"].append(q)
            elif any(term in text for term in ["documentation", "release", "codebook", "esn"]):
                groups["Documentation and Release"].append(q)
            else:
                groups["Other"].append(q)
        return {name: items for name, items in groups.items() if items}

    def get_step_guidance(self, step_num: int) -> str:
        """Get guidance text for a specific step."""
        if step_num not in WORKFLOW_STEPS:
            return "Unknown step"

        step_info = WORKFLOW_STEPS[step_num]

        lines = [
            f"## Step {step_num}: {step_info['name']}",
            "",
            f"**Description:** {step_info['description']}",
            "",
            f"**Automation:** {'High' if step_info['automatable'] else 'Manual'}",
            f"**Requires LLM:** {'Yes' if step_info['requires_llm'] else 'No'}",
        ]

        # Add step-specific guidance
        guidance = {
            0: "Create folder structure from CSES template. Files go in: micro/ (data processing)",
            1: "Check that all required files are present: data file, questionnaire, design report. Mark in tracking sheet.",
            2: "Review the design report for: sample size, sampling method, response rate, fieldwork dates, data collection mode.",
            3: "Check variable list against CSES requirements. Note missing or unusual variables.",
            6: "Run frequency tables on the original data to see available variables and their distributions.",
            7: "For each CSES variable, find the matching source variable and write recode commands.",
            12: "Run inconsistency checks, theoretical checks, and interviewer validation checks.",
            13: "Compile all questions that arose during processing for collaborator clarification."
        }

        if step_num in guidance:
            lines.extend(["", f"**Guidance:** {guidance[step_num]}"])

        return "\n".join(lines)
