"""Workflow standards engine backed by cses_wiki."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.workflow.state import WORKFLOW_STEPS, WorkflowState


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSES_WIKI_ROOT = PROJECT_ROOT / "cses_wiki"


@dataclass(frozen=True)
class StepStandardsProfile:
    step_num: int
    name: str
    wiki_sources: list[str]
    required_inputs: list[str] = field(default_factory=list)
    required_outputs: list[str] = field(default_factory=list)
    human_decision_points: list[str] = field(default_factory=list)
    soft_gate_checks: list[str] = field(default_factory=list)
    collaborator_triggers: list[str] = field(default_factory=list)


@dataclass
class StandardsCheckResult:
    step_num: int
    status: str
    checks: dict[str, bool]
    issues: list[str] = field(default_factory=list)
    wiki_sources: list[str] = field(default_factory=list)
    human_decision_required: bool = False
    ready_for_next_step: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class WorkflowStandardsEngine:
    """Evaluate workflow steps against CSES wiki-backed standards."""

    def __init__(self, working_dir: Path | None = None, wiki_root: Path = CSES_WIKI_ROOT):
        self.working_dir = Path(working_dir) if working_dir else Path.cwd()
        self.wiki_root = wiki_root
        self.documentation_patterns = self._load_pattern("documentation_templates")
        self.syntax_patterns = self._load_pattern("stata_syntax")
        self.validation_patterns = self._load_pattern("validation_checks")
        self.module6_schema = self._load_pattern("module6_schema")

    def get_profile(self, step_num: int) -> StepStandardsProfile:
        profiles = self._profiles()
        return profiles.get(
            step_num,
            StepStandardsProfile(
                step_num=step_num,
                name=WORKFLOW_STEPS.get(step_num, {}).get("name", f"Step {step_num}"),
                wiki_sources=["cses_wiki/topics/operations.md"],
                soft_gate_checks=["Record what was checked and what remains unresolved."],
            ),
        )

    def guidance(self, step_num: int) -> str:
        profile = self.get_profile(step_num)
        lines = [
            f"Step {step_num}: {profile.name}",
            "CSES wiki sources:",
            *[f"- {source}" for source in profile.wiki_sources],
            "Required checks:",
            *[f"- {check}" for check in profile.soft_gate_checks],
        ]
        if profile.human_decision_points:
            lines.extend(["Human decision points:", *[f"- {item}" for item in profile.human_decision_points]])
        return "\n".join(lines)

    def evaluate_step(self, state: WorkflowState, step_num: int, artifacts: list[str] | None = None) -> StandardsCheckResult:
        profile = self.get_profile(step_num)
        artifacts = artifacts or []
        checks: dict[str, bool] = {}
        issues: list[str] = []

        def add(check_id: str, passed: bool, issue: str = ""):
            checks[check_id] = bool(passed)
            if not passed and issue:
                issues.append(issue)

        if step_num == 0:
            add("folder_micro", (self.working_dir / "micro").exists(), "Missing micro folder.")
            add("folder_macro", (self.working_dir / "macro").exists(), "Missing macro folder.")
            add("folder_election_results", (self.working_dir / "Election Results").exists(), "Missing Election Results folder.")
            add("folder_district_data", (self.working_dir / "District Data").exists(), "Missing District Data folder.")
        elif step_num == 1:
            study_design = self._study_design(state)
            add("data_file", bool(state.data_file), "No data file registered.")
            add("questionnaire", bool(state.questionnaire_files), "No questionnaire registered.")
            add("design_report", bool(state.design_report_file), "No design report registered.")
            add("sample_size", bool(study_design.get("sample_size")), "Sample size is not documented.")
            probability_status = str(study_design.get("probability_sample_status") or "").strip().lower()
            add(
                "probability_sample",
                probability_status in {"confirmed_probability_sample", "confirmed_non_probability_sample"},
                "Probability-sample eligibility needs processor review.",
            )
            add("cses_item_coverage", bool(study_design.get("cses_item_coverage")), "CSES item coverage is not documented.")
            processor_decision = str(study_design.get("processor_eligibility_decision") or "").strip().lower()
            add(
                "processor_decision",
                bool(processor_decision) and processor_decision not in {"pending", "tbd", "not specified"},
                "Processor eligibility decision is required.",
            )
        elif step_num in {2, 4}:
            study_design = self._study_design(state)
            for field in ["sample_design", "sample_size", "response_rate", "weighting", "collection_period", "mode"]:
                add(field, bool(study_design.get(field)) and study_design.get(field) != "TBD", f"Study design field missing or TBD: {field}.")
            add("source_citations", self._has_source_evidence(state), "Study design fields need source evidence/citations.")
        elif step_num == 3:
            add("tracking_sheet", bool(state.variable_tracking_file) or self._latest_tracking_sheet() is not None, "No variable tracking sheet found.")
            target_count = (getattr(state, "workflow_tracking", {}) or {}).get("target_count") or len(self.module6_schema.get("variables", []))
            add("full_schema_tracking", target_count >= 200, f"Tracking sheet covers only {target_count} CSES schema variables.")
        elif step_num == 5:
            add(
                "election_results",
                bool(self._find_files(["Election Results/*Election*Results*", "Election Results/*election*results*", "Election Results/*results*.xlsx", "Election Results/*results*.docx", "Election Results/*results*.csv"])),
                "Election results material not found in Election Results/.",
            )
        elif step_num == 6:
            add("frequency_artifact", bool(self._find_files(["micro/frequencies/*", "micro/*freq*"])), "Original-data frequency output not found.")
        elif step_num in {7, 8, 11}:
            do_file = self._latest_do_file()
            add("do_file", do_file is not None, "No CSES micro .do file found.")
            if do_file:
                text = do_file.read_text(encoding="utf-8", errors="replace")
                from src.standards.validators import validate_stata_syntax_text

                syntax = validate_stata_syntax_text(text)
                for check_id, passed in syntax.checks.items():
                    add(f"stata_{check_id}", passed, f"Stata syntax check failed: {check_id}.")
                schema_vars = [item.get("name") for item in self.module6_schema.get("variables", []) if item.get("name")]
                if schema_vars:
                    covered = sum(1 for name in schema_vars if name in text)
                    add("schema_variable_coverage", covered >= len(schema_vars) * 0.9, f"Generated syntax covers {covered}/{len(schema_vars)} schema variables.")
            if step_num == 8:
                add("clean_stata_log", self._has_clean_stata_log(), "Clean Stata execution log not found.")
        elif step_num == 9:
            add("district_review", bool(self._find_files(["*district*", "*constituency*"])), "District data/material not found or not documented.")
        elif step_num == 10:
            add("label_files", bool(self._find_files(["micro/labels/*.do", "micro/Labels/*.do"])), "Micro label .do files not found.")
        elif step_num == 12:
            add("check_files", bool(self._find_files(["micro/data_checks/*.do", "micro/*check*"])), "CSES check files not found.")
            add("check_outputs", bool(self._find_files(["micro/data_checks/*.log", "micro/data_checks/*.smcl", "micro/*check*.log", "micro/*check*.smcl"])), "Check output logs not found.")
        elif step_num == 13:
            add("questions_grouped", True, "")
            add("questions_context", all(q.get("context") for q in state.collaborator_questions), "Collaborator questions need context.")
        elif step_num == 14:
            pending = state.get_pending_questions()
            candidates = getattr(state, "candidate_collaborator_questions", [])
            add("question_status_tracking", True, "")
            add("unresolved_questions_reviewed", not pending, f"{len(pending)} collaborator question(s) remain pending.")
            add("candidate_questions_reviewed", not candidates, f"{len(candidates)} potential collaborator question(s) need processor review.")
        elif step_num == 15:
            add("esn_content", self._log_contains(["Election Study Notes", "Overview of Study Design and Weights"]), "Election Study Notes or study design overview not documented.")
            add("party_leader_appendix", self._log_contains(["Parties and Leaders"]), "Parties and Leaders appendix section not documented.")
        elif step_num == 16:
            pending = state.get_pending_questions()
            candidates = getattr(state, "candidate_collaborator_questions", [])
            add("no_pending_questions", not pending, f"{len(pending)} pending collaborator question(s).")
            add("no_candidate_questions", not candidates, f"{len(candidates)} potential collaborator question(s) need processor review.")
            add("final_dataset", bool(self._find_files(["micro/*final*.dta", "micro/*processed*.dta", "micro/cses-m6_micro_*.dta"])), "Final processed dataset not found.")
            add("release_log", bool(state.log_file) and Path(state.log_file).exists(), "Processing log not found.")

        status = "ready" if all(checks.values()) else "needs_review"
        human_decision_required = any(not value for value in checks.values()) or bool(profile.human_decision_points)
        return StandardsCheckResult(
            step_num=step_num,
            status=status,
            checks=checks,
            issues=issues,
            wiki_sources=profile.wiki_sources,
            human_decision_required=human_decision_required,
            ready_for_next_step=True,
        )

    def final_readiness(self, state: WorkflowState) -> dict:
        results = {str(step): self.evaluate_step(state, step).to_dict() for step in WORKFLOW_STEPS}
        issues = [issue for result in results.values() for issue in result.get("issues", [])]
        return {
            "status": "ready" if not issues else "needs_review",
            "issues": issues,
            "step_results": results,
            "wiki_sources": ["cses_wiki/procedures/module-release-checklist.md"],
        }

    def _load_pattern(self, name: str) -> dict:
        path = self.wiki_root / "patterns" / f"{name}.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def _profiles(self) -> dict[int, StepStandardsProfile]:
        return {
            0: StepStandardsProfile(0, "Set Up Country Folder", ["cses_wiki/topics/operations.md"], soft_gate_checks=["Folder structure follows CSES processing layout."]),
            1: StepStandardsProfile(1, "Check Completeness of Deposit", ["cses_wiki/procedures/study-eligibility-check.md"], human_decision_points=["Processor must record eligibility decision."], soft_gate_checks=["Data, questionnaire, design report, sample size, probability sample evidence, CSES item coverage."]),
            2: StepStandardsProfile(2, "Read Design Report", ["cses_wiki/topics/documentation-standards.md", "cses_wiki/procedures/study-eligibility-check.md"], soft_gate_checks=["Design facts have source evidence."]),
            3: StepStandardsProfile(3, "Fill Variable Tracking Sheet", ["cses_wiki/patterns/documentation_templates.json"], soft_gate_checks=["Tracking sheet captures evidence, confidence, verification, recoding, and questions."]),
            4: StepStandardsProfile(4, "Write Study Design & Weights Overview", ["cses_wiki/topics/documentation-standards.md"], soft_gate_checks=["Study design and weights overview is complete and source-backed."]),
            5: StepStandardsProfile(5, "Request Election Results Table", ["cses_wiki/topics/macro-data.md"], soft_gate_checks=["Election results material exists or request is recorded."]),
            6: StepStandardsProfile(6, "Run Frequencies on Original Data", ["cses_wiki/patterns/stata_syntax.json"], soft_gate_checks=["Frequencies cover all original variables and are saved in standard location."]),
            7: StepStandardsProfile(7, "Process Variables in Stata", ["cses_wiki/patterns/stata_syntax.json"], soft_gate_checks=["Mappings are evidence-backed and human verified before final syntax."]),
            8: StepStandardsProfile(8, "Debug Stata .do File", ["cses_wiki/patterns/validation_checks.json"], soft_gate_checks=["Stata execution is clean or errors are recorded."]),
            9: StepStandardsProfile(9, "Collect and Integrate District Data", ["cses_wiki/procedures/district-data-workflow.md"], soft_gate_checks=["District definitions, files, documentation, and merge readiness are reviewed."]),
            10: StepStandardsProfile(10, "Update Stata Label Files", ["cses_wiki/procedures/value-label-updates.md"], soft_gate_checks=["Micro/macro labels are separated and release-specific syntax is created."]),
            11: StepStandardsProfile(11, "Finish Data Processing", ["cses_wiki/patterns/validation_checks.json"], soft_gate_checks=["Syntax is reproducible, labels applied, source variables handled, and final save exists."]),
            12: StepStandardsProfile(12, "Run Check Files", ["cses_wiki/procedures/module-release-checklist.md"], soft_gate_checks=["Validation, inconsistency, theoretical, missing-value, label, and party-code checks are run."]),
            13: StepStandardsProfile(13, "Write Up Collaborator Questions", ["cses_wiki/procedures/collaborator-question-review.md"], soft_gate_checks=["Questions are grouped, contextualized, and focused."]),
            14: StepStandardsProfile(14, "Follow Up on Collaborator Questions", ["cses_wiki/procedures/collaborator-question-review.md"], soft_gate_checks=["Question responses are tracked and unresolved issues remain visible."]),
            15: StepStandardsProfile(15, "Transfer ESNs to Codebook", ["cses_wiki/topics/documentation-standards.md", "cses_wiki/procedures/party-leader-appendix.md"], soft_gate_checks=["ESNs and appendix format pass documentation checks."]),
            16: StepStandardsProfile(16, "Final Deposit", ["cses_wiki/procedures/module-release-checklist.md"], soft_gate_checks=["Final readiness report covers data, syntax, labels, docs, checks, questions, and TODOs."]),
        }

    def _study_design(self, state: WorkflowState) -> dict:
        try:
            from src.workflow.active_logging import ActiveLogger

            logger = ActiveLogger(state)
            return logger.log_data.study_design if logger.log_data else {}
        except Exception:
            return {}

    def _has_source_evidence(self, state: WorkflowState) -> bool:
        return bool(getattr(state, "wiki_sources", [])) or self._log_contains(["Source:", "Evidence:", "Sampling Evidence"])

    def _latest_tracking_sheet(self) -> Path | None:
        candidates = list((self.working_dir / "micro").glob("**/deposited variables-m6_*.xlsx"))
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    def _latest_do_file(self) -> Path | None:
        candidates = list((self.working_dir / "micro").glob("cses-m6_micro_*.do"))
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    def _find_files(self, patterns: list[str]) -> list[Path]:
        found = []
        for pattern in patterns:
            found.extend(self.working_dir.glob(pattern))
            if not pattern.startswith("**/") and "/" not in pattern and "\\" not in pattern:
                found.extend(self.working_dir.glob(f"**/{pattern}"))
        unique: dict[str, Path] = {}
        for path in found:
            if path.exists():
                unique[str(path.resolve())] = path
        return list(unique.values())

    def _has_clean_stata_log(self) -> bool:
        logs = self._find_files(["micro/*.log", "micro/*.smcl", "micro/**/*.log", "micro/**/*.smcl"])
        for path in logs:
            text = path.read_text(encoding="utf-8", errors="replace").lower()
            if "error" not in text and "r(" not in text:
                return True
        return False

    def _log_contains(self, needles: list[str]) -> bool:
        for path in self._find_files(["micro/*_log.qmd", "micro/*.qmd"]):
            text = path.read_text(encoding="utf-8", errors="replace")
            if all(needle.lower() in text.lower() for needle in needles):
                return True
        return False
