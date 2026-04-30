"""
Workflow State Manager for CSES Processing.

Tracks progress through the 16-step CSES micro-processing workflow.
Persists state to .cses/state.json for resume capability.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a workflow step."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"          # Waiting on something (e.g., collaborator response)
    COMPLETED = "completed"
    SKIPPED = "skipped"          # Not applicable for this study


# The 16 CSES workflow steps (Step 0-16)
WORKFLOW_STEPS = {
    0: {
        "name": "Set Up Country Folder",
        "description": "Create local folder structure from template",
        "automatable": True,
        "requires_llm": False
    },
    1: {
        "name": "Check Completeness of Deposit",
        "description": "Review deposited data and documentation, mark in tracking sheet",
        "automatable": True,
        "requires_llm": True
    },
    2: {
        "name": "Confirm Study Design Facts",
        "description": "Confirm evidence-backed design facts for the processing log",
        "automatable": False,
        "requires_llm": True
    },
    3: {
        "name": "Fill Variable Tracking Sheet",
        "description": "Check variable list against CSES requirements",
        "automatable": True,
        "requires_llm": True
    },
    4: {
        "name": "Write Study Design & Weights Overview",
        "description": "Document study design and weighting methodology in logfile",
        "automatable": True,
        "requires_llm": True
    },
    5: {
        "name": "Request Election Results Table",
        "description": "Contact macro coder for election results for party ordering",
        "automatable": False,
        "requires_llm": False
    },
    6: {
        "name": "Run Frequencies on Original Data",
        "description": "Run frequency tables on deposited data",
        "automatable": True,
        "requires_llm": False
    },
    7: {
        "name": "Process Variables in Stata",
        "description": "Match and recode variables to CSES schema",
        "automatable": True,
        "requires_llm": True
    },
    8: {
        "name": "Debug Stata .do File",
        "description": "Run and debug the generated .do file in Stata, fix errors iteratively",
        "automatable": True,
        "requires_llm": True
    },
    9: {
        "name": "Collect and Integrate District Data",
        "description": "Collect district-level election results, merge to dataset",
        "automatable": False,
        "requires_llm": False
    },
    10: {
        "name": "Update Stata Label Files",
        "description": "Update numeric party code labels",
        "automatable": True,
        "requires_llm": False
    },
    11: {
        "name": "Finish Data Processing",
        "description": "Drop original variables, apply labels, save processed data",
        "automatable": True,
        "requires_llm": False
    },
    12: {
        "name": "Run Check Files",
        "description": "Run inconsistency, theoretical, and validation checks",
        "automatable": True,
        "requires_llm": False
    },
    13: {
        "name": "Write Up Collaborator Questions",
        "description": "Compile clarification questions for collaborators",
        "automatable": True,
        "requires_llm": True
    },
    14: {
        "name": "Follow Up on Collaborator Questions",
        "description": "Track responses, update syntax and documentation",
        "automatable": False,
        "requires_llm": True
    },
    15: {
        "name": "Transfer ESNs to Codebook",
        "description": "Transfer Election Study Notes from log to codebook",
        "automatable": True,
        "requires_llm": True
    },
    16: {
        "name": "Final Deposit",
        "description": "Copy final dataset to Dropbox, email project manager",
        "automatable": True,
        "requires_llm": False
    }
}


# REMOVED: Old STEP_PREREQUISITES dict allowed skipping steps.
# New rule: Step N can ONLY start if Step N-1 is COMPLETED. No exceptions.


@dataclass
class StepState:
    """State of a single workflow step."""
    status: str = "not_started"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    notes: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)  # Files produced

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StepState":
        return cls(**data)


@dataclass
class WorkflowState:
    """
    Complete workflow state for a country study.

    Persisted to .cses/state.json in the working folder.
    """
    # Study identification
    country: str = "Unknown"
    country_code: str = "UNK"
    year: str = "0000"

    # Session tracking
    session_id: str = ""
    created_at: str = ""
    updated_at: str = ""

    # File paths (relative to working directory)
    working_dir: str = ""
    data_file: Optional[str] = None
    questionnaire_files: list[str] = field(default_factory=list)
    codebook_file: Optional[str] = None
    design_report_file: Optional[str] = None

    # Active logging file paths
    log_file: Optional[str] = None
    collaborator_questions_file: Optional[str] = None
    variable_tracking_file: Optional[str] = None  # CSES variable tracking sheet

    # Step states
    steps: dict[str, StepState] = field(default_factory=dict)

    # Current focus
    current_step: int = 0

    # Collaborator questions tracking (legacy - for Step 13 output)
    pending_questions: list[dict] = field(default_factory=list)

    # Collaborator questions with full tracking
    # Each: {id, question, context, step, timestamp, status}
    collaborator_questions: list[dict] = field(default_factory=list)
    candidate_collaborator_questions: list[dict] = field(default_factory=list)

    # Variable mappings (from Step 7)
    mappings: list[dict] = field(default_factory=list)

    # Standards-backed workflow metadata
    standards_checks: dict = field(default_factory=dict)
    wiki_sources: list[dict] = field(default_factory=list)
    processor_decisions: list[dict] = field(default_factory=list)
    final_readiness: dict = field(default_factory=dict)
    evidence_index: dict = field(default_factory=dict)
    evidence_packet_status: str = "missing"
    evidence_packet_path: str = ""
    evidence_manifest_path: str = ""
    last_evidence_refresh: str = ""
    current_phase: str = "setup"
    phase_status: dict = field(default_factory=dict)
    workflow_tracking: dict = field(default_factory=dict)
    input_manifest_path: str = ""
    primary_input_selection: dict = field(default_factory=dict)
    matching_coverage: dict = field(default_factory=dict)
    matching_decisions_path: str = ""
    recoding_coverage: dict = field(default_factory=dict)
    recoding_plans_path: str = ""
    approval_status: dict = field(default_factory=dict)
    stata_execution_status: dict = field(default_factory=dict)
    benchmark_scorecard_path: str = ""

    # Study-specific KB built from deposited files and deterministic data summaries
    study_kb_status: str = "missing"
    study_kb_path: str = ""
    study_kb_updated_at: str = ""
    study_kb_source_manifest: str = ""
    study_kb_model: str = ""
    study_kb_missing_fields: list[str] = field(default_factory=list)
    study_kb_contradictions: list[str] = field(default_factory=list)

    # Question ID counter for generating unique IDs
    _question_counter: int = field(default=0, repr=False)

    def __post_init__(self):
        """Initialize step states if empty."""
        if not self.steps:
            for step_num in WORKFLOW_STEPS:
                self.steps[str(step_num)] = StepState()
        if not self.session_id:
            self.session_id = f"ses_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def get_step(self, step_num: int) -> StepState:
        """Get state for a specific step."""
        key = str(step_num)
        if key not in self.steps:
            self.steps[key] = StepState()
        step_data = self.steps[key]
        if isinstance(step_data, dict):
            self.steps[key] = StepState.from_dict(step_data)
        return self.steps[key]

    def set_step_status(self, step_num: int, status: StepStatus, note: str = None):
        """Update step status."""
        step = self.get_step(step_num)
        step.status = status.value

        if status == StepStatus.IN_PROGRESS and not step.started_at:
            step.started_at = datetime.now(timezone.utc).isoformat()
        elif status == StepStatus.COMPLETED:
            step.completed_at = datetime.now(timezone.utc).isoformat()

        if note:
            step.notes.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {note}")

        self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_step_issue(self, step_num: int, issue: str):
        """Record an issue for a step."""
        step = self.get_step(step_num)
        step.issues.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {issue}")
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_step_artifact(self, step_num: int, artifact_path: str):
        """Record an artifact (output file) for a step."""
        step = self.get_step(step_num)
        if artifact_path not in step.artifacts:
            step.artifacts.append(artifact_path)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def record_standards_check(self, step_num: int, result: dict):
        """Persist a standards check result for a workflow step."""
        self.standards_checks[str(step_num)] = result
        for source in result.get("wiki_sources", []):
            entry = {"step": step_num, "source": source}
            if entry not in self.wiki_sources:
                self.wiki_sources.append(entry)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def record_processor_decision(self, step_num: int, decision: str, context: str = ""):
        """Record a human processor decision or override."""
        self.processor_decisions.append({
            "step": step_num,
            "decision": decision,
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def rebase_paths(self, actual_working_dir: Path):
        """Rebase stored absolute paths when a study folder has moved."""
        actual_dir = str(actual_working_dir.resolve())
        old_dir = self.working_dir or actual_dir
        if old_dir == actual_dir:
            return

        def remap(value):
            if not value:
                return value
            text = str(value)
            try:
                old_path = Path(old_dir)
                new_path = Path(actual_dir)
                try:
                    rel_path = Path(text).relative_to(old_path)
                    return str(new_path / rel_path)
                except ValueError:
                    return text.replace(str(old_path), str(new_path))
            except Exception:
                return text

        self.working_dir = actual_dir
        self.data_file = remap(self.data_file)
        self.codebook_file = remap(self.codebook_file)
        self.design_report_file = remap(self.design_report_file)
        self.log_file = remap(self.log_file)
        self.collaborator_questions_file = remap(self.collaborator_questions_file)
        self.variable_tracking_file = remap(self.variable_tracking_file)
        self.study_kb_path = remap(self.study_kb_path)
        self.study_kb_source_manifest = remap(self.study_kb_source_manifest)
        self.evidence_packet_path = remap(self.evidence_packet_path)
        self.evidence_manifest_path = remap(self.evidence_manifest_path)
        self.input_manifest_path = remap(self.input_manifest_path)
        self.matching_decisions_path = remap(self.matching_decisions_path)
        self.recoding_plans_path = remap(self.recoding_plans_path)
        self.benchmark_scorecard_path = remap(self.benchmark_scorecard_path)
        self.questionnaire_files = [remap(path) for path in self.questionnaire_files or []]

        for step in self.steps.values():
            if isinstance(step, dict):
                artifacts = step.get("artifacts", [])
                step["artifacts"] = [remap(path) for path in artifacts]
            elif hasattr(step, "artifacts"):
                step.artifacts = [remap(path) for path in step.artifacts]

    def add_collaborator_question(self, question: str, context: str, step_num: int) -> str:
        """
        Add a collaborator question with full tracking.

        Args:
            question: The question text
            context: Context for the question
            step_num: Step number where question arose

        Returns:
            Question ID (e.g., "CQ AA1")
        """
        # Generate question ID (CQ AA1, CQ AA2, etc.)
        self._question_counter += 1
        question_id = f"CQ AA{self._question_counter}"

        question_entry = {
            "id": question_id,
            "question": question,
            "context": context,
            "step": step_num,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "pending"
        }

        self.collaborator_questions.append(question_entry)
        self.updated_at = datetime.now(timezone.utc).isoformat()

        return question_id

    def add_candidate_collaborator_question(self, question: str, context: str, step_num: int, missing_items: list[str] = None) -> str:
        """
        Record a potential collaborator question for processor review.

        This does not create an outgoing collaborator question. The processor must
        decide whether the retrieved evidence is sufficient or whether the question
        should be promoted.
        """
        candidate_id = f"PCQ {len(self.candidate_collaborator_questions) + 1}"
        entry = {
            "id": candidate_id,
            "question": question,
            "context": context,
            "step": step_num,
            "missing_items": missing_items or [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "processor_review"
        }
        self.candidate_collaborator_questions.append(entry)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return candidate_id

    def get_pending_questions(self) -> list[dict]:
        """Get all pending (unresolved) collaborator questions."""
        return [q for q in self.collaborator_questions if q.get("status") == "pending"]

    def resolve_question(self, question_id: str, answer: str = None):
        """
        Mark a collaborator question as resolved.

        Args:
            question_id: The question ID (e.g., "CQ AA1")
            answer: Optional answer text
        """
        for q in self.collaborator_questions:
            if q.get("id") == question_id:
                q["status"] = "resolved"
                q["resolved_at"] = datetime.now(timezone.utc).isoformat()
                if answer:
                    q["answer"] = answer
                break

        self.updated_at = datetime.now(timezone.utc).isoformat()

    def get_next_step(self) -> Optional[int]:
        """Get the next step that should be worked on."""
        for step_num in sorted(WORKFLOW_STEPS.keys()):
            step = self.get_step(step_num)
            if step.status in [StepStatus.NOT_STARTED.value, StepStatus.IN_PROGRESS.value]:
                return step_num
        return None

    def check_step_prerequisites(self, step_num: int) -> tuple[bool, str]:
        """
        Check if a step can be started.

        RULE: Step N can only start if Step N-1 is completed.
        ALL steps are required. NO skipping. EVER.
        """
        if step_num == 0:
            return True, "OK"

        # Simple rule: previous step must be done
        prev_step = step_num - 1
        prev_status = self.get_step(prev_step).status

        if prev_status != "completed":
            step_name = WORKFLOW_STEPS.get(prev_step, {}).get("name", f"Step {prev_step}")
            return False, f"Cannot skip steps. Complete Step {prev_step} ({step_name}) first."

        return True, "OK"

    def get_progress_summary(self) -> dict:
        """Get summary of workflow progress."""
        completed = 0
        in_progress = 0
        blocked = 0
        not_started = 0

        for step_num in WORKFLOW_STEPS:
            step = self.get_step(step_num)
            if step.status == StepStatus.COMPLETED.value:
                completed += 1
            elif step.status == StepStatus.IN_PROGRESS.value:
                in_progress += 1
            elif step.status == StepStatus.BLOCKED.value:
                blocked += 1
            else:
                not_started += 1

        total = len(WORKFLOW_STEPS)
        return {
            "total_steps": total,
            "completed": completed,
            "in_progress": in_progress,
            "blocked": blocked,
            "not_started": not_started,
            "percent_complete": (completed / total) * 100
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = {
            "country": self.country,
            "country_code": self.country_code,
            "year": self.year,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "working_dir": self.working_dir,
            "data_file": self.data_file,
            "questionnaire_files": self.questionnaire_files,
            "codebook_file": self.codebook_file,
            "design_report_file": self.design_report_file,
            "log_file": self.log_file,
            "collaborator_questions_file": self.collaborator_questions_file,
            "variable_tracking_file": self.variable_tracking_file,
            "current_step": self.current_step,
            "pending_questions": self.pending_questions,
            "collaborator_questions": self.collaborator_questions,
            "candidate_collaborator_questions": self.candidate_collaborator_questions,
            "_question_counter": self._question_counter,
            "mappings": self.mappings,
            "standards_checks": self.standards_checks,
            "wiki_sources": self.wiki_sources,
            "processor_decisions": self.processor_decisions,
            "final_readiness": self.final_readiness,
            "evidence_index": self.evidence_index,
            "evidence_packet_status": self.evidence_packet_status,
            "evidence_packet_path": self.evidence_packet_path,
            "evidence_manifest_path": self.evidence_manifest_path,
            "last_evidence_refresh": self.last_evidence_refresh,
            "current_phase": self.current_phase,
            "phase_status": self.phase_status,
            "workflow_tracking": self.workflow_tracking,
            "input_manifest_path": self.input_manifest_path,
            "primary_input_selection": self.primary_input_selection,
            "matching_coverage": self.matching_coverage,
            "matching_decisions_path": self.matching_decisions_path,
            "recoding_coverage": self.recoding_coverage,
            "recoding_plans_path": self.recoding_plans_path,
            "approval_status": self.approval_status,
            "stata_execution_status": self.stata_execution_status,
            "benchmark_scorecard_path": self.benchmark_scorecard_path,
            "study_kb_status": self.study_kb_status,
            "study_kb_path": self.study_kb_path,
            "study_kb_updated_at": self.study_kb_updated_at,
            "study_kb_source_manifest": self.study_kb_source_manifest,
            "study_kb_model": self.study_kb_model,
            "study_kb_missing_fields": self.study_kb_missing_fields,
            "study_kb_contradictions": self.study_kb_contradictions,
            "steps": {}
        }
        for key, step in self.steps.items():
            if isinstance(step, StepState):
                data["steps"][key] = step.to_dict()
            else:
                data["steps"][key] = step
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowState":
        """Create from dictionary."""
        steps_data = data.pop("steps", {})
        # Handle private fields that may be in saved data
        question_counter = data.pop("_question_counter", 0)
        state = cls(**data)
        state._question_counter = question_counter
        for key, step_data in steps_data.items():
            if isinstance(step_data, dict):
                state.steps[key] = StepState.from_dict(step_data)
            else:
                state.steps[key] = step_data
        return state

    def save(self, state_dir: Path = None):
        """Save state to .cses/state.json."""
        if state_dir is None:
            state_dir = Path(self.working_dir) / ".cses" if self.working_dir else Path(".cses")
        state_dir.mkdir(parents=True, exist_ok=True)

        state_file = state_dir / "state.json"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved workflow state to {state_file}")

    @classmethod
    def load(cls, working_dir: Path = None) -> Optional["WorkflowState"]:
        """Load state from .cses/state.json."""
        if working_dir is None:
            working_dir = Path.cwd()

        state_file = working_dir / ".cses" / "state.json"
        if not state_file.exists():
            return None

        try:
            with open(state_file, encoding="utf-8") as f:
                data = json.load(f)
            state = cls.from_dict(data)
            state.rebase_paths(working_dir)
            logger.info(f"Loaded workflow state from {state_file}")
            return state
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None


def format_workflow_status(state: WorkflowState) -> str:
    """Format workflow status for display."""
    lines = [
        f"## {state.country} {state.year} - Workflow Status",
        "",
        f"Session: {state.session_id}",
        f"Last updated: {state.updated_at}",
        ""
    ]

    # Show log file if exists
    if state.log_file:
        from pathlib import Path
        log_path = Path(state.log_file)
        lines.append(f"Log file: {log_path.name}")

    # Show pending questions count
    pending = state.get_pending_questions()
    if pending:
        lines.append(f"Pending questions: {len(pending)}")

    if state.log_file or pending:
        lines.append("")

    progress = state.get_progress_summary()
    lines.extend([
        f"### Progress: {progress['percent_complete']:.0f}% complete",
        f"- Completed: {progress['completed']}/{progress['total_steps']}",
        f"- In progress: {progress['in_progress']}",
        f"- Blocked: {progress['blocked']}",
        ""
    ])

    lines.append("### Steps")
    status_markers = {
        "not_started": "[    ]",
        "in_progress": "[....]",
        "blocked": "[WAIT]",
        "completed": "[DONE]",
        "skipped": "[SKIP]"
    }

    for step_num in sorted(WORKFLOW_STEPS.keys()):
        step_info = WORKFLOW_STEPS[step_num]
        step_state = state.get_step(step_num)
        marker = status_markers.get(step_state.status, "[????]")

        line = f"{marker} **Step {step_num}:** {step_info['name']}"
        if step_state.status == "in_progress":
            line += " <- current"
        lines.append(line)

        if step_state.issues:
            lines.append(f"   [!] Issues: {len(step_state.issues)}")

    return "\n".join(lines)
