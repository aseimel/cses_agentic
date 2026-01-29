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
        "name": "Read Design Report",
        "description": "Review design report, verify study meets CSES standards",
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

    # Variable mappings (from Step 7)
    mappings: list[dict] = field(default_factory=list)

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
            "current_step": self.current_step,
            "pending_questions": self.pending_questions,
            "collaborator_questions": self.collaborator_questions,
            "_question_counter": self._question_counter,
            "mappings": self.mappings,
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
        with open(state_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

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
            with open(state_file) as f:
                data = json.load(f)
            state = cls.from_dict(data)
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
