"""Processor-facing workflow phases over canonical CSES steps."""

from __future__ import annotations

from dataclasses import dataclass

from src.workflow.state import StepStatus, WorkflowState


@dataclass(frozen=True)
class WorkflowPhase:
    id: str
    title: str
    steps: tuple[int, ...]


WORKFLOW_PHASES: tuple[WorkflowPhase, ...] = (
    WorkflowPhase("setup", "Setup", (0,)),
    WorkflowPhase("intake", "Intake and Eligibility", (1, 2, 4)),
    WorkflowPhase("inventory", "Source Inventory and Tracking", (3, 5, 6)),
    WorkflowPhase("mapping", "Mapping and Recoding", (7, 8, 11)),
    WorkflowPhase("external", "External Inputs", (9, 10, 13, 14)),
    WorkflowPhase("release", "Release Documentation and Deposit", (12, 15, 16)),
)


STEP_TO_PHASE = {
    step: phase
    for phase in WORKFLOW_PHASES
    for step in phase.steps
}


def phase_for_step(step_num: int) -> WorkflowPhase | None:
    return STEP_TO_PHASE.get(step_num)


def phase_status(state: WorkflowState, phase: WorkflowPhase) -> str:
    statuses = [state.get_step(step).status for step in phase.steps]
    if all(status == StepStatus.COMPLETED.value for status in statuses):
        return "completed"
    if any(status == StepStatus.BLOCKED.value for status in statuses):
        return "blocked"
    if any(status == StepStatus.IN_PROGRESS.value for status in statuses):
        return "in_progress"
    if any(status == StepStatus.COMPLETED.value for status in statuses):
        return "in_progress"
    return "not_started"


def current_phase_id(state: WorkflowState) -> str:
    next_step = state.get_next_step()
    phase = phase_for_step(next_step) if next_step is not None else None
    if phase:
        return phase.id
    return "complete"


def phase_status_payload(state: WorkflowState) -> dict:
    return {
        phase.id: {
            "title": phase.title,
            "status": phase_status(state, phase),
            "steps": list(phase.steps),
        }
        for phase in WORKFLOW_PHASES
    }


def validate_phase_mapping(all_steps: set[int]) -> tuple[bool, str]:
    mapped = [step for phase in WORKFLOW_PHASES for step in phase.steps]
    missing = sorted(all_steps - set(mapped))
    duplicates = sorted({step for step in mapped if mapped.count(step) > 1})
    extra = sorted(set(mapped) - all_steps)
    if missing or duplicates or extra:
        return False, f"missing={missing}; duplicates={duplicates}; extra={extra}"
    return True, "OK"
