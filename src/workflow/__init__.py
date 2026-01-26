"""
CSES Workflow Management Module.

Provides:
- WorkflowState: Track progress through 16-step workflow
- FileOrganizer: Detect and organize collaborator files
- StepExecutor: Execute individual workflow steps
"""

from .state import WorkflowState, StepStatus, WORKFLOW_STEPS
from .organizer import FileOrganizer, DetectedFiles
from .steps import StepExecutor

__all__ = [
    "WorkflowState",
    "StepStatus",
    "WORKFLOW_STEPS",
    "FileOrganizer",
    "DetectedFiles",
    "StepExecutor",
]
