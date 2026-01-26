"""
CSES Data Harmonization Agent Module.

Implements dual-model validation architecture:
1. Original LLM matcher proposes mappings
2. Claude validates each proposal
3. Human makes final decision with both perspectives

Key components:
- validator.py: Claude validation logic
- cses_agent.py: Main agent orchestration
- tool_wrappers.py: Wrappers for existing modules
"""

from .validator import (
    ValidationResult,
    ValidationVerdict,
    validate_proposal,
    validate_proposals,
    format_validation_result,
)

from .cses_agent import (
    CSESAgent,
    DualModelResult,
    run_harmonization,
)

__all__ = [
    # Validator
    "ValidationResult",
    "ValidationVerdict",
    "validate_proposal",
    "validate_proposals",
    "format_validation_result",
    # Agent
    "CSESAgent",
    "DualModelResult",
    "run_harmonization",
]
