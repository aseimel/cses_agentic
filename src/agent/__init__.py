"""
CSES Data Harmonization Agent Module.

Implements dual-model validation architecture:
1. Original LLM matcher proposes mappings
2. Validation LLM validates each proposal
3. Human makes final decision with both perspectives

Key components:
- validator.py: LLM validation logic
- cses_agent.py: Main agent orchestration
- tool_wrappers.py: Wrappers for existing modules
"""
import logging

logger = logging.getLogger(__name__)

# Validator - requires litellm
try:
    from .validator import (
        ValidationResult,
        ValidationVerdict,
        validate_proposal,
        validate_proposals,
        format_validation_result,
    )
except ImportError as e:
    logger.warning(f"Validator not available: {e}")
    ValidationResult = None
    ValidationVerdict = None
    validate_proposal = None
    validate_proposals = None
    format_validation_result = None

# Agent - requires litellm and other modules
try:
    from .cses_agent import (
        CSESAgent,
        DualModelResult,
        run_harmonization,
    )
except ImportError as e:
    logger.warning(f"CSES Agent not available: {e}")
    CSESAgent = None
    DualModelResult = None
    run_harmonization = None

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
