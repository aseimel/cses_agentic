"""
Code generation module for CSES Stata .do files.

This module provides deterministic code generation from a verified tracking sheet.
The workflow is:
1. AI fills tracking sheet with variable matching proposals (Step 7a)
2. Human reviews and verifies the tracking sheet in Excel (Step 7b - manual)
3. This module generates Stata code from the verified sheet (Step 7c)

No AI is used in code generation - it's purely template-based substitution.
"""

from .sheet_reader import TrackingSheetReader, VariableMapping
from .code_templates import StataTemplates
from .generator import StataCodeGenerator
from .recoding_plan import RecodingPlan, RecodingPlanBuilder, StataSyntaxPlanner, PlanDrivenStataSyntaxGenerator

__all__ = [
    'TrackingSheetReader',
    'VariableMapping',
    'StataTemplates',
    'StataCodeGenerator',
    'RecodingPlan',
    'RecodingPlanBuilder',
    'StataSyntaxPlanner',
    'PlanDrivenStataSyntaxGenerator',
]
