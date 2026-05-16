"""
Reference patterns module for CSES code generation.

Provides reference examples from validated .do files (like Sweden_2022)
to guide LLM-powered Stata code generation.
"""

from src.reference.sweden_patterns import (
    load_sweden_patterns,
    get_pattern_for_variable,
    SwedenPatternLoader,
)

__all__ = [
    "load_sweden_patterns",
    "get_pattern_for_variable",
    "SwedenPatternLoader",
]
