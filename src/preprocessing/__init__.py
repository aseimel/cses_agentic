"""
Preprocessing modules for CSES variable mapping.

This package provides Stage 1 of the two-stage pipeline:
- DocumentAggregator: Uses fast LLM to read all documents and create
  per-variable summaries in TOON format.
"""

from .document_aggregator import DocumentAggregator, create_aggregator

__all__ = ['DocumentAggregator', 'create_aggregator']
