"""
Preprocessing modules for CSES variable mapping.

This package provides Stage 0-1 of the multi-stage pipeline:
- DocumentQualityAssessor: Grades document quality before processing
- DocumentAggregator: Uses fast LLM to read all documents and create
  per-variable summaries in TOON format.
"""

from .document_aggregator import DocumentAggregator, create_aggregator
from .quality_assessor import (
    DocumentQualityAssessor,
    DocumentQuality,
    QualityReport,
    QualityScore,
    create_quality_assessor,
)

__all__ = [
    # Document Aggregator
    'DocumentAggregator',
    'create_aggregator',
    # Quality Assessor
    'DocumentQualityAssessor',
    'DocumentQuality',
    'QualityReport',
    'QualityScore',
    'create_quality_assessor',
]
