# Data and document ingestion modules

from .data_loader import DataLoader, DatasetInfo, VariableInfo, load_data
from .doc_parser import DocumentParser, DocumentInfo, QuestionInfo, parse_document
from .context_extractor import (
    AdaptiveContextExtractor,
    ExtractionResult,
    SourceVariableContext,
    extract_context
)

__all__ = [
    # Data loading
    "DataLoader",
    "DatasetInfo",
    "VariableInfo",
    "load_data",
    # Document parsing
    "DocumentParser",
    "DocumentInfo",
    "QuestionInfo",
    "parse_document",
    # Context extraction
    "AdaptiveContextExtractor",
    "ExtractionResult",
    "SourceVariableContext",
    "extract_context",
]
