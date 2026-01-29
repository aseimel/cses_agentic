# Data and document ingestion modules
import logging

logger = logging.getLogger(__name__)

# Data loading - may fail if pandas not available
try:
    from .data_loader import DataLoader, DatasetInfo, VariableInfo, load_data
except ImportError as e:
    logger.warning(f"Data loader not available: {e}")
    DataLoader = None
    DatasetInfo = None
    VariableInfo = None
    load_data = None

# Document parsing
try:
    from .doc_parser import DocumentParser, DocumentInfo, QuestionInfo, parse_document
except ImportError as e:
    logger.warning(f"Document parser not available: {e}")
    DocumentParser = None
    DocumentInfo = None
    QuestionInfo = None
    parse_document = None

# Context extraction
try:
    from .context_extractor import (
        AdaptiveContextExtractor,
        ExtractionResult,
        SourceVariableContext,
        extract_context
    )
except ImportError as e:
    logger.warning(f"Context extractor not available: {e}")
    AdaptiveContextExtractor = None
    ExtractionResult = None
    SourceVariableContext = None
    extract_context = None

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
