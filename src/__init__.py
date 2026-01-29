# CSES Data Harmonization Automation

# Import modules with graceful handling for missing dependencies
import logging
logger = logging.getLogger(__name__)

try:
    from . import ingest
except ImportError as e:
    logger.warning(f"Could not import ingest module: {e}")
    ingest = None

try:
    from . import matching
except ImportError as e:
    logger.warning(f"Could not import matching module: {e}")
    matching = None

try:
    from . import workflow
except ImportError as e:
    logger.warning(f"Could not import workflow module: {e}")
    workflow = None

try:
    from . import agent
except ImportError as e:
    logger.warning(f"Could not import agent module: {e}")
    agent = None

__all__ = ["ingest", "matching", "workflow", "agent"]
