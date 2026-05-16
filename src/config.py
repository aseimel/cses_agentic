"""
Central configuration for CSES CLI.

Model settings are exposed as curated profiles for non-technical users. Users
choose a profile in the GUI; they do not need to know provider-specific model
strings.
"""

import os

from src.model_profiles import DEFAULT_PROFILE_ID, get_profile

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Models are selected through curated profiles. Direct vendor APIs are the
# default routing behavior. OpenWebUI or another OpenAI-compatible endpoint is
# used only when OPENAI_API_BASE is explicitly configured and enabled.
# =============================================================================

# =============================================================================
# ENSEMBLE CONFIGURATION (Multi-Model Quorum Voting)
# =============================================================================

# Ensemble models for quorum voting (3 diverse architectures)
# NOTE: gpt-5 was removed due to JSON parsing failures.
#       Using gpt-4.1 which reliably returns valid JSON.
ACTIVE_MODEL_PROFILE = os.getenv("CSES_MODEL_PROFILE", DEFAULT_PROFILE_ID)
_PROFILE = get_profile(ACTIVE_MODEL_PROFILE)

_ensemble_override = os.getenv("CSES_MATCH_ENSEMBLE_MODELS", "").strip()
ENSEMBLE_MODELS = [
    item.strip()
    for item in _ensemble_override.split(",")
    if item.strip()
] or list(_PROFILE.ensemble_models)

# Quorum threshold (2/3 majority = 66.67%, so use 0.65 to allow it)
QUORUM_THRESHOLD = 0.65

# =============================================================================
# STAGE-SPECIFIC MODEL CONFIGURATION
# =============================================================================

# Model for document quality assessment (fast, cheap)
LLM_MODEL_QUALITY = _PROFILE.quality_model

# Model for preprocessing (fast extraction)
LLM_MODEL_PREPROCESS = os.getenv("CSES_LARGE_TEXT_MODEL") or os.getenv("CSES_STUDY_KB_MODEL") or _PROFILE.preprocess_model

# Model for variable matching (if not using ensemble)
LLM_MODEL_MATCH = os.getenv("CSES_MATCH_FAST_MODEL") or _PROFILE.match_model

# Model for recoding strategy generation
LLM_MODEL_RECODE = os.getenv("CSES_STATA_CODE_MODEL") or _PROFILE.recode_model

# Model for Stata code generation
# NOTE: Using gpt-4.1 instead of gpt-5 due to reliability issues
LLM_MODEL_CODE = os.getenv("CSES_STATA_CODE_MODEL") or _PROFILE.code_model

# Model for validation and conversation
LLM_MODEL_VALIDATE = os.getenv("CSES_AGENTIC_MODEL") or _PROFILE.conversation_model

# Specialized: PDF extraction
LLM_MODEL_PDF = "openai/pdf-extractor"

# =============================================================================
# GENERAL SETTINGS
# =============================================================================

# Temperature for all LLM calls
# Note: GPT-5 models only support temperature=1, so we use 1 for compatibility
# Other models may use 0 for deterministic outputs
LLM_TEMPERATURE = 1

# Maximum iterations for automated debug loop
MAX_DEBUG_ITERATIONS = 5
