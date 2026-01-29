# LLM-based variable matching module
import logging

logger = logging.getLogger(__name__)

# LLM matcher - requires litellm
try:
    from .llm_matcher import (
        LLMMatcher,
        MatchProposal,
        MatchingResult,
        PatternMatcher,
        create_matcher,
    )
except ImportError as e:
    logger.warning(f"LLM matcher not available: {e}")
    LLMMatcher = None
    MatchProposal = None
    MatchingResult = None
    PatternMatcher = None
    create_matcher = None

# Party codes - should work without external deps
try:
    from .party_codes import (
        UN_COUNTRY_CODES,
        COUNTRY_CODE_ALPHA3,
        PartyInfo,
        PartyOrderResult,
        get_un_country_code,
        generate_party_code,
        order_parties_by_vote_share,
        assign_numerical_codes,
        parse_party_data_with_llm,
        generate_party_codes,
        extract_party_results_from_macro,
        prompt_for_party_data,
        generate_party_code_stata_section,
    )
except ImportError as e:
    logger.warning(f"Party codes module not available: {e}")
    UN_COUNTRY_CODES = {}
    COUNTRY_CODE_ALPHA3 = {}
    PartyInfo = None
    PartyOrderResult = None
    get_un_country_code = None
    generate_party_code = None
    order_parties_by_vote_share = None
    assign_numerical_codes = None
    parse_party_data_with_llm = None
    generate_party_codes = None
    extract_party_results_from_macro = None
    prompt_for_party_data = None
    generate_party_code_stata_section = None

__all__ = [
    # LLM Matcher
    "LLMMatcher",
    "MatchProposal",
    "MatchingResult",
    "PatternMatcher",
    "create_matcher",
    # Party Codes
    "UN_COUNTRY_CODES",
    "COUNTRY_CODE_ALPHA3",
    "PartyInfo",
    "PartyOrderResult",
    "get_un_country_code",
    "generate_party_code",
    "order_parties_by_vote_share",
    "assign_numerical_codes",
    "parse_party_data_with_llm",
    "generate_party_codes",
    "extract_party_results_from_macro",
    "prompt_for_party_data",
    "generate_party_code_stata_section",
]
