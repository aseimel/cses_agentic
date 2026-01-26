"""
CSES Party Code Module.

Implements CSES party ordering conventions:
- Parties A-F are ordered by DESCENDING vote share
- Party A = highest vote share, Party B = second highest, etc.
- Minimum threshold: 1% of national vote
- Supplemental parties G-I have no fixed ordering

6-digit party code structure:
[XXX][Y][ZZ]
  |    |  |
  |    |  +-- Party sequence (01-99)
  |    +----- Study number (0 if only one study)
  +---------- UN country code (3 digits)

Examples:
- 756001 = Switzerland (756) + study 0 + party 01 (SVP)
- 276102 = Germany (276) + study 1 + party 02 (CDU/CSU)
"""

import logging
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from litellm import completion

logger = logging.getLogger(__name__)


# UN M49 Country Codes (ISO 3166-1 numeric)
# Source: https://unstats.un.org/unsd/methods/m49/m49alpha.htm
UN_COUNTRY_CODES = {
    # Europe
    "Albania": "008",
    "Andorra": "020",
    "Austria": "040",
    "Belarus": "112",
    "Belgium": "056",
    "Bosnia and Herzegovina": "070",
    "Bulgaria": "100",
    "Croatia": "191",
    "Cyprus": "196",
    "Czech Republic": "203",
    "Czechia": "203",
    "Denmark": "208",
    "Estonia": "233",
    "Finland": "246",
    "France": "250",
    "Germany": "276",
    "Greece": "300",
    "Hungary": "348",
    "Iceland": "352",
    "Ireland": "372",
    "Italy": "380",
    "Latvia": "428",
    "Liechtenstein": "438",
    "Lithuania": "440",
    "Luxembourg": "442",
    "Malta": "470",
    "Moldova": "498",
    "Monaco": "492",
    "Montenegro": "499",
    "Netherlands": "528",
    "North Macedonia": "807",
    "Norway": "578",
    "Poland": "616",
    "Portugal": "620",
    "Romania": "642",
    "Russia": "643",
    "Russian Federation": "643",
    "San Marino": "674",
    "Serbia": "688",
    "Slovakia": "703",
    "Slovenia": "705",
    "Spain": "724",
    "Sweden": "752",
    "Switzerland": "756",
    "Ukraine": "804",
    "United Kingdom": "826",
    "UK": "826",
    "Great Britain": "826",

    # Americas
    "Argentina": "032",
    "Bolivia": "068",
    "Brazil": "076",
    "Canada": "124",
    "Chile": "152",
    "Colombia": "170",
    "Costa Rica": "188",
    "Cuba": "192",
    "Dominican Republic": "214",
    "Ecuador": "218",
    "El Salvador": "222",
    "Guatemala": "320",
    "Haiti": "332",
    "Honduras": "340",
    "Jamaica": "388",
    "Mexico": "484",
    "Nicaragua": "558",
    "Panama": "591",
    "Paraguay": "600",
    "Peru": "604",
    "Puerto Rico": "630",
    "Trinidad and Tobago": "780",
    "United States": "840",
    "USA": "840",
    "Uruguay": "858",
    "Venezuela": "862",

    # Asia
    "Afghanistan": "004",
    "Bangladesh": "050",
    "Bhutan": "064",
    "Cambodia": "116",
    "China": "156",
    "Hong Kong": "344",
    "India": "356",
    "Indonesia": "360",
    "Iran": "364",
    "Iraq": "368",
    "Israel": "376",
    "Japan": "392",
    "Jordan": "400",
    "Kazakhstan": "398",
    "Kuwait": "414",
    "Kyrgyzstan": "417",
    "Lebanon": "422",
    "Malaysia": "458",
    "Mongolia": "496",
    "Myanmar": "104",
    "Nepal": "524",
    "North Korea": "408",
    "Pakistan": "586",
    "Philippines": "608",
    "Saudi Arabia": "682",
    "Singapore": "702",
    "South Korea": "410",
    "Korea": "410",
    "Republic of Korea": "410",
    "Sri Lanka": "144",
    "Syria": "760",
    "Taiwan": "158",
    "Tajikistan": "762",
    "Thailand": "764",
    "Turkey": "792",
    "Turkmenistan": "795",
    "United Arab Emirates": "784",
    "UAE": "784",
    "Uzbekistan": "860",
    "Vietnam": "704",
    "Yemen": "887",

    # Africa
    "Algeria": "012",
    "Angola": "024",
    "Benin": "204",
    "Botswana": "072",
    "Burkina Faso": "854",
    "Cameroon": "120",
    "Egypt": "818",
    "Ethiopia": "231",
    "Ghana": "288",
    "Kenya": "404",
    "Libya": "434",
    "Madagascar": "450",
    "Malawi": "454",
    "Mali": "466",
    "Morocco": "504",
    "Mozambique": "508",
    "Namibia": "516",
    "Niger": "562",
    "Nigeria": "566",
    "Rwanda": "646",
    "Senegal": "686",
    "South Africa": "710",
    "Sudan": "729",
    "Tanzania": "834",
    "Tunisia": "788",
    "Uganda": "800",
    "Zambia": "894",
    "Zimbabwe": "716",

    # Oceania
    "Australia": "036",
    "Fiji": "242",
    "New Zealand": "554",
    "Papua New Guinea": "598",
}

# ISO 3166-1 alpha-3 to UN M49 mapping
COUNTRY_CODE_ALPHA3 = {
    "ALB": "008", "AND": "020", "AUT": "040", "BLR": "112", "BEL": "056",
    "BIH": "070", "BGR": "100", "HRV": "191", "CYP": "196", "CZE": "203",
    "DNK": "208", "EST": "233", "FIN": "246", "FRA": "250", "DEU": "276",
    "GRC": "300", "HUN": "348", "ISL": "352", "IRL": "372", "ITA": "380",
    "LVA": "428", "LIE": "438", "LTU": "440", "LUX": "442", "MLT": "470",
    "MDA": "498", "MCO": "492", "MNE": "499", "NLD": "528", "MKD": "807",
    "NOR": "578", "POL": "616", "PRT": "620", "ROU": "642", "RUS": "643",
    "SMR": "674", "SRB": "688", "SVK": "703", "SVN": "705", "ESP": "724",
    "SWE": "752", "CHE": "756", "UKR": "804", "GBR": "826",
    # Americas
    "ARG": "032", "BOL": "068", "BRA": "076", "CAN": "124", "CHL": "152",
    "COL": "170", "CRI": "188", "CUB": "192", "DOM": "214", "ECU": "218",
    "SLV": "222", "GTM": "320", "HTI": "332", "HND": "340", "JAM": "388",
    "MEX": "484", "NIC": "558", "PAN": "591", "PRY": "600", "PER": "604",
    "PRI": "630", "TTO": "780", "USA": "840", "URY": "858", "VEN": "862",
    # Asia
    "AFG": "004", "BGD": "050", "BTN": "064", "KHM": "116", "CHN": "156",
    "HKG": "344", "IND": "356", "IDN": "360", "IRN": "364", "IRQ": "368",
    "ISR": "376", "JPN": "392", "JOR": "400", "KAZ": "398", "KWT": "414",
    "KGZ": "417", "LBN": "422", "MYS": "458", "MNG": "496", "MMR": "104",
    "NPL": "524", "PRK": "408", "PAK": "586", "PHL": "608", "SAU": "682",
    "SGP": "702", "KOR": "410", "LKA": "144", "SYR": "760", "TWN": "158",
    "TJK": "762", "THA": "764", "TUR": "792", "TKM": "795", "ARE": "784",
    "UZB": "860", "VNM": "704", "YEM": "887",
    # Africa
    "DZA": "012", "AGO": "024", "BEN": "204", "BWA": "072", "BFA": "854",
    "CMR": "120", "EGY": "818", "ETH": "231", "GHA": "288", "KEN": "404",
    "LBY": "434", "MDG": "450", "MWI": "454", "MLI": "466", "MAR": "504",
    "MOZ": "508", "NAM": "516", "NER": "562", "NGA": "566", "RWA": "646",
    "SEN": "686", "ZAF": "710", "SDN": "729", "TZA": "834", "TUN": "788",
    "UGA": "800", "ZMB": "894", "ZWE": "716",
    # Oceania
    "AUS": "036", "FJI": "242", "NZL": "554", "PNG": "598",
}


@dataclass
class PartyInfo:
    """Information about a political party."""
    name: str
    vote_share: float  # As decimal (0.25 = 25%)
    code_letter: str = ""  # A, B, C, D, E, F, G, H, I
    numerical_code: str = ""  # 6-digit code like "756001"
    tier: int = 1  # Electoral tier (1 or 2)
    is_supplemental: bool = False  # True for parties G-I


@dataclass
class PartyOrderResult:
    """Result of party ordering operation."""
    parties: list[PartyInfo] = field(default_factory=list)
    country_code: str = ""
    study_number: int = 0
    electoral_system: str = ""  # "single-tier", "multi-tier-one-vote", "two-votes"
    ordering_basis: str = ""  # Description of how parties were ordered
    errors: list[str] = field(default_factory=list)

    def get_party_by_letter(self, letter: str) -> Optional[PartyInfo]:
        """Get party by its alphabetical code (A-I)."""
        for party in self.parties:
            if party.code_letter == letter:
                return party
        return None

    def get_party_mapping(self) -> dict[str, PartyInfo]:
        """Get mapping of letter codes to party info."""
        return {p.code_letter: p for p in self.parties if p.code_letter}

    def get_numerical_code_mapping(self) -> dict[str, str]:
        """Get mapping of party names to 6-digit codes."""
        return {p.name: p.numerical_code for p in self.parties if p.numerical_code}


def get_un_country_code(country: str) -> Optional[str]:
    """
    Get UN M49 country code from country name or ISO alpha-3 code.

    Args:
        country: Country name (e.g., "Switzerland") or ISO alpha-3 code (e.g., "CHE")

    Returns:
        3-digit UN country code or None if not found
    """
    # Try direct lookup by name (case-insensitive)
    country_lower = country.lower()
    for name, code in UN_COUNTRY_CODES.items():
        if name.lower() == country_lower:
            return code

    # Try ISO alpha-3 lookup (uppercase)
    country_upper = country.upper()
    if country_upper in COUNTRY_CODE_ALPHA3:
        return COUNTRY_CODE_ALPHA3[country_upper]

    # Try partial match
    for name, code in UN_COUNTRY_CODES.items():
        if country_lower in name.lower() or name.lower() in country_lower:
            return code

    return None


def generate_party_code(
    country_un_code: str,
    study_number: int,
    party_sequence: int
) -> str:
    """
    Generate 6-digit CSES party code.

    Args:
        country_un_code: 3-digit UN country code (e.g., "756")
        study_number: Study number within country for same election (usually 0)
        party_sequence: Party sequence number (01-99)

    Returns:
        6-digit party code (e.g., "756001")
    """
    # Ensure country code is 3 digits
    country_un_code = str(country_un_code).zfill(3)

    # Ensure party sequence is 2 digits
    party_seq_str = str(party_sequence).zfill(2)

    return f"{country_un_code}{study_number}{party_seq_str}"


def order_parties_by_vote_share(
    parties: list[dict],
    min_threshold: float = 0.01,
    max_main_parties: int = 6
) -> list[PartyInfo]:
    """
    Order parties by descending vote share, assigning A-F to top parties.

    Args:
        parties: List of dicts with 'name', 'vote_share', optionally 'tier'
        min_threshold: Minimum vote share threshold (default 1%)
        max_main_parties: Maximum number of main parties A-F (default 6)

    Returns:
        List of PartyInfo objects with code letters assigned
    """
    # Filter parties meeting threshold
    eligible = [
        p for p in parties
        if p.get('vote_share', 0) >= min_threshold
    ]

    # Sort by vote share descending
    eligible.sort(key=lambda x: x.get('vote_share', 0), reverse=True)

    result = []
    letters = ['A', 'B', 'C', 'D', 'E', 'F']
    supplemental_letters = ['G', 'H', 'I']

    # Assign main party codes (A-F)
    for i, party in enumerate(eligible[:max_main_parties]):
        if i < len(letters):
            info = PartyInfo(
                name=party.get('name', ''),
                vote_share=party.get('vote_share', 0),
                code_letter=letters[i],
                tier=party.get('tier', 1),
                is_supplemental=False
            )
            result.append(info)

    # Remaining parties can be assigned supplemental codes (G-I)
    remaining = eligible[max_main_parties:]
    for i, party in enumerate(remaining[:3]):
        info = PartyInfo(
            name=party.get('name', ''),
            vote_share=party.get('vote_share', 0),
            code_letter=supplemental_letters[i],
            tier=party.get('tier', 1),
            is_supplemental=True
        )
        result.append(info)

    return result


def assign_numerical_codes(
    parties: list[PartyInfo],
    country_un_code: str,
    study_number: int = 0
) -> list[PartyInfo]:
    """
    Assign 6-digit numerical codes to parties.

    Parties are numbered sequentially starting from 01.

    Args:
        parties: List of PartyInfo objects
        country_un_code: 3-digit UN country code
        study_number: Study number (usually 0)

    Returns:
        Updated list with numerical_code field populated
    """
    for i, party in enumerate(parties):
        party.numerical_code = generate_party_code(
            country_un_code,
            study_number,
            i + 1  # Start from 01
        )
    return parties


def parse_party_data_with_llm(
    file_content: str,
    model: Optional[str] = None
) -> list[dict]:
    """
    Use LLM to semantically extract party information from any format.

    This follows the project philosophy: no format-specific parsing code.
    The LLM interprets whatever document structure is provided.

    Args:
        file_content: Raw text content from any document format
        model: LLM model to use (defaults to LLM_MODEL_MATCH or LLM_MODEL)

    Returns:
        List of dicts with 'name', 'vote_share', optionally 'tier'
    """
    model = model or os.getenv("LLM_MODEL_MATCH") or os.getenv("LLM_MODEL", "openai/gpt-oss:120b")

    print("Analyzing document for party and election results...")

    prompt = f"""Extract political party election results from this document.

DOCUMENT CONTENT:
{file_content[:30000]}

TASK: Find all political parties mentioned with their vote shares.

Return JSON in this EXACT format:
{{"parties": [
    {{"name": "Party Full Name", "vote_share": 0.25, "tier": 1}},
    {{"name": "Another Party", "vote_share": 0.18, "tier": 1}}
]}}

RULES:
1. Extract the PARTY NAME exactly as written
2. Convert vote percentages to decimals (25% -> 0.25)
3. If multiple tiers exist (e.g., constituency vs list votes), note the tier
4. Include ALL parties with at least 0.5% vote share
5. Sort by vote share descending in your output
6. Return ONLY valid JSON, no other text

If you cannot find election results, return: {{"parties": [], "error": "No election results found"}}
"""

    try:
        response = completion(
            model=model,
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = response.choices[0].message.content.strip()

        # Extract JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            result = json.loads(json_match.group(0))
            parties = result.get('parties', [])

            if parties:
                logger.info(f"Extracted {len(parties)} parties from document")
                return parties
            elif result.get('error'):
                logger.warning(f"LLM reported: {result['error']}")
                return []

        logger.warning("Could not parse party data from LLM response")
        return []

    except Exception as e:
        logger.error(f"Error parsing party data: {e}")
        return []


def generate_party_codes(
    country: str,
    study_number: int,
    parties_by_vote_share: list[tuple[str, float]],
    electoral_system: str = "single-tier"
) -> PartyOrderResult:
    """
    Generate complete party codes for a country study.

    This is the main entry point for party code generation.

    Args:
        country: Country name or ISO alpha-3 code
        study_number: Study number (usually 0)
        parties_by_vote_share: List of (party_name, vote_share) tuples
        electoral_system: "single-tier", "multi-tier-one-vote", or "two-votes"

    Returns:
        PartyOrderResult with all party information

    Example:
        >>> result = generate_party_codes(
        ...     country="Switzerland",
        ...     study_number=0,
        ...     parties_by_vote_share=[
        ...         ("SVP", 0.279),
        ...         ("SP", 0.183),
        ...         ("FDP", 0.147),
        ...         ("Die Mitte", 0.141),
        ...         ("Gruene", 0.097),
        ...         ("GLP", 0.077),
        ...     ]
        ... )
        >>> result.parties[0].name
        'SVP'
        >>> result.parties[0].code_letter
        'A'
        >>> result.parties[0].numerical_code
        '756001'
    """
    result = PartyOrderResult()
    result.study_number = study_number
    result.electoral_system = electoral_system

    # Get UN country code
    un_code = get_un_country_code(country)
    if not un_code:
        result.errors.append(f"Unknown country: {country}")
        return result

    result.country_code = un_code

    # Determine ordering basis based on electoral system
    if electoral_system == "single-tier":
        result.ordering_basis = "Vote share in single-tier system"
    elif electoral_system == "multi-tier-one-vote":
        result.ordering_basis = "Tier 1 (lower tier) vote share"
    elif electoral_system == "two-votes":
        result.ordering_basis = "Tier 2 (national list) vote share"
    else:
        result.ordering_basis = "Vote share"

    # Convert input to party dicts
    parties = [
        {"name": name, "vote_share": share}
        for name, share in parties_by_vote_share
    ]

    # Order parties and assign letter codes
    ordered_parties = order_parties_by_vote_share(parties)

    # Assign numerical codes
    ordered_parties = assign_numerical_codes(
        ordered_parties, un_code, study_number
    )

    result.parties = ordered_parties
    return result


def extract_party_results_from_macro(
    working_dir: Path,
    model: Optional[str] = None
) -> Optional[list[dict]]:
    """
    Extract party results from macro folder files.

    Looks for election results in macro/ folder and parses with LLM.

    Args:
        working_dir: Working directory containing macro/ folder
        model: LLM model to use

    Returns:
        List of party dicts or None if not found
    """
    macro_dir = working_dir / "macro"
    if not macro_dir.exists():
        logger.info("No macro/ folder found")
        return None

    # Look for files that might contain election results
    result_patterns = [
        "*election*", "*result*", "*party*", "*vote*",
        "*.xlsx", "*.csv", "*.txt", "*.docx", "*.pdf"
    ]

    for pattern in result_patterns:
        files = list(macro_dir.glob(pattern))
        for file_path in files:
            try:
                # Read file content
                if file_path.suffix.lower() in ['.txt', '.csv']:
                    content = file_path.read_text(encoding='utf-8', errors='replace')
                elif file_path.suffix.lower() == '.xlsx':
                    import pandas as pd
                    df = pd.read_excel(file_path)
                    content = df.to_string()
                elif file_path.suffix.lower() == '.docx':
                    from src.ingest.doc_parser import DocumentParser
                    parser = DocumentParser()
                    doc_info = parser.parse(file_path)
                    content = doc_info.full_text if doc_info else ""
                elif file_path.suffix.lower() == '.pdf':
                    from src.ingest.doc_parser import DocumentParser
                    parser = DocumentParser()
                    doc_info = parser.parse(file_path)
                    content = doc_info.full_text if doc_info else ""
                else:
                    continue

                if content and len(content) > 100:
                    parties = parse_party_data_with_llm(content, model)
                    if parties:
                        logger.info(f"Found party data in {file_path.name}")
                        return parties

            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
                continue

    return None


def prompt_for_party_data() -> str:
    """
    Generate prompt text for user to provide party data.

    Returns:
        Prompt message string
    """
    return """
Party ordering should be coordinated with the macro coder.

To assign party codes (A-F), I need election results with vote shares.

Please provide one of the following:
1. A file with election results (Excel, Word, PDF, or text)
2. Copy-pasted election results table
3. The party order confirmed by the macro coder

Example format:
  Party Name          Vote Share
  Swiss People's Party   27.9%
  Social Democrats       18.3%
  FDP                    14.7%
  The Center             14.1%
  Green Party             9.7%
  Green Liberals          7.7%

Have you discussed the party order with the macro coder?
"""


def generate_party_code_stata_section(
    party_result: PartyOrderResult,
    source_party_var: str = "Q11_party",
    source_value_mapping: Optional[dict[int, str]] = None
) -> str:
    """
    Generate Stata code for party variable assignments.

    Creates the F3011 and related party variable sections for .do file.

    Args:
        party_result: PartyOrderResult with party codes
        source_party_var: Name of source party variable
        source_value_mapping: Optional mapping of source values to party names

    Returns:
        Stata code string
    """
    lines = []

    # Header
    lines.append("")
    lines.append("***************************************************************************")
    lines.append("**>>> PARTY CODES - NUMERICAL (6-DIGIT)")
    lines.append("***************************************************************************")
    lines.append("")
    lines.append(f"* Country code: {party_result.country_code}")
    lines.append(f"* Study number: {party_result.study_number}")
    lines.append(f"* Ordering basis: {party_result.ordering_basis}")
    lines.append("")

    # Party mapping documentation
    lines.append("/*")
    lines.append("Party Code Mapping:")
    for party in party_result.parties:
        if party.code_letter:
            lines.append(f"  {party.numerical_code} = PARTY {party.code_letter}: {party.name} ({party.vote_share*100:.1f}%)")
    lines.append("*/")
    lines.append("")

    # Generate replace statements if mapping provided
    if source_value_mapping:
        lines.append(f"* Recode {source_party_var} to 6-digit CSES party codes")
        lines.append(f"gen long F3011_LH_PL = .")
        lines.append("")

        for source_val, party_name in source_value_mapping.items():
            # Find matching party
            for party in party_result.parties:
                if party.name.lower() in party_name.lower() or party_name.lower() in party.name.lower():
                    lines.append(f"replace F3011_LH_PL = {party.numerical_code} if {source_party_var} == {source_val}")
                    break

        lines.append("")
        lines.append("tab F3011_LH_PL, mis")

    # Alphabetical party variables (F3018_A-I, F3019_A-I)
    lines.append("")
    lines.append("***************************************************************************")
    lines.append("**>>> PARTY CODES - ALPHABETICAL (A-I)")
    lines.append("***************************************************************************")
    lines.append("")
    lines.append("/*")
    lines.append("Alphabetical Party Codes (by vote share):")
    for party in party_result.parties:
        if party.code_letter:
            lines.append(f"  PARTY {party.code_letter}: {party.name}")
    lines.append("*/")
    lines.append("")

    return "\n".join(lines)


# Expose key functions at module level
__all__ = [
    'UN_COUNTRY_CODES',
    'COUNTRY_CODE_ALPHA3',
    'PartyInfo',
    'PartyOrderResult',
    'get_un_country_code',
    'generate_party_code',
    'order_parties_by_vote_share',
    'assign_numerical_codes',
    'parse_party_data_with_llm',
    'generate_party_codes',
    'extract_party_results_from_macro',
    'prompt_for_party_data',
    'generate_party_code_stata_section',
]
