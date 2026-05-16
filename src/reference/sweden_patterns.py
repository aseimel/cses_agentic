"""
CSES reference patterns for Stata code generation.

Runtime code loads compact, distilled examples from cses_wiki/patterns. The
legacy raw-file parser remains as a development fallback only, so installed
users do not need raw example study folders.

Patterns extracted:
- Variable header format: **>>> VARIABLE - DESCRIPTION
- Coding notes with multiple asterisks
- gen/recode/replace patterns
- tab VARIABLE, mis verification
- Cross-tabulation for verification
- Missing value handling
"""

import logging
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VariablePattern:
    """Extracted pattern for a single CSES variable from reference file."""
    variable: str
    description: str
    full_code: str
    source_var: Optional[str] = None
    has_recode: bool = False
    has_crosstab: bool = False
    coding_notes: str = ""
    pattern_type: str = "gen"  # gen, recode, replace, string, calculate


@dataclass
class SwedenPatterns:
    """Collection of patterns extracted from Sweden .do file."""
    variables: dict[str, VariablePattern] = field(default_factory=dict)
    header_template: str = ""
    section_headers: list[str] = field(default_factory=list)
    file_path: Optional[str] = None

    def get_examples_for_category(self, category: str) -> list[VariablePattern]:
        """Get example patterns for a variable category (demographics, survey, etc.)."""
        examples = []
        for var, pattern in self.variables.items():
            if category == "demographics" and var.startswith("F2"):
                examples.append(pattern)
            elif category == "survey" and var.startswith("F3"):
                examples.append(pattern)
            elif category == "admin" and var.startswith("F1"):
                examples.append(pattern)
        return examples[:5]  # Return up to 5 examples

    def get_examples_for_pattern_type(self, pattern_type: str) -> list[VariablePattern]:
        """Get examples that use a specific pattern type (recode, calculate, etc.)."""
        return [p for p in self.variables.values() if p.pattern_type == pattern_type][:3]


class SwedenPatternLoader:
    """Loads distilled reference patterns, with a legacy raw-file fallback."""

    # Pattern to match variable sections
    VARIABLE_PATTERN = re.compile(
        r'\*+\s*\n?\*+\s*>>>\s*(F\d+[_A-Z0-9]*)\s*[-–]\s*([^\n*]+)',
        re.MULTILINE
    )

    # Pattern to extract source variable from gen command
    SOURCE_VAR_PATTERN = re.compile(r'gen\s+\w+\s*=\s*(\w+)')

    def __init__(self, reference_dir: Optional[Path] = None):
        """
        Initialize the pattern loader.

        Args:
            reference_dir: Directory containing reference .do files.
                         If None, looks for Sweden_2022 in project root.
        """
        self.reference_dir = reference_dir
        self._patterns_cache: Optional[SwedenPatterns] = None

    def load_patterns(self, do_file_path: Optional[Path] = None) -> SwedenPatterns:
        """
        Load patterns from the CSES wiki or, for development, a raw .do file.

        Args:
            do_file_path: Explicit path to .do file. If None, auto-detects.

        Returns:
            SwedenPatterns with extracted examples
        """
        if self._patterns_cache is not None:
            return self._patterns_cache

        if do_file_path is None:
            distilled = self._load_distilled_patterns()
            if distilled.variables:
                self._patterns_cache = distilled
                return distilled

        # Find the .do file
        if do_file_path is None:
            do_file_path = self._find_sweden_do_file()

        if do_file_path is None or not do_file_path.exists():
            logger.warning("Sweden reference .do file not found")
            return SwedenPatterns()

        logger.info(f"Loading patterns from: {do_file_path}")

        try:
            content = do_file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Failed to read .do file: {e}")
            return SwedenPatterns()

        patterns = self._parse_do_file(content)
        patterns.file_path = str(do_file_path)
        self._patterns_cache = patterns

        logger.info(f"Loaded {len(patterns.variables)} variable patterns")
        return patterns

    def _load_distilled_patterns(self) -> SwedenPatterns:
        """Load compact examples from cses_wiki/patterns/stata_syntax.json."""
        project_root = Path(__file__).parent.parent.parent
        pattern_path = project_root / "cses_wiki" / "patterns" / "stata_syntax.json"
        if not pattern_path.exists():
            return SwedenPatterns()

        try:
            data = json.loads(pattern_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not load distilled CSES syntax patterns: {e}")
            return SwedenPatterns()

        patterns = SwedenPatterns(
            header_template="",
            section_headers=data.get("common_section_order", []),
            file_path=str(pattern_path),
        )

        for example in data.get("examples", []):
            if example.get("artifact_type") != "micro_syntax":
                continue
            if not patterns.header_template and example.get("header_excerpt"):
                patterns.header_template = example["header_excerpt"]
            for item in example.get("variable_code_examples", []):
                variable = str(item.get("variable", ""))
                if not variable or variable in patterns.variables:
                    continue
                code = str(item.get("code_excerpt", ""))
                patterns.variables[variable] = VariablePattern(
                    variable=variable,
                    description=str(item.get("description", "")),
                    full_code=code,
                    has_recode=str(item.get("pattern_type", "")) in {"recode", "replace"},
                    has_crosstab=bool(re.search(r"\btab\s+\w+\s+" + re.escape(variable), code)),
                    pattern_type=str(item.get("pattern_type", "gen")) or "gen",
                )
        return patterns

    def _find_sweden_do_file(self) -> Optional[Path]:
        """Find the Sweden .do file in common locations."""
        # Try common locations
        search_paths = []

        if self.reference_dir:
            search_paths.append(self.reference_dir)

        # Project root relative paths
        project_root = Path(__file__).parent.parent.parent
        search_paths.extend([
            project_root / "Sweden_2022" / "micro",
            project_root / "Sweden_2022",
            Path.cwd() / "Sweden_2022" / "micro",
            Path.cwd() / "Sweden_2022",
        ])

        for search_dir in search_paths:
            if not search_dir.exists():
                continue

            # Look for the main processing .do file
            for do_file in search_dir.glob("cses-m6_micro_SWE_*.do"):
                if "_OLD" not in str(do_file):
                    return do_file

        return None

    def _parse_do_file(self, content: str) -> SwedenPatterns:
        """Parse .do file content and extract patterns."""
        patterns = SwedenPatterns()

        # Extract header template (first comment block)
        header_match = re.search(r'^/\*+[\s\S]*?\*/', content)
        if header_match:
            patterns.header_template = header_match.group(0)

        # Extract section headers
        section_matches = re.findall(r'\*+\s*\\\\\\\s*([^\n]+)', content)
        patterns.section_headers = section_matches

        # Split content by variable sections
        # Find all variable headers
        variable_sections = self._split_by_variables(content)

        for var_name, section_content in variable_sections.items():
            pattern = self._parse_variable_section(var_name, section_content)
            if pattern:
                patterns.variables[var_name] = pattern

        return patterns

    def _split_by_variables(self, content: str) -> dict[str, str]:
        """Split content into sections by variable."""
        sections = {}

        # Find all variable header positions
        matches = list(self.VARIABLE_PATTERN.finditer(content))

        for i, match in enumerate(matches):
            var_name = match.group(1)
            start_pos = match.start()

            # End position is the start of the next variable section or end of content
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(content)

            section_content = content[start_pos:end_pos]
            sections[var_name] = section_content

        return sections

    def _parse_variable_section(self, var_name: str, content: str) -> Optional[VariablePattern]:
        """Parse a single variable section."""
        # Extract description from header
        header_match = self.VARIABLE_PATTERN.search(content)
        if not header_match:
            return None

        description = header_match.group(2).strip()

        # Extract coding notes (comment blocks with **)
        coding_notes = ""
        notes_match = re.search(r'\*\*\s*Coding\s*Note[s]?:?\s*\n([\s\S]*?)(?=\n\s*(?:gen|recode|replace|tab|\*))', content, re.IGNORECASE)
        if notes_match:
            coding_notes = notes_match.group(1).strip()

        # Determine pattern type and extract source variable
        source_var = None
        pattern_type = "gen"
        has_recode = False
        has_crosstab = False

        if re.search(r'\brecode\s+', content):
            pattern_type = "recode"
            has_recode = True
        elif re.search(r'\breplace\s+', content):
            pattern_type = "replace"
            has_recode = True
        elif re.search(r'gen\s+str', content):
            pattern_type = "string"

        # Extract source variable
        source_match = re.search(r'(?:gen|recode)\s+(?:str\d*\s+)?(\w+)\s*=\s*(\w+)', content)
        if source_match and source_match.group(1) == var_name:
            source_var = source_match.group(2)

        # Check for cross-tabulation
        if re.search(rf'tab\s+\w+\s+{var_name}', content):
            has_crosstab = True

        return VariablePattern(
            variable=var_name,
            description=description,
            full_code=content.strip(),
            source_var=source_var,
            has_recode=has_recode,
            has_crosstab=has_crosstab,
            coding_notes=coding_notes,
            pattern_type=pattern_type
        )


def load_sweden_patterns(do_file_path: Optional[Path] = None) -> SwedenPatterns:
    """
    Load Sweden reference patterns.

    Args:
        do_file_path: Optional explicit path to .do file

    Returns:
        SwedenPatterns with extracted examples
    """
    loader = SwedenPatternLoader()
    return loader.load_patterns(do_file_path)


def get_pattern_for_variable(target_var: str, patterns: Optional[SwedenPatterns] = None) -> str:
    """
    Get a relevant reference pattern for a target variable.

    Args:
        target_var: CSES target variable code (e.g., "F2002")
        patterns: Optional pre-loaded patterns. If None, loads automatically.

    Returns:
        String with example Stata code pattern, or empty string if none found
    """
    if patterns is None:
        patterns = load_sweden_patterns()

    # Direct match
    if target_var in patterns.variables:
        return patterns.variables[target_var].full_code

    # Find similar variable (same prefix)
    prefix = re.match(r'F\d+', target_var)
    if prefix:
        prefix_str = prefix.group(0)
        for var, pattern in patterns.variables.items():
            if var.startswith(prefix_str):
                return pattern.full_code

    # Return a generic example based on variable type
    if target_var.startswith("F2"):
        # Demographics
        examples = patterns.get_examples_for_category("demographics")
        if examples:
            return examples[0].full_code
    elif target_var.startswith("F3"):
        # Survey questions
        examples = patterns.get_examples_for_category("survey")
        if examples:
            return examples[0].full_code

    return ""


def get_examples_for_code_generation(target_var: str, recoding_type: str = "gen") -> str:
    """
    Get formatted examples for LLM code generation prompt.

    Args:
        target_var: CSES target variable code
        recoding_type: Type of recoding needed (gen, recode, replace, calculate)

    Returns:
        Formatted string with example patterns for the LLM prompt
    """
    patterns = load_sweden_patterns()

    if not patterns.variables:
        return "# No reference patterns available"

    examples = []

    # Get examples of the requested pattern type
    type_examples = patterns.get_examples_for_pattern_type(recoding_type)
    for ex in type_examples[:2]:
        examples.append(f"Example ({ex.pattern_type}):\n{ex.full_code[:800]}")

    # Get similar variable examples
    similar = get_pattern_for_variable(target_var, patterns)
    if similar and similar not in [e.full_code for e in type_examples[:2]]:
        examples.append(f"Similar variable example:\n{similar[:800]}")

    if examples:
        return "\n\n---\n\n".join(examples)

    return "# No relevant reference patterns found"
