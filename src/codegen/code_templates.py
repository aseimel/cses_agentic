"""
Stata code templates for CSES variable transformations.

Each transformation type has a corresponding template that generates
deterministic Stata code. Templates follow the Sweden reference .do file patterns.
"""

import re
from dataclasses import dataclass
from typing import Optional

from .sheet_reader import VariableMapping, RecodeRule


@dataclass
class GeneratedCode:
    """Generated Stata code for a variable."""
    cses_var: str
    code: str
    has_verification: bool = True


class StataTemplates:
    """
    Stata code templates for different transformation types.

    All templates follow CSES conventions:
    - Variable header with >>> marker for navigation
    - gen/recode/replace commands
    - tab command for verification
    - Cross-tabulation for recoded variables
    """

    # Variable header template
    HEADER_TEMPLATE = """
***************************************************************************
**>>> {cses_var}  {cses_desc}
***************************************************************************
"""

    # Missing codes by variable type
    MISSING_CODES = {
        # 1-digit missing (demographics, simple scales)
        "single": {"refused": "7", "dk": "8", "missing": "9"},
        # 2-digit missing (most survey questions)
        "double": {"refused": "97", "dk": "98", "missing": "99"},
        # 3-digit missing (occupation, detailed codes)
        "triple": {"refused": "997", "dk": "998", "missing": "999"},
        # 4-digit missing (year of birth, income)
        "quad": {"refused": "9997", "dk": "9998", "missing": "9999"},
    }

    def __init__(self):
        pass

    def get_missing_type(self, cses_var: str) -> str:
        """Determine missing code type based on variable."""
        # Year variables use 4-digit missing
        if cses_var in ("F2001_Y", "F1009", "F1010_Y", "F1011_Y"):
            return "quad"
        # Age uses 4-digit missing
        if cses_var == "F2001_A":
            return "quad"
        # Occupation uses 3-digit missing
        if cses_var == "F2007":
            return "triple"
        # Demographics (F2xxx) generally use single digit
        if cses_var.startswith("F2") and "_" not in cses_var:
            return "single"
        # Survey questions (F3xxx) generally use double digit
        if cses_var.startswith("F3"):
            return "double"
        # Default to double
        return "double"

    def generate_header(self, mapping: VariableMapping) -> str:
        """Generate variable header comment."""
        desc = mapping.cses_desc.upper() if mapping.cses_desc else mapping.cses_var
        return self.HEADER_TEMPLATE.format(
            cses_var=mapping.cses_var,
            cses_desc=desc
        )

    def generate_direct(self, mapping: VariableMapping) -> GeneratedCode:
        """
        Generate code for direct copy (no transformation).

        Pattern: gen TARGET = SOURCE
        """
        lines = [self.generate_header(mapping)]

        if mapping.notes:
            lines.append(f"* Note: {mapping.notes}")

        lines.append(f"gen {mapping.cses_var} = {mapping.source_var}")
        lines.append(f"tab {mapping.cses_var}, mis")

        return GeneratedCode(
            cses_var=mapping.cses_var,
            code="\n".join(lines),
            has_verification=True
        )

    def generate_recode(self, mapping: VariableMapping) -> GeneratedCode:
        """
        Generate code for recode transformation.

        Pattern: gen TARGET = SOURCE
                 recode TARGET (old=new) (old=new) ...
                 tab SOURCE TARGET, mis
        """
        lines = [self.generate_header(mapping)]

        if mapping.notes:
            lines.append(f"* Note: {mapping.notes}")

        # Generate variable
        lines.append(f"gen {mapping.cses_var} = {mapping.source_var}")

        # Build recode command
        if mapping.recode_rules or mapping.missing_rules:
            recode_parts = []

            # Add recode rules
            for rule in mapping.recode_rules:
                recode_parts.append(f"({rule.from_value}={rule.to_value})")

            # Add missing rules
            for rule in mapping.missing_rules:
                recode_parts.append(f"({rule.from_value}={rule.to_value})")

            if recode_parts:
                recode_cmd = f"recode {mapping.cses_var} " + " ".join(recode_parts)
                lines.append(recode_cmd)

        # Cross-tabulation for verification
        lines.append(f"tab {mapping.source_var} {mapping.cses_var}, mis")

        return GeneratedCode(
            cses_var=mapping.cses_var,
            code="\n".join(lines),
            has_verification=True
        )

    def generate_replace(self, mapping: VariableMapping) -> GeneratedCode:
        """
        Generate code for replace transformation (many-to-one mappings).

        Pattern: gen TARGET = .
                 replace TARGET = X if SOURCE == Y
                 replace TARGET = X if SOURCE == Z
                 ...
                 tab SOURCE TARGET, mis
        """
        lines = [self.generate_header(mapping)]

        if mapping.notes:
            lines.append(f"* Note: {mapping.notes}")

        # Initialize with missing
        missing_type = self.get_missing_type(mapping.cses_var)
        missing_val = self.MISSING_CODES[missing_type]["missing"]
        lines.append(f"gen {mapping.cses_var} = .")

        # Add replace statements for each rule
        for rule in mapping.recode_rules:
            comment = f" // {rule.label}" if rule.label else ""
            lines.append(f"replace {mapping.cses_var} = {rule.to_value} if {mapping.source_var} == {rule.from_value}{comment}")

        # Add missing value replacements
        for rule in mapping.missing_rules:
            comment = f" // {rule.label}" if rule.label else ""
            lines.append(f"replace {mapping.cses_var} = {rule.to_value} if {mapping.source_var} == {rule.from_value}{comment}")

        # Handle system missing if not explicitly mapped
        has_sysmiss = any(r.from_value == "." for r in mapping.missing_rules)
        if not has_sysmiss:
            lines.append(f"replace {mapping.cses_var} = {missing_val} if {mapping.source_var} == .")

        # Cross-tabulation for verification
        lines.append(f"tab {mapping.source_var} {mapping.cses_var}, mis")

        return GeneratedCode(
            cses_var=mapping.cses_var,
            code="\n".join(lines),
            has_verification=True
        )

    def generate_calculate(self, mapping: VariableMapping) -> GeneratedCode:
        """
        Generate code for calculated variables.

        This handles special cases like:
        - Age calculation from birth year
        - Arithmetic transformations (e.g., SOURCE - 1)

        Pattern depends on the specific calculation needed.
        """
        lines = [self.generate_header(mapping)]

        if mapping.notes:
            lines.append(f"* Note: {mapping.notes}")

        # Check for common calculation patterns
        source = mapping.source_var

        # Age calculation (F2001_A = election_year - F2001_Y)
        if mapping.cses_var == "F2001_A":
            lines.append(f"gen {mapping.cses_var} = F1009 - F2001_Y if F2001_Y < 9997")
            lines.append(f"replace {mapping.cses_var} = 9997 if F2001_Y == 9997")
            lines.append(f"replace {mapping.cses_var} = 9998 if F2001_Y == 9998")
            lines.append(f"replace {mapping.cses_var} = 9999 if F2001_Y == 9999")
            lines.append(f"tab {mapping.cses_var}, mis")
        else:
            # Generic calculation - user needs to specify in notes or recode_rules
            if mapping.recode_rules:
                # First rule might contain the formula
                formula = mapping.recode_rules[0].from_value
                lines.append(f"gen {mapping.cses_var} = {formula}")
            else:
                # Default to direct copy with comment for manual edit
                lines.append(f"* TODO: Specify calculation formula")
                lines.append(f"gen {mapping.cses_var} = {source}")
            lines.append(f"tab {mapping.cses_var}, mis")

        return GeneratedCode(
            cses_var=mapping.cses_var,
            code="\n".join(lines),
            has_verification=True
        )

    def generate_not_found(self, mapping: VariableMapping) -> GeneratedCode:
        """
        Generate code for variables not found in source data.

        Pattern: gen TARGET = MISSING_CODE
                 tab TARGET, mis
        """
        lines = [self.generate_header(mapping)]

        lines.append(f"* Variable not available in source data")
        if mapping.notes:
            lines.append(f"* Note: {mapping.notes}")

        # Determine appropriate missing code
        missing_type = self.get_missing_type(mapping.cses_var)
        missing_val = self.MISSING_CODES[missing_type]["missing"]

        lines.append(f"gen {mapping.cses_var} = {missing_val}")
        lines.append(f"tab {mapping.cses_var}, mis")

        return GeneratedCode(
            cses_var=mapping.cses_var,
            code="\n".join(lines),
            has_verification=True
        )

    def generate_administrative_information(self, mapping: VariableMapping) -> GeneratedCode:
        """Generate a guarded administrative variable block."""
        lines = [self.generate_header(mapping)]
        if mapping.notes:
            lines.append(f"* Note: {mapping.notes}")

        value = _administrative_value_from_notes(mapping.notes)
        source = mapping.source_var
        source_is_real = source and source not in {
            "ADMINISTRATIVE_INFORMATION",
            "DERIVED_METADATA",
            "EXTERNAL_INPUT_REQUIRED",
            "NOT_FOUND",
        }

        if source_is_real:
            lines.append(f"gen {mapping.cses_var} = {source}")
            lines.append(f"tab {source} {mapping.cses_var}, mis")
        elif value not in {"", None}:
            if _looks_numeric(value):
                lines.append(f"gen {mapping.cses_var} = {value}")
            else:
                escaped = str(value).replace('"', "'")
                lines.append(f'gen str {mapping.cses_var} = "{escaped}"')
            lines.append(f"tab {mapping.cses_var}, mis")
        else:
            missing_type = self.get_missing_type(mapping.cses_var)
            missing_val = self.MISSING_CODES[missing_type]["missing"]
            lines.append("* Administrative value requires processor review before final deposit.")
            lines.append(f"gen {mapping.cses_var} = {missing_val}")
            lines.append(f"tab {mapping.cses_var}, mis")

        return GeneratedCode(
            cses_var=mapping.cses_var,
            code="\n".join(lines),
            has_verification=True
        )

    def generate(self, mapping: VariableMapping) -> GeneratedCode:
        """
        Generate Stata code for a variable mapping.

        Dispatches to the appropriate template based on transform_type.
        """
        if mapping.transform_type == "administrative_information":
            return self.generate_administrative_information(mapping)
        if mapping.transform_type == "not_found" or mapping.source_var in ("NOT_FOUND", ""):
            return self.generate_not_found(mapping)
        elif mapping.transform_type == "direct":
            return self.generate_direct(mapping)
        elif mapping.transform_type == "recode":
            return self.generate_recode(mapping)
        elif mapping.transform_type == "replace":
            return self.generate_replace(mapping)
        elif mapping.transform_type == "calculate":
            return self.generate_calculate(mapping)
        else:
            # Default to direct
            return self.generate_direct(mapping)


def _administrative_value_from_notes(notes: str) -> str:
    match = re.search(r"Administrative value:\s*([^|]+)", notes or "")
    return match.group(1).strip() if match else ""


def _looks_numeric(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", str(value).strip()))
