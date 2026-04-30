"""
Tracking sheet reader for CSES variable mappings.

Reads the extended tracking sheet Excel file that contains:
- CSES target variable mappings
- Source variable information
- Transformation type and recode rules
- Human verification status
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import openpyxl

logger = logging.getLogger(__name__)


# Column indices for the extended tracking sheet (0-indexed)
# Main sheet columns
class SheetColumns:
    """Column positions in the tracking sheet."""
    CSES_DESC = 0       # A: Variable description
    CSES_VAR = 1        # B: CSES target variable code (e.g., F2002)
    SOURCE_VAR = 2      # C: Source variable name
    SOURCE_DESC = 3     # D: Source variable description
    SOURCE_VALUES = 4   # E: Value labels from source
    TRANSFORM = 5       # F: Transformation type (direct/recode/replace/calculate/not_found)
    RECODE_MAP = 6      # G: Value mapping rules (e.g., "1=1; 2=2; .=99")
    MISSING_MAP = 7     # H: Missing value mapping (e.g., ".=99; 97=97")
    CONFIDENCE = 8      # I: AI confidence level (high/medium/low)
    VERIFIED = 9        # J: Human verified (TRUE/FALSE)
    NOTES = 10          # K: Comments/notes


@dataclass
class RecodeRule:
    """A single value transformation rule."""
    from_value: str
    to_value: str
    label: str = ""

    def __str__(self) -> str:
        if self.label:
            return f"{self.from_value}={self.to_value} /* {self.label} */"
        return f"{self.from_value}={self.to_value}"


@dataclass
class VariableMapping:
    """Complete mapping for a single variable."""
    cses_var: str                   # Target CSES variable (e.g., F2002)
    cses_desc: str                  # Description (e.g., "Gender")
    source_var: str                 # Source variable name
    source_desc: str = ""           # Source variable description
    source_values: str = ""         # Value labels string
    transform_type: str = "direct"  # direct, recode, replace, calculate, not_found
    recode_rules: list[RecodeRule] = field(default_factory=list)
    missing_rules: list[RecodeRule] = field(default_factory=list)
    confidence: str = "medium"      # high, medium, low
    verified: bool = False          # Human verified
    notes: str = ""                 # Additional notes

    def is_ready_for_codegen(self) -> bool:
        """Check if this mapping is ready for code generation."""
        # Verified mappings are ready
        if self.verified:
            return True
        return False

    def needs_review(self) -> bool:
        """Check if this mapping needs human review."""
        if self.verified:
            return False
        return True


@dataclass
class TrackingSheet:
    """Complete tracking sheet data."""
    mappings: list[VariableMapping] = field(default_factory=list)
    country_code: str = ""
    year: str = ""
    source_file: str = ""

    def get_mapping(self, cses_var: str) -> Optional[VariableMapping]:
        """Get mapping for a specific CSES variable."""
        for m in self.mappings:
            if m.cses_var == cses_var:
                return m
        return None

    def get_verified_mappings(self) -> list[VariableMapping]:
        """Get all verified mappings."""
        return [m for m in self.mappings if m.verified]

    def get_ready_mappings(self) -> list[VariableMapping]:
        """Get all mappings ready for code generation."""
        return [m for m in self.mappings if m.is_ready_for_codegen()]

    def get_needs_review(self) -> list[VariableMapping]:
        """Get mappings that need human review."""
        return [m for m in self.mappings if m.needs_review()]

    def summary(self) -> str:
        """Get summary of tracking sheet status."""
        total = len(self.mappings)
        verified = len(self.get_verified_mappings())
        ready = len(self.get_ready_mappings())
        not_found = len([m for m in self.mappings if m.transform_type == "not_found"])
        needs_review = len(self.get_needs_review())

        return (
            f"Tracking Sheet: {self.country_code} {self.year}\n"
            f"  Total variables: {total}\n"
            f"  Verified: {verified}\n"
            f"  Ready for codegen: {ready}\n"
            f"  Not found: {not_found}\n"
            f"  Needs review: {needs_review}"
        )


class TrackingSheetReader:
    """
    Reads the extended tracking sheet Excel file.

    The tracking sheet has two formats:
    1. Legacy format (4 columns): VARIABLES, CSES code, CNT_YEAR, REMARKS
    2. Extended format (11 columns): Adds SOURCE_DESC, SOURCE_VALUES, TRANSFORM,
       RECODE_MAP, MISSING_MAP, CONFIDENCE, VERIFIED, NOTES

    This reader handles both formats.
    """

    def __init__(self):
        pass

    def read(self, file_path: Path) -> TrackingSheet:
        """
        Read tracking sheet from Excel file.

        Args:
            file_path: Path to the tracking sheet Excel file

        Returns:
            TrackingSheet with all mappings
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Tracking sheet not found: {file_path}")

        logger.info(f"Reading tracking sheet: {file_path.name}")

        wb = openpyxl.load_workbook(file_path, data_only=True)

        # Find the main sheet (try common names)
        sheet_names = ["deposited variables", "Variables", "Mappings", "Sheet1"]
        ws = None
        for name in sheet_names:
            if name in wb.sheetnames:
                ws = wb[name]
                break
        if ws is None:
            ws = wb.active

        # Parse country code and year from filename
        # Format: deposited variables-m6_CNT_YEAR_DATE.xlsx
        country_code = ""
        year = ""
        name_parts = file_path.stem.split("_")
        if len(name_parts) >= 3:
            country_code = name_parts[1] if len(name_parts) > 1 else ""
            year = name_parts[2] if len(name_parts) > 2 else ""

        # Detect format by checking header row
        header_row = self._find_header_row(ws)
        is_extended = self._is_extended_format(ws, header_row)
        header_map = self._header_map(ws, header_row)

        logger.info(f"Format detected: {'extended' if is_extended else 'legacy'}")

        # Parse mappings
        mappings = []
        start_row = header_row + 1

        for row in range(start_row, ws.max_row + 1):
            mapping = self._parse_row(ws, row, is_extended, header_map)
            if mapping and mapping.cses_var:
                mappings.append(mapping)

        logger.info(f"Read {len(mappings)} variable mappings")

        return TrackingSheet(
            mappings=mappings,
            country_code=country_code,
            year=year,
            source_file=str(file_path)
        )

    def _find_header_row(self, ws) -> int:
        """Find the header row (contains 'VARIABLES' or 'CSES')."""
        for row in range(1, min(10, ws.max_row + 1)):
            cell_a = ws.cell(row=row, column=1).value
            cell_b = ws.cell(row=row, column=2).value
            if cell_a and "VARIABLE" in str(cell_a).upper():
                return row
            if cell_b and "CSES" in str(cell_b).upper():
                return row
        return 1  # Default to row 1

    def _is_extended_format(self, ws, header_row: int) -> bool:
        """Check if this is the extended format with extra columns."""
        # Check if column F (TRANSFORM) has a header
        col_f = ws.cell(row=header_row, column=6).value
        if col_f and "TRANSFORM" in str(col_f).upper():
            return True
        # Check if there are more than 4 columns with data
        if ws.max_column > 6:
            return True
        return False

    def _header_map(self, ws, header_row: int) -> dict[str, int]:
        """Map normalized header text to 1-based column index."""
        mapping = {}
        for col in range(1, ws.max_column + 1):
            value = ws.cell(row=header_row, column=col).value
            if value:
                key = str(value).strip().upper()
                mapping[key] = col
        return mapping

    def _parse_row(self, ws, row: int, is_extended: bool, header_map: dict[str, int] = None) -> Optional[VariableMapping]:
        """Parse a single row into a VariableMapping."""
        header_map = header_map or {}

        def col(*names: str, default: int) -> int:
            for name in names:
                if name in header_map:
                    return header_map[name]
            return default

        # Get CSES variable code (column B)
        cses_var = self._get_cell_value(ws, row, col("CSES CODE", default=2))
        if not cses_var:
            return None

        # Skip header/label rows
        cses_var = str(cses_var).strip()
        if cses_var.upper() in ("CSES CODE", "CSES M6", "CSES M5", ""):
            return None
        if not cses_var.startswith("F"):
            return None

        # Get description (column A)
        cses_desc = self._get_cell_value(ws, row, col("VARIABLES", default=1)) or ""

        # Get source variable (column C)
        source_var = self._get_cell_value(ws, row, col("SOURCE VARIABLE(S)", "SOURCE_VAR", default=3)) or ""
        source_var = str(source_var).strip()

        # Clean up source variable
        if source_var in ("[not yet matched]", "X", "(X = missing)"):
            source_var = "NOT_FOUND"

        if "SOURCE EVIDENCE" in header_map or "DEPENDENCY CLASS" in header_map:
            # Schema-backed tracking sheet.
            source_desc = self._get_cell_value(ws, row, col("SOURCE EVIDENCE", default=4)) or ""
            source_values = ""
            recode_map = self._get_cell_value(ws, row, col("RECODING NOTE", default=7)) or ""
            missing_map = self._get_cell_value(ws, row, col("MISSING VALUE TREATMENT", default=8)) or ""
            confidence = self._get_cell_value(ws, row, col("CONFIDENCE", default=5)) or "medium"
            verified = self._parse_bool(self._get_cell_value(ws, row, col("VERIFIED", default=6)))
            notes_parts = [
                self._get_cell_value(ws, row, col("REMARKS", default=10)) or "",
                self._get_cell_value(ws, row, col("PROCESSOR DECISION", default=15)) or "",
            ]
            notes = " | ".join(str(part) for part in notes_parts if str(part).strip())
            recode_rules = self._parse_recode_map(recode_map)
            missing_rules = self._parse_recode_map(missing_map)
            if source_var in {"NOT_FOUND", "EXTERNAL_INPUT_REQUIRED", "DERIVED_METADATA", ""}:
                transform_type = "not_found"
            elif recode_rules or missing_rules:
                transform_type = "recode"
            else:
                transform_type = "direct"
        elif is_extended:
            # Extended format columns
            source_desc = self._get_cell_value(ws, row, 4) or ""
            source_values = self._get_cell_value(ws, row, 5) or ""
            transform_type = self._get_cell_value(ws, row, 6) or "direct"
            recode_map = self._get_cell_value(ws, row, 7) or ""
            missing_map = self._get_cell_value(ws, row, 8) or ""
            confidence = self._get_cell_value(ws, row, 9) or "medium"
            verified = self._parse_bool(self._get_cell_value(ws, row, 10))
            notes = self._get_cell_value(ws, row, 11) or ""

            # Parse recode and missing rules
            recode_rules = self._parse_recode_map(recode_map)
            missing_rules = self._parse_recode_map(missing_map)
        else:
            # Legacy format - derive what we can
            notes = self._get_cell_value(ws, row, 4) or ""
            source_desc = ""
            source_values = ""
            transform_type = "direct" if source_var and source_var != "NOT_FOUND" else "not_found"
            recode_rules = []
            missing_rules = []
            confidence = "medium"
            verified = False

        # Normalize transform type
        transform_type = str(transform_type).lower().strip()
        if transform_type not in ("direct", "recode", "replace", "calculate", "not_found", "administrative_information"):
            transform_type = "direct"

        if source_var == "NOT_FOUND" or not source_var:
            transform_type = "not_found"

        return VariableMapping(
            cses_var=cses_var,
            cses_desc=str(cses_desc),
            source_var=source_var,
            source_desc=str(source_desc),
            source_values=str(source_values),
            transform_type=transform_type,
            recode_rules=recode_rules,
            missing_rules=missing_rules,
            confidence=str(confidence).lower(),
            verified=verified,
            notes=str(notes)
        )

    def _get_cell_value(self, ws, row: int, col: int):
        """Get cell value, handling None."""
        value = ws.cell(row=row, column=col).value
        return value if value is not None else ""

    def _parse_bool(self, value) -> bool:
        """Parse boolean from various formats."""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        s = str(value).upper().strip()
        return s in ("TRUE", "YES", "1", "X", "Y")

    def _parse_recode_map(self, recode_str: str) -> list[RecodeRule]:
        """
        Parse recode mapping string into RecodeRule list.

        Formats supported:
        - "1=1; 2=2; 3=3"
        - "1=1 (Male); 2=2 (Female)"
        - "(1 2 3 = 1) (4 5 6 = 2)"  - Stata recode format
        """
        if not recode_str:
            return []

        rules = []
        recode_str = str(recode_str).strip()

        # Split by semicolon or newline
        parts = recode_str.replace("\n", ";").split(";")

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Parse "from=to" or "from=to (label)"
            if "=" in part:
                eq_parts = part.split("=", 1)
                from_val = eq_parts[0].strip()
                to_part = eq_parts[1].strip()

                # Check for label in parentheses
                label = ""
                if "(" in to_part and ")" in to_part:
                    paren_start = to_part.find("(")
                    paren_end = to_part.rfind(")")
                    label = to_part[paren_start+1:paren_end].strip()
                    to_val = to_part[:paren_start].strip()
                else:
                    to_val = to_part

                rules.append(RecodeRule(from_value=from_val, to_value=to_val, label=label))

        return rules


def read_tracking_sheet(file_path: Path) -> TrackingSheet:
    """Convenience function to read a tracking sheet."""
    reader = TrackingSheetReader()
    return reader.read(file_path)
