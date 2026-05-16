"""
Deterministic Stata code generator for CSES harmonization.

Reads a verified tracking sheet and generates a complete .do file
using template-based code generation. No AI is used in this process.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .sheet_reader import TrackingSheet, TrackingSheetReader, VariableMapping
from .code_templates import StataTemplates, GeneratedCode
from src.standards.schema import SchemaRegistry

logger = logging.getLogger(__name__)


# Variable ordering for .do file (follows CSES codebook structure)
VARIABLE_ORDER = [
    # ID and administration
    "F1001", "F1002_VER", "F1002_DOI", "F1003_1", "F1003_2", "F1004", "F1005",
    "F1006", "F1006_UN", "F1006_UNALPHA2", "F1006_UNALPHA3", "F1006_NAM",
    "F1007_REG", "F1007_OECD", "F1007_EU", "F1007_VDEM", "F1008", "F1009",
    "F1010_M", "F1010_D", "F1010_Y", "F1010_1", "F1010_2",
    "F1011_M", "F1011_D", "F1011_Y", "F1011_1", "F1011_2",
    "F1019_M", "F1019_D", "F1019_Y",
    "F1100", "F1101_1", "F1101_2", "F1101_3", "F1102",
    "F1103_1", "F1103_2", "F1103_3", "F1104", "F1105_1", "F1105_2", "F1105_3",
    "F1106",
    # Demographics
    "F2001_Y", "F2001_M", "F2001_A",
    "F2001_GG", "F2001_GS", "F2001_GBB", "F2001_GX", "F2001_GY", "F2001_GZ",
    "F2002", "F2003", "F2004", "F2005", "F2006", "F2007", "F2008", "F2009",
    "F2010_1", "F2010_2", "F2011", "F2012", "F2013", "F2014", "F2015", "F2016",
    "F2017", "F2018", "F2020", "F2021",
    # Survey questions
    "F3001",
    "F3002_1", "F3002_2", "F3002_3", "F3002_4", "F3002_5", "F3002_6_1",
    "F3003",
    "F3004_1", "F3004_2", "F3004_3", "F3004_4",
    "F3005_1", "F3005_2", "F3005_3",
    "F3006",
    "F3007_1", "F3007_2", "F3007_3", "F3007_4", "F3007_5", "F3007_6", "F3007_7",
    "F3008_1", "F3008_2",
    "F3009", "F3010",
    "F3011_LH_PL", "F3011_LH_DC", "F3011_LH_PL2", "F3011_UH_PL",
    "F3012_1", "F3012_2", "F3013", "F3014",
    "F3017",
    "F3018_A", "F3018_B", "F3018_C", "F3018_D", "F3018_E", "F3018_F",
    "F3018_G", "F3018_H", "F3018_I",
    "F3019_A", "F3019_B", "F3019_C", "F3019_D", "F3019_E", "F3019_F",
    "F3019_G", "F3019_H", "F3019_I",
    "F3020", "F3021", "F3022", "F3023", "F3024",
]


@dataclass
class GenerationResult:
    """Result of code generation."""
    success: bool
    output_path: Optional[Path] = None
    variables_generated: int = 0
    variables_skipped: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "Success" if self.success else "Failed"
        return (
            f"Code Generation: {status}\n"
            f"  Variables generated: {self.variables_generated}\n"
            f"  Variables skipped: {self.variables_skipped}\n"
            f"  Warnings: {len(self.warnings)}\n"
            f"  Errors: {len(self.errors)}"
        )


class StataCodeGenerator:
    """
    Generates complete Stata .do files from tracking sheets.

    This is a deterministic process - no AI is involved.
    The tracking sheet contains all information needed for code generation.
    """

    def __init__(self):
        self.templates = StataTemplates()
        self.schema = SchemaRegistry()

    def generate(
        self,
        tracking_sheet: TrackingSheet,
        output_path: Path,
        country_name: str = "",
        data_file_path: str = "",
        author: str = "CSES Agent"
    ) -> GenerationResult:
        """
        Generate a complete Stata .do file from a tracking sheet.

        Args:
            tracking_sheet: TrackingSheet with variable mappings
            output_path: Where to write the .do file
            country_name: Full country name for header
            data_file_path: Path to source data file
            author: Author name for header

        Returns:
            GenerationResult with status and statistics
        """
        result = GenerationResult(success=True)
        country_code = tracking_sheet.country_code or "CNT"
        year = tracking_sheet.year or "YEAR"

        if not country_name:
            country_name = country_code

        logger.info(f"Generating Stata code for {country_name} {year}")

        # Build the complete .do file
        sections = []

        # 1. File header
        sections.append(self._generate_header(
            country_code=country_code,
            country_name=country_name,
            year=year,
            author=author,
            date="REPRODUCIBLE"
        ))

        # 2. Open data section
        sections.append(self._generate_open_data(data_file_path))

        # 3. Frequencies section
        sections.append(self._generate_frequencies_section(country_code, year))

        # 4. Variable sections by full schema order.
        generated_names: set[str] = set()
        section_titles = {
            "administration": "ID, WEIGHT, AND ADMINISTRATION VARIABLES",
            "demographics": "DEMOGRAPHIC VARIABLES",
            "survey_questions": "SURVEY QUESTIONS",
            "district_data": "DISTRICT DATA VARIABLES",
            "party_leader_macro": "PARTY, LEADER, AND MACRO LINK VARIABLES",
            "integrated_vote_choice": "INTEGRATED VOTE CHOICE VARIABLES",
            "other": "OTHER CSES VARIABLES",
        }
        for section_id, title in section_titles.items():
            section_vars = [item for item in self.schema.variables if item.section == section_id]
            if not section_vars:
                continue
            sections.append(self._section_header(title))
            for schema_var in section_vars:
                mapping = tracking_sheet.get_mapping(schema_var.name)
                if not mapping:
                    mapping = VariableMapping(
                        cses_var=schema_var.name,
                        cses_desc=schema_var.description,
                        source_var="NOT_FOUND",
                        source_desc="",
                        transform_type="not_found",
                        confidence="processor_review",
                        verified=False,
                        notes=(
                            "Schema-required variable not resolved in tracking sheet. "
                            "Processor review required before final deposit."
                        ),
                    )
                    result.variables_skipped += 1
                    result.warnings.append(f"No mapping for {schema_var.name}")
                code = self.templates.generate(mapping)
                sections.append(code.code)
                generated_names.add(schema_var.name)
                result.variables_generated += 1

        # 5. Footer (drop temp vars, save, close log)
        sections.append(self._generate_footer(country_code, year, generated_names))

        # Write the file
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            full_code = "\n".join(sections)
            output_path.write_text(full_code, encoding="utf-8")
            result.output_path = output_path
            logger.info(f"Generated .do file: {output_path}")
        except Exception as e:
            result.success = False
            result.errors.append(f"Failed to write file: {e}")
            logger.error(f"Failed to write .do file: {e}")

        return result

    def _generate_header(
        self,
        country_code: str,
        country_name: str,
        year: str,
        author: str,
        date: str
    ) -> str:
        """Generate the file header."""
        return f"""/***************************************************************************
**                     Process CSES-M6 Micro-Data                         **
**                     **************************                         **
**                                                                        **
** File Author:      {author:<56}**
** Build:            {date:<56}**
** CSES MODULE 6:    {country_name.upper()} {year:<46}**
**                                                                        **
***************************************************************************/

*-------------------------------------------------------------------------**
****************************************************************************
**\\\\\\              SYNTAX INSTRUCTIONS                                    **
****************************************************************************
*-------------------------------------------------------------------------**
*                                                                         **
* 1) FILE NAVIGATION: The Syntax File can be navigated using different    **
*         combinations of symbols. "\\\\\\\" will direct the user to          **
*         the different sections of the file and ">>>" allows to          **
*         jump from variable to variable.                                 **
*         \\\\\\\\ Section Headings                                            **
*         >>> Variable Headings                                           **
**                                                                        **
* 2) This file is generated deterministically from the tracking sheet.    **
*    Re-running code generation with the same tracking sheet produces     **
*    the same syntax and output names. Review and adjust mappings in the  **
*    tracking sheet rather than hand-editing generated code.              **
**                                                                        **
***************************************************************************/
"""

    def _generate_open_data(self, data_file_path: str) -> str:
        """Generate the open data section."""
        data_suffix = Path(data_file_path).suffix.lower()
        if data_suffix == ".csv":
            open_command = f'import delimited using "{data_file_path}", clear varnames(1) bindquote(strict) stringcols(_all)'
        elif data_suffix in {".xlsx", ".xls"}:
            open_command = f'import excel using "{data_file_path}", clear firstrow'
        else:
            open_command = f'use "{data_file_path}", clear'

        return f"""
*-------------------------------------------------------------------------*
***************************************************************************
**\\\\\\              OPEN DATA
***************************************************************************
*-------------------------------------------------------------------------*

clear
set more off
capture log close

* Open File
* NOTE: Adjust path to match your working directory
{open_command}

* Drop dataset label if present
capture label data ""
"""

    def _generate_frequencies_section(self, country_code: str, year: str) -> str:
        """Generate the frequencies section."""
        return f"""
*-------------------------------------------------------------------------*
***************************************************************************
**\\\\\\              FREQUENCIES OF ORIGINAL DATA
***************************************************************************
*-------------------------------------------------------------------------*

* Run and save original frequencies
capture mkdir "./frequencies"
log using "./frequencies/cses-m6_org-freq_{country_code}_{year}.smcl", replace
foreach var of varlist _all {{
display "Variable `var':"
tab `var', mis
display ""
}}
log close
"""

    def _section_header(self, title: str) -> str:
        """Generate a section header."""
        return f"""
*-------------------------------------------------------------------------*
***************************************************************************
**\\\\\\              {title}
***************************************************************************
*-------------------------------------------------------------------------*
"""

    def _generate_admin_variables(
        self,
        country_code: str,
        country_name: str,
        year: str
    ) -> str:
        """Generate administration variables (F1xxx) with standard values."""
        # These are mostly fixed/calculated values, not from tracking sheet
        return f"""
***************************************************************************
**>>> F1001 - DATASET -> string variable
***************************************************************************

gen str13 F1001 = "CSES-MODULE-6"
tab F1001, mis


***************************************************************************
**>>> F1002_VER - DATASET VERSION -> string variable
***************************************************************************

gen str14 F1002_VER = "VER2025-MMM-DD"
tab F1002_VER, mis


***************************************************************************
**>>> F1002_DOI - DATASET DIGITAL OBJECT IDENTIFIER (DOI) -> string variable
***************************************************************************

gen str35 F1002_DOI = "doi:10.7804/cses.module6.2025-MM-DD"
tab F1002_DOI, mis


***************************************************************************
**>>> F1004 - ID VARIABLE - ELECTION STUDY (ALPHABETIC POLITY)
***************************************************************************

gen str F1004 = "{country_code}_{year}"
tab F1004, mis


***************************************************************************
**>>> F1009 - ID COMPONENT - ELECTION YEAR
***************************************************************************

gen F1009 = {year}
tab F1009, mis


***************************************************************************
**>>> F1106 - MODULE 6 RELEASE CYCLE
***************************************************************************

gen F1106 = 1  // 1=Advance Release 1
tab F1106, mis
"""

    def _generate_footer(self, country_code: str, year: str, generated_names: set[str] | None = None) -> str:
        """Generate the file footer."""
        order_line = ""
        if generated_names:
            ordered = [name for name in self.schema.ordered_names() if name in generated_names]
            chunks = []
            current = []
            for name in ordered:
                current.append(name)
                if len(current) >= 18:
                    chunks.append(" ".join(current))
                    current = []
            if current:
                chunks.append(" ".join(current))
            order_line = "\n".join(["order " + chunks[0]] + ["order " + chunk + ", after(" + chunks[i - 1].split()[-1] + ")" for i, chunk in enumerate(chunks[1:], 1)])
        return f"""
*-------------------------------------------------------------------------*
***************************************************************************
**\\\\\\              FINALIZE AND SAVE
***************************************************************************
*-------------------------------------------------------------------------*

* Apply final CSES variable order
{order_line}

* Drop temporary variables (if any)
capture drop *_temp
capture drop *_str

* Apply release-specific labels and checks when available
capture label define __cses_generated_placeholder 999 "Missing or not available", replace
capture do "./labels/cses-m6_label-updates_{country_code}_{year}.do"
capture do "./data_checks/validation_checks.do"
capture do "./data_checks/missing_value_checks.do"
capture do "./data_checks/party_code_checks.do"
capture do "./data_checks/district_checks.do"

* Keep only CSES release variables
keep {(" ".join([name for name in self.schema.ordered_names() if name in generated_names]) if generated_names else "")}

* Save processed dataset
save "./cses-m6_micro_{country_code}_{year}.dta", replace

* Close log
capture log close

* End of file
"""


def generate_stata_code(
    tracking_sheet_path: Path,
    output_path: Optional[Path] = None,
    country_name: str = "",
    data_file_path: str = "",
    author: str = "CSES Agent"
) -> GenerationResult:
    """
    Convenience function to generate Stata code from a tracking sheet.

    Args:
        tracking_sheet_path: Path to the tracking sheet Excel file
        output_path: Where to write the .do file (auto-generated if not provided)
        country_name: Full country name
        data_file_path: Path to source data file
        author: Author name for header

    Returns:
        GenerationResult with status and file path
    """
    # Read tracking sheet
    reader = TrackingSheetReader()
    tracking_sheet = reader.read(tracking_sheet_path)

    # Auto-generate output path if not provided
    if output_path is None:
        output_name = f"cses-m6_micro_{tracking_sheet.country_code}_{tracking_sheet.year}.do"
        output_path = tracking_sheet_path.parent.parent / output_name

    # Generate code
    generator = StataCodeGenerator()
    return generator.generate(
        tracking_sheet=tracking_sheet,
        output_path=output_path,
        country_name=country_name or tracking_sheet.country_code,
        data_file_path=data_file_path,
        author=author
    )
